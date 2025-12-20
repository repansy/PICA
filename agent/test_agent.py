import math
import numpy as np
from typing import List, Tuple, Deque
from collections import deque
import enviroments.config as cfg
from utils.pica_structures import Vector3D, Plane
from utils.linear_solver import linear_program3, linear_program4

class BCOrcaAgent:
    def __init__(self, id: int, pos: Vector3D, goal: Vector3D, **kwargs):
        self.id = id
        self.pos = pos
        self.goal = goal
        self.radius: float = kwargs.get('radius', cfg.AGENT_RADIUS)
        self.neighbor_dist: float = kwargs.get('neighbor_dist', cfg.NEIGHBOR_DIST)
        self.time_horizon: float = kwargs.get('time_horizon', cfg.TIME_HORIZON)
        self.max_neighbors: int = kwargs.get('max_neighbors', cfg.MAX_NEIGHOBORS)

        # --- 核心改进：P 定义了 速度 和 避让意愿 ---
        # P ∈ [0.1, 1.0]. P越小 -> 越灵活(速度快) -> 责任大(Alpha大)
        self.P: float = np.clip(kwargs.get('P', 0.5), 0.1, 1.0)
        
        # 速度与 P 成反比：P=0.1 速度快(2.0x), P=1.0 速度慢(1.0x)
        # 这样快的智能体才有能力去绕开慢的，防止木桶效应
        self.base_speed = cfg.MAX_SPEED * 0.7
        speed_multiplier = 1.0 + (1.0 - self.P) 
        self.max_speed = self.base_speed * speed_multiplier

        # --- 状态记忆 (仅用于慢脑) ---
        # 记录最近 5 帧 (pos, vel)
        self.history: Deque[Tuple[Vector3D, Vector3D]] = deque(maxlen=5) 
        self.stuck_timer: int = 0  # 记录死锁时长

        # --- 运行时变量 ---
        self.vel = Vector3D()
        self.new_velocity = Vector3D()
        self.pref_velocity = Vector3D()
        self.agent_neighbors: List['BCOrcaAgent'] = []
        self.orca_planes: List[Plane] = []
        self.at_goal = False

    # =================================================================================
    # System 1: 快脑 (Fast Brain) - 纯反射
    # =================================================================================
    def _run_fast_brain(self, other: 'BCOrcaAgent') -> float:
        """
        快脑：基于“能力越强，责任越大”的原则直接计算 Alpha。
        计算复杂度: O(1)
        """
        # 责任权重 = 1 / P (P越小，权重越大，承担的 Alpha 越多)
        w_self = 1.0 / self.P
        w_other = 1.0 / other.P
        
        # 归一化责任分配
        alpha = w_self / (w_self + w_other)
        
        # 加上微小的扰动防止完美对称导致的死锁
        return np.clip(alpha, 0.05, 0.95)

    # =================================================================================
    # System 2: 慢脑 (Slow Brain) - 推理与预测
    # =================================================================================
    def _run_slow_brain(self, other: 'BCOrcaAgent', dist_sq: float) -> Tuple[Vector3D, float]:
        """
        慢脑：处理复杂交互、预测意图、打破死锁。
        返回: (预测后的相对速度, 调整后的 Alpha)
        """
        # 1. 意图预测 (Prediction)
        # 如果对方在过去几帧都在减速或转向，预测它会继续这么做
        # 简单的线性预测往往不够，这里加入惯性权重
        if len(other.history) >= 2:
            prev_pos, prev_vel = other.history[-2]
            curr_pos, curr_vel = other.history[-1]
            accel = curr_vel - prev_vel
            # 预测速度 = 当前速度 + 惯性趋势
            pred_vel = other.vel + accel * 0.5 
        else:
            pred_vel = other.vel

        # 2. 死锁处理 (Deadlock Breaking)
        # 如果我很灵活 (Low P) 且我很堵 (stuck_timer high)，我必须采取极端措施
        # 强制 Alpha = 1.0 (完全避让) 甚至 Alpha > 1.0 (过度避让以拉开空间)
        alpha = self._run_fast_brain(other) # 获取基础 Alpha
        
        if self.stuck_timer > 10 and self.P < 0.5:
            # 我很急且被堵住了 -> 激进避让
            alpha = 1.0
            # 甚至假设对方会加速冲过来，从而让我让出更多空间
            pred_vel = pred_vel * 1.2 
        
        return pred_vel, alpha

    # =================================================================================
    # ORCA 主逻辑
    # =================================================================================
    def compute_new_velocity(self):
        self.orca_planes.clear()
        inv_tau = 1.0 / self.time_horizon
        
        # 1. 检测环境拥挤度，决定启用快脑还是慢脑
        # 简单判定：如果有邻居距离小于 1.5倍半径和，视为拥挤
        is_crowded = False
        safe_dist_sq = (self.radius * 3.0) ** 2
        if len(self.agent_neighbors) > 0:
             if (self.agent_neighbors[0].pos - self.pos).norm_sq() < safe_dist_sq:
                 is_crowded = True

        for other in self.agent_neighbors:
            # --- 决策核心：快慢脑切换 ---
            if is_crowded:
                # 慢脑：消耗更多算力，进行预测和状态分析
                other_vel_opt, alpha = self._run_slow_brain(other, 0.0)
            else:
                # 快脑：直接读数据，套公式，速度极快
                other_vel_opt = other.vel
                alpha = self._run_fast_brain(other)
            
            # --- 以下为标准 ORCA 几何构建 (VO) ---
            relative_pos = other.pos - self.pos
            relative_vel = self.vel - other_vel_opt # 使用（可能被预测修正过的）速度
            dist_sq = relative_pos.norm_sq()
            combined_radius = self.radius + other.radius
            combined_radius_sq = combined_radius**2

            plane = Plane()
            u = Vector3D()

            if dist_sq > combined_radius_sq:
                w = relative_vel - inv_tau * relative_pos
                w_len_sq = w.norm_sq()
                dot = w.dot(relative_pos)
                if dot < 0.0 and dot**2 > combined_radius_sq * w_len_sq:
                    w_len = math.sqrt(w_len_sq)
                    unit_w = w / w_len
                    plane.normal = unit_w
                    u = (combined_radius * inv_tau - w_len) * unit_w
                else:
                    a = dist_sq
                    b = relative_pos.dot(relative_vel)
                    c = relative_vel.norm_sq() - (relative_pos.norm_sq() * relative_vel.norm_sq() - b**2) / (dist_sq - combined_radius_sq)
                    discriminant = b**2 - a * c
                    if discriminant < 0: continue
                    t = (b + math.sqrt(discriminant)) / a
                    ww = relative_vel - t * relative_pos
                    ww_len = ww.norm()
                    unit_ww = ww / ww_len
                    plane.normal = unit_ww
                    u = (combined_radius * t - ww_len) * unit_ww
            else:
                w = relative_vel - (1.0/cfg.TIME_STEP) * relative_pos # 碰撞处理步长
                w_len = w.norm()
                unit_w = w / w_len
                plane.normal = unit_w
                u = (combined_radius * (1.0/cfg.TIME_STEP) - w_len) * unit_w

            # --- 应用 Alpha ---
            # point = v_opt + alpha * u
            # 这里的 v_opt 通常是 self.vel，因为 u 是基于 v_A - v_B 计算的避障向量
            plane.point = self.vel + alpha * u
            self.orca_planes.append(plane)

        # 线性规划求解
        fail_idx, self.new_velocity = linear_program3(self.orca_planes, self.max_speed, self.pref_velocity, False)
        if fail_idx < len(self.orca_planes):
            self.new_velocity = linear_program4(self.orca_planes, fail_idx, self.max_speed, self.new_velocity)

    def update(self, dt: float):
        if self.at_goal:
            self.vel = Vector3D()
            return

        # 更新位置
        self.vel = self.new_velocity
        self.pos += self.vel * dt
        
        # 记录历史 (用于慢脑预测)
        self.history.append((self.pos, self.vel))

        # 更新死锁计时器
        if self.vel.norm_sq() < 0.01 and not self.at_goal:
            self.stuck_timer += 1
        else:
            self.stuck_timer = 0

        # 判断是否到达
        if (self.goal - self.pos).norm_sq() < self.radius**2:
            self.at_goal = True
            self.vel = Vector3D()

    # 标准辅助函数
    def compute_preferred_velocity(self):
        if self.at_goal:
            self.pref_velocity = Vector3D()
            return
        to_goal = self.goal - self.pos
        dist = to_goal.norm()
        if dist < cfg.EPSILON:
             self.pref_velocity = Vector3D()
        else:
            # P越小，Max Speed越大，pref_vel也就越大
            self.pref_velocity = (to_goal / dist) * min(self.max_speed, dist)

    def compute_neighbors(self, all_agents):
        self.agent_neighbors.clear()
        neighbors_dist = []
        range_sq = self.neighbor_dist**2
        for agent in all_agents:
            if agent.id != self.id:
                d_sq = (self.pos - agent.pos).norm_sq()
                if d_sq < range_sq:
                    neighbors_dist.append((d_sq, agent))
        neighbors_dist.sort(key=lambda x: x[0])
        self.agent_neighbors = [a for _, a in neighbors_dist[:self.max_neighbors]]