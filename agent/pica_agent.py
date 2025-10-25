# -*- coding: utf-8 -*-

import math
import random
import numpy as np
from typing import List, Dict, Optional

from utils.pica_structures import Vector3D
from enviroments import config as cfg
# from agent.optimizer import Optimizer

class PicaAgent:
    def __init__(self, id: int, pos: Vector3D, goal: Vector3D, inertia_matrix: np.ndarray, priority: float = 1.0, **kwargs):
        """
        初始化智能体
        
        参数:
            id: 智能体唯一标识符
            pos: 初始位置
            goal: 目标位置
            inertia_matrix: 惯性矩阵，表示智能体的运动特性（质量、转动惯量等）
            priority: 任务优先级，值越高优先级越高
        """
        # --- 核心ID与任务属性 ---
        self.id = id
        self.goal = goal
        self.priority = priority
        self.inertia_matrix = inertia_matrix

        # --- 物理属性 ---
        self.radius = cfg.AGENT_RADIUS
        self.max_speed = cfg.MAX_SPEED
        self.is_colliding = False

        # --- 概率状态表示 ---
        self.pos = pos
        self.vel = Vector3D()
        self.mu_pos = pos
        self.mu_vel = Vector3D()
        self.cov_pos = np.eye(3) * 0.01

        # --- 算法内部状态 ---
        self.should_contour = True
        self.alphas: Dict[int, float] = {}
        self.at_goal = False
        self.optimzer = 1 #TODO:完善快慢脑算法

    def update(self, new_velocity: Vector3D, dt: float):
        """
        更新智能体状态（位置、速度、不确定性）
        
        参数:
            new_velocity: 计算得到的新速度
            dt: 时间步长
        """
        if self.at_goal:
            self.mu_vel = Vector3D(0, 0, 0)
            return

        if new_velocity.norm() > self.max_speed:
            new_velocity = new_velocity.normalized() * self.max_speed
        
        self.mu_vel = new_velocity
        self.mu_pos += self.mu_vel * dt

        process_noise = np.eye(3) * cfg.PROCESS_NOISE_FACTOR
        self.cov_pos += process_noise * dt

        if (self.goal - self.mu_pos).norm() < self.radius:
            self.at_goal = True
            self.mu_vel = Vector3D(0, 0, 0)
        
        self.pos = self.mu_pos
        self.vel = self.mu_vel

    def compute_new_velocity(self, all_agents: List['PicaAgent'], dt: float) -> Vector3D:
        if self.at_goal: return Vector3D()
        
        # 1. 初始化和邻居筛选
        neighbors = self._filter_neighbors(all_agents)
        v_pref = self._break_deadlock(self._get_preferred_velocity(), neighbors)
        
        # --- 中央融合与约束生成 ---
        constraints = []
        for neighbor in neighbors:
            effective_states = self._get_effective_state_for_interaction(neighbor)
            threat_score = self._calculate_threat(self, neighbor)

            # TODO 在此输入更多的参数以满足绘制需要
            # alpha = self.optimzer._optimize_alpha_hybrid(neighbor, threat_score, effective_states, v_pref)
            alpha = 0.5

            constraint = self._create_pica_halfspace_from_states(alpha, effective_states)
            if constraint:
                constraints.append(constraint)
        #TODO linear-program
        v_pica = self._solve_velocity_3d(constraints, v_pref)
        return v_pica
    
    def _get_effective_state_for_interaction(self, neighbor: 'PicaAgent') -> Dict:
            """
            为交互计算保守的"有效状态"（核心创新1：概率感知）
            将不确定的位置表示为高斯分布，并计算最坏情况下的有效位置，将概率问题转化为确定的几何问题。
            参数:
                neighbor: 邻居智能体
            返回:
                Dict: 包含双方有效状态的字典
            """
            dir_to_neighbor = (neighbor.mu_pos - self.mu_pos).normalized()
            if dir_to_neighbor.norm_sq() < 1e-9:
                dir_to_neighbor = Vector3D(1, 0, 0) # 避免除零，使用默认方向
            
            # 计算自身位置在相对方向上的标准差
            std_dev_i = math.sqrt(dir_to_neighbor.to_numpy().T @ self.cov_pos @ dir_to_neighbor.to_numpy())
            # 计算保守的有效位置（沿相对方向向内收缩）
            effective_pos_i = self.mu_pos - dir_to_neighbor * (cfg.UNCERTAINTY_CONFIDENCE_N * std_dev_i)

            # 计算邻居位置在相反方向上的标准差
            dir_to_self = -dir_to_neighbor
            std_dev_j = math.sqrt(dir_to_self.to_numpy().T @ neighbor.cov_pos @ dir_to_self.to_numpy())
            effective_pos_j = neighbor.mu_pos - dir_to_self * (cfg.UNCERTAINTY_CONFIDENCE_N * std_dev_j)

            return {
                "pos_i": effective_pos_i, "vel_i": self.mu_vel, "radius_i": self.radius,
                "pos_j": effective_pos_j, "vel_j": neighbor.mu_vel, "radius_j": neighbor.radius
            }
    
    def _create_pica_halfspace_from_states(self, alpha: float, states: Dict) -> Optional[Dict]:
        """
        创建ORCA半平面约束
        根据责任参数α和有效状态，创建ORCA半平面约束。
        参数:
            alpha: 责任参数
            states: 有效状态字典    
        返回:
            Optional[Dict]: ORCA约束字典（法线和偏移量），如果无效则返回None
        """
        pos_i, vel_i, r_i = states["pos_i"], states["vel_i"], states["radius_i"]
        pos_j, vel_j, r_j = states["pos_j"], states["vel_j"], states["radius_j"]

        rel_pos = pos_j - pos_i
        rel_vel = vel_i - vel_j
        dist_sq = rel_pos.norm_sq()
        combined_radius = r_i + r_j
        combined_radius_sq = combined_radius ** 2

        if dist_sq < combined_radius_sq:
            # 碰撞处理：强制分离
            inv_time_horizon = 1.0 / cfg.TIMESTEP
            u = (rel_pos.normalized() * (combined_radius - rel_pos.norm())) * inv_time_horizon
            normal = u.normalized()
        else:
            inv_tau = 1.0 / cfg.TIMESTEP
            vo_apex = rel_pos * inv_tau # VO锥中心
            vo_radius_sq = (combined_radius * inv_tau) ** 2# VO锥半径平方
            
            w = rel_vel - vo_apex # 相对速度相对于VO锥中心的向量
            w_norm_sq = w.norm_sq()

            dot_product = w.dot(rel_pos)
            if not self.should_contour: #TODO what a import one
                print("2222")
                return None

            if w_norm_sq <= vo_radius_sq:
                normal = w.normalized() if w.norm_sq() > 1e-9 else -rel_pos.normalized()
                u = normal * (math.sqrt(vo_radius_sq) - math.sqrt(w_norm_sq))
            else:
                normal = (w - rel_pos * (dot_product / dist_sq)).normalized()
                u = normal * (w.dot(normal))
        
        u_np = u.to_numpy()                  # 转换为列向量
        M_u_np = self.inertia_matrix @ u_np  # M*u的矩阵运算
        u = Vector3D.from_numpy(M_u_np)  # 转换回向量对象
        
        plane_point = vel_i + u * (1 - alpha)
        offset = plane_point.dot(normal)
        
        return {'normal': normal, 'offset': offset}

    def _calculate_threat(self, agent_i: 'PicaAgent', agent_j: 'PicaAgent') -> float:
        """
        基于状态计算风险评分，综合考虑距离风险和碰撞时间风险。
        参数: states: 状态字典
        返回: float: 风险评分
        """
        states = {"pos_i": agent_i.mu_pos, "vel_i": agent_i.mu_vel, "pos_j": agent_j.mu_pos, "vel_j": agent_j.mu_vel}
        rel_pos = states["pos_j"] - states["pos_i"]
        dist = rel_pos.norm()
        if dist < 1e-6: return float('inf')
        rel_vel = states["vel_i"] - states["vel_j"]
        vel_dot_pos = rel_vel.dot(rel_pos)
        if vel_dot_pos <= 0:
            # 正在远离或无接近分量，只有距离风险
            return cfg.RISK_W_DIST / dist
        rel_vel_sq = rel_vel.norm_sq()
        if rel_vel_sq < 1e-6:
            # 计算碰撞时间（TTC）
            return cfg.RISK_W_DIST / dist
        ttc = vel_dot_pos / rel_vel_sq
        # 综合风险 = 距离风险 + 碰撞时间风险（越短风险越高）
        return cfg.RISK_W_DIST / dist + cfg.RISK_W_TTC / (ttc + 0.1)

    def _solve_velocity_3d(self, constraints: List[Dict], v_pref: Vector3D) -> Vector3D:
        """
        备用计划
        """
        v_new = v_pref
        for _ in range(50):
            for const in constraints:
                n, offset = const['normal'], const['offset']
                n_norm_sq = n.norm_sq()
                if n_norm_sq < 1e-9: continue
                if v_new.dot(n) < offset:
                    v_new += n * (offset - v_new.dot(n)) / n_norm_sq
        if v_new.norm() > self.max_speed:
            v_new = v_new.normalized() * self.max_speed
        return v_new
    
    def _get_preferred_velocity(self) -> Vector3D:
        """
        计算首选速度（指向目标的速度）
        返回: Vector3D: 首选速度向量
        """
        # 速度幅值
        to_goal = self.goal - self.mu_pos
        dist_to_goal = to_goal.norm()
        if dist_to_goal < 1e-6:
            return Vector3D(0, 0, 0)
        pref_speed = min(self.max_speed, dist_to_goal) 
        return to_goal.normalized() * pref_speed

    def _filter_neighbors(self, all_agents: List['PicaAgent']) -> List['PicaAgent']:
        """
        过滤出在交互范围内的邻居
        参数:
            all_agents: 所有智能体列表   
        返回:
            List[Agent]: 在交互范围内的邻居列表
        """
        neighbors = []
        # 计算交互范围（基于时间范围和最大速度 TODO:here maxspeed->speed
        interaction_horizon = cfg.TTC_HORIZON * 2 * self.max_speed
        for agent in all_agents:
            if agent.id == self.id:
                continue
            # 计算距离平方（避免开方计算，提高效率）
            dist_sq = (self.mu_pos - agent.mu_pos).norm_sq()
            if dist_sq < interaction_horizon ** 2:
                neighbors.append(agent)
        return neighbors
    
    def _break_deadlock(self, v_pref: Vector3D, neighbors: List['PicaAgent']) -> Vector3D:
        """
        打破对称死锁, 通过添加微小扰动，避免智能体在场景中僵持。
        参数:
            v_pref: 首选速度
            neighbors: 邻居列表
        返回:
            Vector3D: 可能扰动后的速度
        """
        if self.mu_vel.norm_sq() > 0.1 and neighbors:
            # 转动小角度，随机转动
            perturb_angle = 0.015 * (random.randint(-10, 10))
            # perturb_angle = (self.id % 20 - 10) * 0.015
            # perturb_angle = 0
            c, s = math.cos(perturb_angle), math.sin(perturb_angle)
            px = v_pref.x * c - v_pref.y * s
            py = v_pref.x * s + v_pref.y * c
            v_pref_perturbed = Vector3D(px, py, v_pref.z)
            # 再次限制速度
            if v_pref_perturbed.norm() > self.max_speed:
                return v_pref_perturbed.normalized() * self.max_speed
            return v_pref_perturbed
        return v_pref