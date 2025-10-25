# pica_agent.py
import numpy as np
from typing import List, Dict
import math

from utils.pica_structures import Vector3D, SlowBrainPolicy
from agent.fast_brain import FastBrain
from agent.slow_brain import SlowBrain
import enviroments.config as cfg

class Agent:
    def __init__(self, id: int, pos: Vector3D, goal: Vector3D, inertia_matrix: np.ndarray, priority: float = 1.0):
        # --- 基础属性 ---
        self.id = id
        self.goal = goal
        self.priority = priority
        self.inertia_matrix = inertia_matrix

        self.radius = cfg.AGENT_RADIUS
        self.max_speed = cfg.MAX_SPEED
        self.pos = pos
        self.vel = Vector3D(0, 0, 0)
        self.cov_pos = np.eye(3) * 0.01
        self.at_goal = False
        self.is_colliding = False
        
        # --- 核心模块 ---
        self.fast_brain = FastBrain(self)
        self.slow_brain = SlowBrain(self) # 慢脑需要访问自身属性
        
        # --- 内部状态 ---
        self.v_pref = Vector3D(0,0,0)
        self.rho_smoothed = 0.0
        self.count = 0
        self.slow_policy = SlowBrainPolicy()

    def update(self, new_velocity: Vector3D, dt: float):
        if self.at_goal: self.vel = Vector3D(0,0,0); return
        self.vel = new_velocity
        if self.vel.norm() > self.max_speed:
            self.vel = self.vel.normalized() * self.max_speed
        self.pos += self.vel * dt
        
        process_noise = np.eye(3) * cfg.PROCESS_NOISE_FACTOR
        self.cov_pos += process_noise * dt
        
        if (self.goal - self.pos).norm() < self.radius: self.at_goal = True
        
    def compute_new_velocity(self, all_agents: List['Agent'], dt: float) -> Vector3D:
        """F²-HRA 决策主流程"""
        # 1. 感知
        neighbors = self._filter_neighbors(all_agents)
        self.v_pref = self._get_preferred_velocity()
        
        # 预计更改优化
        
        # initial_risk_scores = {n.id: self._calculate_risk(self, n) for n in neighbors}
        # sorted_neighbors = sorted(neighbors, key=lambda n: initial_risk_scores[n.id], reverse=True)
        
        # critical_neighbors = sorted_neighbors[:cfg.PICA_K]
        # other_neighbors = sorted_neighbors[cfg.PICA_K:]
        
        self._update_local_density(neighbors)
        
        # 2. 慢脑内部思考 (高频)
        self.slow_brain.think(neighbors, dt)
        
        # 4. 决策融合与求解 (低频)
        #    a. 从慢脑获取最新的低频策略
        if self.count % 5 == 0:
            slow_policy = self.slow_brain.get_policy()    
        else:
            slow_policy = self.slow_policy
        self.count += 1
        
        # 3. 快脑构建安全集 (高频)
        safe_constraints = self.fast_brain.compute_safe_velocity_set(neighbors, dt)
        
        #    b. 运行QP求解器
        final_velocity = self._solve_qp(safe_constraints, slow_policy)

        return final_velocity
    
    def _solve_qp(self, constraints: List[Dict], policy: SlowBrainPolicy) -> Vector3D:
        """
        最终的QP求解器。
        在快脑定义的V_safe内，寻找一个最能平衡v_pref和慢脑v_ideal的速度。
        """
        # TODO: 此处应调用一个真正的QP库 (e.g., CVXPY, OSQP)
        # 以下是一个概念性的简化实现
        
        # 1. 确定目标速度 (融合v_pref和v_ideal)
        target_vel = (self.v_pref * cfg.QP_WEIGHT_GOAL + policy.v_ideal * cfg.QP_WEIGHT_IDEAL) / \
                     (cfg.QP_WEIGHT_GOAL + cfg.QP_WEIGHT_IDEAL)
        
        # 2. 在约束内求解 (使用迭代投影法作为QP的简化替代)
        v_new = target_vel
        for _ in range(10):
            for const in constraints:
                n, offset = const['normal'], const['offset']
                if v_new.dot(n) < offset:
                    v_new += n * (offset - v_new.dot(n)) / n.norm_sq()
        
        if v_new.norm() > self.max_speed:
            v_new = v_new.normalized() * self.max_speed
            
        return v_new

    # (此处应包含Agent自身的基础辅助函数: _filter_neighbors, 
    #  _get_preferred_velocity, _get_effective_state_for_interaction, 等。
    #  这些函数与之前版本相同，故省略)
    
    def _get_preferred_velocity(self) -> Vector3D:
        """
        计算首选速度（指向目标的速度）
        返回: Vector3D: 首选速度向量
        """
        # 速度幅值
        to_goal = self.goal - self.pos
        dist_to_goal = to_goal.norm()
        if dist_to_goal < 1e-6:
            return Vector3D(0, 0, 0)
        # TODO :但是无dt是正确的吗,计算速度大小（不超过最大速度）
        pref_speed = min(self.max_speed, dist_to_goal) 
        return to_goal.normalized() * pref_speed

    def _filter_neighbors(self, all_agents: List['Agent']) -> List['Agent']:
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
            dist_sq = (self.pos - agent.pos).norm_sq()
            if dist_sq < interaction_horizon ** 2:
                neighbors.append(agent)
        return neighbors
    
    def _get_effective_state_for_interaction(self, neighbor: 'Agent') -> Dict:
        """
        为交互计算保守的"有效状态"（核心创新1：概率感知）
        将不确定的位置表示为高斯分布，并计算最坏情况下的有效位置，将概率问题转化为确定的几何问题。
        参数:
            neighbor: 邻居智能体
        返回:
            Dict: 包含双方有效状态的字典
        """
        dir_to_neighbor = (neighbor.pos - self.pos).normalized()
        if dir_to_neighbor.norm_sq() < 1e-9:
            dir_to_neighbor = Vector3D(1, 0, 0) # 避免除零，使用默认方向
        
        # 计算自身位置在相对方向上的标准差
        std_dev_i = math.sqrt(dir_to_neighbor.to_numpy().T @ self.cov_pos @ dir_to_neighbor.to_numpy())
        # 计算保守的有效位置（沿相对方向向内收缩）
        effective_pos_i = self.pos - dir_to_neighbor * (cfg.UNCERTAINTY_CONFIDENCE_N * std_dev_i)

        # 计算邻居位置在相反方向上的标准差
        dir_to_self = -dir_to_neighbor
        std_dev_j = math.sqrt(dir_to_self.to_numpy().T @ neighbor.cov_pos @ dir_to_self.to_numpy())
        effective_pos_j = neighbor.pos - dir_to_self * (cfg.UNCERTAINTY_CONFIDENCE_N * std_dev_j)

        return {
            "pos_i": effective_pos_i, "vel_i": self.vel, "radius_i": self.radius,
            "pos_j": effective_pos_j, "vel_j": neighbor.vel, "radius_j": neighbor.radius
        }
        
    def _update_local_density(self, neighbors: List['Agent']):
        """
        更新局部密度估计
        使用高斯核函数估计周围智能体的密度。
        参数:
            neighbors: 邻居列表
        """
        current_rho = 0.0
        for neighbor in neighbors:
            dist_sq = (self.pos - neighbor.pos).norm_sq()
            # 使用高斯核函数计算密度贡献
            current_rho += math.exp(-dist_sq / (2 * cfg.DENSITY_SIGMA**2))
        # 指数平滑更新密度估计
        beta = cfg.DENSITY_BETA_SMOOTHING
        self.rho_smoothed = (1 - beta) * self.rho_smoothed + beta * current_rho
    
    def _calculate_risk(self, agent_i: 'Agent', agent_j: 'Agent') -> float:
        """
        计算两个智能体之间的风险评分
        参数:
            agent_i: 智能体i
            agent_j: 智能体j
        返回:
            float: 风险评分
        """
        states = {"pos_i": agent_i.pos, "vel_i": agent_i.vel, "pos_j": agent_j.pos, "vel_j": agent_j.vel}
        return self._calculate_risk_from_states(states)

    def _calculate_risk_from_states(self, states: Dict) -> float:
        """
        基于状态计算风险评分，综合考虑距离风险和碰撞时间风险。
        参数: states: 状态字典
        返回: float: 风险评分
        """
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
        