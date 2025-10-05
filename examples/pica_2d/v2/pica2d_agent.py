# -*- coding: utf-8 -*-

"""
pica_agent_2d.py

[2D修复版] 实现了混合PICA算法的2D智能体。
基于3D版本简化而来，保留核心逻辑的同时适配2D平面场景。
"""

import math
import numpy as np
from typing import List, Dict, Optional

from utils.pica2d_structures import Vector2D  # 2D向量类
from examples.pica_2d.v2 import config as cfg  # 2D配置文件

class Agent:
    """
    [2D修复版] 实现了混合PICA算法的智能体。
    适配2D平面场景，移除Z轴相关计算，保留核心避障逻辑。
    """
    def __init__(self, id: int, pos: Vector2D, goal: Vector2D, inertia_matrix: np.ndarray, priority: float = 1.0):
        """初始化2D智能体"""
        # 核心属性
        self.id = id
        self.goal = goal
        self.priority = priority
        self.inertia_matrix = inertia_matrix  # 2x2惯性矩阵

        # 物理属性
        self.radius = cfg.AGENT_RADIUS
        self.max_speed = cfg.MAX_SPEED
        self.is_colliding = False

        # 概率状态表示（2D）
        self.pos = pos
        self.vel = Vector2D(0, 0)
        self.mu_pos = pos  # 位置均值
        self.mu_vel = Vector2D(0, 0)  # 速度均值
        self.cov_pos = np.eye(2) * 0.01  # 2D位置协方差矩阵

        # 算法内部状态
        self.rho_smoothed = 0.0  # 平滑后的局部密度
        self.alphas: Dict[int, float] = {}  # 责任分配参数
        self.at_goal = False

    def update(self, new_velocity: Vector2D, dt: float):
        """更新智能体状态（2D位置、速度、不确定性）"""
        if self.at_goal:
            self.mu_vel = Vector2D(0, 0)
            return

        # 速度限制
        if new_velocity.norm() > self.max_speed:
            new_velocity = new_velocity.normalized() * self.max_speed
        
        self.mu_vel = new_velocity
        self.mu_pos += self.mu_vel * dt  # 2D位置更新

        # 更新位置不确定性
        process_noise = np.eye(2) * cfg.PROCESS_NOISE_FACTOR
        self.cov_pos += process_noise * dt

        # 到达目标检测（2D距离）
        if (self.goal - self.mu_pos).norm() < self.radius:
            self.at_goal = True
            self.mu_vel = Vector2D(0, 0)
        
        self.pos = self.mu_pos
        self.vel = self.mu_vel

    def compute_new_velocity(self, all_agents: List['Agent'], dt: float) -> Vector2D:
        """计算新速度（2D核心决策函数）"""
        if self.at_goal:
            return Vector2D(0, 0)

        # 1. 邻居筛选与首选速度计算
        neighbors = self._filter_neighbors(all_agents)
        v_pref = self._get_preferred_velocity()

        # 2. 预处理（打破死锁）
        v_pref = self._break_deadlock(v_pref, neighbors)

        # 3. 风险排序与邻居分类
        initial_risk_scores = {n.id: self._calculate_risk(self, n) for n in neighbors}
        sorted_neighbors = sorted(neighbors, key=lambda n: initial_risk_scores[n.id], reverse=True)
        critical_neighbors = sorted_neighbors[:cfg.PICA_K]
        other_neighbors = sorted_neighbors[cfg.PICA_K:]

        # 4. 更新局部密度
        self._update_local_density(neighbors)

        # 5. 构建约束
        constraints = []
        for neighbor in critical_neighbors:
            effective_states = self._get_effective_state_for_interaction(neighbor)
            risk_score = self._calculate_risk_from_states(effective_states)
            v_pref_j = neighbor._get_preferred_velocity()
            
            self._optimize_alpha_hybrid(neighbor, risk_score, effective_states, v_pref, v_pref_j)
            alpha = self.alphas.get(neighbor.id, 0.5)
            
            constraint = self._create_orca_halfspace_from_states(alpha, effective_states)
            if constraint:
                constraints.append(constraint)

        for neighbor in other_neighbors:
            effective_states = self._get_effective_state_for_interaction(neighbor)
            constraint = self._create_orca_halfspace_from_states(0.5, effective_states)
            if constraint:
                constraints.append(constraint)

        # 6. 求解2D速度
        v_pica = self._solve_velocity_2d(constraints, v_pref)
        return v_pica

    # ================================================================= #
    # ======================= 核心创新实现模块 ======================== #
    # ================================================================= #

    def _get_effective_state_for_interaction(self, neighbor: 'Agent') -> Dict:
        """计算2D保守有效状态（概率感知）"""
        dir_to_neighbor = (neighbor.mu_pos - self.mu_pos).normalized()
        if dir_to_neighbor.norm_sq() < 1e-9:
            dir_to_neighbor = Vector2D(1, 0)  # 2D默认方向（x轴）
        
        # 自身位置不确定性处理
        std_dev_i = math.sqrt(dir_to_neighbor.to_numpy().T @ self.cov_pos @ dir_to_neighbor.to_numpy())
        effective_pos_i = self.mu_pos - dir_to_neighbor * (cfg.UNCERTAINTY_CONFIDENCE_N * std_dev_i)

        # 邻居位置不确定性处理
        dir_to_self = -dir_to_neighbor
        std_dev_j = math.sqrt(dir_to_self.to_numpy().T @ neighbor.cov_pos @ dir_to_self.to_numpy())
        effective_pos_j = neighbor.mu_pos - dir_to_self * (cfg.UNCERTAINTY_CONFIDENCE_N * std_dev_j)

        return {
            "pos_i": effective_pos_i, "vel_i": self.mu_vel, "radius_i": self.radius,
            "pos_j": effective_pos_j, "vel_j": neighbor.mu_vel, "radius_j": neighbor.radius
        }

    def _create_avo_constraint(self, v_candidate: Vector2D, dt: float) -> Optional[Dict]:
        """创建2D加速度约束（物理感知优化）"""
        v_current = self.mu_vel
        accel_needed = (v_candidate - v_current) / dt  # 2D加速度计算
        
        # 检查加速度可行性
        M_inv = np.linalg.inv(self.inertia_matrix)
        accel_potential = accel_needed.to_numpy().T @ M_inv @ accel_needed.to_numpy()
        
        if accel_potential <= cfg.ACCELERATION_MAX**2 + cfg.PICA_EPSILON:
            return None  # 加速度可行

        # 构造速度空间约束
        gradient_at_accel = 2 * M_inv @ accel_needed.to_numpy()
        n_accel = Vector2D(gradient_at_accel[0], gradient_at_accel[1])  # 2D法向量
        n_vel = n_accel / dt
        
        # 计算边界加速度对应的速度
        k_denom_sq = n_accel.to_numpy().T @ self.inertia_matrix @ n_accel.to_numpy()
        k = cfg.ACCELERATION_MAX / math.sqrt(k_denom_sq + cfg.PICA_EPSILON)
        accel_on_boundary = Vector2D.from_numpy(self.inertia_matrix @ n_accel.to_numpy()) * k
        v_on_boundary = v_current + accel_on_boundary * dt

        # 定义半空间约束
        normal = -n_vel
        offset = -v_on_boundary.dot(n_vel)
        
        return {'normal': normal, 'offset': offset}

    def _optimize_alpha_hybrid(self, neighbor: 'Agent', risk_score: float, effective_states: Dict, v_pref_i: Vector2D, v_pref_j: Vector2D):
        """混合优化责任参数α（2D适配）"""
        alpha_heuristic = self._calculate_heuristic_alpha(neighbor)
        alpha_analytical = self._calculate_analytical_alpha(neighbor, effective_states, v_pref_i, v_pref_j)
        
        # 风险加权融合
        w = np.clip(
            (risk_score - cfg.RISK_THRESHOLD_LOW) / (cfg.RISK_THRESHOLD_HIGH - cfg.RISK_THRESHOLD_LOW + 1e-6), 
            0, 1
        )
        alpha_star = (1 - w) * alpha_heuristic + w * alpha_analytical

        # 平滑更新
        alpha_clamped = max(0.0, min(1.0, alpha_star))
        old_alpha = self.alphas.get(neighbor.id, 0.5)
        new_alpha = (1 - cfg.PICA_BETA_DAMPING) * old_alpha + cfg.PICA_BETA_DAMPING * alpha_clamped
        self.alphas[neighbor.id] = new_alpha

    def _calculate_heuristic_alpha(self, neighbor: 'Agent') -> float:
        """启发式责任分配（2D）"""
        epsilon = 1e-6
        my_score = (self.rho_smoothed + 0.1) / (self.priority + epsilon)
        neighbor_score = (neighbor.rho_smoothed + 0.1) / (neighbor.priority + epsilon)
        
        total_score = my_score + neighbor_score
        return 0.5 if total_score < epsilon else my_score / total_score

    def _calculate_analytical_alpha(self, neighbor: 'Agent', states: Dict, v_pref_i: Vector2D, v_pref_j: Vector2D) -> float:
        """解析法责任分配（2D适配）"""
        epsilon = 1e-6
        delta_alpha = 0.01

        # 自身速度梯度计算
        constraint_i_at_0 = self._create_orca_halfspace_from_states(0.0, states)
        constraint_i_at_delta = self._create_orca_halfspace_from_states(delta_alpha, states)
        if not constraint_i_at_0 or not constraint_i_at_delta:
            return 0.5
        
        v_i_at_0 = self._solve_single_constraint(constraint_i_at_0, v_pref_i)
        v_i_at_delta = self._solve_single_constraint(constraint_i_at_delta, v_pref_i)
        a_i_vec = (v_i_at_delta - v_i_at_0) / delta_alpha
        b_i_vec = v_i_at_0

        # 邻居速度梯度计算（2D视角转换）
        states_j_view = {
            "pos_i": states["pos_j"], "vel_i": states["vel_j"], "radius_i": states["radius_j"],
            "pos_j": states["pos_i"], "vel_j": states["vel_i"], "radius_j": states["radius_i"]
        }
        constraint_j_at_0 = self._create_orca_halfspace_from_states(0.0, states_j_view)
        constraint_j_at_delta = self._create_orca_halfspace_from_states(delta_alpha, states_j_view)
        if not constraint_j_at_0 or not constraint_j_at_delta:
            return 0.5
        
        v_j_at_0 = self._solve_single_constraint(constraint_j_at_0, v_pref_j)
        v_j_at_delta = self._solve_single_constraint(constraint_j_at_delta, v_pref_j)
        a_j_vec = (v_j_at_delta - v_j_at_0) / delta_alpha
        b_j_vec = v_j_at_0
        
        # 2D代价函数计算
        M_i, M_j = self.inertia_matrix, neighbor.inertia_matrix
        rho_i, rho_j = self.rho_smoothed, neighbor.rho_smoothed

        term1_i = rho_i * (a_i_vec.to_numpy().T @ M_i @ a_i_vec.to_numpy())
        term1_j = rho_j * (a_j_vec.to_numpy().T @ M_j @ a_j_vec.to_numpy())
        K1 = term1_i + term1_j

        diff_i = b_i_vec - v_pref_i
        diff_j = a_j_vec + b_j_vec - v_pref_j
        term2_i = 2 * rho_i * (a_i_vec.to_numpy().T @ M_i @ diff_i.to_numpy())
        term2_j = -2 * rho_j * (a_j_vec.to_numpy().T @ M_j @ diff_j.to_numpy())
        priority_term = self.priority - neighbor.priority
        K2 = term2_i + term2_j + priority_term
        
        # 求解最优α
        if abs(2 * K1) < epsilon:
            return 1.0 if K2 > 0 else 0.0
        return -K2 / (2 * K1)

    # ================================================================= #
    # ======================= 辅助函数与计算模块 ====================== #
    # ================================================================= #

    def _get_preferred_velocity(self) -> Vector2D:
        """计算2D首选速度（指向目标）"""
        to_goal = self.goal - self.mu_pos
        dist_to_goal = to_goal.norm()
        if dist_to_goal < 1e-6:
            return Vector2D(0, 0)
        pref_speed = min(self.max_speed, dist_to_goal)
        return to_goal.normalized() * pref_speed

    def _filter_neighbors(self, all_agents: List['Agent']) -> List['Agent']:
        """筛选2D交互范围内的邻居"""
        neighbors = []
        interaction_horizon = cfg.TTC_HORIZON * 2 * self.max_speed
        for agent in all_agents:
            if agent.id == self.id:
                continue
            dist_sq = (self.mu_pos - agent.mu_pos).norm_sq()
            if dist_sq < interaction_horizon ** 2:
                neighbors.append(agent)
        return neighbors

    def _break_deadlock(self, v_pref: Vector2D, neighbors: List['Agent']) -> Vector2D:
        """2D平面打破死锁（仅x-y轴旋转）"""
        if self.mu_vel.norm_sq() > 0.01 and neighbors:
            perturb_angle = (self.id % 20 - 10) * 0.015  # 微小旋转角
            c, s = math.cos(perturb_angle), math.sin(perturb_angle)
            # 2D旋转矩阵应用
            px = v_pref.x * c - v_pref.y * s
            py = v_pref.x * s + v_pref.y * c
            v_pref_perturbed = Vector2D(px, py)
            if v_pref_perturbed.norm() > self.max_speed:
                return v_pref_perturbed.normalized() * self.max_speed
            return v_pref_perturbed
        return v_pref

    def _update_local_density(self, neighbors: List['Agent']):
        """更新2D局部密度估计"""
        current_rho = 0.0
        for neighbor in neighbors:
            dist_sq = (self.mu_pos - neighbor.mu_pos).norm_sq()
            current_rho += math.exp(-dist_sq / (2 * cfg.DENSITY_SIGMA**2))
        beta = cfg.DENSITY_BETA_SMOOTHING
        self.rho_smoothed = (1 - beta) * self.rho_smoothed + beta * current_rho

    def _create_orca_halfspace_from_states(self, alpha: float, states: Dict) -> Optional[Dict]:
        """创建2D ORCA半平面约束"""
        pos_i, vel_i, r_i = states["pos_i"], states["vel_i"], states["radius_i"]
        pos_j, vel_j, r_j = states["pos_j"], states["vel_j"], states["radius_j"]

        rel_pos = pos_j - pos_i
        rel_vel = vel_i - vel_j
        dist_sq = rel_pos.norm_sq()
        combined_radius = r_i + r_j
        combined_radius_sq = combined_radius ** 2

        # 碰撞处理
        if dist_sq < combined_radius_sq:
            inv_time_horizon = 1.0 / cfg.TIMESTEP
            u = (rel_pos.normalized() * (combined_radius - rel_pos.norm())) * inv_time_horizon
            normal = u.normalized()
        else:
            inv_tau = 1.0 / cfg.TIMESTEP
            vo_apex = rel_pos * inv_tau
            vo_radius_sq = (combined_radius * inv_tau) ** 2
            
            w = rel_vel - vo_apex
            w_norm_sq = w.norm_sq()

            dot_product = w.dot(rel_pos)
            if dot_product < 0 and dot_product**2 > w_norm_sq * dist_sq:
                return None

            if w_norm_sq <= vo_radius_sq:
                normal = w.normalized() if w.norm_sq() > 1e-9 else -rel_pos.normalized()
                u = normal * (math.sqrt(vo_radius_sq) - math.sqrt(w_norm_sq))
            else:
                normal = (w - rel_pos * (dot_product / dist_sq)).normalized()
                u = normal * (w.dot(normal))
        
        # 构造约束
        plane_point = vel_i + u * alpha
        offset = plane_point.dot(normal)
        
        return {'normal': normal, 'offset': offset}

    def _solve_velocity_2d(self, constraints: List[Dict], v_pref: Vector2D) -> Vector2D:
        """2D速度求解器（迭代投影法）"""
        v_new = v_pref
        for _ in range(50):
            for const in constraints:
                n, offset = const['normal'], const['offset']
                n_norm_sq = n.norm_sq()
                if n_norm_sq < 1e-9:
                    continue
                if v_new.dot(n) < offset:
                    v_new += n * (offset - v_new.dot(n)) / n_norm_sq
        # 速度限制
        if v_new.norm() > self.max_speed:
            v_new = v_new.normalized() * self.max_speed
        return v_new
    
    def _solve_single_constraint(self, constraint: Dict, v_pref: Vector2D) -> Vector2D:
        """求解单个2D约束下的速度"""
        v_new = v_pref
        n, offset = constraint['normal'], constraint['offset']
        n_norm_sq = n.norm_sq()
        if n_norm_sq > 1e-9 and v_new.dot(n) < offset:
            v_new += n * (offset - v_new.dot(n)) / n_norm_sq
        if v_new.norm() > self.max_speed:
            v_new = v_new.normalized() * self.max_speed
        return v_new

    def _calculate_risk(self, agent_i: 'Agent', agent_j: 'Agent') -> float:
        """计算2D风险评分"""
        states = {"pos_i": agent_i.mu_pos, "vel_i": agent_i.mu_vel, 
                 "pos_j": agent_j.mu_pos, "vel_j": agent_j.mu_vel}
        return self._calculate_risk_from_states(states)

    def _calculate_risk_from_states(self, states: Dict) -> float:
        """基于2D状态计算风险（距离+碰撞时间）"""
        rel_pos = states["pos_j"] - states["pos_i"]
        dist = rel_pos.norm()
        if dist < 1e-6:
            return float('inf')
        rel_vel = states["vel_i"] - states["vel_j"]
        vel_dot_pos = rel_vel.dot(rel_pos)
        
        if vel_dot_pos <= 0:
            return cfg.RISK_W_DIST / dist  # 仅距离风险
        
        rel_vel_sq = rel_vel.norm_sq()
        if rel_vel_sq < 1e-6:
            return cfg.RISK_W_DIST / dist
        
        ttc = vel_dot_pos / rel_vel_sq  # 2D碰撞时间
        return cfg.RISK_W_DIST / dist + cfg.RISK_W_TTC / (ttc + 0.1)