# -*- coding: utf-8 -*-

"""
pica_agent.py

[最终版 - 局部可观测性]
实现了混合PICA算法。此版本在严格的局部信息假设下运行：
- 智能体只知道自己的物理/任务属性(M, P)。
- 通过预测/估计机制处理未知的邻居属性。
- 将动力学(AVO)作为内置硬约束，放弃了外部安全网。
"""

import math
import numpy as np
from typing import List, Dict, Optional

from utils.pica_structures import Vector3D
import config as cfg

class Agent:
    def __init__(self, id: int, pos: Vector3D, goal: Vector3D, inertia_matrix: np.ndarray, priority: float = 1.0):
        # --- 核心ID与任务属性 ---
        self.id = id
        self.goal = goal
        self.priority = priority
        self.inertia_matrix = inertia_matrix # 只存储自己的M

        # --- 物理属性 ---
        self.radius = cfg.AGENT_RADIUS
        self.max_speed = cfg.MAX_SPEED
        self.is_colliding = False

        # --- 概率状态表示 ---
        self.pos = pos # for visualization or simple access
        self.vel = Vector3D(0, 0, 0) # for visualization or simple access
        self.mu_pos = pos
        self.mu_vel = Vector3D(0, 0, 0)
        self.cov_pos = np.eye(3) * 0.01

        # --- 算法内部状态 ---
        self.rho_smoothed = 0.0
        self.alphas: Dict[int, float] = {}
        self.at_goal = False

    def update(self, new_velocity: Vector3D, dt: float):
        if self.at_goal:
            self.mu_vel = Vector3D(0, 0, 0)
            self.vel = self.mu_vel
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
        
        # Update simple attributes for easier external access
        self.pos = self.mu_pos
        self.vel = self.mu_vel

    def compute_new_velocity(self, all_agents: List['Agent'], dt: float) -> Vector3D:
        if self.at_goal:
            return Vector3D(0, 0, 0)

        # 1. 初始化
        neighbors = self._filter_neighbors(all_agents)
        v_pref = self._get_preferred_velocity()

        # 如果没有邻居，只需确保自身动作符合动力学
        if not neighbors:
            avo_constraint = self._create_avo_constraint(v_pref, dt)
            if avo_constraint:
                return self._solve_velocity_3d([avo_constraint], v_pref)
            return v_pref

        # 2. 预处理
        v_pref = self._break_deadlock(v_pref, neighbors)
        self._update_local_density(neighbors)

        # 3. 构建统一约束集
        constraints = []
        
        # 3a. 为所有邻居生成ORCA几何约束
        # 注意: 风险评分现在只用于混合权重，不再用于邻居分类
        for neighbor in neighbors:
            effective_states = self._get_effective_state_for_interaction(neighbor)
            risk_score = self._calculate_risk_from_states(effective_states)
            
            # 调用为局部观测调整后的优化器
            self._optimize_alpha_hybrid_local(neighbor, risk_score)
            alpha = self.alphas.get(neighbor.id, 0.5)
            
            constraint = self._create_orca_halfspace_from_states(alpha, effective_states)
            if constraint:
                constraints.append(constraint)

        # 3b. 为自身生成AVO动力学约束
        # 检查首选速度 v_pref 是否在物理上可行
        avo_constraint = self._create_avo_constraint(v_pref, dt)
        if avo_constraint:
            constraints.append(avo_constraint)

        # 4. 统一求解
        return self._solve_velocity_3d(constraints, v_pref)

    # ================================================================= #
    # ================== 核心算法模块 (已为局部观测调整) ================= #
    # ================================================================= #

    def _optimize_alpha_hybrid_local(self, neighbor: 'Agent', risk_score: float):
        """
        [局部可观测版] 混合优化责任参数α。
        使用对邻居属性的估计值来运行优化器。
        """
        # 使用估计值调用核心计算函数
        alpha_heuristic = self._calculate_heuristic_alpha_local(neighbor)
        alpha_analytical = self._calculate_analytical_alpha_local(neighbor)

        w = np.clip((risk_score - cfg.RISK_THRESHOLD_LOW) / (cfg.RISK_THRESHOLD_HIGH - cfg.RISK_THRESHOLD_LOW + 1e-6), 0, 1)
        alpha_star = (1 - w) * alpha_heuristic + w * alpha_analytical

        alpha_clamped = max(0.0, min(1.0, alpha_star))
        old_alpha = self.alphas.get(neighbor.id, 0.5)
        new_alpha = (1 - cfg.PICA_BETA_DAMPING) * old_alpha + cfg.PICA_BETA_DAMPING * alpha_clamped
        self.alphas[neighbor.id] = new_alpha

    def _calculate_heuristic_alpha_local(self, neighbor: 'Agent') -> float:
        """[局部可观测版] 计算启发式α。"""
        epsilon = 1e-6
        # 自身分数是精确的
        my_score = (self.rho_smoothed + 0.1) / (self.priority + epsilon)
        
        # --- 对邻居属性进行估计 ---
        # 假设1: 邻居的优先级与我相同
        neighbor_priority_est = self.priority 
        # 假设2: 邻居感受到的密度与我相似
        neighbor_rho_est = self.rho_smoothed
        
        neighbor_score_est = (neighbor_rho_est + 0.1) / (neighbor_priority_est + epsilon)
        
        total_score = my_score + neighbor_score_est
        if total_score < epsilon:
            return 0.5
        return my_score / total_score

    def _calculate_analytical_alpha_local(self, neighbor: 'Agent') -> float:
        """[局部可观测版] 计算解析法α。"""
        epsilon = 1e-6
        delta_alpha = 0.01
        
        # --- 对邻居属性进行估计 ---
        # 假设3: 邻居的首选速度是其当前速度
        v_pref_j_est = neighbor.mu_vel
        # 假设4: 邻居的惯性是一个标准的、默认的惯性
        M_j_est = cfg.DEFAULT_INERTIA_MATRIX

        # 获取精确的自身属性
        v_pref_i = self._get_preferred_velocity()
        M_i = self.inertia_matrix
        states = self._get_effective_state_for_interaction(neighbor)

        # --- 后续的数学推导与原版完全相同，只是输入变为了估计值 ---
        constraint_i_at_0 = self._create_orca_halfspace_from_states(0.0, states)
        constraint_i_at_delta = self._create_orca_halfspace_from_states(delta_alpha, states)
        if not constraint_i_at_0 or not constraint_i_at_delta: return 0.5

        v_i_at_0 = self._solve_single_constraint(constraint_i_at_0, v_pref_i)
        v_i_at_delta = self._solve_single_constraint(constraint_i_at_delta, v_pref_i)
        a_i_vec = (v_i_at_delta - v_i_at_0) / delta_alpha
        b_i_vec = v_i_at_0

        states_j_view = { "pos_i": states["pos_j"], "vel_i": states["vel_j"], "radius_i": states["radius_j"], "pos_j": states["pos_i"], "vel_j": states["vel_i"], "radius_j": states["radius_i"] }
        constraint_j_at_0 = self._create_orca_halfspace_from_states(0.0, states_j_view)
        constraint_j_at_delta = self._create_orca_halfspace_from_states(delta_alpha, states_j_view)
        if not constraint_j_at_0 or not constraint_j_at_delta: return 0.5
        
        v_j_at_0 = self._solve_single_constraint(constraint_j_at_0, v_pref_j_est)
        v_j_at_delta = self._solve_single_constraint(constraint_j_at_delta, v_pref_j_est)
        a_j_vec = (v_j_at_delta - v_j_at_0) / delta_alpha
        b_j_vec = v_j_at_0
        
        rho_i = self.rho_smoothed
        rho_j_est = self.rho_smoothed # 使用估计的密度

        term1_i = rho_i * (a_i_vec.to_numpy().T @ M_i @ a_i_vec.to_numpy())
        term1_j = rho_j_est * (a_j_vec.to_numpy().T @ M_j_est @ a_j_vec.to_numpy())
        K1 = term1_i + term1_j

        diff_i = b_i_vec - v_pref_i
        diff_j = a_j_vec + b_j_vec - v_pref_j_est
        term2_i = 2 * rho_i * (a_i_vec.to_numpy().T @ M_i @ diff_i.to_numpy())
        term2_j = -2 * rho_j_est * (a_j_vec.to_numpy().T @ M_j_est @ diff_j.to_numpy())
        priority_term = self.priority - self.priority # 使用估计的优先级 (P_j_est = self.priority)
        K2 = term2_i + term2_j + priority_term
        
        if abs(2 * K1) < epsilon:
            my_responsibility = 1.0 if K2 > 0 else 0.0
        else:
            my_responsibility = -K2 / (2 * K1)
        
        return my_responsibility

    def _create_avo_constraint(self, v_candidate: Vector3D, dt: float) -> Optional[Dict]:
        v_current = self.mu_vel
        accel_needed = (v_candidate - v_current) / dt
        
        M_inv = np.linalg.inv(self.inertia_matrix)
        accel_potential = accel_needed.to_numpy().T @ M_inv @ accel_needed.to_numpy()
        
        if accel_potential <= cfg.ACCELERATION_MAX**2 + cfg.PICA_EPSILON:
            return None

        gradient_at_accel = 2 * M_inv @ accel_needed.to_numpy()
        n_accel = Vector3D.from_numpy(Vector3D(0, 0, 0), gradient_at_accel)
        n_vel = n_accel / dt
        
        k_denom_sq = n_accel.to_numpy().T @ self.inertia_matrix @ n_accel.to_numpy()
        k = cfg.ACCELERATION_MAX / math.sqrt(k_denom_sq + cfg.PICA_EPSILON)
        accel_on_boundary = Vector3D.from_numpy(Vector3D(0, 0, 0), self.inertia_matrix @ n_accel.to_numpy()) * k
        
        v_on_boundary = v_current + accel_on_boundary * dt

        normal = -n_vel
        offset = -v_on_boundary.dot(n_vel)
        
        return {'normal': normal, 'offset': offset}
    
    # ================================================================= #
    # ======================= 基础计算模块 (无大改动) =================== #
    # ================================================================= #

    def _get_preferred_velocity(self) -> Vector3D:
        to_goal = self.goal - self.mu_pos
        dist_to_goal = to_goal.norm()
        if dist_to_goal < 1e-6:
            return Vector3D(0, 0, 0)
        # 保持简单：速度大小为到目标的距离和最大速度的较小值
        pref_speed = min(self.max_speed, dist_to_goal) 
        return to_goal.normalized() * pref_speed

    def _filter_neighbors(self, all_agents: List['Agent']) -> List['Agent']:
        neighbors = []
        # 优化：交互范围与自身当前速度相关，更合理
        interaction_horizon = cfg.TTC_HORIZON * (self.mu_vel.norm() + self.max_speed)
        for agent in all_agents:
            if agent.id == self.id:
                continue
            dist_sq = (self.mu_pos - agent.mu_pos).norm_sq()
            if dist_sq < interaction_horizon ** 2:
                neighbors.append(agent)
        return neighbors

    def _break_deadlock(self, v_pref: Vector3D, neighbors: List['Agent']) -> Vector3D:
        if self.mu_vel.norm_sq() > 0.01 and neighbors:
            perturb_angle = (self.id % 20 - 10) * 0.015
            c, s = math.cos(perturb_angle), math.sin(perturb_angle)
            px = v_pref.x * c - v_pref.y * s
            py = v_pref.x * s + v_pref.y * c
            v_pref_perturbed = Vector3D(px, py, v_pref.z)
            if v_pref_perturbed.norm() > self.max_speed:
                return v_pref_perturbed.normalized() * self.max_speed
            return v_pref_perturbed
        return v_pref

    def _update_local_density(self, neighbors: List['Agent']):
        current_rho = 0.0
        for neighbor in neighbors:
            dist_sq = (self.mu_pos - neighbor.mu_pos).norm_sq()
            current_rho += math.exp(-dist_sq / (2 * cfg.DENSITY_SIGMA**2))
        beta = cfg.DENSITY_BETA_SMOOTHING
        self.rho_smoothed = (1 - beta) * self.rho_smoothed + beta * current_rho

    def _create_orca_halfspace_from_states(self, alpha: float, states: Dict) -> Optional[Dict]:
        pos_i, vel_i, r_i = states["pos_i"], states["vel_i"], states["radius_i"]
        pos_j, vel_j, r_j = states["pos_j"], states["vel_j"], states["radius_j"]

        rel_pos = pos_j - pos_i
        rel_vel = vel_i - vel_j
        dist_sq = rel_pos.norm_sq()
        combined_radius = r_i + r_j
        combined_radius_sq = combined_radius ** 2

        if dist_sq < combined_radius_sq:
            inv_time_horizon = 1.0 / cfg.TIMESTEP
            u = (rel_pos.normalized() * (combined_radius - rel_pos.norm())) * inv_time_horizon
            normal = u.normalized() if u.norm_sq() > 0 else rel_pos.normalized()
        else:
            inv_tau = 1.0 / cfg.TTC_HORIZON
            vo_apex = rel_pos * inv_tau
            vo_radius_sq = (combined_radius * inv_tau) ** 2
            
            w = rel_vel - vo_apex
            w_norm_sq = w.norm_sq()

            dot_product = w.dot(rel_pos)
            if dot_product < 0 and dist_sq > 1e-9 and dot_product**2 > w_norm_sq * dist_sq:
                return None

            if w_norm_sq <= vo_radius_sq:
                normal = w.normalized() if w_norm_sq > 1e-9 else -rel_pos.normalized()
                u = normal * (math.sqrt(vo_radius_sq) - math.sqrt(w_norm_sq))
            else:
                normal = (w - rel_pos * (dot_product / (dist_sq + 1e-9))).normalized()
                u = normal * (w.dot(normal))
        
        plane_point = vel_i + u * alpha
        offset = plane_point.dot(normal)
        
        return {'normal': normal, 'offset': offset}
    
    def _get_effective_state_for_interaction(self, neighbor: 'Agent') -> Dict:
        dir_to_neighbor = (neighbor.mu_pos - self.mu_pos).normalized()
        if dir_to_neighbor.norm_sq() < 1e-9:
            dir_to_neighbor = Vector3D(1, 0, 0)
        
        std_dev_i = math.sqrt(dir_to_neighbor.to_numpy().T @ self.cov_pos @ dir_to_neighbor.to_numpy())
        effective_pos_i = self.mu_pos - dir_to_neighbor * (cfg.UNCERTAINTY_CONFIDENCE_N * std_dev_i)

        dir_to_self = -dir_to_neighbor
        std_dev_j = math.sqrt(dir_to_self.to_numpy().T @ neighbor.cov_pos @ dir_to_self.to_numpy())
        effective_pos_j = neighbor.mu_pos - dir_to_self * (cfg.UNCERTAINTY_CONFIDENCE_N * std_dev_j)

        return { "pos_i": effective_pos_i, "vel_i": self.mu_vel, "radius_i": self.radius, "pos_j": effective_pos_j, "vel_j": neighbor.mu_vel, "radius_j": neighbor.radius }

    def _solve_velocity_3d(self, constraints: List[Dict], v_pref: Vector3D) -> Vector3D:
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
    
    def _solve_single_constraint(self, constraint: Dict, v_pref: Vector3D) -> Vector3D:
        v_new = v_pref
        n, offset = constraint['normal'], constraint['offset']
        n_norm_sq = n.norm_sq()
        if n_norm_sq > 1e-9 and v_new.dot(n) < offset:
            v_new += n * (offset - v_new.dot(n)) / n_norm_sq
        if v_new.norm() > self.max_speed:
            v_new = v_new.normalized() * self.max_speed
        return v_new
    
    def _calculate_risk(self, agent_i: 'Agent', agent_j: 'Agent') -> float:
        states = {"pos_i": agent_i.mu_pos, "vel_i": agent_i.mu_vel, "pos_j": agent_j.mu_pos, "vel_j": agent_j.mu_vel}
        return self._calculate_risk_from_states(states)

    def _calculate_risk_from_states(self, states: Dict) -> float:
        rel_pos = states["pos_j"] - states["pos_i"]
        dist = rel_pos.norm()
        if dist < 1e-6: return float('inf')
        rel_vel = states["vel_i"] - states["vel_j"]
        vel_dot_pos = rel_vel.dot(rel_pos)
        if vel_dot_pos <= 0:
            return cfg.RISK_W_DIST / dist
        rel_vel_sq = rel_vel.norm_sq()
        if rel_vel_sq < 1e-6:
            return cfg.RISK_W_DIST / dist
        ttc = vel_dot_pos / rel_vel_sq
        return cfg.RISK_W_DIST / dist + cfg.RISK_W_TTC / (ttc + 0.1)