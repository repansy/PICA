# -*- coding: utf-8 -*-

"""
pica_agent.py

[修复版] 实现了混合PICA (Hybrid Priority and Inertia-aware Collision Avoidance) 算法的智能体。
该版本修正了导致规避能力减弱的逻辑错误，能够正确执行高效的规避动作。

核心创新点：
1. 概率感知与最坏情况建模：将不确定的位置表示为高斯分布，并计算保守的"有效位置"
2. 混合责任分配：结合启发式规则和解析优化，动态计算最优责任参数α
3. 物理感知优化：通过惯性矩阵M考虑智能体的运动特性
4. 最终安全网：使用保守的ORCA规则作为后备保障

"""
import random
import math
import numpy as np
from typing import List, Dict, Optional

from utils.pica_structures import Vector3D
# import enviroments.config as cfg
from examples.pica_3d.v2 import config as cfg

class Agent:
    """
    [修复版] 实现了混合PICA算法的智能体。
    它结合了精确的分析模型和鲁棒的启发式模型，并考虑了状态的不确定性。
    """
    def __init__(self, id: int, pos: Vector3D, goal: Vector3D, inertia_matrix: np.ndarray, priority: float = 1.0):
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
        self.vel = Vector3D(0, 0, 0)
        self.mu_pos = pos
        self.mu_vel = Vector3D(0, 0, 0)
        self.cov_pos = np.eye(3) * 0.01

        # --- 算法内部状态 ---
        self.rho_smoothed = 0.0
        self.alphas: Dict[int, float] = {}
        self.at_goal = False

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

    def compute_new_velocity(self, all_agents: List['Agent'], dt: float) -> Vector3D:
        """
        计算新的速度，核心决策函数
        参数:
            all_agents: 环境中所有智能体的列表
        返回:
            Vector3D: 计算得到的新速度
        """
        if self.at_goal:
            return Vector3D(0, 0, 0)

        # 1. 初始化和邻居筛选
        neighbors = self._filter_neighbors(all_agents)
        v_pref = self._get_preferred_velocity()

        '''
        if not neighbors:
            # 如果没有邻居，仍然需要检查v_pref是否满足自身动力学
            avo_constraint = self._create_avo_constraint(v_pref, dt)
            if avo_constraint:
                # 如果v_pref不可行，则求解一个最接近它的可行速度
                return self._solve_velocity_3d([avo_constraint], v_pref)
            return v_pref
        '''

        # 2. 预处理
        v_pref = self._break_deadlock(v_pref, neighbors)

        initial_risk_scores = {n.id: self._calculate_risk(self, n) for n in neighbors}
        sorted_neighbors = sorted(neighbors, key=lambda n: initial_risk_scores[n.id], reverse=True)
        
        critical_neighbors = sorted_neighbors[:cfg.PICA_K]
        other_neighbors = sorted_neighbors[cfg.PICA_K:]

        self._update_local_density(neighbors)

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

        # 为了下面的avo，牺牲吧
        v_pica = self._solve_velocity_3d(constraints, v_pref)
        return v_pica
        '''
        if self._is_velocity_safe(v_pica, neighbors):
            return v_pica
        else:
            orca_constraints = []
            for neighbor in neighbors:
                effective_states = self._get_effective_state_for_interaction(neighbor)
                constraint = self._create_orca_halfspace_from_states(0.5, effective_states)
                if constraint: orca_constraints.append(constraint)
            return self._solve_velocity_3d(orca_constraints, v_pref)
        '''
        

    # ================================================================= #
    # ======================= 核心创新实现模块 ======================== #
    # ================================================================= #

    def _get_effective_state_for_interaction(self, neighbor: 'Agent') -> Dict:
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

    def _create_avo_constraint(self, v_candidate: Vector3D, dt: float) -> Optional[Dict]:
        """
        [完善版] 如果候选速度违反了加速度约束 (AVO)，则为其生成一个
        最优的线性半空间约束，将整个可行速度集(IAVS)保留下来。
        """
        v_current = self.mu_vel
        
        # 1. 计算到达候选速度所需的加速度
        accel_needed = (v_candidate - v_current) / dt
        
        # 2. 检查此加速度是否可行
        # 我们定义一个势函数 V(a) = a^T * M^-1 * a。可行集是 V(a) <= a_max^2。
        M_inv = np.linalg.inv(self.get_effective_inertia_matrix())
        accel_potential = accel_needed.to_numpy().T @ M_inv @ accel_needed.to_numpy()
        
        if accel_potential <= cfg.ACCELERATION_MAX**2 + cfg.PICA_EPSILON:
            return None  # 加速度可行，无需约束

        # --- 3. 如果不可行，构造最优线性约束 ---
        # 最优切平面的法向量，是指向可行集外部的梯度方向。
        # 梯度 ∇V(a) = 2 * M^-1 * a
        gradient_at_accel = 2 * M_inv @ accel_needed.to_numpy()
        
        # 将梯度向量转换为 Vector3D 对象作为法向量 n
        # 这个法向量 n 现在是在“加速度空间”中
        n_accel = Vector3D(gradient_at_accel[0], gradient_at_accel[1], gradient_at_accel[2])

        # 我们需要将这个约束转换回“速度空间”
        # 关系: v' = v_current + a * dt  =>  a = (v' - v_current) / dt
        # 加速度空间的平面 a · n_accel = c 对应于
        # ((v' - v_current) / dt) · n_accel = c
        # v' · (n_accel / dt) = c + (v_current / dt) · n_accel
        # 所以速度空间的法向量 n_vel = n_accel / dt
        n_vel = n_accel / dt
        
        # 4. 找到椭球边界上与梯度方向最远的点
        # 为了找到切点，我们需要找到在 n_accel 方向上投影最远的可行加速度
        # a_boundary = k * M * n_accel，代入 V(a)=a_max^2 求解k
        # k^2 * (M*n_a)^T * M^-1 * (M*n_a) = a_max^2 => k^2 * n_a^T * M * n_a = a_max^2
        k = cfg.ACCELERATION_MAX / math.sqrt(n_accel.to_numpy().T @ self.get_effective_inertia_matrix() @ n_accel.to_numpy() + cfg.PICA_EPSILON)
        accel_on_boundary = Vector3D.from_numpy(self.get_effective_inertia_matrix() @ n_accel.to_numpy()) * k
        
        # 5. 将边界加速度点转换为速度空间中的切点
        v_on_boundary = v_current + accel_on_boundary * dt

        # 6. 定义半空间
        # 所有可行速度 v' 都必须满足 (v' - v_on_boundary) · n_vel <= 0
        # 即 v' · n_vel <= v_on_boundary · n_vel
        # 我们的求解器需要 v' · n >= offset 的形式，所以翻转法向量
        normal = -n_vel
        offset = -v_on_boundary.dot(n_vel)
        
        return {'normal': normal, 'offset': offset}

    def _optimize_alpha_hybrid(self, neighbor: 'Agent', risk_score: float, effective_states: Dict, v_pref_i: Vector3D, v_pref_j: Vector3D):
        """
        混合优化责任参数α（核心创新2：混合责任分配）结合启发式规则和解析优化，动态计算最优责任参数α。
        根据风险评分加权融合两个模型的结果。
        参数:
            neighbor: 邻居智能体
            risk_score: 风险评分
            effective_states: 有效状态字典
            v_pref_i: 自身的首选速度
            v_pref_j: 邻居的首选速度
        """
        alpha_heuristic = self._calculate_heuristic_alpha(neighbor)
        alpha_analytical = self._calculate_analytical_alpha(neighbor, effective_states, v_pref_i, v_pref_j)
        
        # 3. 根据风险计算混合权重（风险越高，越信赖解析法）
        w = np.clip(
            (risk_score - cfg.RISK_THRESHOLD_LOW) / (cfg.RISK_THRESHOLD_HIGH - cfg.RISK_THRESHOLD_LOW + 1e-6), 
            0, 1
        )
        # 4. 混合两个责任值
        alpha_star = (1 - w) * alpha_heuristic + w * alpha_analytical

        # 5. 钳制责任值到[0,1]范围
        alpha_clamped = max(0.0, min(1.0, alpha_star))
        # 6. 获取上一次的责任值（用于平滑）
        old_alpha = self.alphas.get(neighbor.id, 0.5)
        # 7. 应用阻尼（指数平滑），防止责任值突变
        new_alpha = (1 - cfg.PICA_BETA_DAMPING) * old_alpha + cfg.PICA_BETA_DAMPING * alpha_clamped
        # 8. 存储更新后的责任值
        self.alphas[neighbor.id] = new_alpha

    def _calculate_heuristic_alpha(self, neighbor: 'Agent') -> float:
        """
        计算启发式责任α，基于直观规则：谁周围更拥挤、谁的任务优先级越低，谁就多承担规避责任。
        参数:
            neighbor: 邻居智能体 
        返回:
            float: 启发式责任值α
        """
        epsilon = 1e-6
        my_score = (self.rho_smoothed + 0.1) / (self.priority + epsilon)
        neighbor_score = (neighbor.rho_smoothed + 0.1) / (neighbor.priority + epsilon)
        
        total_score = my_score + neighbor_score
        if total_score < epsilon:
            return 0.5
        
        # BUG FIX 2: alpha 是我方的责任比例，应为 my_score / total_score
        return my_score / total_score

    def _calculate_analytical_alpha(self, neighbor: 'Agent', states: Dict, v_pref_i: Vector3D, v_pref_j: Vector3D) -> float:
        """
        计算解析法责任α（核心创新3：物理感知优化）
        通过构建并最小化代价函数，求解理论上的最优责任α。
        代价函数考虑了双方的运动偏差和物理特性（惯性矩阵）。
        参数:
            neighbor: 邻居智能体
            states: 有效状态字典
            v_pref_i: 自身的首选速度
            v_pref_j: 邻居的首选速度    
        返回:
            float: 解析法责任值α
        """
        epsilon = 1e-6
        delta_alpha = 0.01 # 数值微分步长
        # 1. 计算我方速度关于责任α的梯度（通过数值微分）
        # 1.1 计算α=0时的约束和速度
        constraint_i_at_0 = self._create_orca_halfspace_from_states(0.0, states)
        # 1.2 计算α=δα时的约束和速度
        constraint_i_at_delta = self._create_orca_halfspace_from_states(delta_alpha, states)
        if not constraint_i_at_0 or not constraint_i_at_delta: return 0.5 # 约束无效时返回默认值
        # 1.3 求解两种责任下的速度
        v_i_at_0 = self._solve_single_constraint(constraint_i_at_0, v_pref_i)
        v_i_at_delta = self._solve_single_constraint(constraint_i_at_delta, v_pref_i)
        # 1.4 计算速度关于α的梯度（dv_i/dα）
        a_i_vec = (v_i_at_delta - v_i_at_0) / delta_alpha
        b_i_vec = v_i_at_0

        # 2. 计算邻居速度关于责任α的梯度（注意：邻居的责任α_j = 1 - α_i）
        # 交换视角，从邻居的角度看交互
        states_j_view = {
            "pos_i": states["pos_j"], "vel_i": states["vel_j"], "radius_i": states["radius_j"],
            "pos_j": states["pos_i"], "vel_j": states["vel_i"], "radius_j": states["radius_i"]
        }
        constraint_j_at_0 = self._create_orca_halfspace_from_states(0.0, states_j_view)
        constraint_j_at_delta = self._create_orca_halfspace_from_states(delta_alpha, states_j_view)
        if not constraint_j_at_0 or not constraint_j_at_delta: return 0.5
        
        v_j_at_0 = self._solve_single_constraint(constraint_j_at_0, v_pref_j)
        v_j_at_delta = self._solve_single_constraint(constraint_j_at_delta, v_pref_j)
        a_j_vec = (v_j_at_delta - v_j_at_0) / delta_alpha
        b_j_vec = v_j_at_0
        
        # 3. 构建二次代价函数的系数
        M_i, M_j = self.inertia_matrix, neighbor.inertia_matrix # 惯性矩阵
        rho_i, rho_j = self.rho_smoothed, neighbor.rho_smoothed # 局部密度

        # 3.1 计算二次项系数 K₁
        term1_i = rho_i * (a_i_vec.to_numpy().T @ M_i @ a_i_vec.to_numpy())
        term1_j = rho_j * (a_j_vec.to_numpy().T @ M_j @ a_j_vec.to_numpy())
        K1 = term1_i + term1_j

        # 3.2 计算一次项系数 K₂（简化表示，实际计算更复杂）
        diff_i = b_i_vec - v_pref_i
        diff_j = a_j_vec + b_j_vec - v_pref_j
        term2_i = 2 * rho_i * (a_i_vec.to_numpy().T @ M_i @ diff_i.to_numpy())
        term2_j = -2 * rho_j * (a_j_vec.to_numpy().T @ M_j @ diff_j.to_numpy())
        priority_term = self.priority - neighbor.priority
        K2 = term2_i + term2_j + priority_term
        
        # 4. 求解二次函数最小值
        if abs(2 * K1) < epsilon:
            my_responsibility = 1.0 if K2 > 0 else 0.0
        else:
            my_responsibility = -K2 / (2 * K1)
        
        # BUG FIX 3: alpha* 已经是我方责任，直接返回
        return my_responsibility

    # ================================================================= #
    # ======================= 辅助函数与计算模块 ====================== #
    # ================================================================= #

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
            dist_sq = (self.mu_pos - agent.mu_pos).norm_sq()
            if dist_sq < interaction_horizon ** 2:
                neighbors.append(agent)
        return neighbors

    def _break_deadlock(self, v_pref: Vector3D, neighbors: List['Agent']) -> Vector3D:
        """
        打破对称死锁, 通过添加微小扰动，避免智能体在场景中僵持。
        参数:
            v_pref: 首选速度
            neighbors: 邻居列表
        返回:
            Vector3D: 可能扰动后的速度
        """
        if self.mu_vel.norm_sq() > 0.01 and neighbors:
            # 转动小角度，随机转动
            # perturb_angle = 0.015 * (random.randint(-180, 180))
            perturb_angle = (self.id % 20 - 10) * 0.015
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

    def _update_local_density(self, neighbors: List['Agent']):
        """
        更新局部密度估计
        使用高斯核函数估计周围智能体的密度。
        参数:
            neighbors: 邻居列表
        """
        current_rho = 0.0
        for neighbor in neighbors:
            dist_sq = (self.mu_pos - neighbor.mu_pos).norm_sq()
            # 使用高斯核函数计算密度贡献
            current_rho += math.exp(-dist_sq / (2 * cfg.DENSITY_SIGMA**2))
        # 指数平滑更新密度估计
        beta = cfg.DENSITY_BETA_SMOOTHING
        self.rho_smoothed = (1 - beta) * self.rho_smoothed + beta * current_rho

    def _create_orca_halfspace_from_states(self, alpha: float, states: Dict) -> Optional[Dict]:
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
            if dot_product < 0 and dot_product**2 > w_norm_sq * dist_sq:
                return None

            if w_norm_sq <= vo_radius_sq:
                normal = w.normalized() if w.norm_sq() > 1e-9 else -rel_pos.normalized()
                u = normal * (math.sqrt(vo_radius_sq) - math.sqrt(w_norm_sq))
            else:
                normal = (w - rel_pos * (dot_product / dist_sq)).normalized()
                u = normal * (w.dot(normal))
        
        # BUG FIX 1: The application of the evasion vector 'u' must be additive.
        # A negative sign here inverts the evasion logic, causing weak or incorrect avoidance.
        
        # u_np = u.to_numpy()                  # 转换为列向量
        # M_u_np = self.inertia_matrix @ u_np  # M*u的矩阵运算
        # u = Vector3D.from_numpy(M_u_np)  # 转换回向量对象

        plane_point = vel_i + u * alpha
        offset = plane_point.dot(normal)
        
        return {'normal': normal, 'offset': offset}

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

    '''
    def _is_velocity_safe(self, velocity: Vector3D, neighbors: List['Agent']) -> bool:
        for neighbor in neighbors:
            effective_states = self._get_effective_state_for_interaction(neighbor)
            constraint = self._create_orca_halfspace_from_states(0.5, effective_states)
            if constraint and velocity.dot(constraint['normal']) < constraint['offset'] - cfg.PICA_EPSILON:
                return False
        return True
    '''

    def _calculate_risk(self, agent_i: 'Agent', agent_j: 'Agent') -> float:
        """
        计算两个智能体之间的风险评分
        参数:
            agent_i: 智能体i
            agent_j: 智能体j
        返回:
            float: 风险评分
        """
        states = {"pos_i": agent_i.mu_pos, "vel_i": agent_i.mu_vel, "pos_j": agent_j.mu_pos, "vel_j": agent_j.mu_vel}
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
    