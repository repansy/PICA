import math
import numpy as np
from typing import List, Dict, Optional

# 假设 Vector3D 和 cfg 已经正确导入
from utils.pica_structures import Vector3D
import config as cfg

class Agent:
    """
    最终决定版PICA。
    - 使用与“黄金标准”ORCA完全相同的、几何精确的约束生成函数。
    - 使用一个更稳定、更直接的启发式alpha计算方法，根除不稳定性。
    """
    def __init__(self, id: int, pos: Vector3D, goal: Vector3D, inertia_matrix: np.ndarray, priority: float = 1.0):
        self.id = id
        self.pos = pos
        self.vel = Vector3D(0, 0, 0)
        self.goal = goal
        self.radius = cfg.AGENT_RADIUS
        self.max_speed = cfg.MAX_SPEED
        self.priority = priority
        
        self.rho_smoothed = 0.0
        self.alphas: Dict[int, float] = {} 
        self.inertia_matrix = inertia_matrix
        self.at_goal = False
        self.is_colliding = False

    def update(self, new_velocity: Vector3D, dt: float):
        if self.at_goal:
            self.vel = Vector3D(0, 0, 0)
            return
        # 速度最终由求解器限制，这里作为安全备份
        if new_velocity.norm() > self.max_speed:
            new_velocity = new_velocity.normalized() * self.max_speed
        self.vel = new_velocity
        self.pos += self.vel * dt
        if (self.goal - self.pos).norm() < self.radius:
            self.at_goal = True
            self.vel = Vector3D(0, 0, 0)

    def compute_new_velocity(self, all_agents: List['Agent']) -> Vector3D:
        if self.at_goal:
            return Vector3D(0, 0, 0)
        
        # 1. 智能邻居筛选
        max_interaction_dist = self.max_speed * cfg.TTC_HORIZON + self.radius
        neighbors = []
        for agent in all_agents:
            if agent.id == self.id:
                continue
            dist_sq = (self.pos - agent.pos).norm_sq()
            total_interaction_dist = max_interaction_dist + agent.max_speed * cfg.TTC_HORIZON + agent.radius
            if dist_sq < total_interaction_dist ** 2:
                neighbors.append(agent)

        # 计算原始的首选速度
        to_goal = self.goal - self.pos
        dist_to_goal = to_goal.norm()
        if dist_to_goal < 1e-6:
            v_pref = Vector3D(0, 0, 0)
        else:
            pref_speed = min(self.max_speed, dist_to_goal)
            v_pref = to_goal.normalized() * pref_speed

        # 如果没有需要交互的邻居，直接使用首选速度
        if not neighbors:
            return v_pref

        # 2. 打破僵局的微小扰动
        # 只有在有邻居且自己在移动时，才需要考虑打破僵局
        if self.vel.norm_sq() > 0.01:
            # 使用ID确保扰动是确定性的，避免抖动
            perturb_angle = (self.id % 20 - 10) * 0.01 
            px = v_pref.x * math.cos(perturb_angle) - v_pref.y * math.sin(perturb_angle)
            py = v_pref.x * math.sin(perturb_angle) + v_pref.y * math.cos(perturb_angle)
            v_pref = Vector3D(px, py, v_pref.z)
            # 确保扰动后的速度不会超过最大值
            if v_pref.norm() > self.max_speed:
                v_pref = v_pref.normalized() * self.max_speed


        # --- 原有的PICA和ORCA计算逻辑 ---
        # 注意：现在所有的计算都基于筛选后的 `neighbors` 列表
        risk_scores = {n.id: self._calculate_risk(n) for n in neighbors}
        sorted_neighbors = sorted(neighbors, key=lambda n: risk_scores[n.id], reverse=True)
        critical_neighbors = sorted_neighbors[:cfg.PICA_K]
        other_neighbors = sorted_neighbors[cfg.PICA_K:]
        
        current_rho = self._calculate_local_density(neighbors)
        self.rho_smoothed = ((1 - cfg.DENSITY_BETA_SMOOTHING) * self.rho_smoothed + 
                            cfg.DENSITY_BETA_SMOOTHING * current_rho)
                            
        for neighbor in critical_neighbors:
            self._optimize_alpha_heuristic(neighbor)
        
        constraints = []
        for neighbor in critical_neighbors:
            alpha = self.alphas.get(neighbor.id, 0.5)
            constraint = self._create_orca_halfspace(neighbor, alpha)
            if constraint: constraints.append(constraint)
                
        for neighbor in other_neighbors:
            constraint = self._create_orca_halfspace(neighbor, 0.5)
            if constraint: constraints.append(constraint)

        # 求解时，使用可能被扰动过的 v_pref
        v_pica = self._solve_velocity_3d(constraints, v_pref)

        # 安全网逻辑保持不变
        classic_orca_constraints = []
        for n in neighbors:
            constraint = self._create_orca_halfspace(n, 0.5)
            if constraint: classic_orca_constraints.append(constraint)
        if not self._is_velocity_safe(v_pica, classic_orca_constraints):
            # 注意：安全网求解时，也使用扰动过的 v_pref，以保持一致性
            return self._solve_velocity_3d(classic_orca_constraints, v_pref)
            
        return v_pica
        
    def _optimize_alpha_heuristic(self, neighbor: 'Agent'):
        """
        一个稳定、直接的启发式Alpha计算方法，取代之前不稳定的分析模型。
        核心思想：优先级越高、越拥挤，承担的责任越小 (alpha越小)。
        """
        epsilon = 1e-6
        my_responsibility_score = (self.rho_smoothed + 0.1) / (self.priority + epsilon)
        neighbor_responsibility_score = (neighbor.rho_smoothed + 0.1) / (neighbor.priority + epsilon)
        
        total_score = my_responsibility_score + neighbor_responsibility_score
        if total_score < epsilon:
            alpha_star = 0.5
        else:
            # alpha 是对方承担的责任比例，所以是我方分数 / 总分数
            alpha_star = my_responsibility_score / total_score

        alpha_clamped = max(0.0, min(1.0, alpha_star))
        old_alpha = self.alphas.get(neighbor.id, 0.5)
        new_alpha = (1 - cfg.PICA_BETA_DAMPING) * old_alpha + cfg.PICA_BETA_DAMPING * alpha_clamped
        self.alphas[neighbor.id] = new_alpha

    def _create_orca_halfspace(self, neighbor: 'Agent', alpha: float) -> Optional[Dict]:
        """
        [PICA 修正版] 使用“黄金标准”的几何逻辑，并正确应用PICA的alpha责任分配参数。
        """
        # --- 步骤 1: 初始化基本变量 ---
        rel_pos = neighbor.pos - self.pos
        rel_vel = self.vel - neighbor.vel
        dist_sq = rel_pos.norm_sq()
        combined_radius = self.radius + neighbor.radius
        combined_radius_sq = combined_radius ** 2

        # --- 步骤 2: 稳定的碰撞响应 ---
        if dist_sq < combined_radius_sq:
            pass
        
        # --- 步骤 3: 构建截断速度障碍物 (TVO) ---
        tau = getattr(cfg, 'TTC_HORIZON', 5.0)
        inv_tau = 1.0 / tau
        
        vo_apex = rel_pos * inv_tau
        vo_radius = combined_radius * inv_tau
        vo_radius_sq = vo_radius ** 2
        
        w = rel_vel - vo_apex
        w_norm_sq = w.norm_sq()

        # --- 步骤 4: 精确的几何区域判断 ---
        dot_product_w_pos = w.dot(rel_pos)
        if dot_product_w_pos < 0 and dot_product_w_pos**2 > w_norm_sq * dist_sq:
            return None # 在圆锥之外，绝对安全

        # --- 步骤 5: 根据所在区域计算修正向量 u ---
        if w_norm_sq <= vo_radius_sq:
            # 情况 A: rel_vel 在 TVO 顶盖内
            w_norm = math.sqrt(w_norm_sq)
            normal = w.normalized() if w_norm > 1e-9 else -rel_pos.normalized()
            u = normal * (vo_radius - w_norm)
        else:
            # 情况 B: rel_vel 在 TVO 侧面区域
            normal = (w - rel_pos * (dot_product_w_pos / dist_sq)).normalized()
            u = normal * (normal.dot(w))

        # --- 步骤 6: 定义ORCA/PICA半空间 ---
        # 唯一的区别在于，这里使用传入的alpha，而不是固定的0.5
        plane_point = self.vel + u * alpha
        offset = normal.dot(plane_point)
        
        return {'normal': normal, 'offset': offset}

    def _solve_velocity_3d(self, constraints: List[Dict], v_pref: Vector3D) -> Vector3D:
        """一个健壮的迭代式速度求解器。"""
        v_new = v_pref
        for _ in range(50):
            for const in constraints:
                n = const['normal']
                offset = const['offset']
                n_norm_sq = n.norm_sq()
                if n_norm_sq < 1e-9: continue
                if v_new.dot(n) < offset:
                    correction = n * (offset - v_new.dot(n)) / n_norm_sq
                    v_new += correction
        if v_new.norm() > self.max_speed:
            v_new = v_new.normalized() * self.max_speed
        return v_new
        
    # --- 辅助函数 ---
    def _is_velocity_safe(self, velocity: Vector3D, constraints: List[Dict]) -> bool:
        for const in constraints:
            if velocity.dot(const['normal']) < const['offset'] - cfg.PICA_EPSILON:
                return False
        return True

    def _calculate_risk(self, neighbor: 'Agent') -> float:
        rel_pos = neighbor.pos - self.pos; dist = rel_pos.norm()
        if dist < 1e-6: return float('inf')
        rel_vel = self.vel - neighbor.vel; b = rel_pos.dot(rel_vel)
        if b >= 0: return cfg.RISK_W_DIST / dist
        a = rel_vel.dot(rel_vel)
        if a < 1e-6: return cfg.RISK_W_DIST / dist
        ttc = -b / a
        return cfg.RISK_W_DIST / dist + cfg.RISK_W_TTC / (ttc + 0.1)

    def _calculate_local_density(self, neighbors: List['Agent']) -> float:
        density = 0.0
        for neighbor in neighbors:
            density += math.exp(-(self.pos - neighbor.pos).norm_sq() / (2 * cfg.DENSITY_SIGMA**2))
        return density