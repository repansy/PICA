import math
import random
from typing import List, Dict, Any, Tuple
from collections import deque
import enviroments.config as cfg
from utils.pica_structures import Vector3D, Plane
from utils.linear_solver import linear_program3, linear_program4


# --- B-ORCA 2.0: 融合快慢脑与在线估计的ORCA实现 ---
class BCOrcaAgent:
    def __init__(self, id: int, pos: Vector3D, goal: Vector3D, **kwargs):
        # --- 基础属性 (同原始ORCA) ---
        self.id = id
        self.pos = pos
        self.vel = Vector3D()
        self.goal = goal
        self.radius: float = kwargs.get('radius', cfg.AGENT_RADIUS)
        self.max_speed: float = kwargs.get('max_speed', cfg.MAX_SPEED)
        self.neighbor_dist: float = kwargs.get('neighbor_dist', 15.0)
        self.time_horizon: float = kwargs.get('time_horizon', 5.0)
        self.max_neighbors: int = kwargs.get('max_neighbors', 10)

        # --- 新增：异质性参数 ---
        # 运动能力/惯性矩阵 M = diag(m_x, m_y, m_z)，值越大越难改变速度
        default_m = cfg.M_BASE * (1 + cfg.K_M * self.radius)
        self.M: Vector3D = kwargs.get('M', Vector3D(default_m, default_m, default_m))
        # 任务优先级 P ∈ [0, 1]，值越大越倾向于走直线，很有趣
        self.P: float = kwargs.get('P', 0.1)

        # --- 新增：慢脑相关状态 ---
        self.rho: float = 0.0  # 自身拥挤程度
        # 存储历史轨迹 (pos, vel, dt) 用于邻居状态估计
        self.history: deque = deque(maxlen=cfg.HISTORY_LEN)
        # 存储对邻居的估计和预测结果 {neighbor_id: {data}}
        self.slow_brain_results: Dict[int, Dict[str, Any]] = {}
        # 存储对邻居的意图信念 {neighbor_id: {'inertial': p1, 'goal_oriented': p2}}
        self.neighbor_intentions: Dict[int, Dict[str, float]] = {}

        # --- 内部状态 (同原始ORCA) ---
        self.is_colliding = False
        self.at_goal = False
        self.new_velocity = Vector3D()
        self.pref_velocity = Vector3D()
        self.agent_neighbors: List['BCOrcaAgent'] = []
        self.orca_planes: List[Plane] = []

    # =================================================================================
    # --- 慢脑 (Slow Brain) 模块 ---
    # =================================================================================

    def run_slow_brain(self, all_agents: List['BCOrcaAgent']):
        """
        慢脑主函数：对所有邻居进行状态估计和意图预测。
        在计算新速度之前，为每个智能体调用一次。
        """
        self.slow_brain_results.clear()

        for other in self.agent_neighbors:
            # 1. 在线估计邻居的属性
            p_est, m_est, rho_est = self._estimate_neighbor_properties(other, all_agents)
            
            # 2. 预测邻居的意图速度和置信度
            v_pred, confidence = self._predict_neighbor_intention(other)

            # 3. 计算基于拥挤度的慢脑责任
            alpha_slow = rho_est / (self.rho + rho_est + cfg.EPSILON)

            # 存储所有结果，供快脑使用
            self.slow_brain_results[other.id] = {
                "p_est": p_est, "m_est": m_est, "rho_est": rho_est,
                "v_pred": v_pred, "confidence": confidence,
                "alpha_slow": alpha_slow
            }

    def _estimate_neighbor_properties(self, other: 'BCOrcaAgent', all_agents: List['BCOrcaAgent']) -> Tuple[float, Vector3D, float]:
        """估计邻居 'other' 的 P, M, 和 rho"""
        # 估计 rho_j: 简化为通过 'i' 的视角来观察
        rho_est = 0.0
        for agent in all_agents:
            if agent.id != other.id:
                dist_sq = (other.pos - agent.pos).norm_sq()
                if dist_sq < self.neighbor_dist**2: # 使用一个合理的感知范围
                    rho_est += 1.0 / (dist_sq + cfg.EPSILON)

        # 估计 P_j 和 M_j (需要历史轨迹)
        if not other.history or len(other.history) < 2:
            return other.P, other.M, rho_est # 返回默认值

        # 估计 P_j: 衡量其速度与目标方向的一致性
        alignments = []
        for pos, vel, _ in other.history:
            goal_dir = (other.goal - pos).normalized()
            vel_dir = vel.normalized()
            if not goal_dir.is_zero() and not vel_dir.is_zero():
                alignments.append(max(0, vel_dir.dot(goal_dir))) # clip at 0
        p_est = sum(alignments) / len(alignments) if alignments else other.P

        # 估计 M_j: 衡量其加速度的大小，加速度越小，惯性越大
        accels = []
        for i in range(len(other.history) - 1):
            _, v1, dt1 = other.history[i]
            _, v2, _ = other.history[i+1]
            if dt1 > cfg.EPSILON:
                accel_mag = ((v2 - v1) / dt1).norm()
                accels.append(accel_mag)
        
        avg_accel = sum(accels) / len(accels) if accels else 0
        # M 与加速度大小成反比
        if avg_accel > cfg.EPSILON:
            m_est_mag = 1.0 / avg_accel
            m_est = Vector3D(m_est_mag, m_est_mag, m_est_mag)
        else:
            m_est = other.M # 无法估计，返回默认值

        return p_est, m_est, rho_est
    
    def _predict_neighbor_intention(self, other: 'BCOrcaAgent') -> Tuple[Vector3D, float]:
        """使用贝叶斯推断预测邻居的意图速度和置信度"""
        # 1. 定义意图假设
        v_inertial = other.vel
        v_goal_oriented = (other.goal - other.pos).normalized() * other.max_speed

        # 2. 获取先验概率 (从上一时刻的后验)
        if other.id not in self.neighbor_intentions:
            # 初始化均匀分布
            self.neighbor_intentions[other.id] = {'inertial': 0.5, 'goal_oriented': 0.5}
        prior_inertial = self.neighbor_intentions[other.id]['inertial']
        prior_goal = self.neighbor_intentions[other.id]['goal_oriented']

        # 3. 计算似然度 P(Observation | Intention)
        # 观测是邻居的当前速度 other.vel
        # 假设高斯分布，简化为距离的倒数
        dist_to_inertial = (other.vel - v_inertial).norm_sq()
        dist_to_goal = (other.vel - v_goal_oriented).norm_sq()
        
        likelihood_inertial = 1.0 / (dist_to_inertial + 0.1)
        likelihood_goal = 1.0 / (dist_to_goal + 0.1)

        # 4. 计算后验概率 P(Intention | Observation)
        posterior_inertial_raw = likelihood_inertial * prior_inertial
        posterior_goal_raw = likelihood_goal * prior_goal
        
        norm_factor = posterior_inertial_raw + posterior_goal_raw
        if norm_factor < cfg.EPSILON:
            posterior_inertial, posterior_goal = 0.5, 0.5
        else:
            posterior_inertial = posterior_inertial_raw / norm_factor
            posterior_goal = posterior_goal_raw / norm_factor

        # 5. 更新信念，用于下一时刻
        self.neighbor_intentions[other.id] = {
            'inertial': posterior_inertial, 'goal_oriented': posterior_goal
        }

        # 6. 计算加权的预测速度 (raw)
        v_pred_raw = v_inertial * posterior_inertial + v_goal_oriented * posterior_goal
        
        # 7. 与当前速度平滑，防止突变
        v_pred = other.vel * cfg.BETA_SMOOTHING + v_pred_raw * (1.0 - cfg.BETA_SMOOTHING)

        # 8. 计算置信度
        confidence = max(posterior_inertial, posterior_goal)
        
        return v_pred, confidence
    
    # =================================================================================
    # --- 快脑 (Fast Brain) 模块 与 核心逻辑修改 ---
    # =================================================================================

    def compute_new_velocity(self, dt: float):
        """
        快脑主函数：使用慢脑提供的信息，计算最终的避障速度。
        """
        self.orca_planes.clear()
        inv_time_horizon = 1.0 / self.time_horizon

        for other in self.agent_neighbors:
            # --- MODIFICATION START: 从慢脑获取数据 ---
            if other.id not in self.slow_brain_results: continue # 如果慢脑没有结果，跳过
            
            sb_res = self.slow_brain_results[other.id]
            v_pred_j = sb_res["v_pred"]
            
            # 使用预测的速度，而不是当前速度
            relative_velocity = self.vel - v_pred_j
            # --- MODIFICATION END ---
            
            relative_position = other.pos - self.pos
            dist_sq = relative_position.norm_sq()
            combined_radius = self.radius + other.radius
            combined_radius_sq = combined_radius**2

            plane = Plane()
            u = Vector3D()

            if dist_sq > combined_radius_sq:
                w = relative_velocity - inv_time_horizon * relative_position
                w_length_sq = w.norm_sq()
                dot_product = w.dot(relative_position)

                if dot_product < 0.0 and dot_product**2 > combined_radius_sq * w_length_sq:
                    w_length = math.sqrt(w_length_sq)
                    unit_w = w / w_length
                    plane.normal = unit_w
                    u = (combined_radius * inv_time_horizon - w_length) * unit_w
                else:
                    a = dist_sq
                    b = relative_position.dot(relative_velocity)
                    cross_prod_sq = relative_position.norm_sq() * relative_velocity.norm_sq() - b**2
                    c = relative_velocity.norm_sq() - cross_prod_sq / (dist_sq - combined_radius_sq)
                    discriminant = b**2 - a * c
                    if discriminant < 0: continue
                    t = (b + math.sqrt(discriminant)) / a
                    ww = relative_velocity - t * relative_position
                    ww_length = ww.norm()
                    unit_ww = ww / ww_length
                    plane.normal = unit_ww
                    u = (combined_radius * t - ww_length) * unit_ww
            else:
                inv_time_step = 1.0 / dt
                w = relative_velocity - inv_time_step * relative_position
                w_length = w.norm()
                unit_w = w / w_length
                plane.normal = unit_w
                u = (combined_radius * inv_time_step - w_length) * unit_w
            
            # --- MODIFICATION START: 混合责任分配 ---
            # 1. 快脑责任 (基于估计的物理属性)
            cost_i = cfg.W_P * self.P + cfg.W_M * self.M.norm() + cfg.W_R * self.radius
            cost_j_est = cfg.W_P * sb_res["p_est"] + cfg.W_M * sb_res["m_est"].norm() + cfg.W_R * other.radius
            alpha_fast = cost_j_est / (cost_i + cost_j_est + cfg.EPSILON)

            # 2. 慢脑责任 (已在慢脑中计算)
            alpha_slow = sb_res["alpha_slow"]
            
            # 3. 置信度混合
            confidence = sb_res["confidence"]
            alpha_hybrid = confidence * alpha_slow + (1.0 - confidence) * alpha_fast
            
            # 原来的责任是 0.5，现在替换为混合责任
            plane.point = self.vel + alpha_hybrid * u
            # --- MODIFICATION END ---
            
            self.orca_planes.append(plane)

        fail_plane, self.new_velocity = linear_program3(self.orca_planes, self.max_speed, self.pref_velocity, False)
        if fail_plane < len(self.orca_planes):
            self.new_velocity = linear_program4(self.orca_planes, fail_plane, self.max_speed, self.new_velocity)
    
    # =================================================================================
    # --- 更新和辅助函数 (部分有修改) ---
    # =================================================================================

    def update(self, dt: float):
        """根据 new_velocity 更新智能体的速度和位置，并记录历史"""
        if self.at_goal:
            self.vel = Vector3D()
            return
        
        self.vel = self.new_velocity
        self._break_deadlock() # 保留死锁打破机制
        self.pos += self.vel * dt

        # --- MODIFICATION: 记录历史轨迹，供慢脑使用 ---
        self.history.append((self.pos, self.vel, dt))

        if (self.goal - self.pos).norm_sq() < self.radius**2:
            self.at_goal = True
            self.vel = Vector3D()
            
    def compute_congestion(self):
        """计算自身的拥挤程度 rho"""
        self.rho = 0.0
        for other in self.agent_neighbors:
            dist_sq = (self.pos - other.pos).norm_sq()
            self.rho += 1.0 / (dist_sq + cfg.EPSILON)

    # --- 以下函数与原始OrcaAgent基本一致 ---
    def compute_neighbors(self, all_agents: List['BCOrcaAgent']):
        self.agent_neighbors.clear()
        neighbors_dist_sq = []
        range_sq = self.neighbor_dist**2
        for agent in all_agents:
            if agent.id != self.id:
                dist_sq = (self.pos - agent.pos).norm_sq()
                if dist_sq < range_sq:
                    neighbors_dist_sq.append((dist_sq, agent))
        neighbors_dist_sq.sort(key=lambda x: x[0])
        self.agent_neighbors = [agent for _, agent in neighbors_dist_sq[:self.max_neighbors]]
    
    def compute_preferred_velocity(self):
        if self.at_goal:
            self.pref_velocity = Vector3D()
            return
        to_goal = self.goal - self.pos
        dist_to_goal = to_goal.norm()
        if dist_to_goal < cfg.EPSILON:
             self.pref_velocity = Vector3D()
        else:
            self.pref_velocity = (to_goal / dist_to_goal) * min(self.max_speed, dist_to_goal)
            
    def _break_deadlock(self) -> Vector3D:
        v_pref = self.vel
        if self.vel.norm_sq() < 0.1 and self.at_goal == False:
            # 随机转动小角度
            perturb_angle = 0.5 * (random.randint(-90, 90))
            # perturb_angle = 0
            c, s = math.cos(perturb_angle), math.sin(perturb_angle)
            xp = v_pref.x * c - v_pref.y * s
            yp = v_pref.x * s + v_pref.y * c
            self.vel = Vector3D(xp, yp, v_pref.z)