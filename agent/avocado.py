import math
import random
import numpy as np
from typing import List, Dict, Any, Tuple
from collections import deque
import enviroments.config as cfg
from utils.pica_structures import Vector3D, Plane
from utils.linear_solver import linear_program3, linear_program4

# --- AVOCADO: 融合责任分配与注意力机制的ORCA实现 ---
class AvocadoAgent:
    def __init__(self, id: int, pos: Vector3D, goal: Vector3D, **kwargs):
        # --- 基础属性 (同原始ORCA) ---
        self.id = id
        self.pos = pos
        self.vel = Vector3D()
        self.goal = goal

        self.radius: float = kwargs.get('radius', cfg.AGENT_RADIUS)
        self.max_speed: float = cfg.MAX_SPEED
        self.neighbor_dist: float = kwargs.get('neighbor_dist', cfg.NEIGHBOR_DIST)
        self.time_horizon: float = kwargs.get('time_horizon', cfg.TIME_HORIZON)
        self.max_neighbors: int = kwargs.get('max_neighbors', cfg.MAX_NEIGHOBORS)

        # --- AVOCADO 参数 ---
        self.const_alpha_: float = kwargs.get('const_alpha', 1.0)  # 责任分配系数
        self.max_noise: float = kwargs.get('max_noise', 0.1)  # 最大噪声强度
        self.kappa_: float = kwargs.get('kappa', 14.15)  # 注意力衰减系数
        self.delta_: float = kwargs.get('delta', 0.57)  # 注意力更新速率
        self.d_: float = kwargs.get('d', 2)  # 意见更新速率
        self.a_: float = kwargs.get('a', 0.3)  # 意见强化系数
        self.c_: float = kwargs.get('c', 0.7)  # 意见校正系数
        self.b_: float = kwargs.get('b', 0.0)  # 意见基础值
        self.epsilon_: float = kwargs.get('epsilon', 3.22)  # 意图判断阈值

        # --- 内部状态 (同原始ORCA) ---
        self.is_colliding = False
        self.at_goal = False
        self.new_velocity = Vector3D()
        self.pref_velocity = Vector3D()
        self.agent_neighbors: List['AvocadoAgent'] = []
        self.orca_planes: List[Plane] = []

        # --- AVOCADO状态 ---
        # 注意力及相关状态 {neighbor_id: AlphaU}
        self.alphas_: Dict[int, AlphaU] = {}
        self.prev_velocity = Vector3D() # 上一时刻速度

        # 噪声生成器
        self.distribution = random.uniform

    # =================================================================================
    # --- AVOCADO核心逻辑 ---
    # =================================================================================

    def compute_alphas(self):
        """计算邻居的注意力和意见"""
        if self.const_alpha_ > 1:
            for other in self.agent_neighbors:
                if other.id not in self.alphas_:
                    continue

                alpha_entry = self.alphas_[other.id]

                # Compute attention (u) logic
                radius = other.radius + self.radius
                delta_p = self.pos - other.pos
                delta_v = self.vel - other.vel

                a = delta_v.dot(delta_v)
                b = 2 * delta_v.dot(delta_p)
                c = delta_p.dot(delta_p) - radius * radius

                root = b*b - 4*a*c
                tau = float('inf')

                if root < 0:
                    # No collision at anytime
                    tau = float('inf')
                else:
                    tau = (-b - math.sqrt(root)) / (2*a) if abs(a) > 1e-8 else float('inf')

                    term2 = (-b + math.sqrt(root)) / (2*a) if abs(a) > 1e-8 else -float('inf')

                    if tau < -1e-6 and term2 > 0:
                        # The robot is in collision
                        tau = 1e-6
                    elif tau < -1e-6 or a < 1e-6:
                        # No collision (backwards)
                        tau = float('inf')

                # Update attention
                alpha_entry.attention = (1 - self.delta_) * alpha_entry.attention + \
                                        self.delta_ * math.tanh(self.kappa_ / (tau + 1e-8))

                # Compute alpha_hat
                delta_vt = other.vel - other.prev_velocity
                u_sq = alpha_entry.u.dot(alpha_entry.u)

                # ((delta_vt * u) / (u^2 + epsilon)) * u
                proj_factor = delta_vt.dot(alpha_entry.u) / (u_sq + 1e-6)
                proj_v_u = proj_factor * alpha_entry.u

                proj_norm = proj_v_u.norm()
                u_norm = math.sqrt(u_sq)

                alpha_hat = math.tanh(self.epsilon_ * (proj_norm / (u_norm + 1e-6) - 0.5))

                # Compute opinion (alpha update)
                # Differential equation: (-d*alpha + d*attention*tanh(...) + b) * dt
                term_tanh = math.tanh(self.a_ * alpha_entry.alpha + self.c_ * alpha_hat)
                delta_alpha = (-self.d_ * alpha_entry.alpha +
                               self.d_ * alpha_entry.attention * term_tanh +
                               self.b_) * cfg.TIME_HORIZON  # 使用仿真器的时间步长

                alpha_entry.alpha += delta_alpha

    def compute_new_velocity(self):
        """
        快脑主函数：考虑噪声和注意力，计算最终的避障速度。
        """
        self.orca_planes.clear()
        inv_time_horizon = 1.0 / self.time_horizon if self.time_horizon > 0 else 0.0

        for other in self.agent_neighbors:
            noise_increment = Vector3D()

            # 噪声生成
            if self.const_alpha_ > 1.0:
                # 模拟 C++ 的 std::default_random_engine 和 uniform_real_distribution
                # 注意：Python 的 random 不需要像 C++ 那样每次实例化 engine
                rx = self.distribution(-self.max_noise, self.max_noise)
                ry = self.distribution(-self.max_noise, self.max_noise)
                rz = self.distribution(-self.max_noise, self.max_noise)
                noise_increment = Vector3D(rx, ry, rz)

            # 获取 attention 用于加权噪声
            attention = 0.0
            if other.id in self.alphas_:
                attention = self.alphas_[other.id].attention

            relative_position = other.pos - self.pos
            # Velocity perception affected by noise and attention
            relative_velocity = self.vel - (other.vel + (1 - attention) * noise_increment)

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
                    # Project on cut-off circle
                    w_length = math.sqrt(w_length_sq)
                    unit_w = w / w_length if w_length > 0 else Vector3D()

                    plane.normal = unit_w
                    u = (combined_radius * inv_time_horizon - w_length) * unit_w
                else:
                    # Project on cone
                    a = dist_sq
                    b = relative_position.dot(relative_velocity)
                    cross_prod = relative_position.cross(relative_velocity)
                    c = relative_velocity.norm_sq() - cross_prod.norm_sq() / (dist_sq - combined_radius_sq)

                    discriminant = b**2 - a * c
                    if discriminant < 0: discriminant = 0  # Safety

                    t = (b + math.sqrt(discriminant)) / a
                    ww = relative_velocity - t * relative_position
                    ww_length = ww.norm()
                    unit_ww = ww / ww_length if ww_length > 0 else Vector3D()

                    plane.normal = unit_ww
                    u = (combined_radius * t - ww_length) * unit_ww
            else:
                # Collision
                inv_time_step = 1.0 / cfg.TIME_HORIZON
                w = relative_velocity - inv_time_step * relative_position
                w_length = w.norm()
                unit_w = w / w_length if w_length > 0 else Vector3D()

                plane.normal = unit_w
                u = (combined_radius * inv_time_step - w_length) * unit_w

            # AVOCADO Logic: Modify plane point based on alpha (aggressiveness/opinion)
            if 0.0 < self.const_alpha_ <= 1.0:
                plane.point = self.vel + self.const_alpha_ * u
                self.orca_planes.append(plane)
            elif self.const_alpha_ > 1.0:
                # Retrieve current alpha value
                current_alpha = self.alphas_[other.id].alpha if other.id in self.alphas_ else 0.0

                factor = 1.0 - max(0.0, min(1.0, (current_alpha + 1) * 0.5))
                plane.point = self.vel + factor * u

                # Store u for next step's computation
                if other.id in self.alphas_:
                    self.alphas_[other.id].u = u

                if factor > 0.0:
                    self.orca_planes.append(plane)

        fail_plane, self.new_velocity = linear_program3(self.orca_planes, self.max_speed, self.pref_velocity, False)
        if fail_plane < len(self.orca_planes):
            self.new_velocity = linear_program4(self.orca_planes, fail_plane, self.max_speed, self.new_velocity)

    # =================================================================================
    # --- 更新和辅助函数 (部分有修改) ---
    # =================================================================================

    def update(self, dt: float):
        """根据 new_velocity 更新智能体的速度和位置"""
        if self.at_goal:
            self.vel = Vector3D()
            return

        self.prev_velocity = self.vel  # 记录上一时刻速度
        self.vel = self.new_velocity
        self.pos += self.vel * dt

        if (self.goal - self.pos).norm_sq() < self.radius**2:
            self.at_goal = True
            self.vel = Vector3D()

    def compute_neighbors(self, all_agents: List['AvocadoAgent']):
        """计算邻居列表"""
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
        # 初始化Alphas_ 注意在compute neighbor后初始化才遍历所有邻居
        for agent in self.agent_neighbors:
            if self.const_alpha_ > 1 and agent.id not in self.alphas_:
                a_u = AlphaU()
                a_u.alpha = self.b_ / self.d_ if self.d_ != 0 else 0.0
                a_u.u = Vector3D()
                a_u.attention = 0.0
                self.alphas_[agent.id] = a_u

    def compute_preferred_velocity(self):
        """计算偏好速度"""
        if self.at_goal:
            self.pref_velocity = Vector3D()
            return
        to_goal = self.goal - self.pos
        dist_to_goal = to_goal.norm()
        if dist_to_goal < cfg.EPSILON:
             self.pref_velocity = Vector3D()
        else:
            self.pref_velocity = (to_goal / dist_to_goal) * min(self.max_speed, dist_to_goal)

class AlphaU:
    """存储 AVOCADO 算法中关于邻居的注意力及状态变量"""
    def __init__(self):
        self.u = Vector3D()
        self.alpha = 0.0
        self.attention = 0.0