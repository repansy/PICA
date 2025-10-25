import math
import random
from typing import List
import examples.pica_3d.v2.config as cfg
from utils.pica_structures import Vector3D, Plane
from utils.linear_solver import linear_program3, linear_program4

# --- "原汁原味"的 RVO2-3D Agent 实现 ---
class OrcaAgent:
    def __init__(self, id: int, pos: Vector3D, goal: Vector3D, **kwargs):
        self.id = id
        self.pos = pos
        self.vel = Vector3D()
        self.goal = goal
        # i will come back just in probalic
        # self.priority = priority
        # self.inertia_matrix = inertia_matrix
        
        # 从 kwargs 获取参数，提供默认值
        self.radius: float = cfg.AGENT_RADIUS
        self.max_speed: float = cfg.MAX_SPEED
        self.neighbor_dist: float = kwargs.get('neighbor_dist', 15.0)
        self.time_horizon: float = kwargs.get('time_horizon', 5.0)
        self.max_neighbors: int = kwargs.get('max_neighbors', 10)

        # 内部状态
        self.is_colliding = False
        self.at_goal = False
        self.new_velocity = Vector3D()
        self.pref_velocity = Vector3D()
        self.agent_neighbors: List['OrcaAgent'] = []
        self.orca_planes: List[Plane] = []

    def update(self, dt: float):
        """
        根据 new_velocity 更新智能体的速度和位置。
        这对应 C++ 代码中的 Agent::update()。
        """
        if self.at_goal:
            self.vel = Vector3D()
            return
        
        self.vel = self.new_velocity
        self._break_deadlock()
        self.pos += self.vel * dt

        # 检查是否到达目标
        if (self.goal - self.pos).norm_sq() < self.radius**2:
            self.at_goal = True
            self.vel = Vector3D()

    def _break_deadlock(self) -> Vector3D:
        """
        TODO: 还有一种用法，添加到v_pref上，但是数值要小一些，主要是为了缓解正对面时返回的效果
        打破对称死锁, 通过添加微小扰动，避免智能体在场景中僵持。
        """
        v_pref = self.vel
        if self.vel.norm_sq() < 0.1 and self.at_goal == False:
            # 随机转动小角度
            perturb_angle = 0.5 * (random.randint(-90, 90))
            # perturb_angle = 0
            c, s = math.cos(perturb_angle), math.sin(perturb_angle)
            xp = v_pref.x * c - v_pref.y * s
            yp = v_pref.x * s + v_pref.y * c
            self.vel = Vector3D(xp, yp, v_pref.z)


    def compute_neighbors(self, all_agents: List['OrcaAgent']):
        """
        计算并存储邻居智能体。
        这替代了 C++ 代码中的 Agent::computeNeighbors() 和 KdTree。
        """
        self.agent_neighbors.clear()
        
        # 简单的基于距离的邻居搜索
        neighbors_dist_sq = []
        range_sq = self.neighbor_dist**2
        for agent in all_agents:
            if agent.id != self.id:
                dist_sq = (self.pos - agent.pos).norm_sq()
                if dist_sq < range_sq:
                    neighbors_dist_sq.append((dist_sq, agent))
        
        # 排序并选择最近的 max_neighbors 个
        neighbors_dist_sq.sort(key=lambda x: x[0])
        self.agent_neighbors = [agent for _, agent in neighbors_dist_sq[:self.max_neighbors]]
    
    def compute_preferred_velocity(self):
        """计算朝向目标的期望速度"""
        if self.at_goal:
            self.pref_velocity = Vector3D()
            return

        to_goal = self.goal - self.pos
        dist_to_goal = to_goal.norm()

        if dist_to_goal < cfg.RVO3D_EPSILON:
             self.pref_velocity = Vector3D()
        else:
            # 速度设置为朝向目标，大小不超过 max_speed
            self.pref_velocity = (to_goal / dist_to_goal) * min(self.max_speed, dist_to_goal)

    def compute_new_velocity(self, dt: float):
        """
        计算新的避障速度。
        这对应 C++ 代码中的 Agent::computeNewVelocity()。
        """
        self.orca_planes.clear()
        inv_time_horizon = 1.0 / self.time_horizon

        for other in self.agent_neighbors:
            relative_position = other.pos - self.pos
            relative_velocity = self.vel - other.vel
            dist_sq = relative_position.norm_sq()
            combined_radius = self.radius + other.radius
            combined_radius_sq = combined_radius**2

            plane = Plane()
            u = Vector3D()

            if dist_sq > combined_radius_sq:
                # --- 非碰撞情况 (No collision) ---
                w = relative_velocity - inv_time_horizon * relative_position
                w_length_sq = w.norm_sq()
                dot_product = w.dot(relative_position)

                if dot_product < 0.0 and dot_product**2 > combined_radius_sq * w_length_sq:
                    # Case 1: 投影在截断球面上 (Project on cut-off sphere)
                    w_length = math.sqrt(w_length_sq)
                    unit_w = w / w_length
                    plane.normal = unit_w
                    u = (combined_radius * inv_time_horizon - w_length) * unit_w
                else:
                    # Case 2: 投影在圆锥侧面上 (Project on cone)
                    a = dist_sq
                    b = relative_position.dot(relative_velocity)
                    # 使用向量叉乘的性质简化计算: |a x b|^2 = |a|^2 * |b|^2 - (a . b)^2
                    cross_prod_sq = relative_position.norm_sq() * relative_velocity.norm_sq() - b**2
                    c = relative_velocity.norm_sq() - cross_prod_sq / (dist_sq - combined_radius_sq)
                    
                    discriminant = b**2 - a * c
                    if discriminant < 0: # 理论上不应发生，但作为保护
                        continue 
                    
                    t = (b + math.sqrt(discriminant)) / a
                    ww = relative_velocity - t * relative_position
                    ww_length = ww.norm()
                    unit_ww = ww / ww_length

                    plane.normal = unit_ww
                    u = (combined_radius * t - ww_length) * unit_ww
            else:
                # --- 碰撞情况 (Collision) ---
                inv_time_step = 1.0 / dt
                w = relative_velocity - inv_time_step * relative_position
                w_length = w.norm()
                unit_w = w / w_length

                plane.normal = unit_w
                u = (combined_radius * inv_time_step - w_length) * unit_w
            
            # 核心：每个智能体承担一半的责任
            plane.point = self.vel + 0.5 * u
            self.orca_planes.append(plane)

        # --- 求解最优速度 ---
        fail_plane, self.new_velocity = linear_program3(self.orca_planes, self.max_speed, self.pref_velocity, False)

        if fail_plane < len(self.orca_planes):
            self.new_velocity = linear_program4(self.orca_planes, fail_plane, self.max_speed, self.new_velocity)