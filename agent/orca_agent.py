import math
import numpy as np
from typing import List, Dict, Any, Tuple
import enviroments.config as cfg
from utils.pica2d_structures import Vector2D, Line
from utils.linear_solver import linear_program2, linear_program3
import random

class OrcaAgent:
    """智能体类，实现RVO算法核心功能"""
    def __init__(self, id: int, pos: Vector2D, goal: Vector2D, **kwargs):
        self.id = id
        self.pos = pos
        self.vel = Vector2D()
        self.goal = goal
        
        # 从 kwargs 获取参数，提供默认值
        self.radius: float = kwargs.get('radius', cfg.AGENT_RADIUS)
        self.max_speed: float = cfg.MAX_SPEED
        self.neighbor_dist: float = kwargs.get('neighbor_dist', cfg.NEIGHBOR_DIST)
        self.time_horizon: float = kwargs.get('time_horizon', cfg.TIME_HORIZON)
        self.max_neighbors: int = kwargs.get('max_neighbors', cfg.MAX_NEIGHOBORS)

        # 内部状态
        self.is_colliding = False
        self.at_goal = False
        self.new_velocity = Vector2D()
        self.pref_velocity = Vector2D()
        self.agent_neighbors: List['OrcaAgent'] = []
        self.orca_lines: List['Line'] = []

    def update(self, dt: float):
        """
        根据 new_velocity 更新智能体的速度和位置。
        这对应 C++ 代码中的 Agent::update()。
        """
        if self.at_goal:
            self.vel = Vector2D()
            return
        
        self.vel = self.new_velocity
        self._break_deadlock()
        self.pos += self.vel * dt

        # 检查是否到达目标
        if (self.goal - self.pos).norm_sq() < self.radius**2:
            self.at_goal = True
            self.vel = Vector2D()

    def compute_preferred_velocity(self):
        """计算朝向目标的期望速度"""
        if self.at_goal:
            self.pref_velocity = Vector2D()
            return

        to_goal = self.goal - self.pos
        dist_to_goal = to_goal.norm()

        if dist_to_goal < cfg.EPSILON:
             self.pref_velocity = Vector2D()
        else:
            # 速度设置为朝向目标，大小不超过 max_speed
            self.pref_velocity = (to_goal / dist_to_goal) * min(self.max_speed, dist_to_goal)

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

    def compute_new_velocity(self, time_step):
        """计算新速度"""
        self.orca_lines.clear()
        num_obst_lines = len(self.orca_lines) # 平衡函数使用
        inv_time_horizon = 1.0 / self.time_horizon

        for other in self.agent_neighbors:
            relative_position = other.pos - self.pos
            relative_velocity = self.vel - other.vel
            dist_sq = relative_position.norm_sq()
            combined_radius = self.radius + other.radius
            combined_radiussq = combined_radius **2

            line = Line()
            u = Vector2D()

            if dist_sq > combined_radiussq:
                w = relative_velocity - inv_time_horizon * relative_position
                w_length_sq = w.norm_sq()
                dot_product = w @ relative_position

                if dot_product < 0.0 and (dot_product** 2) > (combined_radiussq * w_length_sq):
                    w_length = math.sqrt(w_length_sq)
                    unit_w = w / w_length
                    line.direction = Vector2D(unit_w.y, -unit_w.x)
                    u = (combined_radius * inv_time_horizon - w_length) * unit_w
                else:
                    leg = math.sqrt(dist_sq - combined_radiussq)
                    if Vector2D.det(relative_position, w) > 0.0:
                        line.direction = Vector2D(
                            relative_position.x * leg - relative_position.y * combined_radius,
                            relative_position.x * combined_radius + relative_position.y * leg
                        ) / dist_sq
                    else:
                        dir_vec = Vector2D(
                            relative_position.x * leg + relative_position.y * combined_radius,
                            -relative_position.x * combined_radius + relative_position.y * leg
                        ) / dist_sq
                        line.direction = -dir_vec
                    u = (relative_velocity @ line.direction) * line.direction - relative_velocity
            else:
                inv_time_step = 1.0 / time_step
                w = relative_velocity - inv_time_step * relative_position
                w_length = w.norm()
                unit_w = w / w_length
                line.direction = Vector2D(unit_w.y, -unit_w.x)
                u = (combined_radius * inv_time_step - w_length) * unit_w

            line.point = self.vel + 0.5 * u
            self.orca_lines.append(line)

        # 求解线性规划得到新速度
        line_fail, self.new_velocity = linear_program2(self.orca_lines, self.max_speed, self.pref_velocity, False)
        if line_fail < len(self.orca_lines):
            self.new_velocity = linear_program3(self.orca_lines, num_obst_lines, line_fail, self.max_speed, self.new_velocity)

    def _break_deadlock(self) -> Vector2D:
        v_pref = self.vel
        if self.vel.norm_sq() < 0.2 and self.at_goal == False:
            # 随机转动小角度
            perturb_angle = 0.05 * (random.randint(-15, 15))
            # perturb_angle = 0
            c, s = math.cos(perturb_angle), math.sin(perturb_angle)
            xp = v_pref.x * c - v_pref.y * s
            yp = v_pref.x * s + v_pref.y * c
            self.vel = Vector2D(xp, yp)