# main.py
import numpy as np
import math
from typing import List

# from agent.pica_agent import Agent # 使用您最终的Agent类
# from agent.orca_agent import OrcaAgent as Agent
from examples.pica_3d.v2.pica_agent import Agent
from examples.pica_3d.v2 import config as cfg

from utils.pica_structures import Vector3D
# import enviroments.config as cfg

def setup_crossing_scenario(num_agents: int) -> List[Agent]:
    """
    [核心异质性验证场景]
    两组不同物理特性的无人机相互交叉。
    - A组: 敏捷、低优先级 (从左到右)
    - B组: 笨重、高优先级 (从下到上)
    """
    print("Setting up: Asymmetric Crossing Scenario")
    agents: List[Agent] = []
    center_x, center_y, z = cfg.WORLD_SIZE[0] / 2, cfg.WORLD_SIZE[1] / 2, cfg.WORLD_SIZE[2] / 2
    spacing = 5
    start_offset = (num_agents // 4) * spacing / 2

    # A组: 敏捷侦察机 (从左到右)
    for i in range(num_agents // 2):
        start_pos = Vector3D(0, center_y - start_offset + i * spacing, z)
        goal_pos = Vector3D(cfg.WORLD_SIZE[0], center_y - start_offset + i * spacing, z)
        
        # 敏捷: 惯性矩阵为单位矩阵
        inertia_matrix = np.eye(3) * 1.0
        # 低优先级
        priority = 1.0
        
        agents.append(Agent(id=i, pos=start_pos, goal=goal_pos, inertia_matrix=inertia_matrix, priority=priority))

    # B组: 重型运输机 (从下到上)
    for i in range(num_agents // 2):
        start_pos = Vector3D(center_x - start_offset + i * spacing, 0, z)
        goal_pos = Vector3D(center_x - start_offset + i * spacing, cfg.WORLD_SIZE[1], z)

        # 笨重: 惯性矩阵惩罚非前向运动
        inertia_matrix = np.diag([1.0, 50.0, 50.0]) # 假设Y是其前进方向
        # 高优先级
        priority = 10.0

        agents.append(Agent(id=i + num_agents // 2, pos=start_pos, goal=goal_pos, inertia_matrix=inertia_matrix, priority=priority))

    return agents

def setup_circle_scenario_2d(num_agents: int) -> List[Agent]:
    """
    [经典基准场景]
    所有智能体在一个2D圆环上，目标是其对跖点。
    """
    print("Setting up: 2D Circle (Antipodal) Scenario")
    agents: List[Agent] = []
    center_x, center_y, z = cfg.WORLD_SIZE[0] / 2, cfg.WORLD_SIZE[1] / 2, cfg.WORLD_SIZE[2] / 2
    radius = min(center_x, center_y) * 0.8

    for i in range(num_agents):
        angle = 2 * math.pi * i / num_agents
        start_pos = Vector3D(center_x + radius * math.cos(angle), 
                             center_y + radius * math.sin(angle), 
                             z)
        goal_pos = Vector3D(center_x - radius * math.cos(angle), 
                            center_y - radius * math.sin(angle), 
                            z)
        
        # 在此场景中，我们可以混合不同类型的智能体来测试
        if i % 2 == 0:
            # 偶数ID: 敏捷型
            inertia = np.eye(3) * 1.0
            priority = 1.0
        else:
            # 奇数ID: 稍笨重型
            inertia = np.diag([5.0, 5.0, 5.0])
            priority = 5.0

        agents.append(Agent(id=i, pos=start_pos, goal=goal_pos, inertia_matrix=inertia, priority=priority))
        
    return agents

def setup_sphere_scenario_3d(num_agents: int) -> List[Agent]:
    """
    [3D扩展场景]
    所有智能体在一个3D球面上，目标是其对跖点。
    """
    print("Setting up: 3D Sphere (Antipodal) Scenario")
    agents: List[Agent] = []
    center = Vector3D(cfg.WORLD_SIZE[0]/2, cfg.WORLD_SIZE[1]/2, cfg.WORLD_SIZE[2]/2)
    radius = min(center.x, center.y, center.z) * 0.8
    
    # 使用斐波那契晶格在球面上均匀分布点
    points = []
    phi = math.pi * (3. - math.sqrt(5.))  # 黄金角
    for i in range(num_agents):
        y = 1 - (i / float(num_agents - 1)) * 2  # y goes from 1 to -1
        r = math.sqrt(1 - y * y)  # radius at y
        theta = phi * i  # golden angle increment
        x = math.cos(theta) * r
        z = math.sin(theta) * r
        points.append(Vector3D(x, y, z))

    for i in range(num_agents):
        start_vec = points[i]
        start_pos = center + start_vec * radius
        goal_pos = center - start_vec * radius

        # 随机分配异质性
        inertia = np.diag(np.random.uniform(1, 10, 3))
        priority = np.random.uniform(1, 10)
        
        agents.append(Agent(id=i, pos=start_pos, goal=goal_pos, inertia_matrix=inertia, priority=priority))
        
    return agents

def setup_ellipsoid_scenario_3d(num_agents: int) -> List[Agent]:
    """
    [更复杂的3D场景]
    智能体在一个椭球表面上，目标是对跖点，强制在不同维度上产生不同密度的交互。
    """
    print("Setting up: 3D Ellipsoid (Antipodal) Scenario")
    agents: List[Agent] = []
    center = Vector3D(cfg.WORLD_SIZE[0]/2, cfg.WORLD_SIZE[1]/2, cfg.WORLD_SIZE[2]/2)
    # 一个在X轴上被拉长的椭球
    radii = Vector3D(center.x * 0.9, center.y * 0.5, center.z * 0.7)

    # 同样使用斐波那契晶格，但之后根据椭球半径进行缩放
    points = []
    phi = math.pi * (3. - math.sqrt(5.))
    for i in range(num_agents):
        y = 1 - (i / float(num_agents - 1)) * 2
        r = math.sqrt(1 - y * y)
        theta = phi * i
        x = math.cos(theta) * r
        z = math.sin(theta) * r
        points.append(Vector3D(x, y, z))

    for i in range(num_agents):
        start_vec = points[i]
        start_pos = center + Vector3D(start_vec.x * radii.x, start_vec.y * radii.y, start_vec.z * radii.z)
        goal_pos = center - Vector3D(start_vec.x * radii.x, start_vec.y * radii.y, start_vec.z * radii.z)
        
        inertia = np.diag(np.random.uniform(1, 10, 3))
        priority = np.random.uniform(1, 10)

        agents.append(Agent(id=i, pos=start_pos, goal=goal_pos, inertia_matrix=inertia, priority=priority))
        
    return agents

# 场景生成函数的字典
scenario_factory = {
    'CROSSING': setup_crossing_scenario,
    'CIRCLE_2D': setup_circle_scenario_2d,
    'SPHERE_3D': setup_sphere_scenario_3d,
    'ELLIPSOID_3D': setup_ellipsoid_scenario_3d
}