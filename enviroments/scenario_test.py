import math 
import numpy as np

from typing import List,Dict,Any
from enviroments import config as cfg

from utils.pica_structures import Vector3D
# from agent.orca_agent import OrcaAgent as Agent
# from agent.test_agent import TestAgent as Agent
from agent.pivo_agent import BCOrcaAgent as Agent

class SphereScenario:
    """
    一个高度可配置的球形对跖点场景，专为测试异质性智能体设计。
    """
    def __init__(
        self,
        agent_groups: List[Dict[str, Any]],
        num_agents: int = cfg.NUM_AGENTS,
        world_size: tuple = cfg.WORLD_SIZE,
        seed: int = 42
    ):
        """
        初始化场景。

        Args:
            agent_groups (List[Dict[str, Any]]): 
                一个定义异质智能体组的列表。每个字典代表一个组，包含：
                - 'ratio' (float): 该组智能体占总数的比例。
                - 'params' (Dict): 该组智能体的属性，如 'radius', 'P', 'M'。
            num_agents (int): 场景中的智能体总数。
            world_size (tuple): 仿真世界的大小 (x, y, z)。
            seed (int): 用于控制随机性的种子。
        """
        self.num_agents = num_agents
        self.world_size = world_size
        self.seed = seed
        self.agent_groups = agent_groups
        np.random.seed(self.seed)

        # 检查比例总和是否为1
        total_ratio = sum(group['ratio'] for group in self.agent_groups)
        if not math.isclose(total_ratio, 1.0):
            raise ValueError(f"所有智能体组的比例总和必须为1，当前为: {total_ratio}")

        # 计算场景中心和球体半径
        self.center = Vector3D(world_size[0] / 2, world_size[1] / 2, world_size[2] / 2)
        self.radius = min(self.center.x, self.center.y, self.center.z) * 0.8

        # 生成均匀分布在球面上的点
        self.sphere_points = self._generate_fibonacci_lattice_points()

    def _generate_fibonacci_lattice_points(self) -> List[Vector3D]:
        """
        使用斐波那契晶格算法在单位球面上生成均匀分布的点。
        这能确保智能体的初始布局形成一个规整的正球形。
        """
        points = []
        phi = math.pi * (3. - math.sqrt(5.))  # 黄金角

        for i in range(self.num_agents):
            y = 1 - (i / float(self.num_agents - 1)) * 2  # y 坐标从 1 线性下降到 -1
            radius_at_y = math.sqrt(1 - y * y)  # 在高度y处的圆环半径
            theta = phi * i  # 当前点的黄金角

            x = math.cos(theta) * radius_at_y
            z = math.sin(theta) * radius_at_y
            points.append(Vector3D(x, y, z))
        
        # 随机打乱点以分配给不同组
        np.random.shuffle(points)
        return points

    def create_agents(self) -> List[Agent]:
        """
        根据 agent_groups 的定义创建并返回智能体列表。
        """
        agents = []
        current_agent_idx = 0

        for group_info in self.agent_groups:
            ratio = group_info['ratio']
            params = group_info['params']
            num_in_group = int(round(self.num_agents * ratio))

            # 处理最后一个组，确保总数正确
            if group_info == self.agent_groups[-1]:
                num_in_group = self.num_agents - len(agents)

            for _ in range(num_in_group):
                if current_agent_idx >= self.num_agents:
                    break
                
                # 获取起点和终点
                start_vec = self.sphere_points[current_agent_idx]
                start_pos = self.center + start_vec * self.radius
                goal_pos = self.center - start_vec * self.radius

                # 创建智能体实例，传入特定参数
                agent = Agent(
                    id=current_agent_idx,
                    pos=start_pos,
                    goal=goal_pos,
                    **params  # 将字典中的所有参数解包传入
                )
                agents.append(agent)
                current_agent_idx += 1
        
        return agents

class CircleScenario:
    """
    一个高度可配置的圆形对跖点场景，专为测试异质性智能体设计。
    """
    def __init__(
        self,
        agent_groups: List[Dict[str, Any]],
        num_agents: int = cfg.NUM_AGENTS,
        world_size: tuple = cfg.WORLD_SIZE,
        seed: int = 42
    ):
        """
        初始化场景。

        Args:
            agent_groups (List[Dict[str, Any]]): 
                一个定义异质智能体组的列表。每个字典代表一个组，包含：
                - 'ratio' (float): 该组智能体占总数的比例。
                - 'params' (Dict): 该组智能体的属性，如 'radius', 'P', 'M'。
            num_agents (int): 场景中的智能体总数。
            world_size (tuple): 仿真世界的大小 (x, y, z)。
            seed (int): 用于控制随机性的种子。
        """
        self.num_agents = num_agents
        self.world_size = world_size
        self.seed = seed
        self.agent_groups = agent_groups
        np.random.seed(self.seed)

        # 检查比例总和是否为1
        total_ratio = sum(group['ratio'] for group in self.agent_groups)
        if not math.isclose(total_ratio, 1.0):
            raise ValueError(f"所有智能体组的比例总和必须为1，当前为: {total_ratio}")

        # 计算场景中心和球体半径
        self.center = Vector3D(world_size[0] / 2, world_size[1] / 2, world_size[1] / 2)
        self.radius = min(self.center.x, self.center.y, self.center.z) * 0.8

        # 生成均匀分布在球面上的点
        self.sphere_points = self._generate_fibonacci_lattice_points()

    def _generate_fibonacci_lattice_points(self) -> List[Vector3D]:
        points = []
        for i in range(self.num_agents):
            # 角度均匀分布（0到2π）
            theta = 2 * math.pi * i / self.num_agents
            # 单位圆上的点
            x = math.cos(theta)
            y = math.sin(theta)
            points.append(Vector3D(x, y, 0))
        
        # 随机打乱点的顺序，确保分组均匀分布
        np.random.shuffle(points)
        return points

    def create_agents(self) -> List[Agent]:
        """
        根据 agent_groups 的定义创建并返回智能体列表。
        """
        agents = []
        current_agent_idx = 0

        for group_info in self.agent_groups:
            ratio = group_info['ratio']
            params = group_info['params']
            num_in_group = int(round(self.num_agents * ratio))

            # 处理最后一个组，确保总数正确
            if group_info == self.agent_groups[-1]:
                num_in_group = self.num_agents - len(agents)

            for _ in range(num_in_group):
                if current_agent_idx >= self.num_agents:
                    break
                
                # 获取起点和终点
                start_vec = self.sphere_points[current_agent_idx]
                start_pos = self.center + start_vec * self.radius
                goal_pos = self.center - start_vec * self.radius

                # 创建智能体实例，传入特定参数
                agent = Agent(
                    id=current_agent_idx,
                    pos=start_pos,
                    goal=goal_pos,
                    **params  # 将字典中的所有参数解包传入
                )
                agents.append(agent)
                current_agent_idx += 1
        
        return agents

# 示例 : 初期test测试
discrete_groups = [
    {'ratio': 0.3, 'params': {'radius': 2.0, 'P': 0.7, 'M': Vector3D(3.0, 3.0, 3.0)}},
    {'ratio': 0.5, 'params': {'radius': 1.0, 'P': 0.5, 'M': Vector3D(2.0, 2.0, 2.0)}},
    {'ratio': 0.2,'params': {'radius': 0.5, 'P': 0.2, 'M': Vector3D(1.0, 1.0, 1.0)}},
]
dynamic_groups = [
    {'ratio': 0.5, 'params': {'radius': 0.5, 'P': 0.2, 'M': Vector3D(1.0, 1.0, 1.0)}},
    {'ratio': 0.5, 'params': {'radius': 0.5, 'P': 0.1, 'M': Vector3D(1.0, 1.0, 1.0)}},
]
radius_groups = [
    {'ratio': 0.5, 'params': {'radius': 2.0, 'P': 0.5, 'M': Vector3D(3.0, 3.0, 3.0)}},
    {'ratio': 0.5, 'params': {'radius': 0.5, 'P': 0.5, 'M': Vector3D(1.0, 1.0, 1.0)}},
]
role_based_groups = [
    {'ratio': 0.4, 'params': {'radius': 0.8, 'P': 0.8, 'M': Vector3D(2.0, 2.0, 2.0)}},
    {'ratio': 0.6, 'params': {'radius': 0.4, 'P': 0.3, 'M': Vector3D(0.5, 0.5, 0.5)}},
]

# test ： genelized-test

# 通用测试

test_groups = [
    {'ratio': 0.5, 'params': {'radius': 1.0, 'P': 0.5, 'M': Vector3D(1.0, 1.0, 1.0)}},
    {'ratio': 0.5, 'params': {'radius': 1.0, 'P': 0.5, 'M': Vector3D(1.0, 1.0, 1.0)}},
]

R1_groups = [
    {'ratio': 0.5, 'params': {'radius': 2.0, 'P': 0.5, 'M': Vector3D(1.0, 1.0, 1.0)}},
    {'ratio': 0.5, 'params': {'radius': 2.0, 'P': 0.5, 'M': Vector3D(1.0, 1.0, 1.0)}},
]

R2_groups = [
    {'ratio': 0.5, 'params': {'radius': 0.5, 'P': 0.5, 'M': Vector3D(1.0, 1.0, 1.0)}},
    {'ratio': 0.5, 'params': {'radius': 0.5, 'P': 0.5, 'M': Vector3D(1.0, 1.0, 1.0)}},
]

R3_groups = [
    {'ratio': 0.5, 'params': {'radius': 2.0, 'P': 0.5, 'M': Vector3D(1.0, 1.0, 1.0)}},
    {'ratio': 0.5, 'params': {'radius': 1.0, 'P': 0.5, 'M': Vector3D(1.0, 1.0, 1.0)}},
]

R4_groups = [
    {'ratio': 0.5, 'params': {'radius': 2.0, 'P': 0.5, 'M': Vector3D(1.0, 1.0, 1.0)}},
    {'ratio': 0.5, 'params': {'radius': 0.5, 'P': 0.5, 'M': Vector3D(1.0, 1.0, 1.0)}},
]

R5_groups = [
    {'ratio': 0.5, 'params': {'radius': 1.0, 'P': 0.5, 'M': Vector3D(1.0, 1.0, 1.0)}},
    {'ratio': 0.5, 'params': {'radius': 0.5, 'P': 0.5, 'M': Vector3D(1.0, 1.0, 1.0)}},
]

## PIVO专属测试
P1_groups = [
    {'ratio': 0.5, 'params': {'radius': 1.0, 'P': 0.4, 'M': Vector3D(1.0, 1.0, 1.0)}},
    {'ratio': 0.5, 'params': {'radius': 1.0, 'P': 0.6, 'M': Vector3D(1.0, 1.0, 1.0)}},
]

P2_groups = [
    {'ratio': 0.5, 'params': {'radius': 1.0, 'P': 0.3, 'M': Vector3D(1.0, 1.0, 1.0)}},
    {'ratio': 0.5, 'params': {'radius': 1.0, 'P': 0.7, 'M': Vector3D(1.0, 1.0, 1.0)}},
]

P3_groups = [
    {'ratio': 0.5, 'params': {'radius': 1.0, 'P': 0.2, 'M': Vector3D(1.0, 1.0, 1.0)}},
    {'ratio': 0.5, 'params': {'radius': 1.0, 'P': 0.8, 'M': Vector3D(1.0, 1.0, 1.0)}},
]

P4_groups = [
    {'ratio': 0.5, 'params': {'radius': 1.0, 'P': 0.1, 'M': Vector3D(1.0, 1.0, 1.0)}},
    {'ratio': 0.5, 'params': {'radius': 1.0, 'P': 0.9, 'M': Vector3D(1.0, 1.0, 1.0)}},
]

P5_groups = [
    {'ratio': 0.5, 'params': {'radius': 1.0, 'P': 0.5, 'M': Vector3D(1.0, 1.0, 1.0)}},
    {'ratio': 0.5, 'params': {'radius': 1.0, 'P': 0.2, 'M': Vector3D(1.0, 1.0, 1.0)}},
]

P6_groups = [
    {'ratio': 0.5, 'params': {'radius': 1.0, 'P': 0.5, 'M': Vector3D(1.0, 1.0, 1.0)}},
    {'ratio': 0.5, 'params': {'radius': 1.0, 'P': 0.4, 'M': Vector3D(1.0, 1.0, 1.0)}},
]

P7_groups = [
    {'ratio': 0.5, 'params': {'radius': 1.0, 'P': 0.5, 'M': Vector3D(1.0, 1.0, 1.0)}},
    {'ratio': 0.5, 'params': {'radius': 1.0, 'P': 0.6, 'M': Vector3D(1.0, 1.0, 1.0)}},
]

P8_groups = [
    {'ratio': 0.5, 'params': {'radius': 1.0, 'P': 0.5, 'M': Vector3D(1.0, 1.0, 1.0)}},
    {'ratio': 0.5, 'params': {'radius': 1.0, 'P': 0.8, 'M': Vector3D(1.0, 1.0, 1.0)}},
]


M1_groups = [
    {'ratio': 0.5, 'params': {'radius': 1.0, 'P': 0.5, 'M': Vector3D(2.0, 2.0, 2.0)}},
    {'ratio': 0.5, 'params': {'radius': 1.0, 'P': 0.5, 'M': Vector3D(1.0, 1.0, 1.0)}},
]

M2_groups = [
    {'ratio': 0.5, 'params': {'radius': 1.0, 'P': 0.5, 'M': Vector3D(0.5, 0.5, 0.5)}},
    {'ratio': 0.5, 'params': {'radius': 1.0, 'P': 0.5, 'M': Vector3D(1.0, 1.0, 1.0)}},
]

M3_groups = [
    {'ratio': 0.5, 'params': {'radius': 1.0, 'P': 0.5, 'M': Vector3D(2.0, 2.0, 2.0)}},
    {'ratio': 0.5, 'params': {'radius': 1.0, 'P': 0.5, 'M': Vector3D(0.5, 0.5, 0.5)}},
]

M4_groups = [
    {'ratio': 0.5, 'params': {'radius': 1.0, 'P': 0.5, 'M': Vector3D(2.0, 2.0, 2.0)}},
    {'ratio': 0.5, 'params': {'radius': 1.0, 'P': 0.5, 'M': Vector3D(2.0, 2.0, 2.0)}},
]

M5_groups = [
    {'ratio': 0.5, 'params': {'radius': 1.0, 'P': 0.5, 'M': Vector3D(0.5, 0.5, 0.5)}},
    {'ratio': 0.5, 'params': {'radius': 1.0, 'P': 0.5, 'M': Vector3D(0.5, 0.5, 0.5)}},
]

# 混合参数比较场景1

RM1_groups = [
    {'ratio': 0.5, 'params': {'radius': 1.0, 'P': 0.5, 'M': Vector3D(0.5, 0.5, 0.5)}},
    {'ratio': 0.5, 'params': {'radius': 1.0, 'P': 0.5, 'M': Vector3D(0.5, 0.5, 0.5)}},
]

RM2_groups = [
    {'ratio': 0.5, 'params': {'radius': 1.0, 'P': 0.5, 'M': Vector3D(0.5, 0.5, 0.5)}},
    {'ratio': 0.5, 'params': {'radius': 1.0, 'P': 0.5, 'M': Vector3D(0.5, 0.5, 0.5)}},
]

RM3_groups = [
    {'ratio': 0.5, 'params': {'radius': 1.0, 'P': 0.5, 'M': Vector3D(0.5, 0.5, 0.5)}},
    {'ratio': 0.5, 'params': {'radius': 1.0, 'P': 0.5, 'M': Vector3D(0.5, 0.5, 0.5)}},
]

RM4_groups = [
    {'ratio': 0.5, 'params': {'radius': 1.0, 'P': 0.5, 'M': Vector3D(0.5, 0.5, 0.5)}},
    {'ratio': 0.5, 'params': {'radius': 1.0, 'P': 0.5, 'M': Vector3D(0.5, 0.5, 0.5)}},
]

RM5_groups = [
    {'ratio': 0.5, 'params': {'radius': 1.0, 'P': 0.5, 'M': Vector3D(0.5, 0.5, 0.5)}},
    {'ratio': 0.5, 'params': {'radius': 1.0, 'P': 0.5, 'M': Vector3D(0.5, 0.5, 0.5)}},
]

RM6_groups = [
    {'ratio': 0.5, 'params': {'radius': 1.0, 'P': 0.5, 'M': Vector3D(0.5, 0.5, 0.5)}},
    {'ratio': 0.5, 'params': {'radius': 1.0, 'P': 0.5, 'M': Vector3D(0.5, 0.5, 0.5)}},
]

# 混合参数比较场景2

PR1_groups = [
    {'ratio': 0.5, 'params': {'radius': 1.0, 'P': 0.5, 'M': Vector3D(0.5, 0.5, 0.5)}},
    {'ratio': 0.5, 'params': {'radius': 1.0, 'P': 0.5, 'M': Vector3D(0.5, 0.5, 0.5)}},
]

# 比例掺杂场景

SphereScenario_factory = {    
    # 'SPHERE_DISCRETE': lambda: HeterogeneousSphereScenario(agent_groups=discrete_groups, num_agents=cfg.NUM_AGENTS).create_agents(),
    # 'SPHERE_DYNAMIC': lambda: HeterogeneousSphereScenario(agent_groups=dynamic_groups, num_agents=cfg.NUM_AGENTS).create_agents(),
    # 'SPHERE_INERTIA' : lambda: HeterogeneousSphereScenario(agent_groups=radius_groups, num_agents=cfg.NUM_AGENTS).create_agents(),
    # 'SPHERE_ROLE_BASED': lambda: HeterogeneousSphereScenario(agent_groups=role_based_groups, num_agents=cfg.NUM_AGENTS).create_agents(),
    # 'Baseline': lambda: SphereScenario(agent_groups=test_groups, num_agents=cfg.NUM_AGENTS).create_agents(),
    'R1': lambda: SphereScenario(agent_groups=R1_groups, num_agents=cfg.NUM_AGENTS).create_agents(),
    'R2': lambda: SphereScenario(agent_groups=R2_groups, num_agents=cfg.NUM_AGENTS).create_agents(),
    'R3': lambda: SphereScenario(agent_groups=R3_groups, num_agents=cfg.NUM_AGENTS).create_agents(),
    'R4': lambda: SphereScenario(agent_groups=R4_groups, num_agents=cfg.NUM_AGENTS).create_agents(),
    'R5': lambda: SphereScenario(agent_groups=R5_groups, num_agents=cfg.NUM_AGENTS).create_agents(),
    'P1': lambda: SphereScenario(agent_groups=P1_groups, num_agents=cfg.NUM_AGENTS).create_agents(),
    'P2': lambda: SphereScenario(agent_groups=P2_groups, num_agents=cfg.NUM_AGENTS).create_agents(),
    'P3': lambda: SphereScenario(agent_groups=P3_groups, num_agents=cfg.NUM_AGENTS).create_agents(),
    'P4': lambda: SphereScenario(agent_groups=P4_groups, num_agents=cfg.NUM_AGENTS).create_agents(),
    'P5': lambda: SphereScenario(agent_groups=P5_groups, num_agents=cfg.NUM_AGENTS).create_agents(),
    'P6': lambda: SphereScenario(agent_groups=P6_groups, num_agents=cfg.NUM_AGENTS).create_agents(),
    'P7': lambda: SphereScenario(agent_groups=P7_groups, num_agents=cfg.NUM_AGENTS).create_agents(),
    'P8': lambda: SphereScenario(agent_groups=P8_groups, num_agents=cfg.NUM_AGENTS).create_agents(),
    'M1': lambda: SphereScenario(agent_groups=M1_groups, num_agents=cfg.NUM_AGENTS).create_agents(),
    'M2': lambda: SphereScenario(agent_groups=M2_groups, num_agents=cfg.NUM_AGENTS).create_agents(),
    'M3': lambda: SphereScenario(agent_groups=M3_groups, num_agents=cfg.NUM_AGENTS).create_agents(),
    'M4': lambda: SphereScenario(agent_groups=M4_groups, num_agents=cfg.NUM_AGENTS).create_agents(),
    'M5': lambda: SphereScenario(agent_groups=M5_groups, num_agents=cfg.NUM_AGENTS).create_agents(),
}

CircleScenario_factory = {    
    'Baseline': lambda: CircleScenario(agent_groups=test_groups, num_agents=cfg.NUM_AGENTS).create_agents(),
    'R1':lambda: CircleScenario(agent_groups=R1_groups, num_agents=cfg.NUM_AGENTS).create_agents(),
    'R2':lambda: CircleScenario(agent_groups=R2_groups, num_agents=cfg.NUM_AGENTS).create_agents(),
    'R3':lambda: CircleScenario(agent_groups=R3_groups, num_agents=cfg.NUM_AGENTS).create_agents(),
    'R4':lambda: CircleScenario(agent_groups=R4_groups, num_agents=cfg.NUM_AGENTS).create_agents(),
    'R5':lambda: CircleScenario(agent_groups=R5_groups, num_agents=cfg.NUM_AGENTS).create_agents(),
}
