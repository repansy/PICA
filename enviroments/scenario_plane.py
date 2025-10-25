import numpy as np
import math
from typing import List, Tuple
from utils.pica_structures import Vector3D
from agent.pica_agent import PicaAgent as Agent
# from examples.pica_3d.v2.pica_agent import Agent
# from agent.orca_agent import OrcaAgent as Agent
from examples.pica_3d.v2 import config as cfg


class PlaneScenario:
    """平面场景基类：限制智能体在指定平面内运动（xy/yz/xz）"""
    def __init__(self, 
                 num_agents: int,  # 2或4个智能体
                 plane: str = "xy",  # 指定平面：xy/yz/xz
                 seed: int = 42):
        self.num_agents = num_agents
        # self.num_pairs = num_agents // 2  # 对向组数
        self.plane = plane  # 平面约束
        self.seed = seed
        np.random.seed(seed)  # 种子控制位置生成
        
        # 平面固定坐标（如xy平面固定z=中间值）
        self.fixed_coord = 25
        
        # 计算中心和半径
        self.center = self._get_center()
        self.radius = min(cfg.WORLD_SIZE) * 0.6
        self.start_positions, self.goal_positions = self._generate_opposing_positions()

    def _get_center(self) -> Vector3D:
        """根据平面设置场景中心（固定一个坐标）"""
        if self.plane == "xy":
            return Vector3D(cfg.WORLD_SIZE[0]/2, cfg.WORLD_SIZE[1]/2, self.fixed_coord)
        elif self.plane == "yz":
            return Vector3D(self.fixed_coord, cfg.WORLD_SIZE[1]/2, cfg.WORLD_SIZE[2]/2)
        else:  # xz
            return Vector3D(cfg.WORLD_SIZE[0]/2, self.fixed_coord, cfg.WORLD_SIZE[2]/2)

    def _generate_opposing_positions(self) -> Tuple[List[Vector3D], List[Vector3D]]:
        """
        通用对向点生成：
        1. 均匀分配N/2个角度（0~2π）
        2. 每个角度生成一对对称点（中心两侧）
        3. 目标点为对向点的起点
        """
        starts = []
        goals = []
        
        # 生成均匀分布的角度（支持任意偶数对）
        angles = np.linspace(0, 2*math.pi, self.num_agents, endpoint=False)
        
        for angle in angles:
            # 根据平面计算坐标（以xy平面为例，其他平面类似）
            if self.plane == "xy":
                x1 = self.center.x + self.radius * math.cos(angle)
                y1 = self.center.y + self.radius * math.sin(angle)
                x2 = self.center.x - self.radius * math.cos(angle)
                y2 = self.center.y - self.radius * math.sin(angle)
                pos1 = Vector3D(x1, y1, self.fixed_coord)
                pos2 = Vector3D(x2, y2, self.fixed_coord)
            
            elif self.plane == "yz":
                y1 = self.center.y + self.radius * math.cos(angle)
                z1 = self.center.z + self.radius * math.sin(angle)
                y2 = self.center.y - self.radius * math.cos(angle)
                z2 = self.center.z - self.radius * math.sin(angle)
                pos1 = Vector3D(self.fixed_coord, y1, z1)
                pos2 = Vector3D(self.fixed_coord, y2, z2)
            
            else:  # xz平面
                x1 = self.center.x + self.radius * math.cos(angle)
                z1 = self.center.z + self.radius * math.sin(angle)
                x2 = self.center.x - self.radius * math.cos(angle)
                z2 = self.center.z - self.radius * math.sin(angle)
                pos1 = Vector3D(x1, self.fixed_coord, z1)
                pos2 = Vector3D(x2, self.fixed_coord, z2)
            
            # 添加一对起点（pos1和pos2）
            starts.append(pos1)
            starts.append(pos2)
            # 目标为对向点的起点
            goals.append(pos2)
            goals.append(pos1)
        return starts, goals

    def create_agents_with_heterogeneity(self, hetero_type: str) -> List[Agent]:
        """
        创建带异质性的智能体
        hetero_type: 
            - "M_diff": 仅惯性矩阵不同
            - "P_diff": 仅优先级不同
            - "MP_diff": M和P均不同
        """
        agents = []
        # 定义异质性参数（M为惯性矩阵，P为优先级）
        if hetero_type == "M_diff":
            # M不同：前半为灵活（I），后半为笨重（10I），P相同
            Ms = [np.eye(3) for _ in range(self.num_agents//2)] + \
                 [np.eye(3)*10 for _ in range(self.num_agents - self.num_agents//2)]
            Ps = [50 for _ in range(self.num_agents)]  # P相同
        
        elif hetero_type == "P_diff":
            # P不同：前半低优先级（1），后半高优先级（100），M相同
            Ms = [np.eye(3) for _ in range(self.num_agents)]  # M相同
            Ps = [1 for _ in range(self.num_agents//2)] + [100 for _ in range(self.num_agents - self.num_agents//2)]
        
        elif hetero_type == "MP_diff":
            # M和P均不同：(灵活,低P) vs (笨重,高P)
            Ms = [np.eye(3) for _ in range(self.num_agents//2)] + \
                 [np.eye(3)*10 for _ in range(self.num_agents - self.num_agents//2)]
            Ps = [1 for _ in range(self.num_agents//2)] + [100 for _ in range(self.num_agents - self.num_agents//2)]
        
        # 创建智能体
        for i in range(self.num_agents):
            agents.append(Agent(
                id=i,
                pos=self.start_positions[i],
                goal=self.goal_positions[i],
                inertia_matrix=Ms[i],
                priority=Ps[i]
            ))
        return agents


# 场景工厂：快速创建不同配置的场景
plane_scenario_factory = {
    # 2个智能体，xy平面，不同异质性类型
    "2_agents_xy_M": lambda seed=42: PlaneScenario(2, "xy", seed).create_agents_with_heterogeneity("M_diff"),
    "2_agents_xy_P": lambda seed=42: PlaneScenario(2, "xy", seed).create_agents_with_heterogeneity("P_diff"),
    "2_agents_xy_MP": lambda seed=42: PlaneScenario(2, "xy", seed).create_agents_with_heterogeneity("MP_diff"),
    
    # 4个智能体，xy平面，不同异质性类型
    "4_agents_xy_M": lambda seed=42: PlaneScenario(4, "xy", seed).create_agents_with_heterogeneity("M_diff"),
    "4_agents_xy_P": lambda seed=42: PlaneScenario(4, "xy", seed).create_agents_with_heterogeneity("P_diff"),
    "4_agents_xy_MP": lambda seed=42: PlaneScenario(4, "xy", seed).create_agents_with_heterogeneity("MP_diff"),


    # 8个智能体，xy平面，不同异质性类型
    "8_agents_xy_M": lambda seed=42: PlaneScenario(8, "xy", seed).create_agents_with_heterogeneity("M_diff"),
    "8_agents_xy_P": lambda seed=42: PlaneScenario(8, "xy", seed).create_agents_with_heterogeneity("P_diff"),
    "8_agents_xy_MP": lambda seed=42: PlaneScenario(8, "xy", seed).create_agents_with_heterogeneity("MP_diff"),
    
    # 支持其他平面（如xz），只需修改plane参数
    # "2_agents_xz_M": lambda seed=42: PlaneScenario(2, "xz", seed).create_agents_with_heterogeneity("M_diff")
}
