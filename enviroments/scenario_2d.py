import numpy as np
import math
from typing import List
from utils.pica2d_structures import Vector2D  # 假设存在2D向量类
from examples.pica_2d.v2.pica2d_agent import Agent  # 2D智能体类
from examples.pica_2d.v2 import config as cfg


class DiskScenario:
    """2D圆形平面场景（对应3D球体的赤道投影）"""
    def __init__(self):
        self.center = Vector2D(
            cfg.WORLD_SIZE[0] / 2,  # 2D世界宽度
            cfg.WORLD_SIZE[1] / 2   # 2D世界高度
        )
        self.radius = min(self.center.x, self.center.y) * 0.8  # 圆形区域半径

    def _generate_equidistant_points(self) -> List[Vector2D]:
        """生成圆上均匀分布的点（用于对跖点场景）"""
        points = []
        for i in range(cfg.NUM_AGENTS):
            theta = 2 * math.pi * i / cfg.NUM_AGENTS  # 均匀角度分布
            x = self.radius * math.cos(theta)
            y = self.radius * math.sin(theta)
            points.append(Vector2D(x, y))
        return points

    def create_agents(self) -> List[Agent]:
        """创建2D智能体，起点和目标为对跖点（圆心对称）"""
        agents = []
        points = self._generate_equidistant_points()
        
        for i in range(cfg.NUM_AGENTS):
            # 起点：圆上某点
            start_pos = self.center + points[i]
            # 目标点：对跖点（圆心对称点）
            goal_pos = self.center - points[i]
            
            # 2D惯性矩阵（仅XY方向）
            if i % 2 == 0:
                # 低惯性（灵活）
                inertia = np.diag([1.0, 1.0])
                priority = 1.0
            else:
                # 高惯性（笨重）
                inertia = np.diag([5.0, 5.0])
                priority = 3.0
            
            agents.append(Agent(
                id=i,
                pos=start_pos,
                goal=goal_pos,
                inertia_matrix=inertia,
                priority=priority
            ))
        return agents


# 2D场景工厂
scenario_factory_2d = {
    'DISK_ANTIPODAL': lambda: DiskScenario().create_agents(),  # 对跖点圆形场景
    # 'DISK_RANDOM': lambda: DiskScenario().create_random_agents()  # 可扩展随机场景
}