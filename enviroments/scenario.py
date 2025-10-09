# scenario.py

# from agent.pica_agent import Agent # 使用您最终的Agent类
# from agent.orca_agent import OrcaAgent as Agent
import numpy as np
import math
from typing import List, Dict
from utils.pica_structures import Vector3D
from examples.pica_3d.v2.pica_agent import Agent
from examples.pica_3d.v2 import config as cfg


class BaseSphereScenario:
    """球形场景基类，封装通用逻辑"""
    def __init__(self):
        self.center = Vector3D(
            cfg.WORLD_SIZE[0]/2,
            cfg.WORLD_SIZE[1]/2,
            cfg.WORLD_SIZE[2]/2
        )
        self.radius = min(self.center.x, self.center.y, self.center.z) * 0.8
        self.points = self._generate_fibonacci_lattice()  # 预生成球面均匀点

    def _generate_fibonacci_lattice(self) -> List[Vector3D]:
        """生成斐波那契晶格点，均匀分布在单位球面上"""
        points = []
        phi = math.pi * (3. - math.sqrt(5.))  # 黄金角
        for i in range(cfg.NUM_AGENTS):
            y = 1 - (i / float(cfg.NUM_AGENTS - 1)) * 2  # y从1到-1
            r = math.sqrt(1 - y * y)  # y处的圆半径
            theta = phi * i  # 黄金角增量
            x = math.cos(theta) * r
            z = math.sin(theta) * r
            points.append(Vector3D(x, y, z))
        return points

    def _get_start_goal_pos(self, idx: int) -> (Vector3D, Vector3D): # type: ignore
        """获取第idx个智能体的起点和对跖点目标"""
        start_vec = self.points[idx]
        start_pos = self.center + start_vec * self.radius
        goal_pos = self.center - start_vec * self.radius
        return start_pos, goal_pos

    def create_agents(self) -> List[Agent]:
        """子类需实现该方法，定义权限和惯性分配逻辑"""
        raise NotImplementedError("子类必须实现create_agents方法")


class DiscreteLevelSphereScenario(BaseSphereScenario):
    """方案1：离散层级权限（高/中/低三级）"""
    def __init__(self, high_ratio=0.3, mid_ratio=0.5):
        super().__init__()
        self.high_count = int(cfg.NUM_AGENTS * high_ratio)
        self.mid_count = int(cfg.NUM_AGENTS * mid_ratio)
        self.low_count = cfg.NUM_AGENTS - self.high_count - self.mid_count

    def create_agents(self) -> List[Agent]:
        agents = []
        for i in range(cfg.NUM_AGENTS):
            start_pos, goal_pos = self._get_start_goal_pos(i)
            
            # 权限与惯性分配（高权限对应强抗磁性）
            if i < self.high_count:
                priority = 100.0
                inertia = np.diag([1.0, 50.0, 50.0])  # 难横向移动（抗干扰）
            elif i < self.high_count + self.mid_count:
                priority = 50.0
                inertia = np.diag([1.0, 10.0, 10.0])   # 中等抗干扰
            else:
                priority = 10.0
                inertia = np.eye(3) * 1.0              # 易机动（弱抗干扰）
            
            agents.append(Agent(
                id=i,
                pos=start_pos,
                goal=goal_pos,
                inertia_matrix=inertia,
                priority=priority
            ))
        return agents


class DynamicContinuousSphereScenario(BaseSphereScenario):
    """方案2：连续权限"""
    def __init__(self, base_priority_range=(10, 100)):
        super().__init__()
        self.base_priorities = np.random.uniform(
            base_priority_range[0],
            base_priority_range[1],
            cfg.NUM_AGENTS
        )

    def create_agents(self) -> List[Agent]:
        agents = []
        for i in range(cfg.NUM_AGENTS):
            start_pos, goal_pos = self._get_start_goal_pos(i)
            # 初始惯性（统一敏捷）
            inertia = np.eye(3) * 1.0
            # 基础优先级（连续分布）
            priority = self.base_priorities[i]
            
            agents.append(Agent(
                id=i,
                pos=start_pos,
                goal=goal_pos,
                inertia_matrix=inertia,
                priority=priority
            ))
        return agents


class RoleBasedSphereScenario(BaseSphereScenario):
    """方案3：功能角色绑定权限（物理特性与权限强耦合）"""
    def __init__(self, role_ratio: Dict[str, float] = None):
        super().__init__()
        self.role_ratio = role_ratio or {
            "heavy": 0.3,   # 重型载荷机（高权限+高惯性）
            "agile": 0.5,   # 敏捷侦察机（低权限+低惯性）
            "emergency": 0.2  # 应急机（中权限+动态响应）
        }

    def create_agents(self) -> List[Agent]:
        agents = []
        role_counts = {
            role: int(cfg.NUM_AGENTS * ratio)
            for role, ratio in self.role_ratio.items()
        }
        # 处理整数分配误差
        role_counts["agile"] += cfg.NUM_AGENTS - sum(role_counts.values())

        idx = 0
        # 1. 重型载荷机：高权限+难机动
        for _ in range(role_counts["heavy"]):
            start_pos, goal_pos = self._get_start_goal_pos(idx)
            agents.append(Agent(
                id=idx,
                pos=start_pos,
                goal=goal_pos,
                inertia_matrix=np.diag([10.0, 10.0, 10.0]),  # 高惯性
                priority=200.0
            ))
            idx += 1

        # 2. 敏捷侦察机：低权限+高机动
        for _ in range(role_counts["agile"]):
            start_pos, goal_pos = self._get_start_goal_pos(idx)
            agents.append(Agent(
                id=idx,
                pos=start_pos,
                goal=goal_pos,
                inertia_matrix=np.eye(3) * 1.0,  # 低惯性
                priority=50.0
            ))
            idx += 1

        # 3. 应急机：中权限+中等机动
        for _ in range(role_counts["emergency"]):
            start_pos, goal_pos = self._get_start_goal_pos(idx)
            agents.append(Agent(
                id=idx,
                pos=start_pos,
                goal=goal_pos,
                inertia_matrix=np.diag([5.0, 5.0, 5.0]),  # 中等惯性
                priority=150.0
            ))
            idx += 1

        return agents


# 场景工厂：关联场景名称与创建函数
scenario_factory = {
    # 球形权限场景
    'SPHERE_DISCRETE': lambda: DiscreteLevelSphereScenario().create_agents(),
    'SPHERE_DYNAMIC': lambda: DynamicContinuousSphereScenario().create_agents(),
    'SPHERE_ROLE_BASED': lambda: RoleBasedSphereScenario().create_agents()
}


# 原有场景函数（setup_crossing_scenario等）保持不变，此处省略
#   'CROSSING': setup_crossing_scenario,
#   'CIRCLE_2D': setup_circle_scenario_2d,
#   'ELLIPSOID_3D': setup_ellipsoid_scenario_3d,