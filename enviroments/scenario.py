# scenario.py

# from agent.pica_agent import Agent # 使用您最终的Agent类
# from agent.orca_agent import OrcaAgent as Agent
import numpy as np
import math
from typing import List, Dict
from utils.pica_structures import Vector3D
# from examples.pica_3d.v2.pica_agent import Agent
from agent.orca_agent import OrcaAgent as Agent
from examples.pica_3d.v2 import config as cfg


class BaseSphereScenario:
    """球形场景基类，封装通用逻辑"""
    def __init__(self, seed: int = 42):
        self.seed = seed
        np.random.seed(self.seed)
        self.center = Vector3D(
            cfg.WORLD_SIZE[0]/2,
            cfg.WORLD_SIZE[1]/2,
            cfg.WORLD_SIZE[2]/2
        )
        self.radius = min(self.center.x, self.center.y, self.center.z) * 0.8
        # self.points = self._generate_fibonacci_lattice()  # 预生成球面均匀点
        self.points = self._generate_seed_points()

    def _generate_seed_points(self)->List[Vector3D]:
    # def _generate_fibonacci_lattice(self) -> List[Vector3D]:
        """
        TODO 需要随机种子重新辅助生成的位置
        # 生成斐波那契晶格点，均匀分布在单位球面上
        points = []
        phi = math.pi * (3. - math.sqrt(5.))  # 黄金角
        for i in range(cfg.NUM_AGENTS):
            y = 1 - (i / float(cfg.NUM_AGENTS - 1)) * 2  # y从1到-1
            r = math.sqrt(1 - y * y)  # y处的圆半径
            theta = phi * i  # 黄金角增量
            x = math.cos(theta) * r
            z = math.sin(theta) * r
            points.append(Vector3D(x, y, z))
        """
        points = []
        for _ in range(cfg.NUM_AGENTS):
            # 用种子控制的随机数生成phi和costheta
            phi = np.random.uniform(0, 2 * math.pi)  # 方位角：0~2π
            costheta = np.random.uniform(-1, 1)  # 极角余弦：-1~1（对应极角0~π）
            theta = math.acos(costheta)  # 极角
            
            # 计算球面坐标（单位球）
            x = math.sin(theta) * math.cos(phi)
            y = math.sin(theta) * math.sin(phi)
            z = math.cos(theta)
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


class InertiaSphereScenario(BaseSphereScenario):
    """针对惯性矩阵进行测试"""
    def __init__(self, high_ratio = 0.5):
        super().__init__()
        self.high_count = int(cfg.NUM_AGENTS * high_ratio)

    def create_agents(self):
        agents = []
        priority = 1.0
        for i in range(cfg.NUM_AGENTS):
            start_pos, goal_pos = self._get_start_goal_pos(i)
            if i < self.high_count:
                inertia = np.diag([10.0, 10.0, 10.0]) 
            else:
                inertia = np.diag([1.0, 1.0, 1.0])            
            agents.append(Agent(
                id=i,
                pos=start_pos,
                goal=goal_pos,
                inertia_matrix=inertia,
                priority=priority
            ))
        return agents

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
            elif i < self.high_count + self.mid_count:
                priority = 50.0
            else:
                priority = 10.0
            
            inertia = np.eye(3) * 1.0
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


# scenario.py（新增部分）
class CylinderOvertakeScenario:
    """3D圆柱形流道超车场景：同方向运动，异质无人机"""
    def __init__(self, seed: int = 42, num_agents: int = 20):
        self.seed = seed
        np.random.seed(seed)
        self.num_agents = num_agents
        self.radius = 5.0  # 流道半径
        self.length = 50.0  # 流道长度
        self.center = Vector3D(0, 0, 0)  # 流道起点中心

    def _generate_initial_positions(self) -> List[Vector3D]:
        """生成径向（x,y）和轴向（z）初始位置，避免碰撞"""
        positions = []
        for _ in range(self.num_agents):
            # 径向位置：随机分布在圆内（种子控制）
            r = np.random.uniform(0, self.radius)
            theta = np.random.uniform(0, 2*np.pi)
            x = r * np.cos(theta)
            y = r * np.sin(theta)
            # 轴向位置：起始区域[0, 10)，避免重叠（种子控制间距）
            z = np.random.uniform(0, 10) + len(positions)*2  # 最小间距2m
            positions.append(Vector3D(x, y, z))
        return positions

    def create_agents(self) -> List[Agent]:
        agents = []
        positions = self._generate_initial_positions()
        # 异质性设置：30%高能力，70%低能力（种子控制比例）
        is_high_cap = np.random.choice(
            [True, False], size=self.num_agents, p=[0.3, 0.7]
        )
        
        for i in range(self.num_agents):
            pos = positions[i]
            goal = Vector3D(pos.x, pos.y, self.length)  # 沿z轴到流道末端
            if is_high_cap[i]:
                # 高能力：小尺寸（碰撞半径0.5）、高敏捷（M=I）、快速度（5m/s）
                inertia = np.eye(3)
                v_pref = 5.0
                radius = 0.5
            else:
                # 低能力：大尺寸（碰撞半径1.0）、低敏捷（M=10I）、慢速度（2m/s）
                inertia = np.eye(3) * 10.0
                v_pref = 2.0
                radius = 1.0
            
            agents.append(Agent(
                id=i,
                pos=pos,
                goal=goal,
                inertia_matrix=inertia,
                priority=1.0,  # 权限相同，仅能力差异
                radius=radius,
                v_pref=v_pref  # 新增：理想速度参数
            ))
        return agents


# 场景工厂：关联场景名称与创建函数
scenario_factory = {
    # 球形权限场景
    'SPHERE_INERTIA' : lambda: InertiaSphereScenario().create_agents(),
    'SPHERE_DISCRETE': lambda: DiscreteLevelSphereScenario().create_agents(),
    'SPHERE_DYNAMIC': lambda: DynamicContinuousSphereScenario().create_agents(),
    'SPHERE_ROLE_BASED': lambda: RoleBasedSphereScenario().create_agents(),
    # 'OVERTAKE': lambda: CylinderOvertakeScenario().create_agents()
}
