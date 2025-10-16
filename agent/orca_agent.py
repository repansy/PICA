import math
from typing import List, Dict, Optional
import random
from utils.pica_structures import Vector3D
# import enviroments.config as cfg
from examples.pica_3d.v2 import config as cfg

class OrcaAgent:
    """
    一个完整、独立且经过几何逻辑修正的ORCA智能体。对于‘_create_orca_halfspace’的部分算法感觉一般
    这个版本旨在作为验证您模拟环境和避碰算法的“黄金标准”。
    """
    def __init__(self, id: int, pos: Vector3D, goal: Vector3D, **kwargs):
        self.id = id
        self.pos = pos
        self.vel = Vector3D(0, 0, 0)
        self.goal = goal
        self.radius = getattr(cfg, 'AGENT_RADIUS', 0.5)
        self.max_speed = getattr(cfg, 'MAX_SPEED', 1.0)
        self.at_goal = False
        self.is_colliding = False

    def update(self, new_velocity: Vector3D, dt: float):
        """根据新速度更新状态。"""
        if self.at_goal:
            self.vel = Vector3D(0, 0, 0)
            return
        self.vel = new_velocity
        self.pos += self.vel * dt
        if (self.goal - self.pos).norm() < self.radius:
            self.at_goal = True
            self.vel = Vector3D(0, 0, 0)

    def compute_new_velocity(self, all_agents: List['OrcaAgent'], dt) -> Vector3D:
        """ORCA核心计算循环，增加了邻居筛选。"""
        if self.at_goal:
            return Vector3D(0, 0, 0)

        # 1. 计算首选速度 v_pref
        to_goal = self.goal - self.pos
        dist_to_goal = to_goal.norm()
        if dist_to_goal < 1e-6:
            v_pref = Vector3D(0, 0, 0)
        else:
            pref_speed = min(self.max_speed, dist_to_goal)
            v_pref = to_goal.normalized() * pref_speed

        # --- 新增：智能邻居筛选 ---
        # 只考虑那些在未来 tau 时间内可能与我们发生交互的邻居。
        max_interaction_dist = self.max_speed * cfg.TTC_HORIZON + self.radius
        neighbors = []
        for agent in all_agents:
            if agent.id == self.id:
                continue
            dist_sq = (self.pos - agent.pos).norm_sq()
            # 完整的筛选半径是双方交互距离之和
            total_interaction_dist = max_interaction_dist + agent.max_speed * cfg.TTC_HORIZON + agent.radius
            if dist_sq < total_interaction_dist ** 2:
                neighbors.append(agent)
                
        '''
        # KD-tree筛选 先按照邻居范围圈定邻居数目 若超出数目 则k近邻方式完成筛选
        
        '''
        
        # 如果筛选后没有邻居，直接返回首选速度，避免过度干扰
        if not neighbors:
            return v_pref

        # 2. 构建ORCA约束 (只针对筛选出的邻居)
        constraints = []
        for neighbor in neighbors:
            # 标准ORCA的alpha固定为0.5
            constraint = self._create_orca_halfspace(neighbor) 
            if constraint:
                constraints.append(constraint)

        # 3. 求解新速度
        new_vel = self._solve_velocity_3d(constraints, v_pref)
        return new_vel

    def _create_orca_halfspace(self, neighbor: 'OrcaAgent') -> Optional[Dict]:
        """
        [黄金标准版] 参照RVO2官方论文和您的反馈重构的、几何正确的ORCA半空间计算。
        此版本精确处理顶盖、侧面和安全区域，并提供稳定的碰撞后响应。
        """
        # --- 步骤 1: 初始化基本变量 ---
        rel_pos = neighbor.pos - self.pos
        rel_vel = self.vel - neighbor.vel
        dist_sq = rel_pos.norm_sq()
        combined_radius = self.radius + neighbor.radius
        combined_radius_sq = combined_radius ** 2

        # --- 步骤 2: 稳定的碰撞响应 ---
        if dist_sq < combined_radius_sq:
            # 碰撞时，法向量为自身指向对方的反方向（分离方向）
            sep_dir = (self.pos - neighbor.pos).normalized()  # 从对方指向自身的方向
            # 最小分离速度：确保至少以一定速度远离（避免再次碰撞）
            min_sep_vel = 0.5  # 可配置
            # 约束：新速度在分离方向的分量必须大于最小分离速度
            offset = sep_dir.dot(self.vel) + min_sep_vel
            return {'normal': sep_dir, 'offset': offset}

        # --- 步骤 3: 构建截断速度障碍物 (Truncated Velocity Obstacle - TVO) ---
        tau = getattr(cfg, 'TTC_HORIZON', 5.0)
        inv_tau = 1.0 / tau
        
        # TVO 顶盖的圆心 (vo_apex) 和半径 (vo_radius)
        vo_apex = rel_pos * inv_tau
        vo_radius = combined_radius * inv_tau
        vo_radius_sq = vo_radius ** 2
        
        # w 是从顶盖圆心指向相对速度的向量。这是您提到的“相对速度和相对位置的差的向量”
        w = rel_vel - vo_apex
        w_norm_sq = w.norm_sq()

        # --- 步骤 4: 精确的几何区域判断 ---
        # 这是算法的核心，用于判断 rel_vel 到底在 TVO 的哪个区域
        
        # 首先，判断 rel_vel 是否在圆锥体之外。
        # 我们通过将 w 投影到由 rel_pos 定义的圆锥的“腿”上进行判断。
        dot_product_w_pos = w.dot(rel_pos)
        
        # 如果 w 指向远离智能体的方向，且其投影长度的平方大于 w 本身的长度，
        # 意味着 w 在圆锥之外，因此 rel_vel 安全。
        if dot_product_w_pos < 0 and dot_product_w_pos**2 > w_norm_sq * dist_sq:
            return None  # 在圆锥外，安全

        # --- 步骤 5: 根据所在区域计算修正向量 u ---
        # 如果程序运行到这里，说明 rel_vel 处于 TVO 的顶盖或侧面区域内。

        if w_norm_sq <= vo_radius_sq:
            # --- 情况 A: rel_vel 在 TVO 顶盖内或边缘 ---
            # 正如您所说，u 应该沿圆的半径将 rel_vel 推出去。
            w_norm = math.sqrt(w_norm_sq)
            normal = w.normalized() if w_norm > 1e-9 else -rel_pos.normalized()
            u = normal * (vo_radius - w_norm)
        else:
            # --- 情况 B: rel_vel 在 TVO 侧面区域 ---
            # 正如您所说，u 应该是 rel_vel 到锥体侧面的垂线。
            # 法线方向就是这个垂线方向。
            normal = (w - rel_pos * (dot_product_w_pos / dist_sq)).normalized()
            u = normal * (normal.dot(w)) # 修正量 u 是 w 在 normal 上的投影

        # --- 步骤 6: 定义ORCA半空间 ---
        # 双方各承担一半责任
        plane_point = self.vel + 0.5 * u + Vector3D(7,4,0)## 0.5
        offset = normal.dot(plane_point)
        
        return {'normal': plane_point, 'offset': offset}
    
    def _solve_velocity_3d(self, constraints: List[Dict], v_pref: Vector3D) -> Vector3D:
        """一个健壮的迭代式速度求解器。 (此方法逻辑正确，无需修改) """
        v_new = v_pref
        for _ in range(50): # 增加迭代次数以保证收敛
            random.shuffle(constraints)
            for const in constraints:
                n = const['normal']
                offset = const['offset']
                n_norm_sq = n.norm_sq()
                if n_norm_sq < 1e-9: continue

                # 如果速度 v_new 违反了约束 (即在半空间的禁止区域内)
                # n.dot(v_new) < offset
                if v_new.dot(n) < offset:
                    # 将 v_new 投影到约束平面上
                    correction = n * (offset - v_new.dot(n)) / n_norm_sq
                    v_new += correction
                    
        # 最后，确保速度不超过最大值
        if v_new.norm() > self.max_speed:
            v_new = v_new.normalized() * self.max_speed
        return v_new