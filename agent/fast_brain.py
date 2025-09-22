# fast_brain.py
import numpy as np
from typing import List, Dict, Optional
import math

# 导入Agent类以进行类型提示，避免循环导入问题
from typing import TYPE_CHECKING
if TYPE_CHECKING:
    from pica_agent import Agent

from utils.pica_structures import Vector3D
import enviroments.config as cfg

class FastBrain:
    """
    战术脑：高频反应。
    负责构建一个智能的、非对称的、物理可行的安全速度集。
    """
    def __init__(self, self_agent: 'Agent'):
        """
        初始化慢脑。
        Args:
            self_agent (Agent): 对拥有此慢脑的Agent对象的引用，用于访问其自身状态。
        """
        self._self_agent = self_agent
        
    def compute_safe_velocity_set(self, neighbors: List['Agent'], dt: float) -> List[Dict]:
        """
        主函数：计算并返回所有硬约束（RVO + AVO）。
        """
        
        # 1. 计算所有邻居的启发式责任
        alphas_h = {n.id: self._calculate_heuristic_alpha(n) for n in neighbors}

        # 2. 构建非对称RVO约束
        rvo_constraints = []
        for neighbor in neighbors:
            alpha = alphas_h[neighbor.id]
            # TODO: 需要一个_get_effective_state_for_interaction的实现
            effective_states = self._self_agent._get_effective_state_for_interaction(neighbor)
            constraint = self._create_rvo_halfspace(alpha, effective_states)
            if constraint:
                rvo_constraints.append(constraint)
        
        final_constraints = rvo_constraints
            
        return final_constraints

    def _calculate_heuristic_alpha(self, neighbor: 'Agent') -> float:
        """计算我方(self_agent)应承担的启发式责任"""
        c_inertia_i = np.trace(self._self_agent.inertia_matrix)
        c_inertia_j_est = np.trace(neighbor.inertia_matrix) # 使用估计
        p_i = self._self_agent.priority
        p_j_est = neighbor.priority # 使用估计
        rho_i = self._self_agent.rho_smoothed
        rho_j_est = neighbor.rho_smoothed

        term_inertia = c_inertia_i / (c_inertia_i + c_inertia_j_est + 1e-6)
        term_priority = p_i / (p_i + p_j_est + 1e-6)
        term_rho = rho_i / (rho_i + rho_j_est + 1e-6)
        
        # 责任与能力成反比(惯性大则能力弱)，与优先级成反比
        my_responsibility = cfg.ALPHA_WEIGHT_INERTIA * term_inertia + \
                            cfg.ALPHA_WEIGHT_PRIORITY * (1.0 - term_priority) + \
                            cfg.ALPHA_WEIGHT_RHO * term_rho
        
        return np.clip(my_responsibility, 0.0, 1.0)


    def _create_rvo_halfspace(self, alpha: float, states: Dict) -> Optional[Dict]:
        """
        创建ORCA半平面约束
        根据责任参数α和有效状态，创建ORCA半平面约束。
        参数:
            alpha: 责任参数
            states: 有效状态字典    
        返回:
            Optional[Dict]: ORCA约束字典（法线和偏移量），如果无效则返回None
        """
        self = self._self_agent
        
        pos_i, vel_i, r_i = states["pos_i"], states["vel_i"], states["radius_i"]
        pos_j, vel_j, r_j = states["pos_j"], states["vel_j"], states["radius_j"]

        rel_pos = pos_j - pos_i
        rel_vel = vel_i - vel_j
        dist_sq = rel_pos.norm_sq()
        combined_radius = r_i + r_j
        combined_radius_sq = combined_radius ** 2

        if dist_sq < combined_radius_sq:
            # 碰撞处理：强制分离
            inv_time_horizon = 1.0 / cfg.TIMESTEP
            u = (rel_pos.normalized() * (combined_radius - rel_pos.norm())) * inv_time_horizon
            normal = u.normalized()
        else:
            inv_tau = 1.0 / cfg.TIMESTEP
            vo_apex = rel_pos * inv_tau # VO锥中心
            vo_radius_sq = (combined_radius * inv_tau) ** 2# VO锥半径平方
            
            w = rel_vel - vo_apex # 相对速度相对于VO锥中心的向量
            w_norm_sq = w.norm_sq()

            dot_product = w.dot(rel_pos)
            if dot_product < 0 and dot_product**2 > w_norm_sq * dist_sq:
                return None

            if w_norm_sq <= vo_radius_sq:
                normal = w.normalized() if w.norm_sq() > 1e-9 else -rel_pos.normalized()
                u = normal * (math.sqrt(vo_radius_sq) - math.sqrt(w_norm_sq))
            else:
                normal = (w - rel_pos * (dot_product / dist_sq)).normalized()
                u = normal * (w.dot(normal))
        
        # BUG FIX 1: The application of the evasion vector 'u' must be additive.
        # A negative sign here inverts the evasion logic, causing weak or incorrect avoidance.
        plane_point = vel_i + u * alpha
        offset = plane_point.dot(normal)
        
        return {'normal': normal, 'offset': offset}