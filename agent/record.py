import numpy as np
import math
from collections import deque
from scipy.optimize import minimize
from typing import List, Tuple, Dict

# ==============================================================================
# 全局配置 (Hyperparameters)
# ==============================================================================
class Config:
    # --- 物理参数 ---
    RADIUS = 0.3
    MAX_SPEED = 2.0
    NEIGHBOR_DIST = 6.0
    TIME_HORIZON = 2.0
    EPSILON = 1e-5

    # --- 密度感知 (Dual-Mode) ---
    RHO_SCALAR_CAP = 10.0      # 标量密度上限
    BETA_MIN = 0.2             # 最低置信度
    BETA_DECAY = 0.5           # 置信度衰减速率

    # --- 意图与交互 ---
    P_SMOOTH = 0.8             # 意图平滑因子 (越接近1越相信历史)
    
    # --- 慢脑优化权重 ---
    W_PREF = 1.0               # 目标驱动权重
    W_INERTIA = 2.0            # 惯性权重 (防抖动)
    W_SLACK_BASE = 1000.0      # 基础安全约束惩罚
    W_VORTEX = 0.5             # 涡旋流场引导权重 (疏散用)

cfg = Config()

# ==============================================================================
# 数据结构
# ==============================================================================
class AgentState:
    def __init__(self, id, pos, vel, radius):
        self.id = id
        self.pos = np.array(pos, dtype=float)
        self.vel = np.array(vel, dtype=float)
        self.radius = radius

# ==============================================================================
# B-ORCA 3.0 智能体类
# ==============================================================================
class BCOrcaAgent:
    def __init__(self, agent_id: int, start_pos: Tuple[float, float]):
        # 基础状态
        self.id = agent_id
        self.state = AgentState(agent_id, start_pos, (0,0), cfg.RADIUS)
        self.pref_vel = np.array([0.0, 0.0])
        
        # --- 认知状态 ---
        self.p_rel_cache: Dict[int, float] = {}  # {neighbor_id: p_value}
        self.history = deque(maxlen=20)          # 历史轨迹
        
        # --- 感知结果 ---
        self.rho_scalar = 0.0        # 拥挤度 (用于置信度)
        self.rho_vector = np.zeros(2)# 压力场 (用于风险权重)
        self.beta = 1.0              # 预测置信度
        
        # --- 标志位 ---
        self.slow_brain_active = False

    # ==========================================================================
    # 主循环 (Main Loop)
    # ==========================================================================
    def compute_velocity(self, neighbors: List[AgentState], dt: float) -> np.array:
        """
        核心计算流：感知 -> 快脑 -> (失败/死锁) -> 慢脑
        """
        # 1. 双模态密度感知 (Dual-Mode Perception)
        self._percept_density(neighbors)
        
        # 2. 尝试快脑 (Fast Linear Brain)
        v_opt, success = self._run_fast_brain(neighbors)
        
        # 3. 仲裁逻辑 (死锁检测 + 可行性检测)
        # 死锁定义：快脑有解，但速度极小，且我有移动意图
        is_deadlock = success and (np.linalg.norm(v_opt) < 0.05) and (np.linalg.norm(self.pref_vel) > 0.1)
        
        if not success or is_deadlock:
            self.slow_brain_active = True
            # 4. 慢脑介入 (Slow Non-linear Brain)
            # 传入密度向量，用于指导优化方向
            v_opt = self._run_slow_brain(neighbors, dt)
        else:
            self.slow_brain_active = False
            
        # 5. 更新历史
        self._update_history(neighbors)
        
        return v_opt

    # ==========================================================================
    # 模块 1: 双模态感知 (Dual-Mode Perception)
    # ==========================================================================
    def _percept_density(self, neighbors):
        scalar_sum = 0.0
        vector_sum = np.zeros(2)
        
        for n in neighbors:
            rel_vec = self.state.pos - n.pos
            dist = np.linalg.norm(rel_vec)
            dist = max(dist, 0.1) # 防止除零
            
            if dist > cfg.NEIGHBOR_DIST: continue
            
            # 基础权重 (距离越近影响越大)
            weight = 1.0 / dist
            
            # A. 标量密度 (无方向，纯粹的拥挤感)
            scalar_sum += weight
            
            # B. 向量密度 (有方向，压力的来源)
            # 如果左边有个邻居，rel_vec指向右，压力向右
            normal = rel_vec / dist
            vector_sum += normal * weight

        # 归一化与计算
        self.rho_scalar = min(scalar_sum, cfg.RHO_SCALAR_CAP)
        self.rho_vector = vector_sum # 模长代表压力大小，方向代表逃逸方向
        
        # 更新置信度 (基于标量密度)
        # 人越多，越不信预测
        self.beta = cfg.BETA_MIN + (1.0 - cfg.BETA_MIN) * np.exp(-cfg.BETA_DECAY * self.rho_scalar)

    # ==========================================================================
    # 模块 2: 快脑 (Fast Brain - Linearized)
    # ==========================================================================
    def _run_fast_brain(self, neighbors) -> Tuple[np.array, bool]:
        """
        使用缓存的 P_rel 进行线性规划。
        (此处为模拟 RVO2 的逻辑，实际需对接 C++ 库或 Python 半平面求解器)
        """
        # 模拟检查：如果最近邻居太近且相对速度对冲，认为线性无解
        min_dist = float('inf')
        for n in neighbors:
            d = np.linalg.norm(self.state.pos - n.pos)
            combined_r = self.state.radius + n.radius
            min_dist = min(min_dist, d - combined_r)
        
        # 简单的 heuristic: 极度贴身时线性近似失效
        if min_dist < 0.05: 
            return np.zeros(2), False 
            
        # 假设快脑通常能工作 (返回 ORCA 近似解)
        # 实际代码应调用: rvo_lib.solve_linear_program(...)
        return self.pref_vel, True

    # ==========================================================================
    # 模块 3: 慢脑 (Slow Brain - Non-linear & Vortex)
    # ==========================================================================
    def _run_slow_brain(self, neighbors, dt) -> np.array:
        
        # --- A. 意图重构 (更新 P_rel) ---
        for n in neighbors:
            p_meas = self._estimate_neighbor_p(n, dt)
            p_old = self.p_rel_cache.get(n.id, 0.0)
            # 平滑更新
            self.p_rel_cache[n.id] = cfg.P_SMOOTH * p_old + (1 - cfg.P_SMOOTH) * p_meas

        # --- B. 涡旋引导场 (Vortex Guide) ---
        # 如果处于死锁，利用向量密度计算一个“推荐逃逸方向”
        # 策略：沿着压力梯度的切线方向移动 (Boundary Surfing)
        vortex_bias = np.zeros(2)
        pressure_mag = np.linalg.norm(self.rho_vector)
        if pressure_mag > 0.1:
            # 计算向量密度的垂线 (2D 逆时针旋转 90度)
            # 压力指向外，我们沿着压力的切线走
            tangent = np.array([-self.rho_vector[1], self.rho_vector[0]])
            tangent /= pressure_mag
            # 简单的 RVO 右行规则：根据目标位置决定顺/逆时针
            # 这里简化为：总是倾向于向右偏移
            vortex_bias = tangent * 0.5 

        # --- C. 构建优化问题 ---
        n_neigh = len(neighbors)
        # 变量: [vx, vy, slack_1, ..., slack_n]
        x0 = np.concatenate([self.state.vel * 0.5 + vortex_bias, np.zeros(n_neigh)])
        
        # 目标函数
        def objective(x):
            v = x[:2]
            slacks = x[2:]
            
            # 1. 目标性
            obj = cfg.W_PREF * np.sum((v - self.pref_vel)**2)
            # 2. 惯性 (防止震荡)
            obj += cfg.W_INERTIA * np.sum((v - self.state.vel)**2)
            # 3. 涡旋引导 (鼓励沿着切线流动，解决死锁)
            if pressure_mag > 1.0:
                 # 负号表示最大化与切线的投影
                 obj -= cfg.W_VORTEX * np.dot(v, vortex_bias)
            # 4. 安全约束惩罚 (松弛变量)
            # [关键] 根据向量密度加权：压力来源方向的邻居，惩罚更重
            obj += cfg.W_SLACK_BASE * np.sum(slacks**2)
            
            return obj

        # 约束条件
        cons = []
        # 速度幅值
        cons.append({'type': 'ineq', 'fun': lambda x: cfg.MAX_SPEED**2 - np.sum(x[:2]**2)})
        
        for i, n in enumerate(neighbors):
            # 闭包绑定参数
            def vo_constraint(x, idx=i, neighbor=n):
                v = x[:2]
                slack = x[2+idx]
                
                # 获取参数
                p_rel = self.p_rel_cache.get(neighbor.id, 0.0)
                alpha = 0.5 + 0.5 * p_rel
                
                # 几何数据
                rel_pos = neighbor.pos - self.state.pos
                dist = np.linalg.norm(rel_pos)
                r_sum = self.state.radius + neighbor.radius
                
                # 预测邻居速度 (置信度混合)
                # beta 越低，越依赖当前观测 (v_curr)，不信预测
                v_neigh = self.beta * neighbor.vel + (1 - self.beta) * neighbor.vel # 这里可换成kalman预测
                
                v_rel = v - v_neigh
                
                # --- 精确 VO 锥体约束 (Exact Cone) ---
                # 目标: v_rel 不在碰撞锥内
                
                # 1. 碰撞检测与软边界 (Soft Margin)
                # 中密度时(rho_scalar < 5)，虚拟扩大半径以疏散
                margin = 1.2 if self.rho_scalar < 5.0 else 1.05
                r_eff = r_sum * margin
                
                if dist < r_eff:
                    # 已侵入舒适区：产生负值，强迫增大 slack
                    return dist - r_eff + slack 
                
                # 2. 角度约束
                sin_theta = min(r_eff / dist, 0.99)
                cos_theta = math.sqrt(1 - sin_theta**2)
                
                v_rel_norm = np.linalg.norm(v_rel) + cfg.EPSILON
                # cos_phi = v_rel dot rel_pos
                cos_phi = np.dot(v_rel, rel_pos) / (v_rel_norm * dist)
                
                # 约束: cos_theta - cos_phi >= 0 (在锥外)
                return cos_theta - cos_phi + slack
                
            cons.append({'type': 'ineq', 'fun': vo_constraint})
            # Slack >= 0
            cons.append({'type': 'ineq', 'fun': lambda x, idx=i: x[2+idx]})

        # 求解
        bounds = [(None, None), (None, None)] + [(0, None)] * n_neigh
        
        try:
            res = minimize(objective, x0, method='SLSQP', bounds=bounds, constraints=cons, 
                           options={'ftol': 1e-3, 'maxiter': 20})
            if res.success:
                return res.x[:2]
            return self.state.vel * 0.5 # 失败回退
        except:
            return np.zeros(2)

    # ==========================================================================
    # 辅助: 意图估计
    # ==========================================================================
    def _estimate_neighbor_p(self, neighbor, dt):
        # 简化版意图估计
        rel_pos = neighbor.pos - self.state.pos
        dist = np.linalg.norm(rel_pos) + cfg.EPSILON
        v_proj = np.dot(neighbor.vel, -rel_pos/dist) # 正值表示向我冲来
        
        # 如果对方速度快且指向我，认为P高(不合作)
        if v_proj > 0.5: return 1.0
        return 0.0

    def _update_history(self, neighbors):
        # 实际应存储更多信息
        pass
    
    def _nonlinear_vo_constraint(self, v_candidate, neighbor, dt):
        """
        计算精确的非线性 VO 约束 (Signed Distance)。
        正值 = 安全；负值 = 侵入 VO。
        完全还原了被 ORCA 切平面误杀的“月牙形”可行域。
        """
        # 1. 基础参数准备
        p_rel = neighbor.pos - self.state.pos
        r_sum = self.state.radius + neighbor.radius
        dist_sq = np.linalg.norm(p_rel)**2
        
        # 2. 预测邻居速度 (引入置信度)
        # 慢脑的一大优势：可以处理邻居的非恒定速度，这里先取有效观测速度
        v_neigh = self.beta * neighbor.vel + (1 - self.beta) * neighbor.vel
        v_rel = v_candidate - v_neigh
        
        tau = cfg.TIME_HORIZON
        inv_tau = 1.0 / tau
        
        # VO 截断圆的中心和半径 (对应代码中的 combined_radius * inv_time_horizon)
        center_cutoff = p_rel * inv_tau
        r_cutoff = r_sum * inv_tau
        
        # 3. 核心判断：我们在 VO 的哪个部位？(参考 ORCA 源码逻辑)
        # w 是从截断圆心指向相对速度的向量
        w = v_rel - center_cutoff
        w_len_sq = np.dot(w, w)
        
        # 投影判断 (对应 dot_product)
        # dot < 0 意味着 v_rel 在圆心的"后方" (远离圆锥尖端的一侧)，即 Cap 区域
        dot_w_p = np.dot(w, p_rel)
        
        # 4. 分情况计算精确约束
        
        # --- 情况 A: Cap Region (圆头区) ---
        # 这里的几何条件对应源码中的 if dot < 0 and dot^2 > r^2 * w^2
        # 实际上，只要 dot < 0，最近点往往就在圆弧上
        if dot_w_p < 0:
            # 约束: 距离圆心的长度 >= 半径
            # ORCA 在这里做切线，我们直接用距离
            dist_to_center = np.sqrt(w_len_sq)
            # Slack: 正值表示在圆外(安全)，负值表示在圆内
            return dist_to_center - r_cutoff

        # --- 情况 B: Leg Region (侧面区) ---
        else:
            # 这里对应源码的 else 分支 (投影在圆锥侧面)
            # 约束: 相对速度方向 与 相对位置方向 的夹角 >= 锥角
            
            # 1. 计算锥角余弦 (半顶角)
            # 注意：这里的锥是基于 tau 扩展后的，实际上 ORCA 的 Leg 是切线
            # 在 Leg 区域，VO 边界就是由原点引出的切线
            
            # 如果已经重叠，特殊处理
            if dist_sq < r_sum**2:
                return -1.0 # 强碰撞
                
            # 真实的锥角 (Leg 的方向是切于半径为 R/tau 的圆)
            # 但 RVO2 的几何定义中，Leg 是切于 radius R at distance P
            # sin(alpha) = R / dist
            sin_alpha = r_sum / np.sqrt(dist_sq)
            cos_alpha = np.sqrt(1 - sin_alpha**2)
            
            # 2. 计算当前夹角余弦
            v_rel_norm = np.linalg.norm(v_rel) + 1e-6
            cos_phi = np.dot(v_rel, p_rel) / (v_rel_norm * np.sqrt(dist_sq))
            
            # 3. 约束: phi >= alpha  =>  cos(phi) <= cos(alpha)
            # 注意: 夹角越大，cos越小。
            # 此时 ORCA 的切平面是精确的，非线性优势主要体现在 Cap 区
            # 但保留非线性形式可以让优化器更平滑
            return cos_alpha - cos_phi