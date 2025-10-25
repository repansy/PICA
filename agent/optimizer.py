

class Optimizer:
    '''
    目前仅有
    '''
    def __init__(self, num_particles=100):
        self.num_particles = num_particles
        # belief_model[neighbor_id] = {'particles': [(g, P), ...], 'weights': [w, ...]}
        self.belief_model = {}

    def run(self, self_agent, neighbors, dt):
        # --- 1. 对每个邻居更新信念模型 ---
        for neighbor in neighbors:
            if neighbor.id not in self.belief_model:
                self._initialize_particles(neighbor.id)
            
            # 预测步 (简单模型：状态不变 + 噪声)
            self._predict_step(neighbor.id)
            
            # 更新步 (核心)
            observed_velocity = neighbor.mu_vel
            self._update_step(self_agent, neighbor, observed_velocity)
            
            # 重采样 (如果需要)
            self._resample_if_needed(neighbor.id)

        # --- 2. 基于更新后的信念模型，生成决策输出 ---
        slow_brain_outputs = {}
        for neighbor in neighbors:
            particles = self.belief_model[neighbor.id]['particles']
            weights = self.belief_model[neighbor.id]['weights']
            
            # 计算期望优先级
            p_expected = sum(w * p for (g, p), w in zip(particles, weights))
            
            # 计算解析法 alpha
            alpha_a = self._calculate_analytical_alpha(self_agent, neighbor, p_expected)
            
            # 计算期望目标和轨迹
            g_expected = sum((g * w for (g, p), w in zip(particles, weights)), Vector3D())
            
            # 决策轮廓策略
            should_contour = self._decide_contour_strategy(self_agent, neighbor, g_expected, neighbors)
            
            slow_brain_outputs[neighbor.id] = {'alpha_a': alpha_a, 'should_contour': should_contour}
            
        return slow_brain_outputs

    def _predict_step(self, neighbor_id):
        # 简单地给每个粒子的状态增加一点噪声
        # ...
        pass
        
    def _update_step(self, self_agent, neighbor, observed_velocity):
        particles = self.belief_model[neighbor.id]['particles']
        weights = self.belief_model[neighbor.id]['weights']
        
        new_weights = []
        for i, (g_hypo, p_hypo) in enumerate(particles):
            # 1. 模拟邻居在当前假设下的决策
            v_pref_hypo = (g_hypo - neighbor.mu_pos).normalized() * neighbor.max_speed
            # ... (运行一个简化的决策模型，得到 v_pred_hypo)
            # 这个模型可以使用 self_agent 的信息，因为它模拟的是交互
            v_pred_hypo = self._simulate_neighbor_decision(self_agent, neighbor, v_pref_hypo, p_hypo)

            # 2. 计算似然
            likelihood = math.exp(- (observed_velocity - v_pred_hypo).norm_sq() / (2 * cfg.OBSERVATION_NOISE**2) )
            
            new_weights.append(weights[i] * likelihood)
        
        # 3. 更新并归一化权重
        total_weight = sum(new_weights)
        self.belief_model[neighbor.id]['weights'] = [w / total_weight for w in new_weights]

    def _initialize_particles(self, neighbor_id):
        # 初始化时，可以在一个较大范围内均匀撒点
        # 例如，目标点可以在一个以邻居为中心的圆环上，优先级在[0,2]之间
        # ...
        pass

    def _resample_if_needed(self, neighbor_id):
        # 当有效粒子数 (1 / sum(w^2)) 小于阈值时，进行重采样
        # ...
        pass
        
    def _simulate_neighbor_decision(self, self_agent, neighbor, v_pref_hypo, p_hypo):
        # 这是一个简化的、运行在“慢脑”中的模拟器
        # 它计算邻居 j 在面对 i 时，如果它的期望速度是 v_pref_hypo, 优先级是 p_hypo
        # 那么它最可能的速度是什么
        # ...
        return v_pref_hypo # 最简单的模型，可以更复杂


def _optimize_alpha_hybrid(self, neighbor: 'PicaAgent', threat_score: float, effective_states: Dict, v_pref: Vector3D):
    """
    混合优化责任参数α（核心创新2：混合责任分配）结合启发式规则和解析优化，动态计算最优责任参数α。
    根据风险评分加权融合两个模型的结果。
    参数:
        neighbor: 邻居智能体
        risk_score: 风险评分
        effective_states: 有效状态字典
        v_pref_i: 自身的首选速度
    """
    alpha_heuristic = self._calculate_heuristic_alpha(neighbor)
    alpha_analytical = self._calculate_analytical_alpha(neighbor, effective_states, v_pref)
    
    w = np.clip(
        (threat_score - cfg.RISK_THRESHOLD_LOW) / (cfg.RISK_THRESHOLD_HIGH - cfg.RISK_THRESHOLD_LOW + 1e-6), 
        0, 1
    )
    alpha_star = (1 - w) * alpha_heuristic + w * alpha_analytical
    self.alphas[neighbor.id] = alpha_star




def _calculate_heuristic_alpha(self, neighbor: 'PicaAgent') -> float:
    """
    计算启发式责任α，基于直观规则：谁周围更拥挤、谁的任务优先级越低，谁就多承担规避责任。
    参数:
        neighbor: 邻居智能体 
    返回:
        float: 启发式责任值α
    """
    epsilon = 1e-6
    my_score = (self.rho_smoothed + 0.1) / (self.priority + epsilon)
    neighbor_score = (neighbor.rho_smoothed + 0.1) / (neighbor.priority + epsilon)
    
    total_score = my_score + neighbor_score
    if total_score < epsilon:
        return 0.5
    
    # BUG FIX 2: alpha 是我方的责任比例，应为 my_score / total_score
    return my_score / total_score

def _calculate_analytical_alpha(self, neighbor: 'PicaAgent', states: Dict, v_pref_i: Vector3D) -> float:
    """
    计算解析法责任α（核心创新3：物理感知优化）
    通过构建并最小化代价函数，求解理论上的最优责任α。
    代价函数考虑了双方的运动偏差和物理特性（惯性矩阵）。
    参数:
        neighbor: 邻居智能体
        states: 有效状态字典
        v_pref_i: 自身的首选速度
        v_pref_j: 邻居的首选速度    
    返回:
        float: 解析法责任值α
    """
    v_pref_j = neighbor._get_preferred_velocity()

    epsilon = 1e-6
    delta_alpha = 0.01 # 数值微分步长
    # 1. 计算我方速度关于责任α的梯度（通过数值微分）
    # 1.1 计算α=0时的约束和速度
    constraint_i_at_0 = self._create_orca_halfspace_from_states(0.0, states)
    # 1.2 计算α=δα时的约束和速度
    constraint_i_at_delta = self._create_orca_halfspace_from_states(delta_alpha, states)
    if not constraint_i_at_0 or not constraint_i_at_delta: return 0.5 # 约束无效时返回默认值
    # 1.3 求解两种责任下的速度
    v_i_at_0 = self._solve_single_constraint(constraint_i_at_0, v_pref_i)
    v_i_at_delta = self._solve_single_constraint(constraint_i_at_delta, v_pref_i)
    # 1.4 计算速度关于α的梯度（dv_i/dα）
    a_i_vec = (v_i_at_delta - v_i_at_0) / delta_alpha
    b_i_vec = v_i_at_0

    # 2. 计算邻居速度关于责任α的梯度（注意：邻居的责任α_j = 1 - α_i）
    # 交换视角，从邻居的角度看交互
    states_j_view = {
        "pos_i": states["pos_j"], "vel_i": states["vel_j"], "radius_i": states["radius_j"],
        "pos_j": states["pos_i"], "vel_j": states["vel_i"], "radius_j": states["radius_i"]
    }
    constraint_j_at_0 = self._create_orca_halfspace_from_states(0.0, states_j_view)
    constraint_j_at_delta = self._create_orca_halfspace_from_states(delta_alpha, states_j_view)
    if not constraint_j_at_0 or not constraint_j_at_delta: return 0.5
    
    v_j_at_0 = self._solve_single_constraint(constraint_j_at_0, v_pref_j)
    v_j_at_delta = self._solve_single_constraint(constraint_j_at_delta, v_pref_j)
    a_j_vec = (v_j_at_delta - v_j_at_0) / delta_alpha
    b_j_vec = v_j_at_0
    
    # 3. 构建二次代价函数的系数
    M_i, M_j = self.inertia_matrix, neighbor.inertia_matrix # 惯性矩阵
    rho_i, rho_j = self.rho_smoothed, neighbor.rho_smoothed # 局部密度

    # 3.1 计算二次项系数 K₁
    term1_i = rho_i * (a_i_vec.to_numpy().T @ M_i @ a_i_vec.to_numpy())
    term1_j = rho_j * (a_j_vec.to_numpy().T @ M_j @ a_j_vec.to_numpy())
    K1 = term1_i + term1_j

    # 3.2 计算一次项系数 K₂（简化表示，实际计算更复杂）
    diff_i = b_i_vec - v_pref_i
    diff_j = a_j_vec + b_j_vec - v_pref_j
    term2_i = 2 * rho_i * (a_i_vec.to_numpy().T @ M_i @ diff_i.to_numpy())
    term2_j = -2 * rho_j * (a_j_vec.to_numpy().T @ M_j @ diff_j.to_numpy())
    priority_term = self.priority - neighbor.priority
    K2 = term2_i + term2_j + priority_term
    
    # 4. 求解二次函数最小值
    if abs(2 * K1) < epsilon:
        my_responsibility = 1.0 if K2 > 0 else 0.0
    else:
        my_responsibility = -K2 / (2 * K1)
    
    # BUG FIX 3: alpha* 已经是我方责任，直接返回
    return my_responsibility


def _slow_brain_run(self, neighbors: List['PicaAgent']) -> Dict:
    # ... (实现预测、解析法alpha)
    
    output = {}
    for neighbor in neighbors:
        # ... (alpha_a 计算)
        
        # 轮廓策略决策
        # 假设已完成预测得到 pred_states
        pred_states = self._predict_interaction(neighbor)
        score = self._calculate_contour_score(pred_states)
        should_contour = score > cfg.CONTOUR_THRESHOLD
        
        output[neighbor.id] = {'alpha_a': 0.5, 'should_contour': should_contour} # 占位
    return output

def _calculate_contour_score(self, pred_states: Dict) -> float:
    """
    计算激活轮廓策略的分数。
    """
    rel_pos = pred_states['rel_pos']
    rel_vel = pred_states['rel_vel']
    
    if rel_pos.norm() < 1e-6 or rel_vel.norm() < 1e-6: return 0.0

    # 切向度 (1 - |cos(theta)|)
    tangentiality = 1.0 - abs(rel_vel.dot(rel_pos)) / (rel_vel.norm() * rel_pos.norm())
    
    # 分离度 (越远离，分越高)
    separation = 1 / (1 + math.exp(-0.5 * rel_vel.dot(rel_pos))) # Sigmoid
    
    # 组合分数 (可以根据需求调整权重)
    # 在远离(separation>0.5)或接近垂直(tangentiality>0.8)时得分高
    score = separation + tangentiality 
    return score