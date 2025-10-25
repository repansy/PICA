# slow_brain.py
import math
import numpy as np
from collections import deque
from typing import List, Dict

# 导入Agent类以进行类型提示，避免循环导入问题
from typing import TYPE_CHECKING
if TYPE_CHECKING:
    from pica_agent import Agent

from utils.pica_structures import Vector3D
import enviroments.config as cfg

class SlowBrainPolicy:
    """
    慢脑输出的战略策略。
    这是一个数据容器，用于在慢脑和快脑之间传递信息。
    """
    # 具有长远协同节奏的理想速度
    v_ideal: Vector3D = Vector3D(0, 0, 0)
    
    # 用于QP成本函数的动态权重矩阵，体现了战略偏好
    m_cost: np.ndarray = np.eye(3)

class NeighborBelief:
    """
    慢脑对某个邻居的内部信念模型。
    存储了通过时频分析得出的关于邻居行为模式的认知。
    """
    # 邻居速度轨迹中的主导行为频率 (Hz)
    omega: float = 0.0
    
    # 对应主导频率的相位 (radians)
    phi: float = 0.0
    
    # 对这个频率-相位模型的置信度 (0 to 1)
    confidence: float = 0.0

class SlowBrain:
    """
    战略脑 (The Deliberative Brain):
    - 内部高频思考：持续观察环境，更新对邻居行为模式的信念。
    - 外部低频输出：仅在对自己的规划有足够信心时，才输出一个具有远见的战略策略。
    
    这个模块负责处理需要时间预见性和复杂协同的战略性问题。
    """
    def __init__(self, self_agent: 'Agent'):
        """
        初始化慢脑。
        
        Args:
            self_agent (Agent): 对拥有此慢脑的Agent对象的引用，用于访问其自身状态。
        """
        self._self_agent = self_agent
        self.policy = SlowBrainPolicy()
        
        # 历史缓冲区现在存储完整的状态字典
        self.history_buffer = deque(maxlen=cfg.HISTORY_BUFFER_SIZE)
        
        self.neighbor_beliefs: Dict[int, NeighborBelief] = {}
        self.timer = 0.0

    def think(self, neighbors: List['Agent'], dt: float):
        """
        慢脑的内部思考循环，随高频主循环调用。
        
        Args:
            neighbors (List['Agent']): 当前感知到的所有邻居。
            dt (float): 高频循环的时间步长。
        """
        self.timer += dt
        
        # 1. 记录历史状态
        self.history_buffer.append({
            'pos': self._self_agent.pos, 
            'vel': self._self_agent.vel,
            # 存储邻居的快照，用于后续分析
            'neighbors_snapshot': {n.id: {'pos': n.pos, 'vel': n.vel} for n in neighbors}
        })
        
        # 2. (高频) 更新对邻居的信念
        self._update_beliefs(neighbors)

        # 3. (高频) 生成一个“候选”的战略策略
        candidate_policy = self._generate_candidate_policy(neighbors, dt)

        # 4. (低频) 根据门控条件，决定是否用候选策略更新最终输出
        if self._should_output_policy():
            self.policy = candidate_policy
            self.timer = 0.0

    def get_policy(self) -> SlowBrainPolicy:
        """供外部(快脑)调用的接口，获取最新的、经过门控的战略策略。"""
        return self.policy

    def _update_beliefs(self, neighbors: List['Agent']):
        """
        [非简化] 基于历史轨迹，通过FFT更新对邻居行为模式的信念。
        """
        if len(self.history_buffer) < cfg.HISTORY_BUFFER_SIZE:
            return # 历史数据不足

        for neighbor in neighbors:
            if neighbor.id not in self.neighbor_beliefs:
                self.neighbor_beliefs[neighbor.id] = NeighborBelief()

            # --- 步骤 1: 提取用于分析的信号 ---
            # 我们分析“规避速度”：即邻居速度在其与我方相对位置的垂直平面上的投影。
            # 这个信号最能反映邻居的规避动作，而不是其朝向目标的主体运动。
            evasion_signal = []
            valid_history = True
            for frame in self.history_buffer:
                if neighbor.id in frame['neighbors_snapshot']:
                    my_pos = frame['pos']
                    neighbor_pos = frame['neighbors_snapshot'][neighbor.id]['pos']
                    neighbor_vel = frame['neighbors_snapshot'][neighbor.id]['vel']
                    
                    rel_pos = neighbor_pos - my_pos
                    if rel_pos.norm_sq() < 1e-6: 
                        evasion_signal.append(0.0)
                        continue

                    # 计算速度在相对位置上的投影
                    v_parallel = rel_pos * (neighbor_vel.dot(rel_pos) / rel_pos.norm_sq())
                    # 规避速度是垂直分量
                    v_evasion = neighbor_vel - v_parallel
                    evasion_signal.append(v_evasion.norm())
                else:
                    valid_history = False
                    break
            
            if not valid_history or len(evasion_signal) < cfg.HISTORY_BUFFER_SIZE:
                continue

            # --- 步骤 2: FFT 时频分析 ---
            signal = np.array(evasion_signal)
            # 应用汉宁窗以减少频谱泄漏
            windowed_signal = signal * np.hanning(len(signal))
            
            fft_result = np.fft.fft(windowed_signal - np.mean(windowed_signal))
            fft_freq = np.fft.fftfreq(len(signal), d=cfg.TIMESTEP)

            # 找到主导频率 (忽略直流分量[0]和奈奎斯特频率)
            n_samples = len(signal)
            valid_indices = range(1, n_samples // 2)
            if not valid_indices: continue

            idx = np.argmax(np.abs(fft_result[valid_indices])) + 1
            
            dominant_freq = abs(fft_freq[idx])
            dominant_phase = np.angle(fft_result[idx])
            
            # --- 步骤 3: 计算置信度 ---
            # 置信度 = 主导频率的能量 / (总能量 - 直流分量能量)
            # 这衡量了信号的周期性有多强。
            total_power = np.sum(np.abs(fft_result[1:n_samples//2])**2)
            dominant_power = np.abs(fft_result[idx])**2
            confidence = dominant_power / (total_power + 1e-6) if total_power > 0 else 0.0

            # --- 步骤 4: 平滑地更新信念 ---
            belief = self.neighbor_beliefs[neighbor.id]
            # 使用指数移动平均 (EMA) 进行平滑更新
            ema_alpha = 0.3
            belief.omega = (1 - ema_alpha) * belief.omega + ema_alpha * dominant_freq
            belief.phi = (1 - ema_alpha) * belief.phi + ema_alpha * dominant_phase
            belief.confidence = (1 - ema_alpha) * belief.confidence + ema_alpha * confidence

    def _generate_candidate_policy(self, neighbors: List['Agent'], dt) -> SlowBrainPolicy:
        """
        [非简化] 基于当前对世界的信念，生成一个候选的战略策略。
        """
        policy = SlowBrainPolicy()
        v_pref = self._self_agent.v_pref
        
        # 1. 频率空间协调 (Kuramoto Model)
        # Kuramoto模型: d(phi_i)/dt = omega_i + K/N * sum(sin(phi_j - phi_i))
        # 我们计算一个“理想相位变化率”，用于调整v_pref的方向
        
        my_phase_dot = 0.0 # 自身内在角速度（假定为0）
        coupling_term = 0.0
        
        if neighbors:
            total_confidence = sum(self.neighbor_beliefs.get(n.id, NeighborBelief()).confidence for n in neighbors)
            
            for neighbor in neighbors:
                belief = self.neighbor_beliefs.get(neighbor.id, NeighborBelief())
                
                # 耦合强度 K 与邻居的置信度和风险相关
                risk = self._self_agent._calculate_risk(self._self_agent, neighbor)
                # 只有当对邻居的认知清晰时，才进行强耦合
                coupling_strength_k = cfg.OSCILLATOR_COUPLING_K * belief.confidence * risk

                # 权重是该邻居的置信度在总置信度中的占比
                weight = belief.confidence / (total_confidence + 1e-6)
                
                # 目标是反相 (错开节奏)，目标相位差为 π
                target_phase_difference = np.pi
                current_phase_difference = belief.phi - 0 # 假设自身相位为0
                
                coupling_term += weight * coupling_strength_k * np.sin(current_phase_difference - target_phase_difference)

        # 理想的相位变化率 = 内在角速度 + 耦合项
        target_phase_velocity = my_phase_dot + coupling_term

        # 2. 生成理想速度 v_ideal
        # 将相位变化率映射为对 v_pref 的一个旋转调整
        # target_phase_velocity > 0: 鼓励顺时针旋转 (等待/延迟)
        # target_phase_velocity < 0: 鼓励逆时针旋转 (加速/抢先)
        # 这是一个启发式映射，可以根据需要设计得更复杂
        rotation_angle = np.clip(target_phase_velocity * dt, -0.5, 0.5) # 限制单步最大旋转角度

        c, s = np.cos(rotation_angle), np.sin(rotation_angle)
        vx, vy = v_pref.x, v_pref.y
        v_ideal_x = vx * c - vy * s
        v_ideal_y = vx * s + vy * c
        
        policy.v_ideal = Vector3D(v_ideal_x, v_ideal_y, v_pref.z)
        if policy.v_ideal.norm() > self._self_agent.max_speed:
            policy.v_ideal = policy.v_ideal.normalized() * self._self_agent.max_speed
        
        # 3. 生成成本矩阵 M_cost
        # M_cost 可以动态调整，以反映战略。例如，如果风险很高，
        # 可以增加 M_cost 的整体尺度，使得QP求解器更不愿意偏离v_ideal和v_pref。
        # 简化实现：成本矩阵就是自身的物理惯性矩阵。
        policy.m_cost = self._self_agent.inertia_matrix
        
        return policy

    def _should_output_policy(self) -> bool:
        """判断是否应该更新输出策略的门控机制。"""
        time_gate_open = self.timer >= cfg.SLOW_BRAIN_OUTPUT_PERIOD
        
        if not self.neighbor_beliefs:
            return time_gate_open

        # 只有当至少有一个邻居的认知达到高置信度时，才考虑置信度门控
        max_confidence = max(b.confidence for b in self.neighbor_beliefs.values())
        confidence_gate_open = max_confidence > cfg.CONFIDENCE_THRESHOLD
        
        return time_gate_open or confidence_gate_open