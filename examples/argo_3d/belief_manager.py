'''
意图:
    记忆: self.neighbors字典就是长期记忆。只要一个邻居在字典里，我们就“记得”它。
    
    信念更新: predict_all体现了时间的流逝会让信念变得模糊（协方差增大）。
    update_from_measurement则体现了新信息会使信念变得清晰（协方差减小）。
    
    遗忘机制: forget_lost_neighbors是主动遗忘。
    这对于处理动态变化的集群（有成员加入/离开）至关重要，防止Agent对一个早已离开的“幽灵”进行无效避障。
    
改进：
    完整的EKF实现: 充实了predict_all和update_from_measurement的数学细节，
    包括状态转移矩阵F、过程噪声Q、观测矩阵H以及标准的卡尔曼增益更新步骤。

    参数化噪声: PROCESS_NOISE现在是可调参数，代表了我们对“邻居会多大程度上偏离匀速运动”的假设。
'''

# agent/belief_manager.py
import numpy as np
from typing import Dict
from utils.argo_structures import BeliefState, AgentConfig, State

# Constants
FORGET_THRESHOLD_SECONDS = 5.0
PROCESS_NOISE_POS_STD = 0.05 # How much we expect neighbors to deviate from constant velocity
PROCESS_NOISE_VEL_STD = 0.1

class BeliefManager:
    def __init__(self, agent_id: int):
        self.agent_id = agent_id
        self.neighbors: Dict[int, BeliefState] = {}
        # 6x6 Process Noise Covariance Matrix Q
        self.Q = np.diag([PROCESS_NOISE_POS_STD**2]*3 + [PROCESS_NOISE_VEL_STD**2]*3)
        # 3x6 Observation Matrix H
        self.H = np.block([np.eye(3), np.zeros((3, 3))])

    def _get_state_transition_matrix(self, dt: float) -> np.ndarray:
        F = np.eye(6)
        F[0:3, 3:6] = np.eye(3) * dt
        return F

    def predict_all(self, current_time: float):
        """EKF predict step for all known neighbors."""
        for belief in self.neighbors.values():
            dt = current_time - belief.last_update_time
            if dt <= 0: continue
            
            F = self._get_state_transition_matrix(dt)
            
            belief.mean = F @ belief.mean
            belief.covariance = F @ belief.covariance @ F.T + self.Q * dt # Scale Q by dt
            # Note: last_update_time does NOT change here
            
    def update_from_measurement(self, neighbor_id: int, measurement: Dict, current_time: float):
        """EKF update step when a new measurement arrives."""
        measured_state: State = measurement["state"]
        neighbor_config: AgentConfig = measurement["config"]
        
        # Observation noise covariance R
        R = np.eye(3) * neighbor_config.sensor_noise_std**2
        z = measured_state.pos # We only observe position

        if neighbor_id not in self.neighbors:
            # Initialize belief for a new neighbor
            mean = np.concatenate([measured_state.pos, measured_state.vel])
            covariance = np.eye(6) * 0.1 # Start with some uncertainty
            self.neighbors[neighbor_id] = BeliefState(mean, covariance, current_time, neighbor_config)
            return

        belief = self.neighbors[neighbor_id]
        
        # --- EKF Update Logic ---
        # Innovation (residual)
        y = z - self.H @ belief.mean
        # Innovation covariance
        S = self.H @ belief.covariance @ self.H.T + R
        # Kalman Gain
        K = belief.covariance @ self.H.T @ np.linalg.inv(S)
        
        # Update mean and covariance
        belief.mean = belief.mean + K @ y
        belief.covariance = (np.eye(6) - K @ self.H) @ belief.covariance
        belief.last_update_time = current_time

    def forget_lost_neighbors(self, current_time: float):
        """Removes neighbors that haven't been heard from for too long."""
        lost_neighbors = [
            nid for nid, belief in self.neighbors.items()
            if current_time - belief.last_update_time > FORGET_THRESHOLD_SECONDS
        ]
        for nid in lost_neighbors:
            del self.neighbors[nid]

    def get_current_beliefs(self) -> Dict[int, BeliefState]:
        return self.neighbors