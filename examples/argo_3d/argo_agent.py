'''
意图: 
    Agent类是顶层协调者。它不关心EKF或梯度下降的具体数学，只负责调用正确的模块，完成“感知-决策”的循环。
'''

# agent/agent.py
import numpy as np
from typing import Dict, List

from .belief_manager import BeliefManager
from .optimization import solve_argo_3d_velocity
from utils.argo_structures import State, AgentConfig, Obstacle

class Agent:
    def __init__(self, id: int, initial_state: State, config: AgentConfig, goal: np.ndarray, config_params: Dict):
        self.id = id
        self.config = config
        self.state = initial_state
        self.goal = goal
        
        # --- Store global parameters ---
        self.config_params = config_params
        
        self.belief_manager = BeliefManager(self.id)
        self.current_velocity_command = initial_state.vel
        self.is_at_goal = False
        
        # --- The "black box recorder" for visualization ---
        self.last_decision_context = {}

    def update_beliefs(self, measurements: Dict, current_time: float, dt: float):
        """Update beliefs about the world based on new info."""
        # Note: In the previous version, dt was passed but not used in predict_all.
        # The logic is now corrected inside belief_manager to use timestamps.
        self.belief_manager.predict_all(current_time)
        
        for neighbor_id, measurement in measurements.items():
            self.belief_manager.update_from_measurement(neighbor_id, measurement, current_time)
        
        self.belief_manager.forget_lost_neighbors(current_time)

    def get_pref_velocity(self) -> np.ndarray:
        """Calculates the preferred velocity towards the goal."""
        dist_to_goal = np.linalg.norm(self.goal - self.state.pos)
        
        # Check if agent has arrived
        if dist_to_goal < self.config.radius:
            self.is_at_goal = True
            return np.zeros(3)
        
        direction_to_goal = (self.goal - self.state.pos) / dist_to_goal
        
        # Decelerate when approaching the goal
        speed = min(self.config.pref_speed, dist_to_goal)
        
        return direction_to_goal * speed

    def compute_new_velocity(self, nearby_obstacles: List[Obstacle]):
        """Run the ARGO-3D optimization to decide the next velocity command."""
        if self.is_at_goal:
            self.current_velocity_command = np.zeros(3)
            return

        current_beliefs = self.belief_manager.get_current_beliefs()
        pref_vel = self.get_pref_velocity()
        
        optimal_velocity = solve_argo_3d_velocity(
            current_state=self.state,
            pref_velocity=pref_vel,
            neighbor_beliefs=current_beliefs,
            nearby_obstacles=nearby_obstacles,
            config=self.config,
            config_params=self.config_params,
            # *** Pass the context dictionary to be populated ***
            debug_context=self.last_decision_context
        )
        self.current_velocity_command = optimal_velocity