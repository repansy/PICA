# config.py
import numpy as np
from utils.argo_structures import AgentConfig, Obstacle

# --- Simulation Parameters ---
SIM_DURATION_SECONDS = 12000.0
SIM_TIME_STEP = 0.1
SENSOR_RANGE = 20.0
COMM_LOSS_PROB = 0.1 # 10% packet loss

# --- ARGO-3D Social Behavior Parameters ---
# Encapsulate them in a dictionary for easy passing
CONFIG_PARAMS = {
    # For Responsibility Calculation
    'ALPHA_0': 0.5,
    'GAMMA': 0.3,
    'ALPHA_MIN': 0.1,
    # For Flow-based Scaling Matrix
    'KAPPA': 0.6,
    'LAMBDA': 0.4,
    # From Simulator, for crowdedness calculation
    'SENSOR_RANGE': SENSOR_RANGE
}

# --- Agent Default Properties ---
DEFAULT_AGENT_CONFIG = AgentConfig(
    radius=0.5,
    max_speed=2.0,
    pref_speed=1.5,
    max_accel=1.0,
    priority=1.0,
    sensor_noise_std=0.05 # 5cm position noise
)

# --- Scenario Definition ---
def get_scenario_agents():
    """Defines the agents for the simulation scenario."""
    agents = {
        # Agent 0: Standard agent
        0: {"initial_pos": np.array([-10.0, 0.0, 5.0]), "goal": np.array([10.0, 0.0, 5.0]), "config": DEFAULT_AGENT_CONFIG},
        
        # Agent 1: Another standard agent, on a collision course with Agent 0
        1: {"initial_pos": np.array([10.0, 0.0, 5.0]), "goal": np.array([-10.0, 0.0, 5.0]), "config": DEFAULT_AGENT_CONFIG},

        # Agent 2: High-priority, faster agent moving vertically
        2: {"initial_pos": np.array([0.0, -10.0, 10.0]), "goal": np.array([0.0, 10.0, 0.0]),
            "config": AgentConfig(radius=0.7, max_speed=3.0, pref_speed=2.5, max_accel=1.5, priority=2.0, sensor_noise_std=0.02)},
            
        # Agent 3: Slower, lower-priority agent
        3: {"initial_pos": np.array([0.0, 10.0, 5.0]), "goal": np.array([0.0, -10.0, 5.0]),
            "config": AgentConfig(radius=0.4, max_speed=1.5, pref_speed=1.0, max_accel=0.8, priority=0.5, sensor_noise_std=0.1)},
    }
    return agents

def get_scenario_obstacles():
    """Defines the static obstacles for the simulation."""
    obstacles = [
        Obstacle(pos=np.array([0.0, 0.0, 6.0]), radius=1.5),
        Obstacle(pos=np.array([5.0, 3.0, 4.0]), radius=1.0)
    ]
    return obstacles