'''
意图分析:
    信息分配: _get_measurements是关键。
    它不是简单地广播所有信息，而是模拟了真实世界中谁能看到谁，以及看到的质量如何。
    
    严格分离: 注意_step函数的顺序：感知 -> 决策 -> 执行。
    在决策环节，所有agent是并行计算的，它们基于t时刻的信念，计算出t+dt时刻的指令。
    在物理更新环节，模拟器才将这些指令应用到真值上。这避免了时序上的逻辑错误。
    
改进：    
    KD-Tree集成: _update_kdtree 和 _get_measurements 现在使用scipy.spatial.KDTree。
    query_ball_point方法可以极快地找出指定范围内的所有邻居，避免了O(N²)的暴力搜索。

    静态障碍物管理: 增加了Obstacle列表和对应的KD-Tree，
    并通过_get_nearby_obstacles方法将局部障碍物信息传递给Agent。

    通信仿真: _get_measurements现在还模拟了通信丢包 (comm_loss_prob)和传感器噪声(sensor_noise_std)，
    使得仿真环境更加真实。

    run_step方法: 将原_step公开为run_step，使其更符合典型的仿真器API。
    物理更新部分也更加精细，正确地应用了加速度和速度限制。
'''

# simulator/simulator.py
import numpy as np
from typing import Dict, List
from scipy.spatial import KDTree

from agent.argo_agent import Agent
from utils.argo_structures import State, Obstacle

class Simulator:
    def __init__(self, agents: Dict[int, Agent], obstacles: List[Obstacle], sensor_range: float, comm_loss_prob: float):
        self.agents = agents
        self.static_obstacles = obstacles
        self.time = 0.0

        # --- Simulation Control Parameters ---
        self.sensor_range = sensor_range           # Max distance an agent can see others
        self.comm_loss_prob = comm_loss_prob     # Probability of a measurement packet being lost

        # KD-Tree for efficient neighbor search
        self.agent_kdtree: KDTree = None
        # We can also build a KD-Tree for obstacles if they are numerous
        self.obstacle_kdtree: KDTree = None 
        self._build_obstacle_kdtree()

    def _build_obstacle_kdtree(self):
        if not self.static_obstacles:
            return
        obstacle_positions = [obs.pos for obs in self.static_obstacles]
        self.obstacle_kdtree = KDTree(obstacle_positions)

    def _update_kdtree(self):
        """Rebuilds the KD-Tree with current agent positions."""
        agent_positions = [agent.state.pos for agent in self.agents.values()]
        if agent_positions:
            self.agent_kdtree = KDTree(agent_positions)

    def _get_measurements(self) -> Dict[int, Dict]:
        """
        Simulates imperfect sensing using the KD-Tree.
        Returns a dictionary: {agent_id: {neighbor_id: measurement, ...}, ...}
        """
        self._update_kdtree()
        all_measurements = {agent_id: {} for agent_id in self.agents.keys()}
        agent_list = list(self.agents.values())

        for i, agent in enumerate(agent_list):
            # Query the KD-Tree for neighbors within sensor_range
            indices = self.agent_kdtree.query_ball_point(agent.state.pos, self.sensor_range)

            for j in indices:
                if i == j: continue # Don't sense yourself
                
                neighbor = agent_list[j]
                
                # Simulate communication packet loss
                if np.random.rand() < self.comm_loss_prob:
                    continue

                # Create a noisy measurement of the neighbor's true state
                noise = np.random.normal(0, neighbor.config.sensor_noise_std, size=3)
                measured_pos = neighbor.state.pos + noise
                # Velocity can also be noisy, but for simplicity, we assume it's measured directly
                measurement = State(pos=measured_pos, vel=neighbor.state.vel)
                
                all_measurements[agent.id][neighbor.id] = {
                    "state": measurement,
                    "config": neighbor.config # Pass config info through communication
                }
        
        return all_measurements

    def _get_nearby_obstacles(self, agent: Agent) -> List[Obstacle]:
        """Finds static obstacles near an agent."""
        if not self.obstacle_kdtree:
            return []
        
        indices = self.obstacle_kdtree.query_ball_point(agent.state.pos, self.sensor_range)
        return [self.static_obstacles[i] for i in indices]

    def run_step(self, dt: float):
        """Runs a single simulation step."""
        
        # 1. Simulate Communication (Perception)
        all_measurements = self._get_measurements()

        # 2. Agent Decision Making Loop
        for agent in self.agents.values():
            # a. Get local environment info
            agent_measurements = all_measurements.get(agent.id, {})
            nearby_obstacles = self._get_nearby_obstacles(agent)
            
            # b. Update internal belief model
            agent.update_beliefs(agent_measurements, self.time, dt)
            
            # c. Make decision
            # --- THIS IS THE CORRECTED PART ---
            # We no longer calculate pref_vel here. We just tell the agent to compute its command.
            # The agent will handle its own pref_vel calculation internally.
            agent.compute_new_velocity(nearby_obstacles)
            # --- END OF CORRECTION ---

        # 3. Update Physics (Execution)
        # ... (This part remains the same) ...
        for agent in self.agents.values():
            command_vel = agent.current_velocity_command
            
            accel_needed = (command_vel - agent.state.vel) / dt
            accel_norm = np.linalg.norm(accel_needed)
            if accel_norm > agent.config.max_accel:
                accel_needed = (accel_needed / accel_norm) * agent.config.max_accel
            
            achievable_vel = agent.state.vel + accel_needed * dt
            
            speed = np.linalg.norm(achievable_vel)
            if speed > agent.config.max_speed:
                achievable_vel = (achievable_vel / speed) * agent.config.max_speed

            agent.state.vel = achievable_vel
            agent.state.pos += agent.state.vel * dt
            
        self.time += dt

    # (Helper methods like _get_pref_velocity remain the same)