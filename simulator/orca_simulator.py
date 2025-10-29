# simulator/simulator.py
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from typing import List, Set, Tuple, Union, Optional # <-- IMPORT UNION HERE
import csv
import os

# Use a generic type hint that works for both Agent and OrcaAgent
# from examples.pica_3d.v2.pica_agent import Agent as PicaAgent
from agent.pivo_agent import BCOrcaAgent as PicaAgent
from agent.orca_agent import OrcaAgent
Agent = Union[PicaAgent, OrcaAgent] 

# Here is the important cfg
# from examples.pica_3d.v2 import config as cfg
import enviroments.config as cfg

class Simulator:
    """Manages the overall simulation, including agent states, collision detection, and visualization."""
    def __init__(self, agents: List[Agent]):
        self.agents = agents
        self.time = 0.0
        self.dt = 1/cfg.TIME_HORIZON
        self.plot_counter = 0

        # Collision Tracking
        self.total_collision_events = 0
        self.currently_colliding_pairs: Set[Tuple[int, int]] = set()

        # CSV文件记录
        self.csv_file: Optional[csv.writer] = None
        self.csv_file_handle = None
        self.csv_file_2: Optional[csv.writer] = None
        self.csv_file_handle_2 = None
        if cfg.RECORD_TRAJECTORY:
            self._setup_csv_file()
            
        if cfg.VISUALIZE:
            self.fig = plt.figure(figsize=(10, 10))
            self.ax = self.fig.add_subplot(111, projection='3d')
            plt.ion() # Interactive mode on
            self.visualize() # Initial plot
            
    def _setup_csv_file(self):
        """创建并初始化CSV文件，写入表头"""
        # 确保目录存在
        os.makedirs(os.path.dirname(cfg.TRAJECTORY_FILE), exist_ok=True)
        os.makedirs(os.path.dirname(cfg.TRAJECTORY_FILE_2), exist_ok=True)
        os.makedirs(os.path.dirname(cfg.TRAJECTORY_FILE_3), exist_ok=True)

        # 创建CSV文件并写入表头
        self.csv_file_handle = open(cfg.TRAJECTORY_FILE, 'w', newline='')
        self.csv_file = csv.writer(self.csv_file_handle)

        self.csv_file_handle_3 = open(cfg.TRAJECTORY_FILE_3, 'w', newline='')
        self.csv_file_3 = csv.writer(self.csv_file_handle_3)
        
        # 创建表头 [Agent0_x,Agent0_y,Agent0_z,Agent1_x,Agent1_y,Agent1_z,...]
        header = []
        header_3 = []

        for i in range(len(self.agents)):
            header.extend([f'Agent{i}_x', f'Agent{i}_y', f'Agent{i}_z'])
            header_3.extend([f'Agent{i}_R', f'Agent{i}_M', f'Agent{i}_P'])

        self.csv_file.writerow(header)
        self.csv_file_3.writerow(header_3)

        row = []
        for agent in self.agents:
            row.extend([agent.radius, 1.0, 0.5])
        self.csv_file_3.writerow(row)

    def _write_positions_to_csv(self):
        """将当前所有智能体的位置写入CSV文件"""
        if self.csv_file is None:
            return
            
        row = []
        for agent in self.agents:
            row.extend([agent.pos.x, agent.pos.y, agent.pos.z])
        
        self.csv_file.writerow(row)
        
    def step(self):
        """Advances the simulation by one timestep."""
        # 1. Compute all new velocities first.
        for agent in self.agents:
            agent.compute_neighbors(self.agents)
            agent.compute_preferred_velocity()

        for agent in self.agents:
            agent.compute_new_velocity()
        
        # 2. Update all agents' positions simultaneously.
        for agent in self.agents:
            agent.update(self.dt)
            
        # 3. Check for collisions AFTER moving.
        self._check_for_collisions()
            
        self.time += self.dt
        
        # 4. 记录位置和alpha到CSV文件
        if cfg.RECORD_TRAJECTORY:
            self._write_positions_to_csv()
        
        # 4. Visualize the new state.
        if cfg.VISUALIZE and self.plot_counter % cfg.PLOT_FREQUENCY == 0:
            self.visualize()
        self.plot_counter += 1

    def _check_for_collisions(self):
        """Checks all pairs of agents for collisions and prints warnings."""
        newly_colliding_pairs: Set[Tuple[int, int]] = set()
        for agent in self.agents:
            agent.is_colliding = False

        for i in range(len(self.agents)):
            for j in range(i + 1, len(self.agents)):
                agent1 = self.agents[i]
                agent2 = self.agents[j]

                dist = (agent1.pos - agent2.pos).norm()
                min_safe_dist = agent1.radius + agent2.radius

                if dist < min_safe_dist:
                    agent1.is_colliding = True
                    agent2.is_colliding = True
                    
                    pair = tuple(sorted((agent1.id, agent2.id)))
                    newly_colliding_pairs.add(pair)

                    if pair not in self.currently_colliding_pairs:
                        self.total_collision_events += 1
                        print(f"\n\033[91m"  # Start red text
                              f"--- COLLISION DETECTED! ---"
                              f"\nTime: {self.time:.2f}s"
                              f"\nAgents: {agent1.id} and {agent2.id}"
                              f"\nDistance: {dist:.3f}m (min safe: {min_safe_dist:.3f}m)"
                              f"\n\033[0m") # End red text
        
        self.currently_colliding_pairs = newly_colliding_pairs

    def visualize(self):
        """Renders the current state of the 3D simulation with collision highlighting."""
        self.ax.clear()
        
        ws = cfg.WORLD_SIZE
        self.ax.set_xlim(0, ws[0])
        self.ax.set_ylim(0, ws[1])
        self.ax.set_zlim(0, ws[2])
        self.ax.set_xlabel("X (m)")
        self.ax.set_ylabel("Y (m)")
        self.ax.set_zlabel("Z (m)")

        # default_colors = ['b', 'g', 'c', 'm', 'y', 'k'] * (cfg.NUM_AGENTS // 6 + 1)
        # 定义基础颜色列表（可根据需要扩展）
        base_colors = ['b', 'g', 'c', 'm', 'y', 'k', 'orange', 'purple', 'pink', 'brown']
        # 用于存储 (优先级, 惯性特征) 到颜色的映射
        color_map = {}
        # 用于跟踪已使用的颜色索引
        color_index = 0
        for _, agent in enumerate(self.agents):
            if hasattr(agent, 'M'):
                # 提取惯性矩阵的特征（这里用对角线元素作为能力特征）,惯性矩阵主要通过对角线元素体现不同方向的惯性
                inertia_feature = agent.M.norm_sq()
                # 组合优先级和惯性特征作为唯一标识
                agent_key = (agent.P, inertia_feature)
            else:
                # 对于无 inertia_matrix 的 OrcaAgent，使用固定标识
                agent_key = ("orca",)  # 用元组确保可哈希

            # 如果是新的组合，分配一个新颜色
            if agent_key not in color_map:
                if agent_key == ("orca",):
                    # OrcaAgent 固定使用灰色
                    color_map[agent_key] = 'gray'
                else:
                    color_map[agent_key] = base_colors[color_index % len(base_colors)]
                    color_index += 1
            
            agent_color = 'r' if agent.is_colliding else color_map[agent_key]
            marker_size = 150 if agent.is_colliding else 100

            self.ax.scatter(agent.pos.x, agent.pos.y, agent.pos.z, color=agent_color, marker='o', s=marker_size)
            self.ax.scatter(agent.goal.x, agent.goal.y, agent.goal.z, color=color_map[agent_key], marker='x', s=150)
            self.ax.quiver(agent.pos.x, agent.pos.y, agent.pos.z, 
                           agent.vel.x, agent.vel.y, agent.vel.z, 
                           length=1.5, color=agent_color)

        self.ax.set_title(f"3D Simulation | Time: {self.time:.2f}s | Collisions: {self.total_collision_events}")
        plt.draw()
        plt.pause(0.001)

    def all_agents_at_goal(self):
        """Checks if all agents have reached their goals."""
        return all(agent.at_goal for agent in self.agents)
    
    def __del__(self):
        """确保CSV文件在对象销毁时被正确关闭"""
        if self.csv_file_handle is not None:
            self.csv_file_handle.close()