# simulator/2d_simulator.py
import matplotlib.pyplot as plt
import csv
import os
from typing import List, Set, Tuple, Optional
from examples.pica_2d.v2.pica2d_agent import Agent
import examples.pica_2d.v2.config as cfg

class Simulator:
    def __init__(self, agents: List[Agent]):
        self.agents = agents
        self.time = 0.0
        self.dt = cfg.TIMESTEP
        self.plot_counter = 0

        # 碰撞跟踪（2D平面碰撞）
        self.total_collision_events = 0
        self.currently_colliding_pairs: Set[Tuple[int, int]] = set()

        # CSV记录（仅X、Y坐标）
        self.csv_file: Optional[csv.writer] = None
        self.csv_file_handle = None
        if cfg.RECORD_TRAJECTORY:
            self._setup_csv_file()
            
        if cfg.VISUALIZE:
            self.fig, self.ax = plt.subplots(figsize=(8, 8))
            plt.ion()
            self.visualize()

    def _setup_csv_file(self):
        """CSV表头改为[Agent0_x, Agent0_y, Agent1_x, Agent1_y, ...]"""
        os.makedirs(os.path.dirname(cfg.TRAJECTORY_FILE), exist_ok=True)
        self.csv_file_handle = open(cfg.TRAJECTORY_FILE, 'w', newline='')
        self.csv_file = csv.writer(self.csv_file_handle)
        header = []
        for i in range(len(self.agents)):
            header.extend([f'Agent{i}_x', f'Agent{i}_y'])
        self.csv_file.writerow(header)

    def _write_positions_to_csv(self):
        """写入2D位置"""
        if self.csv_file is None:
            return
        row = []
        for agent in self.agents:
            row.extend([agent.pos.x, agent.pos.y])
        self.csv_file.writerow(row)

    def step(self):
        """2D仿真步骤（移除Z轴物理更新）"""
        # 1. 计算新速度
        new_velocities = {agent.id: agent.compute_new_velocity(self.agents, self.dt) 
                          for agent in self.agents}
        
        # 2. 更新位置（仅X、Y方向）
        for agent in self.agents:
            agent.update(new_velocities[agent.id], self.dt)
        
        # 3. 2D碰撞检测（平面距离）
        self._check_for_collisions()
        
        self.time += self.dt
        if cfg.RECORD_TRAJECTORY:
            self._write_positions_to_csv()
        if cfg.VISUALIZE and self.plot_counter % cfg.PLOT_FREQUENCY == 0:
            self.visualize()
        self.plot_counter += 1

    def _check_for_collisions(self):
        """2D碰撞检测（基于X、Y距离）"""
        newly_colliding_pairs = set()
        for agent in self.agents:
            agent.is_colliding = False

        for i in range(len(self.agents)):
            for j in range(i + 1, len(self.agents)):
                agent1, agent2 = self.agents[i], self.agents[j]
                dist = (agent1.pos - agent2.pos).norm()  # 2D距离
                min_safe_dist = agent1.radius + agent2.radius
                if dist < min_safe_dist:
                    agent1.is_colliding = True
                    agent2.is_colliding = True
                    pair = tuple(sorted((agent1.id, agent2.id)))
                    newly_colliding_pairs.add(pair)
                    if pair not in self.currently_colliding_pairs:
                        self.total_collision_events += 1
        self.currently_colliding_pairs = newly_colliding_pairs

    def visualize(self):
        """2D可视化（移除Z轴）"""
        self.ax.clear()
        ws = cfg.WORLD_SIZE  # 2D世界尺寸 (x_max, y_max)
        self.ax.set_xlim(0, ws[0])
        self.ax.set_ylim(0, ws[1])
        self.ax.set_xlabel("X (m)")
        self.ax.set_ylabel("Y (m)")

        # 绘制智能体和目标（仅X、Y）
        for agent in self.agents:
            color = 'r' if agent.is_colliding else 'b'
            self.ax.scatter(agent.pos.x, agent.pos.y, color=color, s=100)
            self.ax.scatter(agent.goal.x, agent.goal.y, color='g', marker='x')
            self.ax.quiver(agent.pos.x, agent.pos.y, agent.vel.x, agent.vel.y, color=color)

        self.ax.set_title(f"2D Simulation | Time: {self.time:.2f}s | Collisions: {self.total_collision_events}")
        plt.draw()
        plt.pause(0.001)

    def all_agents_at_goal(self):
        """Checks if all agents have reached their goals."""
        return all(agent.at_goal for agent in self.agents)
    
    def __del__(self):
        """确保CSV文件在对象销毁时被正确关闭"""
        if self.csv_file_handle is not None:
            self.csv_file_handle.close()