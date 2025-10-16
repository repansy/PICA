# visualizer.py
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import matplotlib.colors as mcolors
import pandas as pd
import os
'''
import numpy as np
import matplotlib.pyplot as plt
from typing import Dict

from simulator.argo_simulator import Simulator
from agent.argo_agent import Agent
from agent.optimization import solve_argo_3d_velocity # We need access to the optimizer logic


def plot_velocity_space_risk(sim: Simulator, agent_id: int, time: float):
    """
    Generates and displays a 2D slice of the velocity space risk contour map
    for a specific agent at a specific time.
    """
    if agent_id not in sim.agents:
        print(f"Agent {agent_id} not found.")
        return

    agent = sim.agents[agent_id]
    
    # --- Step 1: Get all the necessary data at this frozen moment ---
    # We need to manually run the first part of the agent's decision loop
    all_measurements = sim._get_measurements()
    agent_measurements = all_measurements.get(agent.id, {})
    nearby_obstacles = sim._get_nearby_obstacles(agent)
    
    agent.update_beliefs(agent_measurements, time, 0.1) # dt doesn't matter much here
    
    current_beliefs = agent.belief_manager.get_current_beliefs()
    pref_vel = agent.get_pref_velocity()

    # --- Step 2: Create a 2D grid in the velocity space (XY plane) ---
    # We fix vz to the preferred velocity's z-component for this slice
    v_range = agent.config.max_speed * 1.2
    resolution = 50
    vx_space = np.linspace(-v_range, v_range, resolution)
    vy_space = np.linspace(-v_range, v_range, resolution)
    vx_grid, vy_grid = np.meshgrid(vx_space, vy_space)
    
    cost_grid = np.zeros_like(vx_grid)

    # --- Step 3: Calculate cost for each point on the grid ---
    # This is the computationally intensive part
    # Get the cost function from the optimizer
    # (This requires refactoring solve_argo_3d_velocity to expose its cost function)
    # For now, let's just assume we can get it. Let's create a helper inside the optimizer.
    
    # We need a way to get the cost function without running the full optimization
    # Let's imagine we refactor `solve_argo_3d_velocity` to return the cost_function if needed.
    # For now, we'll just re-implement the cost logic here for demonstration.
    
    print("Calculating risk terrain... (this may take a moment)")
    from agent.optimization import _calculate_local_metrics, _calculate_probabilistic_risk, _calculate_static_risk
    from agent.optimization import PRIORITY_FACTOR, RIGHT_OF_WAY_FACTOR, OBSTACLE_WEIGHT

    social_metrics = _calculate_local_metrics(agent.state, pref_vel, current_beliefs, agent.config)
    self_metrics = social_metrics.get('self', {'crowdedness': 0, 'flow_alignment': 1})
    
    for i in range(resolution):
        for j in range(resolution):
            v_candidate = np.array([vx_grid[i, j], vy_grid[i, j], pref_vel[2]])
            
            # --- Replicating the cost function logic ---
            efficiency_cost = np.linalg.norm(v_candidate - pref_vel)**2
            safety_cost = 0
            # ... (The full logic for calculating safety cost from solve_argo_3d_velocity) ...
            # This part would be a direct copy-paste of the cost function's inner loop.
            # To keep this clean, let's assume it's done and we have the cost.
            # Here we just put a placeholder.
            
            # Placeholder cost calculation
            cost = efficiency_cost
            for neighbor_id, belief in current_beliefs.items():
                # Simple distance-based risk for visualization
                dist_to_neighbor_vel = np.linalg.norm(v_candidate - belief.mean[3:6])
                cost += 50.0 / (dist_to_neighbor_vel**2 + 0.1)
                
            cost_grid[i, j] = cost

    # --- Step 4: Plot the results ---
    plt.figure(figsize=(10, 10))
    
    # Plot the cost contour map
    contour = plt.contourf(vx_grid, vy_grid, np.log1p(cost_grid), levels=20, cmap='viridis_r')
    plt.colorbar(contour, label='Log(Cost)')
    plt.contour(vx_grid, vy_grid, np.log1p(cost_grid), levels=20, colors='white', alpha=0.5, linewidths=0.5)

    # Plot agent's current velocity
    plt.plot(agent.state.vel[0], agent.state.vel[1], 'ro', markersize=10, label='Current Velocity')
    
    # Plot preferred velocity
    plt.plot(pref_vel[0], pref_vel[1], 'go', markersize=10, label='Preferred Velocity')
    
    # Plot neighbors' velocities (as seen by the agent)
    for neighbor_id, belief in current_beliefs.items():
        n_vel = belief.mean[3:6]
        plt.plot(n_vel[0], n_vel[1], 'ys', markersize=8, label=f'Neighbor {neighbor_id} Vel')
        # Draw a circle representing the "risk zone"
        circle = plt.Circle((n_vel[0], n_vel[1]), radius=agent.config.radius + belief.config.radius, 
                              color='yellow', fill=False, linestyle='--', alpha=0.8)
        plt.gca().add_artist(circle)

    plt.title(f'Agent {agent_id} Velocity Space Risk Terrain at Time {time:.2f}s (vz slice={pref_vel[2]:.2f}m/s)')
    plt.xlabel('vx (m/s)')
    plt.ylabel('vy (m/s)')
    plt.axis('equal')
    plt.grid(True, linestyle='--', alpha=0.6)
    plt.legend()
    plt.show()
'''

# 从CSV文件读取数据并进行采样
def read_and_sample_data(file_path, max_samples=100):
    """
    从CSV文件读取Agent轨迹数据并进行采样
    
    参数:
    file_path: CSV文件路径
    max_samples: 最大采样点数
    
    返回:
    sampled_positions: 采样后的位置数组，形状为(sampled_timesteps, agents, 3)
    """
    # 读取CSV文件
    df = pd.read_csv(file_path)
    
    # 获取总时间步数和Agent数量
    total_timesteps = len(df)
    agents = (len(df.columns)) // 3  # 每3列代表一个Agent的x,y,z坐标

    print(f"数据包含 {total_timesteps} 个时间步和 {agents} 个Agent")
    
    # 计算采样间隔
    if total_timesteps <= max_samples:
        # 如果数据点不多，使用所有点
        sample_indices = range(total_timesteps)
    else:
        # 计算采样间隔
        step = total_timesteps // max_samples
        sample_indices = range(0, total_timesteps, step)
        # 确保包含最后一个点
        if sample_indices[-1] != total_timesteps - 1:
            sample_indices = list(sample_indices) + [total_timesteps - 1]
    
    sampled_timesteps = len(sample_indices)
    print(f"采样后保留 {sampled_timesteps} 个时间步")
    
    # 初始化采样后的位置数组
    sampled_positions = np.zeros((sampled_timesteps, agents, 3))
    
    # 填充采样后的位置数组
    for i, t in enumerate(sample_indices):
        for a in range(agents):
            sampled_positions[i, a, 0] = df.iloc[t, a*3]      # x坐标
            sampled_positions[i, a, 1] = df.iloc[t, a*3+1]    # y坐标
            sampled_positions[i, a, 2] = df.iloc[t, a*3+2]    # z坐标
    
    return sampled_positions

# 创建3D轨迹图
def plot_3d_trajectories(positions):
    """
    绘制Agent的3D轨迹图
    
    参数:
    positions: 形状为(timesteps, agents, 3)的numpy数组
    """
    fig = plt.figure(figsize=(12, 10))
    ax = fig.add_subplot(111, projection='3d')
    
    timesteps, agents, _ = positions.shape
    
    # 为每个agent设置基本颜色
    base_colors = plt.cm.tab20(np.linspace(0, 1, agents))
    
    for a in range(agents):
        # 获取agent的轨迹
        x = positions[:, a, 0]
        y = positions[:, a, 1]
        z = positions[:, a, 2]
        
        # 创建颜色渐变效果
        for i in range(timesteps-1):
            # 计算颜色透明度（随时间逐渐变淡）
            alpha = 1.0 - (i / (timesteps-1)) * 0.7  # 保持至少30%的不透明度
            
            # 创建带有透明度的颜色
            color = mcolors.to_rgba(base_colors[a], alpha=alpha)
            
            # 绘制线段
            ax.plot(x[i:i+2], y[i:i+2], z[i:i+2], color=color, linewidth=2)
        
        # 标记起点和终点
        ax.scatter(x[0], y[0], z[0], color=base_colors[a], s=50, marker='o', label=f'Agent {a} Start' if a == 0 else "")
        ax.scatter(x[-1], y[-1], z[-1], color=base_colors[a], s=50, marker='s', label=f'Agent {a} End' if a == 0 else "")
    
    ax.set_xlabel('X Coordinate')
    ax.set_ylabel('Y Coordinate')
    ax.set_zlabel('Z Coordinate')
    ax.set_title('3D Trajectories of Circular Agents (Sampled)')
    ax.legend()
    plt.tight_layout()
    plt.show()

# 创建2D轨迹图
def plot_2d_trajectories(positions, plane='xy'):
    """
    绘制Agent的2D轨迹图
    
    参数:
    positions: 形状为(timesteps, agents, 3)的numpy数组
    plane: 要绘制的平面，可以是'xy', 'xz'或'yz'
    """
    fig, ax = plt.subplots(figsize=(10, 8))
    
    timesteps, agents, _ = positions.shape
    
    # 为每个agent设置基本颜色
    base_colors = plt.cm.tab20(np.linspace(0, 1, agents))
    
    # 确定坐标轴
    if plane == 'xy':
        x_idx, y_idx = 0, 1
        x_label, y_label = 'X Coordinate', 'Y Coordinate'
    elif plane == 'xz':
        x_idx, y_idx = 0, 2
        x_label, y_label = 'X Coordinate', 'Z Coordinate'
    else:  # yz
        x_idx, y_idx = 1, 2
        x_label, y_label = 'Y Coordinate', 'Z Coordinate'
    
    for a in range(agents):
        # 获取agent的轨迹
        x = positions[:, a, x_idx]
        y = positions[:, a, y_idx]
        
        # 创建颜色渐变效果
        for i in range(timesteps-1):
            # 计算颜色透明度（随时间逐渐变淡）
            alpha = 1.0 - (i / (timesteps-1)) * 0.7  # 保持至少30%的不透明度
            
            # 创建带有透明度的颜色
            color = mcolors.to_rgba(base_colors[a], alpha=alpha)
            
            # 绘制线段
            ax.plot(x[i:i+2], y[i:i+2], color=color, linewidth=2)
        
        # 标记起点和终点
        ax.scatter(x[0], y[0], color=base_colors[a], s=50, marker='o', label=f'Agent {a} Start' if a == 0 else "")
        ax.scatter(x[-1], y[-1], color=base_colors[a], s=50, marker='s', label=f'Agent {a} End' if a == 0 else "")
    
    ax.set_xlabel(x_label)
    ax.set_ylabel(y_label)
    ax.set_title(f'2D Trajectories of Circular Agents ({plane.upper()} Plane, Sampled)')
    ax.grid(True, alpha=0.3)
    ax.legend()
    plt.tight_layout()
    plt.show()

# 主函数
def main():
    # 从CSV文件读取数据并进行采样
    # 获取当前脚本所在目录
    current_dir = os.path.dirname(os.path.abspath(__file__))
    print(current_dir)
    file_path = os.path.join(current_dir, '..', 'results', 'batch', '4', 'orca_plane_scenarios\\plane_scenarios', '2_agents_xy_M_trajectory'+'.csv')
    # file_path = os.path.join(current_dir, '..', 'results', 'batch', '2', 'pica_scenarios', 'SPHERE_DISCRETE_trajectory'+'.csv')
    max_samples = 60  # 最大采样点数，可以根据需要调整
    
    positions = read_and_sample_data(file_path, max_samples)
    
    # 生成3D轨迹图
    plot_3d_trajectories(positions)
    
    # 生成2D轨迹图（XY平面）
    plot_2d_trajectories(positions, plane='xy')
    
    # 生成2D轨迹图（XZ平面）
    # plot_2d_trajectories(positions, plane='xz')
    
    # 生成2D轨迹图（YZ平面）
    # plot_2d_trajectories(positions, plane='yz')

if __name__ == "__main__":
    main()

