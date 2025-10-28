# visualizer.py
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import matplotlib.colors as mcolors
import pandas as pd
import os

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
    # file_path = os.path.join(current_dir, '..', 'results', 'test-trajectory'+'.csv')
    # file_path = os.path.join(current_dir, '..', 'results', 'batch', '1', 'plane_scenarios', '2_agents_xy_M_trajectory'+'.csv')
    file_path = os.path.join(current_dir, '..', 'results', 'batch', '2', 'SPHERE_DISCRETE_trajectory'+'.csv')
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

