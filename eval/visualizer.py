# visualizer.py
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import matplotlib.colors as mcolors
import pandas as pd
import os
from matplotlib.patches import Circle
import matplotlib.cm as cm

# 从CSV文件读取数据并进行采样，返回agent_info列表
def read_and_sample_data_with_attributes(pos_file_path, static_file_path, max_samples=100):
    """
    从CSV文件读取Agent轨迹数据和静态属性，并进行采样
    
    参数:
    pos_file_path: 位置CSV文件路径
    static_file_path: 静态属性CSV文件路径
    max_samples: 最大采样点数
    
    返回:
    agents: 包含完整信息的智能体列表
    """
    # 读取位置数据
    pos_df = pd.read_csv(pos_file_path)
    static_df = pd.read_csv(static_file_path)
    
    # 获取总时间步数和Agent数量
    total_timesteps = len(pos_df)
    num_agents = len(static_df.columns) // 3  # 每3列代表一个Agent的R,M,P

    print(f"数据包含 {total_timesteps} 个时间步和 {num_agents} 个Agent")
    
    # 计算采样间隔
    if total_timesteps <= max_samples:
        sample_indices = range(total_timesteps)
    else:
        step = total_timesteps // max_samples
        sample_indices = range(0, total_timesteps, step)
        if sample_indices[-1] != total_timesteps - 1:
            sample_indices = list(sample_indices) + [total_timesteps - 1]
    
    sampled_timesteps = len(sample_indices)
    print(f"采样后保留 {sampled_timesteps} 个时间步")
    
    # 构建agent_info列表
    agents = []
    for i in range(num_agents):
        agent_info = {
            'id': i,
            'radius': static_df[f'Agent{i}_R'].iloc[0],
            'inertia': static_df[f'Agent{i}_M'].iloc[0],
            'authority': static_df[f'Agent{i}_P'].iloc[0],
            'trajectory': np.zeros((sampled_timesteps, 3))
        }
        
        # 填充采样后的轨迹数据
        for idx, t in enumerate(sample_indices):
            agent_info['trajectory'][idx] = [
                pos_df[f'Agent{i}_x'].iloc[t],
                pos_df[f'Agent{i}_y'].iloc[t],
                pos_df[f'Agent{i}_z'].iloc[t]
            ]
        
        agents.append(agent_info)
    
    return agents

# 创建3D轨迹图（基于agent_info）
def plot_3d_trajectories_with_attributes(agents, attribute='radius', show_spheres=False):
    """
    绘制带属性的Agent 3D轨迹图
    
    参数:
    agents: 智能体信息列表
    attribute: 用于着色的属性 ('radius', 'inertia', 'authority')
    show_spheres: 是否显示表示半径的球体
    """
    fig = plt.figure(figsize=(14, 10))
    ax = fig.add_subplot(111, projection='3d')
    
    # 提取属性值用于颜色映射
    attr_values = [agent[attribute] for agent in agents]
    norm = mcolors.Normalize(vmin=min(attr_values), vmax=max(attr_values))
    cmap = cm.viridis
    
    for agent in agents:
        trajectory = agent['trajectory']
        timesteps = len(trajectory)
        attr_value = agent[attribute]
        color = cmap(norm(attr_value))
        
        # 提取轨迹坐标
        x = trajectory[:, 0]
        y = trajectory[:, 1]
        z = trajectory[:, 2]
        
        # 绘制轨迹线
        for i in range(timesteps-1):
            alpha = 1.0 - (i / (timesteps-1)) * 0.7
            line_color = mcolors.to_rgba(color, alpha=alpha)
            ax.plot(x[i:i+2], y[i:i+2], z[i:i+2], 
                   color=line_color, linewidth=2, alpha=0.8)
        
        # 标记起点和终点
        ax.scatter(x[0], y[0], z[0], color=color, s=100, marker='o', 
                  label=f'Agent {agent["id"]} Start' if agent["id"] == 0 else "")
        ax.scatter(x[-1], y[-1], z[-1], color=color, s=100, marker='s', 
                  label=f'Agent {agent["id"]} End' if agent["id"] == 0 else "")
        
        # 如果显示球体，在关键点绘制表示半径的球体
        if show_spheres:
            # 在轨迹的几个关键点绘制球体
            key_points = [0, timesteps//2, timesteps-1]
            for point in key_points:
                if point < timesteps:
                    # 绘制球体（简化表示）
                    u = np.linspace(0, 2 * np.pi, 10)
                    v = np.linspace(0, np.pi, 10)
                    sphere_x = agent['radius'] * np.outer(np.cos(u), np.sin(v)) + x[point]
                    sphere_y = agent['radius'] * np.outer(np.sin(u), np.sin(v)) + y[point]
                    sphere_z = agent['radius'] * np.outer(np.ones(np.size(u)), np.cos(v)) + z[point]
                    
                    ax.plot_surface(sphere_x, sphere_y, sphere_z, 
                                   color=color, alpha=0.3, linewidth=0)
    
    # 添加颜色条
    sm = cm.ScalarMappable(cmap=cmap, norm=norm)
    sm.set_array([])
    cbar = plt.colorbar(sm, ax=ax, shrink=0.8, aspect=20)
    cbar.set_label(attribute.capitalize(), rotation=270, labelpad=15)
    
    ax.set_xlabel('X Coordinate')
    ax.set_ylabel('Y Coordinate')
    ax.set_zlabel('Z Coordinate')
    ax.set_title(f'3D Trajectories Colored by {attribute.capitalize()}')
    ax.legend()
    plt.tight_layout()
    plt.show()

# 创建2D轨迹图（基于agent_info）
def plot_2d_trajectories_with_attributes(agents, plane='xy', attribute='radius'):
    """
    绘制带属性的Agent 2D轨迹图
    
    参数:
    agents: 智能体信息列表
    plane: 要绘制的平面 ('xy', 'xz', 'yz')
    attribute: 用于着色的属性
    """
    fig, ax = plt.subplots(figsize=(12, 8))
    
    # 提取属性值用于颜色映射
    attr_values = [agent[attribute] for agent in agents]
    norm = mcolors.Normalize(vmin=min(attr_values), vmax=max(attr_values))
    cmap = cm.viridis
    
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
    
    for agent in agents:
        trajectory = agent['trajectory']
        timesteps = len(trajectory)
        attr_value = agent[attribute]
        color = cmap(norm(attr_value))
        
        # 提取轨迹坐标
        x = trajectory[:, x_idx]
        y = trajectory[:, y_idx]
        
        # 绘制轨迹线
        for i in range(timesteps-1):
            alpha = 1.0 - (i / (timesteps-1)) * 0.7
            line_color = mcolors.to_rgba(color, alpha=alpha)
            ax.plot(x[i:i+2], y[i:i+2], color=line_color, linewidth=2, alpha=0.8)
        
        # 标记起点和终点，并用半径大小表示点的大小
        start_size = 50 + agent['radius'] * 100  # 根据半径调整点大小
        ax.scatter(x[0], y[0], color=color, s=start_size, marker='o', 
                  alpha=0.7, label=f'Agent {agent["id"]} Start' if agent["id"] == 0 else "")
        ax.scatter(x[-1], y[-1], color=color, s=start_size, marker='s', 
                  alpha=0.7, label=f'Agent {agent["id"]} End' if agent["id"] == 0 else "")
        
        # 在轨迹上绘制表示半径的圆
        key_points = [0, timesteps//4, timesteps//2, 3*timesteps//4, timesteps-1]
        for point in key_points:
            if point < timesteps:
                circle = Circle((x[point], y[point]), agent['radius'] * 0.5, 
                               fill=False, color=color, alpha=0.5, linewidth=1)
                ax.add_patch(circle)
    
    # 添加颜色条
    sm = cm.ScalarMappable(cmap=cmap, norm=norm)
    sm.set_array([])
    cbar = plt.colorbar(sm, ax=ax, shrink=0.8, aspect=20)
    cbar.set_label(attribute.capitalize(), rotation=270, labelpad=15)
    
    ax.set_xlabel(x_label)
    ax.set_ylabel(y_label)
    ax.set_title(f'2D Trajectories Colored by {attribute.capitalize()} ({plane.upper()} Plane)')
    ax.grid(True, alpha=0.3)
    ax.legend()
    ax.set_aspect('equal', adjustable='datalim')
    plt.tight_layout()
    plt.show()

# 属性分布可视化
def plot_attributes_distribution(agents):
    """绘制智能体属性分布图"""
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    
    attributes = ['radius', 'inertia', 'authority']
    colors = ['skyblue', 'lightcoral', 'lightgreen']
    
    for idx, (attr, color) in enumerate(zip(attributes, colors)):
        values = [agent[attr] for agent in agents]
        agent_ids = [agent['id'] for agent in agents]
        
        axes[idx].bar(agent_ids, values, color=color, alpha=0.7)
        axes[idx].set_xlabel('Agent ID')
        axes[idx].set_ylabel(attr.capitalize())
        axes[idx].set_title(f'{attr.capitalize()} Distribution')
        axes[idx].grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.show()

# 轨迹属性相关性分析
def plot_trajectory_attribute_correlation(agents):
    """分析轨迹长度与属性的相关性"""
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    
    attributes = ['radius', 'inertia', 'authority']
    colors = ['red', 'blue', 'green']
    
    # 计算每个智能体的轨迹长度
    path_lengths = []
    for agent in agents:
        trajectory = agent['trajectory']
        length = 0
        for i in range(len(trajectory)-1):
            length += np.linalg.norm(trajectory[i+1] - trajectory[i])
        path_lengths.append(length)
    
    for idx, (attr, color) in enumerate(zip(attributes, colors)):
        attr_values = [agent[attr] for agent in agents]
        
        # 散点图显示相关性
        axes[idx].scatter(attr_values, path_lengths, color=color, alpha=0.6, s=60)
        
        # 添加趋势线
        if len(attr_values) > 1:
            z = np.polyfit(attr_values, path_lengths, 1)
            p = np.poly1d(z)
            axes[idx].plot(attr_values, p(attr_values), color=color, linestyle='--', alpha=0.8)
        
        axes[idx].set_xlabel(attr.capitalize())
        axes[idx].set_ylabel('Path Length')
        axes[idx].set_title(f'Path Length vs {attr.capitalize()}')
        axes[idx].grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.show()

# 主函数
def main():
    # 获取当前脚本所在目录
    current_dir = os.path.dirname(os.path.abspath(__file__))
    
    # 文件路径
    pos_file = os.path.join(current_dir, '..', 'results', 'batch', '2', 'SPHERE_DYNAMIC_trajectory.csv')
    static_file = os.path.join(current_dir, '..', 'results', 'batch', '2', 'SPHERE_DYNAMIC_RMPsetting.csv')  # 假设静态属性文件
    
    max_samples = 60
    
    try:
        # 读取带属性的数据
        agents = read_and_sample_data_with_attributes(pos_file, static_file, max_samples)
        
        print(f"成功加载 {len(agents)} 个智能体的数据")
        print(f"属性范围 - 半径: {min(a['radius'] for a in agents):.2f}~{max(a['radius'] for a in agents):.2f}")
        print(f"属性范围 - 惯性: {min(a['inertia'] for a in agents):.2f}~{max(a['inertia'] for a in agents):.2f}")
        print(f"属性范围 - 权限: {min(a['authority'] for a in agents):.2f}~{max(a['authority'] for a in agents):.2f}")
        
        # 生成各种可视化
        print("\n生成3D轨迹图（按半径着色）...")
        plot_3d_trajectories_with_attributes(agents, attribute='radius', show_spheres=True)
        
        print("生成3D轨迹图（按惯性着色）...")
        plot_3d_trajectories_with_attributes(agents, attribute='inertia')
        
        print("生成2D轨迹图（XY平面，按权限着色）...")
        plot_2d_trajectories_with_attributes(agents, plane='xy', attribute='authority')
        
        print("生成属性分布图...")
        plot_attributes_distribution(agents)
        
        print("生成轨迹属性相关性图...")
        plot_trajectory_attribute_correlation(agents)
        
    except FileNotFoundError as e:
        print(f"文件未找到: {e}")
        print("请检查文件路径是否正确")
    except Exception as e:
        print(f"发生错误: {e}")

if __name__ == "__main__":
    main()