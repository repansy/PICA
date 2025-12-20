import numpy as np
import matplotlib.pyplot as plt
from matplotlib.collections import LineCollection
import pandas as pd
import os
from scipy.ndimage import gaussian_filter1d 

# ==========================================
# 1. 数据读取模块 (保持之前的双文件读取逻辑)
# ==========================================
def read_and_sample_dual_data(pos_path, resp_path, max_samples=100):
    df_pos = pd.read_csv(pos_path)
    df_resp = pd.read_csv(resp_path)
    
    total_timesteps = len(df_pos)
    # 动态获取 Agent 数量 (位置文件每3列一个Agent)
    agents = len(df_pos.columns) // 3  
    
    print(f"数据包含 {total_timesteps} 时间步, {agents} 个 Agent")
    
    # 采样逻辑
    if total_timesteps <= max_samples:
        sample_indices = range(total_timesteps)
    else:
        step = total_timesteps // max_samples
        sample_indices = list(range(0, total_timesteps, step))
        if sample_indices[-1] != total_timesteps - 1:
            sample_indices.append(total_timesteps - 1)
    
    sampled_timesteps = len(sample_indices)
    
    # 按照变量大小创建全0矩阵
    sampled_positions = np.zeros((sampled_timesteps, agents, 3))
    sampled_responsibilities = np.zeros((sampled_timesteps, agents))
    
    # 转为 numpy 加速处理
    pos_data_all = df_pos.values[sample_indices]
    resp_data_all = df_resp.values[sample_indices]

    for a in range(agents):
        sampled_positions[:, a, 0] = pos_data_all[:, a*3]     # x
        sampled_positions[:, a, 1] = pos_data_all[:, a*3+1]   # y
        sampled_positions[:, a, 2] = pos_data_all[:, a*3+2]   # z
        
        # 提取责任 h (每4列的第3个)
        if (a*4 + 2) < resp_data_all.shape[1]:
            sampled_responsibilities[:, a] = resp_data_all[:, a*4 + 2]
        else:
            sampled_responsibilities[:, a] = 0.0
            
    return sampled_positions, sampled_responsibilities

# ==========================================
# 2. 2D 轨迹绘制模块 (根据你的参考代码修改)
# ==========================================
def plot_2d_trajectories_xy(positions, responsibilities):
    """
    绘制 XY 平面的 2D 轨迹，颜色代表责任分数。
    
    参数:
    positions: (timesteps, agents, 3)
    responsibilities: (timesteps, agents)
    """
    # 创建画布
    fig, ax = plt.subplots(figsize=(10, 8))
    
    timesteps, agents, _ = positions.shape
    
    # 设置颜色映射 (RdYlBu_r: 蓝=低, 黄=中, 红=高)
    cmap = plt.get_cmap('RdYlBu_r') 
    norm = plt.Normalize(0, 1)
    
    print("正在绘制 2D XY 视图...")

    for a in range(agents):
        # 1. 只获取 XY 轴数据
        x = positions[:, a, 0]
        y = positions[:, a, 1]
        raw_h = responsibilities[:, a]
        
        # 2. 数据平滑 (避免颜色跳变太僵硬)
        h_smooth = gaussian_filter1d(raw_h, sigma=2.0)
        h_smooth = np.clip(h_smooth, 0, 1)
        
        # 3. 准备 LineCollection 所需的数据结构
        # 将点集转换为线段集: (N, 2) -> (N-1, 2, 2)
        points = np.array([x, y]).T.reshape(-1, 1, 2)
        segments = np.concatenate([points[:-1], points[1:]], axis=1)
        
        # 4. 计算线宽 (>0.5 加粗)
        # 注意 segments 的长度比 points 少 1，所以取 h_smooth[:-1]
        segment_h = h_smooth[:-1] + 0.4
        segment_h = np.ones_like(segment_h) * 0.5  # 默认颜色为中间值 (黄色)
        
        widths = np.where(segment_h > 0.5, 2.5, 1.5)
        
        # 5. 创建并添加 LineCollection
        lc = LineCollection(segments, cmap=cmap, norm=norm)
        lc.set_array(segment_h)  # 设置颜色依据
        lc.set_linewidth(widths) # 设置线宽
        lc.set_alpha(0.9)        # 设置不透明度
        
        ax.add_collection(lc)
        
        # 6. 标记起点和终点 (颜色跟随责任值)
        start_color = cmap(norm(h_smooth[0]))
        end_color = cmap(norm(h_smooth[-1]))
        
        ax.scatter(x[0], y[0], c=[start_color], s=60, marker='o', 
                   edgecolors='white', zorder=10, 
                   label=f'Start' if a == 0 else "")
        
        ax.scatter(x[-1], y[-1], c=[end_color], s=80, marker='o', 
                   edgecolors='black', zorder=10, 
                   label=f'End' if a == 0 else "")

    # LineCollection 不会自动更新坐标轴范围，必须手动设置
    # 获取所有数据的最大最小值来确定范围
    all_x = positions[:, :, 0].flatten()
    all_y = positions[:, :, 1].flatten()
    
    # 稍微加一点留白
    pad_x = (all_x.max() - all_x.min()) * 0.05
    pad_y = (all_y.max() - all_y.min()) * 0.05
    
    ax.set_xlim(all_x.min() - pad_x, all_x.max() + pad_x)
    ax.set_ylim(all_y.min() - pad_y, all_y.max() + pad_y)
    
    # 装饰 (参考你的代码风格)
    ax.set_xlabel('X Coordinate')
    ax.set_ylabel('Y Coordinate')
    ax.set_title('2D Trajectories (XY Plane) - Colored by Responsibility')
    ax.grid(True, linestyle='--', alpha=0.3)
    ax.set_aspect('equal') # 保证 XY 比例一致，物体不会变形
    
    # 图例
    ax.legend(loc='upper left')
    
    # 颜色条
    sm = plt.cm.ScalarMappable(cmap=cmap, norm=norm)
    sm.set_array([])
    plt.colorbar(sm, ax=ax, label='Responsibility Score (>0.5 Thick)')
    
    plt.tight_layout()
    save_fig_path = "orca.png"
    plt.savefig(save_fig_path, dpi=300, bbox_inches='tight')
    print(f"图片已保存到：{save_fig_path}")
    plt.show()

# ==========================================
# 主程序
# ==========================================
def main():
    current_dir = os.path.dirname(os.path.abspath(__file__))
    results_dir = os.path.join(current_dir, '..', 'results\\PicaBatch3')
    pos_file = os.path.join(results_dir, 'cP3_trajectory.csv')
    resp_file = os.path.join(results_dir, 'cP3_alpha.csv') 
    
    max_samples = 150
    
    if not os.path.exists(pos_file):
        print("未找到文件，请检查路径。")
        return

    # 1. 读取数据
    pos, resp = read_and_sample_dual_data(pos_file, resp_file, max_samples)
    
    # 2. 绘制 2D XY
    plot_2d_trajectories_xy(pos, resp)

if __name__ == "__main__":
    main()