import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from mpl_toolkits.mplot3d.art3d import Line3DCollection
import pandas as pd
import os
from scipy.ndimage import gaussian_filter1d 

# ==========================================
# 1. 数据读取 (保持不变)
# ==========================================
def read_and_sample_dual_data(pos_path, resp_path, max_samples=100):
    df_pos = pd.read_csv(pos_path)
    df_resp = pd.read_csv(resp_path)
    
    total_timesteps = len(df_pos)
    agents = len(df_pos.columns) // 3  
    
    print(f"数据包含 {total_timesteps} 个时间步和 {agents} 个Agent")
    
    if total_timesteps <= max_samples:
        sample_indices = range(total_timesteps)
    else:
        step = total_timesteps // max_samples
        sample_indices = range(0, total_timesteps, step)
        if sample_indices[-1] != total_timesteps - 1:
            sample_indices = list(sample_indices) + [total_timesteps - 1]
    
    sampled_timesteps = len(sample_indices)
    
    sampled_positions = np.zeros((sampled_timesteps, agents, 3))
    sampled_responsibilities = np.zeros((sampled_timesteps, agents))
    
    pos_data_all = df_pos.values[sample_indices]
    resp_data_all = df_resp.values[sample_indices]

    for a in range(agents):
        sampled_positions[:, a, 0] = pos_data_all[:, a*3]
        sampled_positions[:, a, 1] = pos_data_all[:, a*3+1]
        sampled_positions[:, a, 2] = pos_data_all[:, a*3+2]
        
        # 提取 h 值 (每4列一组，取第3个)
        if (a*4 + 2) < resp_data_all.shape[1]:
            sampled_responsibilities[:, a] = resp_data_all[:, a*4 + 2]
        else:
            sampled_responsibilities[:, a] = 0.0
            
    return sampled_positions, sampled_responsibilities

# ==========================================
# 2. 核心绘图 (修改了颜色映射)
# ==========================================
def plot_trajectory_with_emphasis(positions, responsibilities):
    fig = plt.figure(figsize=(14, 10))
    ax = fig.add_subplot(111, projection='3d')
    
    timesteps, agents, _ = positions.shape
    
    # -------------------------------------------------------------
    # [修改点] 调整颜色映射 (Colormap)
    # 'RdYlBu_r' 表示 Red-Yellow-Blue reversed
    # 效果: 0.0(蓝) -> 0.5(黄) -> 1.0(红)
    # -------------------------------------------------------------
    cmap = plt.get_cmap('RdYlBu_r') 
    norm = plt.Normalize(0, 1)

    print("正在绘制...")

    for a in range(agents):
        x = positions[:, a, 0]
        y = positions[:, a, 1]
        z = positions[:, a, 2]
        raw_h = responsibilities[:, a]
        
        # --- 1. 数据平滑 ---
        h_smooth = gaussian_filter1d(raw_h, sigma=2.0)
        h_smooth = np.clip(h_smooth, 0, 1)

        # --- 2. 构建线段 ---
        points = np.array([x, y, z]).T.reshape(-1, 1, 3)
        segments = np.concatenate([points[:-1], points[1:]], axis=1)
        
        segment_h = h_smooth[:-1]
        
        # segment_h = np.ones_like(segment_h) * 0.5  # 默认颜色为中间值 (黄色)

        # --- 3. 视觉强化 (>0.5 部分) ---
        widths = np.where(segment_h > 0.5, 2.5, 1.2)
        
        # 创建集合
        lc = Line3DCollection(segments, cmap=cmap, norm=norm)
        lc.set_array(segment_h)  
        lc.set_linewidth(widths) 
        lc.set_alpha(0.9)        
        
        ax.add_collection(lc)
        
        # --- 4. 起点终点 ---
        start_color = cmap(norm(h_smooth[0]))
        end_color = cmap(norm(h_smooth[-1]))
        
        # 起点
        ax.scatter(x[0], y[0], z[0], c=[start_color], s=50, 
                   marker='o', edgecolors='white', linewidth=0.8,
                   label=f'Start' if a == 0 else "")
        
        # 终点
        ax.scatter(x[-1], y[-1], z[-1], c=[end_color], s=70, 
                   marker='o', edgecolors='black', linewidth=0.8,
                   label=f'End' if a == 0 else "")

    # --- 设置 ---
    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_zlabel('Z')
    ax.set_title(f'Trajectory Responsibility Map\n(Thick Line: Score > 0.5)')
    
    ax.legend(loc='upper left')

    # 颜色条
    sm = plt.cm.ScalarMappable(cmap=cmap, norm=norm)
    sm.set_array([])
    # 可以在这里添加 ticks 参数明确标注 0, 0.5, 1
    cbar = plt.colorbar(sm, ax=ax, shrink=0.7, pad=0.1, ticks=[0, 0.5, 1])
    cbar.set_label('Responsibility Score (Blue=Low, Red=High)')
    
    plt.tight_layout()
    save_fig_path = "trajectory_responsibility_map-1.png"
    plt.savefig(save_fig_path, dpi=300, bbox_inches='tight')
    print(f"图片已保存到：{save_fig_path}")
    plt.show()

# ==========================================
# 主程序
# ==========================================
def main():
    current_dir = os.path.dirname(os.path.abspath(__file__))
    # 注意：Windows路径可能需要根据实际情况调整斜杠
    results_dir = os.path.join(current_dir, '..', 'results\\')#, 'PicaBatch2')
    
    pos_file = os.path.join(results_dir, 'test_trajectory.csv') 
    resp_file = os.path.join(results_dir, 'test_alpha.csv') 
    
    max_samples = 150 
    
    if not os.path.exists(pos_file) or not os.path.exists(resp_file):
        print(f"Error: 文件未找到。\n{pos_file}\n{resp_file}")
        print(">>> 生成模拟数据演示...")
        mock_pos, mock_resp = generate_mock_data()
        plot_trajectory_with_emphasis(mock_pos, mock_resp)
        return

    positions, responsibilities = read_and_sample_dual_data(pos_file, resp_file, max_samples)
    plot_trajectory_with_emphasis(positions, responsibilities)

# --- 模拟数据 ---
def generate_mock_data():
    steps = 150
    agents = 3
    pos = np.zeros((steps, agents, 3))
    resp = np.zeros((steps, agents))
    t = np.linspace(0, 10, steps)
    
    for i in range(agents):
        pos[:, i, 0] = np.sin(t + i) * 5
        pos[:, i, 1] = np.cos(t + i) * 5
        pos[:, i, 2] = t * 2
        
        base_h = np.zeros(steps)
        base_h[60:90] = 0.9 
        base_h[90:120] = 0.4 
        resp[:, i] = base_h
        
    return pos, resp

if __name__ == "__main__":
    main()