import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import matplotlib.colors as mcolors
import matplotlib.cm as cm
import pandas as pd
import os
import re
import argparse
from matplotlib.patches import Circle


class AgentVisualizer:
    """智能体轨迹与属性可视化工具类"""
    
    def __init__(self, pos_file, static_file, max_samples=100):
        """
        初始化可视化工具
        
        参数:
            pos_file: 轨迹CSV文件路径（每行对应一个时间步，列格式为Agent{i}_x/y/z）
            static_file: 静态属性CSV文件路径（列格式为Agent{i}_R/M/P）
            max_samples: 最大采样时间步数（控制轨迹平滑度与性能）
        """
        self.pos_file = pos_file
        self.static_file = static_file
        self.max_samples = max_samples
        self.agents = self._load_and_process_data()  # 加载并预处理数据
        self._validate_agents()  # 验证数据有效性
        
    
    def _load_and_process_data(self):
        """加载轨迹与静态属性数据，返回处理后的agent列表"""
        # 读取CSV文件
        try:
            pos_df = pd.read_csv(self.pos_file)
            static_df = pd.read_csv(self.static_file)
        except FileNotFoundError as e:
            raise FileNotFoundError(f"文件不存在: {e.filename}")
        except Exception as e:
            raise RuntimeError(f"读取文件失败: {str(e)}")
        
        # 提取所有智能体ID（通过静态属性列名匹配）
        agent_ids = self._extract_agent_ids(static_df.columns)
        if not agent_ids:
            raise ValueError("未找到有效智能体属性列（需包含Agent{i}_R/M/P格式）")
        num_agents = len(agent_ids)
        print(f"检测到 {num_agents} 个智能体（ID: {sorted(agent_ids)}）")
        
        # 时间步采样（均匀分布）
        total_timesteps = len(pos_df)
        sample_indices = self._sample_timesteps(total_timesteps)
        print(f"时间步采样: 原始{total_timesteps} → 采样后{len(sample_indices)}")
        
        # 构建智能体数据（包含轨迹与属性）
        agents = []
        for agent_id in sorted(agent_ids):
            # 提取静态属性（R:半径, M:惯性, P:权限）
            try:
                radius = static_df[f'Agent{agent_id}_R'].iloc[0]
                inertia = static_df[f'Agent{agent_id}_M'].iloc[0]
                authority = static_df[f'Agent{agent_id}_P'].iloc[0]
            except KeyError as e:
                raise KeyError(f"智能体{agent_id}的属性列不存在: {e}")
            
            # 提取轨迹数据（x,y,z）
            trajectory = []
            for t in sample_indices:
                try:
                    x = pos_df[f'Agent{agent_id}_x'].iloc[t]
                    y = pos_df[f'Agent{agent_id}_y'].iloc[t]
                    z = pos_df[f'Agent{agent_id}_z'].iloc[t]
                    trajectory.append([x, y, z])
                except KeyError as e:
                    raise KeyError(f"智能体{agent_id}的轨迹列不存在: {e}")
            
            agents.append({
                'id': agent_id,
                'radius': float(radius),
                'inertia': float(inertia),
                'authority': float(authority),
                'trajectory': np.array(trajectory, dtype=np.float32)
            })
        
        return agents
    
    
    def _extract_agent_ids(self, columns):
        """从列名中提取智能体ID（匹配Agent{i}_R/M/P格式）"""
        agent_ids = set()
        pattern = re.compile(r'Agent(\d+)_(R|M|P)')  # 正则匹配智能体ID和属性
        for col in columns:
            match = pattern.match(col)
            if match:
                agent_ids.add(int(match.group(1)))
        return sorted(agent_ids)
    
    
    def _sample_timesteps(self, total_timesteps):
        """均匀采样时间步，确保分布均匀"""
        if total_timesteps <= self.max_samples:
            return np.arange(total_timesteps)
        # 线性均匀采样（包含起点和终点）
        return np.unique(np.linspace(0, total_timesteps-1, self.max_samples, dtype=int))
    
    
    def _validate_agents(self):
        """验证智能体数据有效性"""
        if not self.agents:
            raise ValueError("未加载到任何智能体数据")
        for agent in self.agents:
            if len(agent['trajectory']) == 0:
                raise ValueError(f"智能体{agent['id']}的轨迹数据为空")
            # 检查属性值是否为正数
            if agent['radius'] <= 0 or agent['inertia'] <= 0:
                raise ValueError(f"智能体{agent['id']}的半径/惯性必须为正数")
    
    
    def _get_color_mapper(self, attribute):
        """获取属性到颜色的映射器（处理属性值相同的情况）"""
        attr_values = [agent[attribute] for agent in self.agents]
        vmin, vmax = min(attr_values), max(attr_values)
        # 避免属性值全部相同导致的归一化错误
        if vmin == vmax:
            vmin -= 1e-6
            vmax += 1e-6
        norm = mcolors.Normalize(vmin=vmin, vmax=vmax)
        return cm.viridis, norm, attr_values
    
    
    def plot_3d_trajectories(self, attribute='radius', show_spheres=False, sphere_density=15):
        """
        绘制3D轨迹图（按指定属性着色）
        
        参数:
            attribute: 着色属性（'radius'/'inertia'/'authority'）
            show_spheres: 是否显示表示半径的球体
            sphere_density: 球体点云密度（越大越精细，默认15）
        """
        if attribute not in ['radius', 'inertia', 'authority']:
            raise ValueError("属性必须为'radius'/'inertia'/'authority'")
        
        fig = plt.figure(figsize=(12, 10))
        ax = fig.add_subplot(111, projection='3d')
        cmap, norm, _ = self._get_color_mapper(attribute)
        
        for agent in self.agents:
            traj = agent['trajectory']
            attr_val = agent[attribute]
            color = cmap(norm(attr_val))
            
            # 绘制轨迹线（带时间衰减效果）
            x, y, z = traj[:, 0], traj[:, 1], traj[:, 2]
            for i in range(len(traj)-1):
                alpha = 1.0 - (i / (len(traj)-1)) * 0.8  # 从起点到终点逐渐透明
                ax.plot(x[i:i+2], y[i:i+2], z[i:i+2], 
                       color=mcolors.to_rgba(color, alpha=alpha), 
                       linewidth=2)
            
            # 标记起点（圆形）和终点（方形）
            if agent['id'] == self.agents[0]['id']:  # 仅第一个智能体添加图例
                ax.scatter(x[0], y[0], z[0], color=color, s=100, marker='o', label='Start')
                ax.scatter(x[-1], y[-1], z[-1], color=color, s=100, marker='s', label='End')
            else:
                ax.scatter(x[0], y[0], z[0], color=color, s=100, marker='o')
                ax.scatter(x[-1], y[-1], z[-1], color=color, s=100, marker='s')
            
            # 绘制表示半径的球体（用点云模拟，高效且美观）
            if show_spheres:
                self._plot_3d_sphere(ax, agent['radius'], traj[::len(traj)//3],  # 取3个关键帧
                                   color=color, density=sphere_density)
        
        # 添加颜色条和标题
        cbar = fig.colorbar(cm.ScalarMappable(cmap=cmap, norm=norm), 
                           ax=ax, shrink=0.8, aspect=20)
        cbar.set_label(attribute.capitalize(), rotation=270, labelpad=20)
        
        # 调整坐标轴比例（避免变形）
        ax.set_box_aspect([np.ptp(x), np.ptp(y), np.ptp(z)])
        ax.set_xlabel('X', labelpad=10)
        ax.set_ylabel('Y', labelpad=10)
        ax.set_zlabel('Z', labelpad=10)
        ax.set_title(f'3D Trajectories (Colored by {attribute.capitalize()})', pad=20)
        ax.legend()
        plt.tight_layout()
        plt.show()
    
    
    def _plot_3d_sphere(self, ax, radius, positions, color, density=15):
        """用点云绘制3D球体（高效替代表面绘制）"""
        phi, theta = np.mgrid[0:np.pi:density*1j, 0:2*np.pi:density*1j]
        sphere_x = radius * np.sin(phi) * np.cos(theta)
        sphere_y = radius * np.sin(phi) * np.sin(theta)
        sphere_z = radius * np.cos(phi)
        
        for (x0, y0, z0) in positions:
            ax.scatter(sphere_x + x0, sphere_y + y0, sphere_z + z0,
                      color=mcolors.to_rgba(color, alpha=0.1), s=1, marker='o')
    
    
    def plot_2d_trajectories(self, plane='xy', attribute='radius'):
        """
        绘制2D轨迹图（指定平面投影，按属性着色）
        
        参数:
            plane: 投影平面（'xy'/'xz'/'yz'）
            attribute: 着色属性（'radius'/'inertia'/'authority'）
        """
        if plane not in ['xy', 'xz', 'yz']:
            raise ValueError("平面必须为'xy'/'xz'/'yz'")
        if attribute not in ['radius', 'inertia', 'authority']:
            raise ValueError("属性必须为'radius'/'inertia'/'authority'")
        
        fig, ax = plt.subplots(figsize=(10, 8))
        cmap, norm, _ = self._get_color_mapper(attribute)
        
        # 确定坐标轴索引和标签
        plane_map = {'xy': (0, 1, 'X', 'Y'), 
                    'xz': (0, 2, 'X', 'Z'), 
                    'yz': (1, 2, 'Y', 'Z')}
        x_idx, y_idx, x_label, y_label = plane_map[plane]
        
        for agent in self.agents:
            traj = agent['trajectory']
            attr_val = agent[attribute]
            color = cmap(norm(attr_val))
            
            # 提取平面坐标
            x = traj[:, x_idx]
            y = traj[:, y_idx]
            
            # 绘制轨迹线（带时间衰减）
            for i in range(len(traj)-1):
                alpha = 1.0 - (i / (len(traj)-1)) * 0.8
                ax.plot(x[i:i+2], y[i:i+2], 
                       color=mcolors.to_rgba(color, alpha=alpha), 
                       linewidth=2)
            
            # 标记起点和终点（点大小关联半径）
            point_size = 50 + agent['radius'] * 100  # 半径越大，点越大
            if agent['id'] == self.agents[0]['id']:
                ax.scatter(x[0], y[0], color=color, s=point_size, marker='o', label='Start')
                ax.scatter(x[-1], y[-1], color=color, s=point_size, marker='s', label='End')
            else:
                ax.scatter(x[0], y[0], color=color, s=point_size, marker='o')
                ax.scatter(x[-1], y[-1], color=color, s=point_size, marker='s')
            
            # 绘制表示半径的圆（自适应坐标尺度）
            self._plot_2d_circles(ax, agent['radius'], x, y, color)
        
        # 添加颜色条和标题
        cbar = fig.colorbar(cm.ScalarMappable(cmap=cmap, norm=norm), 
                           ax=ax, shrink=0.8, aspect=20)
        cbar.set_label(attribute.capitalize(), rotation=270, labelpad=20)
        
        ax.set_xlabel(x_label)
        ax.set_ylabel(y_label)
        ax.set_title(f'2D Trajectories ({plane.upper()} Plane, Colored by {attribute.capitalize()})')
        ax.grid(alpha=0.3)
        ax.legend()
        ax.set_aspect('equal', adjustable='datalim')  # 等比例显示
        plt.tight_layout()
        plt.show()
    
    
    def _plot_2d_circles(self, ax, radius, x, y, color):
        """在2D轨迹关键帧绘制表示半径的圆（自适应坐标尺度）"""
        # 计算坐标范围，动态调整圆的大小
        x_range = ax.get_xlim()[1] - ax.get_xlim()[0]
        y_range = ax.get_ylim()[1] - ax.get_ylim()[0]
        scale = min(x_range, y_range) * 0.01  # 取坐标范围的1%作为缩放基准
        circle_radius = radius * scale
        
        # 在轨迹的5个关键帧绘制圆
        key_indices = np.linspace(0, len(x)-1, 5, dtype=int)
        for idx in key_indices:
            circle = Circle((x[idx], y[idx]), circle_radius,
                           fill=False, color=color, alpha=0.5, linewidth=1.5)
            ax.add_patch(circle)


def main():
    # 解析命令行参数
    '''
    parser = argparse.ArgumentParser(description='智能体轨迹与属性可视化工具')
    parser.add_argument('--pos_file', required=True, default='F:\CodeRepo\PICA\results\PicaBatch\SphereBaseline\Baseline_trajectory.csv',help='轨迹CSV文件路径（格式：Agent{i}_x/y/z）')
    parser.add_argument('--static_file', required=True, default='F:\CodeRepo\PICA\results\PicaBatch\SphereBaseline\Baseline_RMPsetting.csv',help='静态属性CSV文件路径（格式：Agent{i}_R/M/P）')
    parser.add_argument('--max_samples', type=int, default=20, help='最大采样时间步数（默认100）')
    parser.add_argument('--attribute', default='radius', 
                       choices=['radius', 'inertia', 'authority'], 
                       help='着色属性（默认radius）')
    args = parser.parse_args()
    '''

    choices=['radius', 'inertia', 'authority']

    # try:
    # 初始化可视化工具
    visualizer = AgentVisualizer(
        pos_file='F:\\CodeRepo\\PICA\\results\\PicaBatch\\circle\\R1_trajectory.csv',
        static_file='F:\\CodeRepo\\PICA\\results\\PicaBatch\\circle\\R1_RMPsetting.csv',
        max_samples=20
    )
    
    # 生成可视化结果
    print("绘制3D轨迹图...")
    visualizer.plot_3d_trajectories(attribute=choices[0], show_spheres=True)
    
    print("绘制2D轨迹图（XY平面）...")
    visualizer.plot_2d_trajectories(plane='xy', attribute=choices[0])
    
    print("绘制2D轨迹图（XZ平面）...")
    visualizer.plot_2d_trajectories(plane='xz', attribute=choices[0])
        
    # except Exception as e:
    #     print(f"可视化失败: {str(e)}", file=sys.stderr)
    #     exit(1)


if __name__ == "__main__":
    # import sys
    main()