import os
import numpy as np
import pandas as pd
from scipy.spatial import distance, cKDTree
from scipy.stats import variation

# 参数配置
# 时间间隔0.1
TIME_HORIZON = 10.0
CONGESTION_DISTANCE = 15.0
CONGESTION_DENSITY = 0.005      # 拥堵密度阈值 0.5
MIN_DIST_THRESHOLD = 0.125     # 有效移动距离阈值 1.0
JERK_THRESHOLD = 0.1          # 加速度变化阈值 
CONGESTION_DURATION_THRESHOLD = 5  # 最小拥堵持续时间

class AgentMotionAnalyzer:
    def __init__(self, file_path, static_file_path, dt=1/TIME_HORIZON):
        self.df = pd.read_csv(file_path)
        self.static_df = pd.read_csv(static_file_path)
        self.num_timesteps = len(self.df)
        self.dt = dt
        
        # 读取 总列数除以3即可
        self.num_agents = int(len(self.df.columns)/3)
        # 校验列数是否为3的倍数（确保数据格式正确）
        if len(self.df.columns) % 3 != 0:
            raise ValueError(f"CSV文件列数({len(self.df.columns)})不是3的倍数，数据格式错误")
        
        self.positions = self._load_positions()
        self.static_attributes = self._load_static_attributes()
        
    def _load_positions(self):
        """加载位置数据并重塑为三维数组 (时间步, 智能体, 坐标)"""
        positions = np.zeros((self.num_timesteps, self.num_agents, 3))
        for i in range(self.num_agents):
            positions[:, i, 0] = self.df[f'Agent{i}_x']
            positions[:, i, 1] = self.df[f'Agent{i}_y']
            positions[:, i, 2] = self.df[f'Agent{i}_z']
        return positions
    
    def _load_static_attributes(self):
        """加载智能体静态属性：半径R、惯性M、权限P"""
        static_attrs = np.zeros((self.num_agents, 3))  # [R, M, P]
        
        for i in range(self.num_agents):
            static_attrs[i, 0] = self.static_df[f'Agent{i}_R']  # 半径
            static_attrs[i, 1] = self.static_df[f'Agent{i}_M']  # 惯性
            static_attrs[i, 2] = self.static_df[f'Agent{i}_P']  # 权限
            
        return static_attrs

    def analyze_motion_efficiency(self):
        
        """基于惯性的运动分析"""
        inertia_effects = []
        for i in range(self.num_agents):
            inertia = self.static_attributes[i, 1]
            # 计算加速度变化
            accelerations = []
            for t in range(1, self.num_timesteps-1):
                accel = (self.positions[t+1, i] - 2*self.positions[t, i] + self.positions[t-1, i]) / (self.dt**2)
                accelerations.append(np.linalg.norm(accel))
            
            # 惯性对加速度变化的影响
            avg_accel = np.mean(accelerations) if accelerations else 0
            inertia_effect = avg_accel / (inertia + 1e-6)  # 避免除零
            inertia_effects.append(inertia_effect)
        
        """运动效率分析：路径优化与流畅度"""
        path_lengths = np.zeros(self.num_agents)
        movement_times = np.zeros(self.num_agents)
        
        # 计算路径长度和运动时间
        for t in range(self.num_timesteps - 1):
            displacements = np.linalg.norm(
                self.positions[t+1] - self.positions[t], axis=1
            )
            path_lengths += displacements
            movement_times[displacements > MIN_DIST_THRESHOLD] += 1
        
        # 计算绕行系数
        path_ratios = []
        for i in range(self.num_agents):
            straight_dist = np.linalg.norm(
                self.positions[-1, i] - self.positions[0, i]
            )
            if straight_dist > 1e-5:
                path_ratios.append(path_lengths[i] / straight_dist)
        
        return {
            "avg_movement_time": np.mean(movement_times),
            "avg_path_ratio": np.mean(path_ratios) if path_ratios else 0,
            "inertia_effects": np.mean(inertia_effects)
        }
    
    def analyze_spatial_behavior(self):
        """空间行为分析：分布与协同"""
        nn_distances = []
        congestion_durations = np.zeros(self.num_agents)
        congestion_start_times = -np.ones(self.num_agents)
        
        for t in range(self.num_timesteps):
            current_pos = self.positions[t]
            
            # 最近邻距离
            tree = cKDTree(current_pos)
            nn_dists, _ = tree.query(current_pos, k=2)
            nn_distances.extend(nn_dists[:, 1])
            
            # 拥堵检测
            dist_matrix = distance.cdist(current_pos, current_pos, 'euclidean')
            for i in range(self.num_agents):
                # 计算半径内的邻居数量 (不包括智能体自身)
                count = np.sum(dist_matrix[i] < CONGESTION_DISTANCE) - 1
                # 使用三维球体体积公式计算
                volume = (4/3) * np.pi * (CONGESTION_DISTANCE**3) # <-- 三维体积计算
                
                # 防止体积为0导致除法错误
                if volume > 0:
                    density = count / volume
                else:
                    density = 0
                
                if density > CONGESTION_DENSITY:
                    if congestion_start_times[i] < 0:
                        congestion_start_times[i] = t
                else:
                    if congestion_start_times[i] >= 0:
                        duration = t - congestion_start_times[i]
                        if duration >= CONGESTION_DURATION_THRESHOLD:
                            congestion_durations[i] += duration
                        congestion_start_times[i] = -1
        
        # 处理未结束的拥堵
        for i in range(self.num_agents):
            if congestion_start_times[i] >= 0:
                duration = self.num_timesteps - 1 - congestion_start_times[i]
                if duration >= CONGESTION_DURATION_THRESHOLD:
                    congestion_durations[i] += duration
        
        # 逸散程度 (空间分布均匀性)
        all_positions = self.positions.reshape(-1, 3)
        x_min, x_max = np.min(all_positions[:, 0]), np.max(all_positions[:, 0])
        y_min, y_max = np.min(all_positions[:, 1]), np.max(all_positions[:, 1])
        z_min, z_max = np.min(all_positions[:, 2]), np.max(all_positions[:, 2]) 

        grid_size = 10
        x_bins = np.linspace(x_min, x_max, grid_size + 1)
        y_bins = np.linspace(y_min, y_max, grid_size + 1)
        z_bins = np.linspace(z_min, z_max, grid_size + 1) 
        
        # <-- 使用了X, Y, Z三轴的立体直方图
        density_map, _ = np.histogramdd(
            all_positions, 
            bins=[x_bins, y_bins, z_bins]
        )
                
        flat_density = density_map.flatten()
        valid_densities = flat_density[flat_density > 0]
        dispersion = variation(valid_densities) if valid_densities.size > 0 else 0
        
        return {
            "avg_nn_distance": np.mean(nn_distances) if nn_distances else 0,
            "dispersion": dispersion,
            "avg_congestion_duration_seconds": np.mean(congestion_durations) * self.dt
        }
    
    def generate_report(self):
        """生成综合分析报告"""
        efficiency = self.analyze_motion_efficiency()
        spatial = self.analyze_spatial_behavior()
        
        return {
            "motion_efficiency": efficiency,
            "spatial_behavior": spatial,
            "total_time": self.num_timesteps
        }

def simgle_test():
# 获取当前脚本所在目录
    current_dir = os.path.dirname(os.path.abspath(__file__))
    # print(current_dir)
    target_path = os.path.join(current_dir, '..', 'results', 'ORCA', 'c-20-trajectory-3d'+'.csv')
    # agent_positions_0814  stl-traj-s traj-ca-s-1 trajectories
    # 812  86 30
    
    analyzer = AgentMotionAnalyzer(target_path, 20) 
    
    report = analyzer.generate_report()
    
    print("群体智能体运动分析报告")
    print("=" * 40)
    print(f"总运行时间: {report['total_time']}时间步")
    
    print("\n运动效率:")
    print(f"- 平均运动时间: {report['motion_efficiency']['avg_movement_time']:.2f}时间步")
    print(f"- 惯性对加速度变化: {report['motion_efficiency']['inertia_effects']:.4f}")
    print(f"- 路径优化率: {report['motion_efficiency']['avg_path_ratio']:.4f}")
    
    print("\n空间行为:")
    print(f"- 平均邻近距离: {report['spatial_behavior']['avg_nn_distance']:.4f}")
    print(f"- 空间分布均匀性: {report['spatial_behavior']['dispersion']:.4f}")


def batch_analyze_scenarios(input_dir, output_summary):
    """
    批量分析所有场景的CSV文件并生成汇总结果
    input_dir: 存放场景CSV的目录
    output_summary: 汇总结果输出路径
    """
    # 获取所有场景CSV文件
    csv_files_1 = [f for f in os.listdir(input_dir) if f.endswith("_trajectory.csv")]
    summary_data = []
    
    for file in csv_files_1:
        # 提取场景名（从文件名中解析，如"CROSSING_trajectory.csv" -> "CROSSING"）
        scenario = file.replace("_trajectory.csv", "")
        file_2 = file.replace("_trajectory.csv", "_RMPsetting.csv")
        file_path = os.path.join(input_dir, file)
        file_path_2 = os.path.join(input_dir, file_2)
        print(f"\n===== 分析场景: {scenario} =====")
        
        # 假设所有场景的智能体数量相同，若不同需根据场景调整
        analyzer = AgentMotionAnalyzer(file_path, file_path_2)
        report = analyzer.generate_report()
        
        # 解析报告数据为一行记录
        row = {
            "scenario": scenario,
            "total_time": report["total_time"],
            # 运动效率指标
            "avg_movement_time": report["motion_efficiency"]["avg_movement_time"],
            "avg_jerk": report["motion_efficiency"]["inertia_effects"],
            "avg_path_ratio": report["motion_efficiency"]["avg_path_ratio"],
            # 空间行为指标
            "avg_nn_distance": report["spatial_behavior"]["avg_nn_distance"],
            "dispersion": report["spatial_behavior"]["dispersion"],
        }
        summary_data.append(row)
    
    # 写入汇总CSV
    df = pd.DataFrame(summary_data)
    df.to_csv(output_summary, index=False)
    print(f"\n汇总结果已保存至: {output_summary}")

# 批量运行入口
if __name__ == "__main__":
    current_dir = os.path.dirname(os.path.abspath(__file__))
    # 场景CSV存放目录（与batch_run.py的输出目录对应）
    # SPHERE_ROLE_BASED # SPHERE_DYNAMIC # SPHERE_DYNAMIC
    input_dir = os.path.join(current_dir, "..", "results", "batch\\8")
    # 汇总结果输出路径
    output_summary = os.path.join(current_dir, "..", "results", "summary_results_1.csv")
    batch_analyze_scenarios(input_dir, output_summary)
