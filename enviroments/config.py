# config.py
####### 注意每个file cfg
import numpy as np

# --- 仿真参数 ---
TIMESTEP = 0.1 # 快脑循环周期 (dt_f)
SCENARIO = 'SPHERE_3D'
NUM_AGENTS = 20
SIMULATION_TIME = 300

# --- 仿真与可视化 ---
VISUALIZE = True
# 每隔N个时间步更新一次图像，以加速仿真
PLOT_FREQUENCY = 5
# 是否记录轨迹到CSV文件
RECORD_TRAJECTORY = True
# CSV文件保存路径
TRAJECTORY_FILE = "results/s-20-trajectory.csv"
WORLD_SIZE = (50, 50, 50) # meters (x, y, z)

# --- Agent 物理属性 ---
AGENT_RADIUS = 0.5
MAX_SPEED = 2.0
ACCELERATION_MAX = 4.0 # m/s^2
DEFAULT_INERTIA_MATRIX = np.eye(3) * 2.0 # 用于估计未知邻居

# --- 快脑参数 ---
TTC_HORIZON = 8.0   # 8.0 seconds
PICA_EPSILON = 1e-5
# 启发式alpha的权重
ALPHA_WEIGHT_INERTIA = 0.3
ALPHA_WEIGHT_PRIORITY = 0.3
ALPHA_WEIGHT_RHO = 0.4

# --- 慢脑参数 ---
HISTORY_BUFFER_SIZE = 50 # 5秒的历史 @ 10Hz
SLOW_BRAIN_OUTPUT_PERIOD = 0.5 
CONFIDENCE_THRESHOLD = 0.7 
OSCILLATOR_COUPLING_K = 1.0

# --- QP求解器成本函数权重 ---
QP_WEIGHT_GOAL = 1.0 # v_pref 的权重
QP_WEIGHT_IDEAL = 1.5 # v_ideal 的权重

# --- 概率保守参数 ---
DENSITY_SIGMA = 5.0
UNCERTAINTY_CONFIDENCE_N = 1.5 
DENSITY_BETA_SMOOTHING = 0.7
PROCESS_NOISE_FACTOR = 0.001 

# --- 风险评估与混合模型参数 ---
# 风险评分中距离和TTC的权重。这些是绝对权重，无需归一化。
RISK_W_DIST = 1.0
RISK_W_TTC = 2.5