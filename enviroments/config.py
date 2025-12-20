# config.py

# --- Simulation Scenario ---
# 'antipodal_sphere': Agents start on a sphere and travel to the opposite point.
# 'random': Agents start at random positions with random goals.
SCENARIO =  'sP4'  
RESULT_DIR = 'G:\\CodeRepo\\PICA\\results\\PicaBatch4'
# 是否记录轨迹到CSV文件
RECORD_TRAJECTORY = True
# CSV文件保存路径
TRAJECTORY_FILE = "results/test_trajectory.csv"
TRAJECTORY_FILE_2 = "results/test_alpha.csv"
TRAJECTORY_FILE_3 = "results/test_RMP.csv"

# --- 仿真与可视化 ---
VISUALIZE = False
# 每隔N个时间步更新一次图像，以加速仿真
PLOT_FREQUENCY = 5

# --- Simulation Parameters ---
NUM_AGENTS = 32
SIMULATION_TIME = 300  # seconds
WORLD_SIZE = (100, 100, 100) # meters (x, y, z)

# --- Agent Physical Properties ---
ACCELERATION_MAX = 2.0
NEIGHBOR_DIST = 15.0
TIME_HORIZON = 10.0
TIME_STEP = 10.0
MAX_NEIGHOBORS = 10

# --- B-ORCA 2.0 核心配置 ---

# 智能体物理属性与异质性建模
AGENT_RADIUS = 0.5         # 智能体基础半径 (m)
MAX_SPEED = 2.0            # 智能体最大速度 (m/s)
M_BASE = 1.0               # 基础惯性/运动能力值
K_M = 0.5                  # 惯性随半径增长的系数 (用于R-M绑定)

# 慢脑：在线估计与意图预测
HISTORY_LEN = 10           # 存储历史轨迹的长度，用于估计邻居属性

# 数值稳定性
EPSILON = 1e-6             # 用于避免除零等计算问题的微小正数
# --- 密度感知 (Dual-Mode) ---
BETA_MIN = 0.0             # 最低置信度

# --- 意图与交互 ---
P_SMOOTH = 0.8             # 意图平滑因子 (越接近1越相信历史)

# --- 慢脑优化权重 ---
W_PREF = 1.0               # 目标驱动权重
W_INERTIA = 2.0            # 惯性权重 (防抖动)
W_SLACK = 1000.0           # 基础安全约束惩罚
