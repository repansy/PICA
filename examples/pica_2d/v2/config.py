# config.py
import numpy as np

# --- Simulation Scenario ---
# 'antipodal_sphere': Agents start on a sphere and travel to the opposite point.
# 'random': Agents start at random positions with random goals.
SCENARIO_2D =  'DISK_ANTIPODAL'
RESULT_DIR = 'F:\\CodeRepo\\PICA\\results\\pica_batch\\2'

# --- Simulation Parameters ---
NUM_AGENTS = 8
TIMESTEP = 0.1  # seconds
SIMULATION_TIME = 300  # seconds
WORLD_SIZE = (50, 50, 50) # meters (x, y, z)

# --- Agent Physical Properties ---
AGENT_RADIUS = 0.5  # meters
MAX_SPEED = 2.0     # meters/second
ACCELERATION_MAX = 2.0
DEFAULT_INERTIA_MATRIX = np.diag([3.0, 3.0, 0.5])

# --- [新] 概率不确定性建模 (Probabilistic Uncertainty Modeling) ---
# 该模块取代了原有的 UNCERTAINTY_BUFFER，提供了更动态、更真实的建模方式
# 过程噪声，模拟不确定性随时间的自然增长
PROCESS_NOISE_FACTOR = 0.001 
# 在计算有效状态时使用的置信水平（N个标准差），值越高越保守
UNCERTAINTY_CONFIDENCE_N = 2.5 

# --- PICA 核心算法参数 ---
# 对风险最高的K个邻居启动完整的混合PICA优化
PICA_K = 5 
# Alpha值的更新阻尼因子，防止振荡，保持系统稳定性
PICA_BETA_DAMPING = 0.25 
# 用于浮点数比较和数值计算的微小量，保证稳定性
PICA_EPSILON = 1e-5
# 规避时的时间视界(tau)。值越大，规避动作越平滑、越有预见性。原1.0s太短，易导致急促反应。
TTC_HORIZON = 2.0 # seconds 

# --- 风险评估与混合模型参数 ---
# 风险评分中距离和TTC的权重。这些是绝对权重，无需归一化。
RISK_W_DIST = 1.0
RISK_W_TTC = 2.5
# 混合模型的风险阈值，决定解析法和启发式的权重
RISK_THRESHOLD_LOW = 5.0  # 低于此风险，完全使用启发式
RISK_THRESHOLD_HIGH = 20.0 # 高于此风险，完全使用解析法

# --- 局部密度计算参数 ---
# 密度计算的高斯核宽度，即“感知”拥挤程度的范围
DENSITY_SIGMA = 5.0 
# 密度值的平滑因子，防止因邻居瞬时移动造成密度剧烈波动
DENSITY_BETA_SMOOTHING = 0.7 

# --- 仿真与可视化 ---
VISUALIZE = True
# 每隔N个时间步更新一次图像，以加速仿真
PLOT_FREQUENCY = 5
# 是否记录轨迹到CSV文件
RECORD_TRAJECTORY = True
# CSV文件保存路径
TRAJECTORY_FILE = "results/c-20-trajectory.csv"


# --- (已废弃) ---
# UNCERTAINTY_BUFFER = 0.15 # 已被更先进的概率不确定性模型取代,对吗