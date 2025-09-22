'''
意图：
    使用dataclass可以方便地创建清晰、类型化的数据容器，避免在代码中到处使用裸tuple或dict。
    BeliefState是关键，它封装了我们对一个邻居的所有概率性知识。

改进：
    为AgentConfig添加了priority和sensor_noise_std，这是实现非对称和异质性的关键参数。

    在BeliefState中加入了config，这样Agent在评估与邻居的风险时，能知道对方的属性（如半径、优先级）。

    增加了Obstacle结构，为静态障碍物做准备
'''

# utils/structures.py
import numpy as np
from dataclasses import dataclass, field

@dataclass
class State:
    pos: np.ndarray = field(default_factory=lambda: np.zeros(3))
    vel: np.ndarray = field(default_factory=lambda: np.zeros(3))

@dataclass
class AgentConfig:
    radius: float
    max_speed: float
    max_accel: float
    pref_speed: float
    # --- Asymmetry & Heterogeneity Params ---
    priority: float = 1.0       # Higher is more important
    sensor_noise_std: float = 0.1 # Position measurement noise

@dataclass
class BeliefState:
    mean: np.ndarray             # 6D state vector [px,py,pz,vx,vy,vz]
    covariance: np.ndarray      # 6x6 covariance matrix
    last_update_time: float
    config: AgentConfig         # Store neighbor's config for asymmetry calcs

# --- Static Obstacle Structures ---
@dataclass
class Obstacle:
    # For now, we only support spherical obstacles
    pos: np.ndarray
    radius: float