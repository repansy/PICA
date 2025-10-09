# utils/structures.py
import math
import numpy as np

class Vector3D:
    """A complete 3D vector class with standard operations."""
    def __init__(self, x, y, z):
        self.x = x
        self.y = y
        self.z = z
    
    @classmethod
    def from_numpy(cls, np_array):
        return Vector3D(np_array[0],np_array[1],np_array[2])
    
    def __add__(self, other):
        return Vector3D(self.x + other.x, self.y + other.y, self.z + other.z)

    def __sub__(self, other):
        return Vector3D(self.x - other.x, self.y - other.y, self.z - other.z)

    def __mul__(self, scalar):
        return Vector3D(self.x * scalar, self.y * scalar, self.z * scalar)
    
    # Allows for scalar * vector multiplication
    def __rmul__(self, scalar):
        return self.__mul__(scalar)

    def __truediv__(self, scalar):
        if scalar == 0:
            return Vector3D(0, 0, 0)
        return Vector3D(self.x / scalar, self.y / scalar, self.z / scalar)
    
    # Unary negation (-vector)
    def __neg__(self):
        return Vector3D(-self.x, -self.y, -self.z)
    
    def __repr__(self):
        return f"Vector3D({self.x:.2f}, {self.y:.2f}, {self.z:.2f})"

    def norm(self):
        return math.sqrt(self.x**2 + self.y**2 + self.z**2)
        
    def norm_sq(self):
        return self.x**2 + self.y**2 + self.z**2

    def normalized(self):
        mag = self.norm()
        if mag < 1e-9: # Epsilon check for safety
            return Vector3D(0, 0, 0)
        return self / mag

    def dot(self, other):
        return self.x * other.x + self.y * other.y + self.z * other.z
    
    # --- NEWLY ADDED FUNCTION ---
    def cross(self, other):
        """Computes the cross product with another vector."""
        return Vector3D(
            self.y * other.z - self.z * other.y,
            self.z * other.x - self.x * other.z,
            self.x * other.y - self.y * other.x
        )
        
    def to_numpy(self):
        """Converts the vector to a NumPy array for matrix operations."""
        return np.array([self.x, self.y, self.z])


class SlowBrainPolicy:
    """
    慢脑输出的战略策略。
    这是一个数据容器，用于在慢脑和快脑之间传递信息。
    """
    # 具有长远协同节奏的理想速度
    v_ideal: Vector3D = Vector3D(0, 0, 0)
    
    # 用于QP成本函数的动态权重矩阵，体现了战略偏好
    m_cost: np.ndarray = np.eye(3)

class NeighborBelief:
    """
    慢脑对某个邻居的内部信念模型。
    存储了通过时频分析得出的关于邻居行为模式的认知。
    """
    # 邻居速度轨迹中的主导行为频率 (Hz)
    omega: float = 0.0
    
    # 对应主导频率的相位 (radians)
    phi: float = 0.0
    
    # 对这个频率-相位模型的置信度 (0 to 1)
    confidence: float = 0.0
    