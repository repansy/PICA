# utils/2d_structures.py
import math
import numpy as np

class Vector2D:
    """2D向量类，替代3D的Vector3D"""
    def __init__(self, x: float = 0, y: float = 0):
        self.x = x
        self.y = y

    def norm(self) -> float:
        """计算向量模长（2D平面距离）"""
        return math.hypot(self.x, self.y)

    def norm_sq(self) -> float:
        """计算模长平方（避免开方，提高效率）"""
        return self.x**2 + self.y**2

    def normalized(self) -> 'Vector2D':
        """返回单位向量"""
        n = self.norm()
        if n < 1e-9:
            return Vector2D(0, 0)
        return Vector2D(self.x / n, self.y / n)

    def dot(self, other: 'Vector2D') -> float:
        """点积计算"""
        return self.x * other.x + self.y * other.y
    
    # Unary negation (-vector)
    def __neg__(self):
        return Vector2D(-self.x, -self.y)
    # Allows for scalar * vector multiplication
    def __rmul__(self, scalar):
        return self.__mul__(scalar)

    def __truediv__(self, scalar):
        if scalar == 0:
            return Vector2D(0, 0, 0)
        return Vector2D(self.x / scalar, self.y / scalar)

    def __sub__(self, other: 'Vector2D') -> 'Vector2D':
        return Vector2D(self.x - other.x, self.y - other.y)

    def __add__(self, other: 'Vector2D') -> 'Vector2D':
        return Vector2D(self.x + other.x, self.y + other.y)

    def __mul__(self, scalar: float) -> 'Vector2D':
        return Vector2D(self.x * scalar, self.y * scalar)

    @classmethod
    def from_numpy(cls, arr: list) -> 'Vector2D':
        """从numpy数组转换（适配优化器输出）"""
        return cls(arr[0], arr[1])
    
    def to_numpy(self):
        """转换为 numpy 数组用于矩阵运算"""
        return np.array([self.x, self.y])