import examples.pica_3d.v2.config as cfg
from typing import List, Tuple
from utils.pica_structures import Vector3D, Plane, Line
import math


# --- 3D 线性规划求解器 (直接从 C++ 翻译) ---
# 这些函数与上一回答中的相同，因为它们是标准 RVO2-3D 库的一部分
def linear_program1(planes: List[Plane], plane_no: int, line: Line, radius: float, opt_velocity: Vector3D, direction_opt: bool) -> Tuple[bool, Vector3D]:
    dot_product = line.point.dot(line.direction)
    discriminant = dot_product**2 + radius**2 - line.point.norm_sq()

    if discriminant < 0.0:
        return False, Vector3D()

    sqrt_discriminant = math.sqrt(discriminant)
    t_left = -dot_product - sqrt_discriminant
    t_right = -dot_product + sqrt_discriminant

    for i in range(plane_no):
        numerator = (planes[i].point - line.point).dot(planes[i].normal)
        denominator = line.direction.dot(planes[i].normal)

        if denominator**2 <= cfg.RVO3D_EPSILON:
            if numerator > 0.0:
                return False, Vector3D()
            continue

        t = numerator / denominator
        if denominator >= 0.0:
            t_left = max(t_left, t)
        else:
            t_right = min(t_right, t)

        if t_left > t_right:
            return False, Vector3D()
    
    result = Vector3D()
    if direction_opt:
        if opt_velocity.dot(line.direction) > 0.0:
            result = line.point + t_right * line.direction
        else:
            result = line.point + t_left * line.direction
    else:
        t = line.direction.dot(opt_velocity - line.point)
        if t < t_left:
            result = line.point + t_left * line.direction
        elif t > t_right:
            result = line.point + t_right * line.direction
        else:
            result = line.point + t * line.direction
            
    return True, result

def linear_program2(planes: List[Plane], plane_no: int, radius: float, opt_velocity: Vector3D, direction_opt: bool) -> Tuple[bool, Vector3D]:
    plane = planes[plane_no]
    plane_dist = plane.point.dot(plane.normal)
    plane_dist_sq = plane_dist**2
    radius_sq = radius**2

    if plane_dist_sq > radius_sq:
        return False, Vector3D()

    plane_radius_sq = radius_sq - plane_dist_sq
    plane_center = plane_dist * plane.normal
    
    result = Vector3D()
    if direction_opt:
        plane_opt_velocity = opt_velocity - opt_velocity.dot(plane.normal) * plane.normal
        plane_opt_velocity_length_sq = plane_opt_velocity.norm_sq()
        if plane_opt_velocity_length_sq <= cfg.RVO3D_EPSILON:
            result = plane_center
        else:
            result = plane_center + math.sqrt(plane_radius_sq / plane_opt_velocity_length_sq) * plane_opt_velocity
    else:
        result = opt_velocity + ((plane.point - opt_velocity).dot(plane.normal)) * plane.normal
        if result.norm_sq() > radius_sq:
            plane_result = result - plane_center
            plane_result_length_sq = plane_result.norm_sq()
            result = plane_center + math.sqrt(plane_radius_sq / plane_result_length_sq) * plane_result

    for i in range(plane_no):
        if planes[i].normal.dot(planes[i].point - result) > 0.0:
            cross_product = planes[i].normal.cross(plane.normal)
            if cross_product.norm_sq() <= cfg.RVO3D_EPSILON:
                return False, Vector3D()

            line = Line()
            line.direction = cross_product.normalized()
            line_normal = line.direction.cross(plane.normal)
            line.point = plane.point + (((planes[i].point - plane.point).dot(planes[i].normal)) / (line_normal.dot(planes[i].normal))) * line_normal
            
            success, result = linear_program1(planes, i, line, radius, opt_velocity, direction_opt)
            if not success:
                return False, Vector3D()

    return True, result

def linear_program3(planes: List[Plane], radius: float, opt_velocity: Vector3D, direction_opt: bool) -> Tuple[int, Vector3D]:
    result = Vector3D()
    if direction_opt:
        result = opt_velocity.normalized() * radius
    elif opt_velocity.norm_sq() > radius**2:
        result = opt_velocity.normalized() * radius
    else:
        result = opt_velocity

    for i in range(len(planes)):
        if planes[i].normal.dot(planes[i].point - result) > 0.0:
            temp_result = result
            success, result = linear_program2(planes, i, radius, opt_velocity, direction_opt)
            if not success:
                result = temp_result
                return i, result
    
    return len(planes), result

def linear_program4(planes: List[Plane], begin_plane: int, radius: float, current_vel: Vector3D) -> Vector3D:
    distance = 0.0
    result = current_vel

    for i in range(begin_plane, len(planes)):
        if planes[i].normal.dot(planes[i].point - result) > distance:
            proj_planes = []
            for j in range(i):
                plane = Plane()
                cross_product = planes[j].normal.cross(planes[i].normal)

                if cross_product.norm_sq() <= cfg.RVO3D_EPSILON:
                    if planes[i].normal.dot(planes[j].normal) > 0.0:
                        continue
                    else:
                        plane.point = 0.5 * (planes[i].point + planes[j].point)
                else:
                    line_normal = cross_product.cross(planes[i].normal)
                    plane.point = planes[i].point + (((planes[j].point - planes[i].point).dot(planes[j].normal)) / (line_normal.dot(planes[j].normal))) * line_normal
                
                plane.normal = (planes[j].normal - planes[i].normal).normalized()
                proj_planes.append(plane)

            temp_result = result
            fail_plane, new_result = linear_program3(proj_planes, radius, planes[i].normal, True)
            
            if fail_plane < len(proj_planes):
                result = temp_result
            else:
                result = new_result
            
            distance = planes[i].normal.dot(planes[i].point - result)
    return result
