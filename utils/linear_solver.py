import math
from utils.pica2d_structures import Vector2D, Line
from enviroments import config as cfg

def linear_program1(lines, line_no, radius, opt_velocity, direction_opt):
    """求解一维线性规划"""
    line = lines[line_no]
    dot_product = line.point @ line.direction
    discriminant = dot_product **2 + radius** 2 - line.point.norm_sq()

    if discriminant < 0.0:
        return (False, Vector2D())

    sqrt_discriminant = math.sqrt(discriminant)
    t_left = -dot_product - sqrt_discriminant
    t_right = -dot_product + sqrt_discriminant

    for i in range(line_no):
        other_line = lines[i]
        denominator = Vector2D.det(line.direction, other_line.direction)
        numerator = Vector2D.det(other_line.direction, line.point - other_line.point)

        if math.fabs(denominator) <= cfg.EPSILON:
            if numerator < 0.0:
                return (False, Vector2D())
            continue

        t = numerator / denominator

        if denominator >= 0.0:
            t_right = min(t_right, t)
        else:
            t_left = max(t_left, t)

        if t_left > t_right:
            return (False, Vector2D())

    if direction_opt:
        if opt_velocity @ line.direction > 0.0:
            result = line.point + line.direction * t_right
        else:
            result = line.point + line.direction * t_left
    else:
        t = line.direction @ (opt_velocity - line.point)
        if t < t_left:
            result = line.point + line.direction * t_left
        elif t > t_right:
            result = line.point + line.direction * t_right
        else:
            result = line.point + line.direction * t

    return (True, result)


def linear_program2(lines, radius, opt_velocity, direction_opt):
    """求解二维线性规划"""
    if direction_opt:
        result = opt_velocity * radius
    else:
        if opt_velocity.norm_sq() > radius **2:
            result = opt_velocity.normalized() * radius
        else:
            result = opt_velocity

    for i in range(len(lines)):
        line = lines[i]
        if Vector2D.det(line.direction, line.point - result) > 0.0:
            temp_result = result
            success, new_result = linear_program1(lines, i, radius, opt_velocity, direction_opt)
            if not success:
                return (i, temp_result)
            result = new_result

    return (len(lines), result)


def linear_program3(lines, num_obst_lines, begin_line, radius, result):
    """求解三维线性规划"""
    distance = 0.0
    for i in range(begin_line, len(lines)):
        line = lines[i]
        current_det = Vector2D.det(line.direction, line.point - result)
        if current_det > distance:
            proj_lines = lines[:num_obst_lines]

            for j in range(num_obst_lines, i):
                line_j = lines[j]
                determinant = Vector2D.det(line.direction, line_j.direction)

                if math.fabs(determinant) <= cfg.EPSILON:
                    if line.direction @ line_j.direction > 0.0:
                        continue
                    proj_point = (line.point + line_j.point) * 0.5
                else:
                    t = Vector2D.det(line_j.direction, line.point - line_j.point) / determinant
                    proj_point = line.point + line.direction * t

                proj_dir = (line_j.direction - line.direction).normalized()
                proj_lines.append(Line(proj_point, proj_dir))

            temp_result = result
            opt_vel = Vector2D(-line.direction.y, line.direction.x)
            line_fail, new_result = linear_program2(proj_lines, radius, opt_vel, True)
            
            if line_fail < len(proj_lines):
                result = temp_result
            else:
                result = new_result

            distance = Vector2D.det(line.direction, line.point - result)

    return result