from utils.pica_structures import Vector3D

# test ： genelized-test

# 通用测试

test_groups = [
    {'ratio': 0.5, 'params': {'radius': 1.0, 'P': 0.5, 'M': Vector3D(1.0, 1.0, 1.0)}},
    {'ratio': 0.5, 'params': {'radius': 1.0, 'P': 0.5, 'M': Vector3D(1.0, 1.0, 1.0)}},
]

R1_groups = [
    {'ratio': 0.5, 'params': {'radius': 2.0, 'P': 0.5, 'M': Vector3D(1.0, 1.0, 1.0)}},
    {'ratio': 0.5, 'params': {'radius': 2.0, 'P': 0.5, 'M': Vector3D(1.0, 1.0, 1.0)}},
]

R2_groups = [
    {'ratio': 0.5, 'params': {'radius': 0.5, 'P': 0.5, 'M': Vector3D(1.0, 1.0, 1.0)}},
    {'ratio': 0.5, 'params': {'radius': 0.5, 'P': 0.5, 'M': Vector3D(1.0, 1.0, 1.0)}},
]

R3_groups = [
    {'ratio': 0.5, 'params': {'radius': 2.0, 'P': 0.5, 'M': Vector3D(1.0, 1.0, 1.0)}},
    {'ratio': 0.5, 'params': {'radius': 1.0, 'P': 0.5, 'M': Vector3D(1.0, 1.0, 1.0)}},
]

R4_groups = [
    {'ratio': 0.5, 'params': {'radius': 2.0, 'P': 0.5, 'M': Vector3D(1.0, 1.0, 1.0)}},
    {'ratio': 0.5, 'params': {'radius': 0.5, 'P': 0.5, 'M': Vector3D(1.0, 1.0, 1.0)}},
]

R5_groups = [
    {'ratio': 0.5, 'params': {'radius': 1.0, 'P': 0.5, 'M': Vector3D(1.0, 1.0, 1.0)}},
    {'ratio': 0.5, 'params': {'radius': 0.5, 'P': 0.5, 'M': Vector3D(1.0, 1.0, 1.0)}},
]

## PIVO专属测试
P1_groups = [
    {'ratio': 0.5, 'params': {'radius': 1.0, 'P': 0.4, 'M': Vector3D(1.0, 1.0, 1.0)}},
    {'ratio': 0.5, 'params': {'radius': 1.0, 'P': 0.6, 'M': Vector3D(1.0, 1.0, 1.0)}},
]

P2_groups = [
    {'ratio': 0.5, 'params': {'radius': 1.0, 'P': 0.3, 'M': Vector3D(1.0, 1.0, 1.0)}},
    {'ratio': 0.5, 'params': {'radius': 1.0, 'P': 0.7, 'M': Vector3D(1.0, 1.0, 1.0)}},
]

P3_groups = [
    {'ratio': 0.5, 'params': {'radius': 1.0, 'P': 0.2, 'M': Vector3D(1.0, 1.0, 1.0)}},
    {'ratio': 0.5, 'params': {'radius': 1.0, 'P': 0.8, 'M': Vector3D(1.0, 1.0, 1.0)}},
]

P4_groups = [
    {'ratio': 0.5, 'params': {'radius': 1.0, 'P': 0.1, 'M': Vector3D(1.0, 1.0, 1.0)}},
    {'ratio': 0.5, 'params': {'radius': 1.0, 'P': 0.9, 'M': Vector3D(1.0, 1.0, 1.0)}},
]

P5_groups = [
    {'ratio': 0.5, 'params': {'radius': 1.0, 'P': 0.5, 'M': Vector3D(1.0, 1.0, 1.0)}},
    {'ratio': 0.5, 'params': {'radius': 1.0, 'P': 0.2, 'M': Vector3D(1.0, 1.0, 1.0)}},
]

P6_groups = [
    {'ratio': 0.5, 'params': {'radius': 1.0, 'P': 0.5, 'M': Vector3D(1.0, 1.0, 1.0)}},
    {'ratio': 0.5, 'params': {'radius': 1.0, 'P': 0.4, 'M': Vector3D(1.0, 1.0, 1.0)}},
]

P7_groups = [
    {'ratio': 0.5, 'params': {'radius': 1.0, 'P': 0.5, 'M': Vector3D(1.0, 1.0, 1.0)}},
    {'ratio': 0.5, 'params': {'radius': 1.0, 'P': 0.6, 'M': Vector3D(1.0, 1.0, 1.0)}},
]

P8_groups = [
    {'ratio': 0.5, 'params': {'radius': 1.0, 'P': 0.5, 'M': Vector3D(1.0, 1.0, 1.0)}},
    {'ratio': 0.5, 'params': {'radius': 1.0, 'P': 0.8, 'M': Vector3D(1.0, 1.0, 1.0)}},
]


M1_groups = [
    {'ratio': 0.5, 'params': {'radius': 1.0, 'P': 0.5, 'M': Vector3D(2.0, 2.0, 2.0)}},
    {'ratio': 0.5, 'params': {'radius': 1.0, 'P': 0.5, 'M': Vector3D(1.0, 1.0, 1.0)}},
]

M2_groups = [
    {'ratio': 0.5, 'params': {'radius': 1.0, 'P': 0.5, 'M': Vector3D(0.5, 0.5, 0.5)}},
    {'ratio': 0.5, 'params': {'radius': 1.0, 'P': 0.5, 'M': Vector3D(1.0, 1.0, 1.0)}},
]

M3_groups = [
    {'ratio': 0.5, 'params': {'radius': 1.0, 'P': 0.5, 'M': Vector3D(2.0, 2.0, 2.0)}},
    {'ratio': 0.5, 'params': {'radius': 1.0, 'P': 0.5, 'M': Vector3D(0.5, 0.5, 0.5)}},
]

M4_groups = [
    {'ratio': 0.5, 'params': {'radius': 1.0, 'P': 0.5, 'M': Vector3D(2.0, 2.0, 2.0)}},
    {'ratio': 0.5, 'params': {'radius': 1.0, 'P': 0.5, 'M': Vector3D(2.0, 2.0, 2.0)}},
]

M5_groups = [
    {'ratio': 0.5, 'params': {'radius': 1.0, 'P': 0.5, 'M': Vector3D(0.5, 0.5, 0.5)}},
    {'ratio': 0.5, 'params': {'radius': 1.0, 'P': 0.5, 'M': Vector3D(0.5, 0.5, 0.5)}},
]

# 混合参数比较场景1，扭转测试，简单单调测试，PM的baseline，R为1.0，P为0.5，M为0.5/2.0
PM1_groups = [
    {'ratio': 0.5, 'params': {'radius': 1.0, 'P': 0.3, 'M': Vector3D(0.5, 0.5, 0.5)}},
    {'ratio': 0.5, 'params': {'radius': 1.0, 'P': 0.7, 'M': Vector3D(2.0, 2.0, 2.0)}},
]

PM2_groups = [
    {'ratio': 0.5, 'params': {'radius': 1.0, 'P': 0.7, 'M': Vector3D(0.5, 0.5, 0.5)}},
    {'ratio': 0.5, 'params': {'radius': 1.0, 'P': 0.3, 'M': Vector3D(2.0, 2.0, 2.0)}},
]

PM3_groups = [
    {'ratio': 0.5, 'params': {'radius': 1.0, 'P': 0.5, 'M': Vector3D(0.5, 0.5, 0.5)}},
    {'ratio': 0.5, 'params': {'radius': 1.0, 'P': 0.8, 'M': Vector3D(2.0, 2.0, 2.0)}},
]

PM4_groups = [
    {'ratio': 0.5, 'params': {'radius': 1.0, 'P': 0.5, 'M': Vector3D(0.5, 0.5, 0.5)}},
    {'ratio': 0.5, 'params': {'radius': 1.0, 'P': 0.2, 'M': Vector3D(2.0, 2.0, 2.0)}},
]

## 扭转测试，RP的baseline, R为1.0，P为0.3/0.7, M为1.0
RP1_groups = [
    {'ratio': 0.5, 'params': {'radius': 2.0, 'P': 0.7, 'M': Vector3D(1.0, 1.0, 1.0)}},
    {'ratio': 0.5, 'params': {'radius': 1.0, 'P': 0.3, 'M': Vector3D(1.0, 1.0, 1.0)}},
]

RP2_groups = [
    {'ratio': 0.5, 'params': {'radius': 1.0, 'P': 0.7, 'M': Vector3D(1.0, 1.0, 1.0)}},
    {'ratio': 0.5, 'params': {'radius': 0.5, 'P': 0.3, 'M': Vector3D(1.0, 1.0, 1.0)}},
]

RP3_groups = [
    {'ratio': 0.5, 'params': {'radius': 2.0, 'P': 0.7, 'M': Vector3D(1.0, 1.0, 1.0)}},
    {'ratio': 0.5, 'params': {'radius': 0.5, 'P': 0.3, 'M': Vector3D(1.0, 1.0, 1.0)}},
]

RP4_groups = [
    {'ratio': 0.5, 'params': {'radius': 2.0, 'P': 0.3, 'M': Vector3D(1.0, 1.0, 1.0)}},
    {'ratio': 0.5, 'params': {'radius': 1.0, 'P': 0.7, 'M': Vector3D(1.0, 1.0, 1.0)}},
]

RP5_groups = [
    {'ratio': 0.5, 'params': {'radius': 1.0, 'P': 0.3, 'M': Vector3D(1.0, 1.0, 1.0)}},
    {'ratio': 0.5, 'params': {'radius': 0.5, 'P': 0.7, 'M': Vector3D(1.0, 1.0, 1.0)}},
]

RP6_groups = [
    {'ratio': 0.5, 'params': {'radius': 2.0, 'P': 0.3, 'M': Vector3D(1.0, 1.0, 1.0)}},
    {'ratio': 0.5, 'params': {'radius': 0.5, 'P': 0.7, 'M': Vector3D(1.0, 1.0, 1.0)}},
]

## 同质的不用测，扭转测试，baseline为M系列
RM1_groups = [
    {'ratio': 0.5, 'params': {'radius': 2.0, 'P': 0.5, 'M': Vector3D(2.0, 2.0, 2.0)}},
    {'ratio': 0.5, 'params': {'radius': 1.0, 'P': 0.5, 'M': Vector3D(1.0, 1.0, 1.0)}},
]

RM2_groups = [
    {'ratio': 0.5, 'params': {'radius': 1.0, 'P': 0.5, 'M': Vector3D(1.0, 1.0, 1.0)}},
    {'ratio': 0.5, 'params': {'radius': 0.5, 'P': 0.5, 'M': Vector3D(0.5, 0.5, 0.5)}},
]

RM3_groups = [
    {'ratio': 0.5, 'params': {'radius': 2.0, 'P': 0.5, 'M': Vector3D(2.0, 2.0, 2.0)}},
    {'ratio': 0.5, 'params': {'radius': 0.5, 'P': 0.5, 'M': Vector3D(0.5, 0.5, 0.5)}},
]

RM4_groups = [
    {'ratio': 0.5, 'params': {'radius': 1.0, 'P': 0.5, 'M': Vector3D(2.0, 2.0, 2.0)}},
    {'ratio': 0.5, 'params': {'radius': 2.0, 'P': 0.5, 'M': Vector3D(1.0, 1.0, 1.0)}},
]

RM5_groups = [
    {'ratio': 0.5, 'params': {'radius': 0.5, 'P': 0.5, 'M': Vector3D(1.0, 1.0, 1.0)}},
    {'ratio': 0.5, 'params': {'radius': 1.0, 'P': 0.5, 'M': Vector3D(0.5, 0.5, 0.5)}},
]

RM6_groups = [
    {'ratio': 0.5, 'params': {'radius': 0.5, 'P': 0.5, 'M': Vector3D(2.0, 2.0, 2.0)}},
    {'ratio': 0.5, 'params': {'radius': 2.0, 'P': 0.5, 'M': Vector3D(0.5, 0.5, 0.5)}},
]

# 混合参数比较场景2

RPM1_groups = [
    {'ratio': 0.5, 'params': {'radius': 1.0, 'P': 0.7, 'M': Vector3D(1.0, 1.0, 1.0)}},
    {'ratio': 0.5, 'params': {'radius': 0.5, 'P': 0.3, 'M': Vector3D(0.5, 0.5, 0.5)}},
]

RPM2_groups = [
    {'ratio': 0.5, 'params': {'radius': 0.5, 'P': 0.8, 'M': Vector3D(2.0, 2.0, 2.0)}},
    {'ratio': 0.5, 'params': {'radius': 2.0, 'P': 0.2, 'M': Vector3D(0.5, 0.5, 0.5)}}
]

RP10_groups = [
    {'ratio': 0.5, 'params': {'radius': 2.0, 'P': 0.2, 'M': Vector3D(1.0, 1.0, 1.0)}},
    {'ratio': 0.5, 'params': {'radius': 0.5, 'P': 0.9, 'M': Vector3D(1.0, 1.0, 1.0)}}
]

# 比例掺杂场景 baseline：0.3混合比例，R为1.0，P为0.5，V为1.0
P9_groups = [
    {'ratio': 0.33, 'params': {'radius': 1.0, 'P': 0.3, 'M': Vector3D(1.0, 1.0, 1.0)}},
    {'ratio': 0.33, 'params': {'radius': 1.0, 'P': 0.5, 'M': Vector3D(1.0, 1.0, 1.0)}},
    {'ratio': 0.34, 'params': {'radius': 1.0, 'P': 0.7, 'M': Vector3D(1.0, 1.0, 1.0)}}
]

# 单调测试

P10_groups = [
    {'ratio': 0.5, 'params': {'radius': 1.0, 'P': 0.1, 'M': Vector3D(1.0, 1.0, 1.0)}},
    {'ratio': 0.5, 'params': {'radius': 1.0, 'P': 0.2, 'M': Vector3D(1.0, 1.0, 1.0)}},
]

P11_groups = [
    {'ratio': 0.5, 'params': {'radius': 1.0, 'P': 0.3, 'M': Vector3D(1.0, 1.0, 1.0)}},
    {'ratio': 0.5, 'params': {'radius': 1.0, 'P': 0.4, 'M': Vector3D(1.0, 1.0, 1.0)}},
]

P12_groups = [
    {'ratio': 0.5, 'params': {'radius': 1.0, 'P': 0.5, 'M': Vector3D(1.0, 1.0, 1.0)}},
    {'ratio': 0.5, 'params': {'radius': 1.0, 'P': 0.6, 'M': Vector3D(1.0, 1.0, 1.0)}},
]

# RPM测试
RPM3_groups = [
    {'ratio': 0.5, 'params': {'radius': 2.0, 'P': 0.1, 'M': Vector3D(0.5, 0.5, 0.5)}},
    {'ratio': 0.5, 'params': {'radius': 0.5, 'P': 0.2, 'M': Vector3D(2.0, 2.0, 2.0)}},
]

RPM4_groups = [
    {'ratio': 0.5, 'params': {'radius': 2.0, 'P': 0.3, 'M': Vector3D(0.5, 0.5, 0.5)}},
    {'ratio': 0.5, 'params': {'radius': 0.5, 'P': 0.4, 'M': Vector3D(1.0, 1.0, 1.0)}},
]

# 比例分配测试
M_groups = [
    {'ratio': 0.33, 'params': {'radius': 1.0, 'P': 0.5, 'M': Vector3D(0.5, 0.5, 0.5)}},
    {'ratio': 0.33, 'params': {'radius': 1.0, 'P': 0.5, 'M': Vector3D(1.0, 1.0, 1.0)}},
    {'ratio': 0.34, 'params': {'radius': 1.0, 'P': 0.5, 'M': Vector3D(2.0, 2.0, 2.0)}},
]

R_groups = [
    {'ratio': 0.33, 'params': {'radius': 0.5, 'P': 0.5, 'M': Vector3D(1.0, 1.0, 1.0)}},
    {'ratio': 0.33, 'params': {'radius': 1.0, 'P': 0.5, 'M': Vector3D(1.0, 1.0, 1.0)}},
    {'ratio': 0.34, 'params': {'radius': 2.0, 'P': 0.5, 'M': Vector3D(1.0, 1.0, 1.0)}}
]

RPM5_groups = [
    {'ratio': 0.33, 'params': {'radius': 2.0, 'P': 0.3, 'M': Vector3D(2.0, 2.0, 2.0)}},
    {'ratio': 0.67, 'params': {'radius': 0.5, 'P': 0.5, 'M': Vector3D(0.5, 0.5, 0.5)}},
]

RPM6_groups = [
    {'ratio': 0.33, 'params': {'radius': 0.5, 'P': 0.7, 'M': Vector3D(0.5, 0.5, 0.5)}},
    {'ratio': 0.33, 'params': {'radius': 1.0, 'P': 0.5, 'M': Vector3D(1.0, 1.0, 1.0)}},
    {'ratio': 0.34, 'params': {'radius': 2.0, 'P': 0.3, 'M': Vector3D(2.0, 2.0, 2.0)}},
]