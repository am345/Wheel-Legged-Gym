#!/usr/bin/env python3
"""
测试修改后的MJCF模型几何
"""

import mujoco
import numpy as np

# 加载模型
model = mujoco.MjModel.from_xml_path('resources/robots/serialleg/mjcf/serialleg_simple.xml')
data = mujoco.MjData(model)

# 重置
mujoco.mj_resetData(model, data)

print("测试MJCF模型几何\n")
print("=" * 60)

# 设置关节位置为零
data.qpos[7:13] = 0.0
mujoco.mj_forward(model, data)

print("\n零位 (所有关节=0):")
print(f"  基座位置: {data.qpos[0:3]}")
print(f"  关节位置: {data.qpos[7:13]}")

# 获取轮子位置
l_wheel_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, 'l_wheel_Link')
r_wheel_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, 'r_wheel_Link')

l_wheel_pos = data.xpos[l_wheel_id]
r_wheel_pos = data.xpos[r_wheel_id]

print(f"  左轮位置: {l_wheel_pos}")
print(f"  右轮位置: {r_wheel_pos}")

# 计算虚拟腿长度
base_pos = data.qpos[0:3]
l_leg_length = np.linalg.norm(l_wheel_pos - base_pos)
r_leg_length = np.linalg.norm(r_wheel_pos - base_pos)

print(f"  左腿长度: {l_leg_length:.4f} m")
print(f"  右腿长度: {r_leg_length:.4f} m")
print(f"  预期长度: {0.167 + 0.200:.4f} m (l1+l2)")

# 测试默认位置
print("\n默认位置 (theta1=0.4, theta2=0.25):")
data.qpos[7] = 0.4   # lf0
data.qpos[8] = 0.25  # lf1
data.qpos[10] = 0.4  # rf0
data.qpos[11] = 0.25 # rf1
mujoco.mj_forward(model, data)

l_wheel_pos = data.xpos[l_wheel_id]
r_wheel_pos = data.xpos[r_wheel_id]

print(f"  左轮位置: {l_wheel_pos}")
print(f"  右轮位置: {r_wheel_pos}")

# 手动计算VMC
l1, l2 = 0.167, 0.200
theta1, theta2 = 0.4, 0.25
end_x = l1 * np.cos(theta1) - l2 * np.sin(theta1 + theta2)
end_y = l1 * np.sin(theta1) + l2 * np.cos(theta1 + theta2)
L0 = np.sqrt(end_x**2 + end_y**2)
theta0 = np.arctan2(end_x, end_y)

print(f"\n  VMC计算:")
print(f"    end_x={end_x:.4f}, end_y={end_y:.4f}")
print(f"    L0={L0:.4f}, theta0={theta0:.4f} rad ({np.degrees(theta0):.1f}°)")

# 从MuJoCo位置计算
base_to_wheel = l_wheel_pos - base_pos
print(f"\n  MuJoCo位置:")
print(f"    基座到轮子: {base_to_wheel}")
print(f"    距离: {np.linalg.norm(base_to_wheel):.4f}")

print("\n" + "=" * 60)
