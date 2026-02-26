#!/usr/bin/env python3
"""
找到让腿伸直向下的关节角度
"""

import numpy as np
from scipy.optimize import fsolve

l1, l2 = 0.167, 0.200

# 目标:end_x=0, end_y=l1+l2
def equations(angles):
    theta1, theta2 = angles
    end_x = l1 * np.cos(theta1) - l2 * np.sin(theta1 + theta2)
    end_y = l1 * np.sin(theta1) + l2 * np.cos(theta1 + theta2)
    return [end_x, end_y - (l1 + l2)]

# 求解
solution = fsolve(equations, [0, 0])
theta1_straight, theta2_straight = solution

print(f"让腿伸直向下的角度:")
print(f"  theta1 = {theta1_straight:.4f} rad ({np.degrees(theta1_straight):.1f}°)")
print(f"  theta2 = {theta2_straight:.4f} rad ({np.degrees(theta2_straight):.1f}°)")

# 验证
end_x = l1 * np.cos(theta1_straight) - l2 * np.sin(theta1_straight + theta2_straight)
end_y = l1 * np.sin(theta1_straight) + l2 * np.cos(theta1_straight + theta2_straight)
L0 = np.sqrt(end_x**2 + end_y**2)
theta0 = np.arctan2(end_x, end_y)

print(f"\n验证:")
print(f"  end_x = {end_x:.6f}")
print(f"  end_y = {end_y:.6f}")
print(f"  L0 = {L0:.4f} m (预期 {l1+l2:.4f})")
print(f"  theta0 = {theta0:.6f} rad ({np.degrees(theta0):.1f}°)")

# 另一个解:theta1=π/2, theta2=-π/2
print(f"\n另一个解 (theta1=π/2, theta2=-π/2):")
theta1, theta2 = np.pi/2, -np.pi/2
end_x = l1 * np.cos(theta1) - l2 * np.sin(theta1 + theta2)
end_y = l1 * np.sin(theta1) + l2 * np.cos(theta1 + theta2)
L0 = np.sqrt(end_x**2 + end_y**2)
theta0 = np.arctan2(end_x, end_y)
print(f"  end_x = {end_x:.6f}")
print(f"  end_y = {end_y:.6f}")
print(f"  L0 = {L0:.4f} m")
print(f"  theta0 = {theta0:.6f} rad ({np.degrees(theta0):.1f}°)")
