#!/usr/bin/env python3
"""
简单测试VMC运动学计算
不依赖IsaacGym,直接对比公式
"""

import numpy as np
import sys
sys.path.append('.')

from mujoco_sim.vmc_kinematics import VMCKinematics

# 初始化VMC
vmc = VMCKinematics(l1=0.167, l2=0.200)

print("测试VMC运动学计算\n")
print("=" * 60)

# 测试用例
test_cases = [
    ([0.0, 0.0], "零位 (腿伸直向下)"),
    ([0.4, 0.25], "默认位置"),
    ([0.5, 0.0], "theta1=0.5, theta2=0"),
    ([0.0, 0.5], "theta1=0, theta2=0.5"),
    ([np.pi/4, 0.0], "theta1=45°, theta2=0"),
    ([0.0, np.pi/4], "theta1=0, theta2=45°"),
]

for (theta1, theta2), desc in test_cases:
    L0, theta0 = vmc.forward_kinematics(theta1, theta2)

    print(f"\n{desc}")
    print(f"  输入: theta1={theta1:.4f} rad ({np.degrees(theta1):.1f}°), "
          f"theta2={theta2:.4f} rad ({np.degrees(theta2):.1f}°)")
    print(f"  输出: L0={L0:.4f} m, theta0={theta0:.4f} rad ({np.degrees(theta0):.1f}°)")

    # 手动验证
    l1, l2 = 0.167, 0.200
    end_x = l1 * np.cos(theta1) - l2 * np.sin(theta1 + theta2)
    end_y = l1 * np.sin(theta1) + l2 * np.cos(theta1 + theta2)
    L0_check = np.sqrt(end_x**2 + end_y**2)
    theta0_check = np.arctan2(end_x, end_y)

    print(f"  验证: end_x={end_x:.4f}, end_y={end_y:.4f}")
    print(f"  验证: L0={L0_check:.4f}, theta0={theta0_check:.4f}")

    # 检查是否匹配
    if abs(L0 - L0_check) < 1e-6 and abs(theta0 - theta0_check) < 1e-6:
        print(f"  ✓ 计算正确")
    else:
        print(f"  ✗ 计算错误!")

print("\n" + "=" * 60)
print("\n预期行为:")
print("  - 零位 (0,0): 腿伸直向下, L0 ≈ 0.367m (l1+l2), theta0 ≈ 0 rad")
print("  - theta0 > 0: 腿向前倾")
print("  - theta0 < 0: 腿向后倾")
print("  - L0越小,腿越弯曲")
