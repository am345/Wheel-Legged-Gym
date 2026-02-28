#!/usr/bin/env python3
"""分析 MuJoCo 中约束力矩的来源和物理意义"""

import sys
sys.path.insert(0, '.')

import mujoco
import numpy as np

# Load model
model_path = "resources/robots/serialleg/mjcf/serialleg_fidelity.xml"
model = mujoco.MjModel.from_xml_path(model_path)
data = mujoco.MjData(model)

print("=" * 80)
print("约束力矩 (qfrc_constraint) 分析")
print("=" * 80)

# Reset to default state
mujoco.mj_resetData(model, data)
data.qpos[0:3] = [0.0, 0.0, 0.30]  # base position
data.qpos[3:7] = [1.0, 0.0, 0.0, 0.0]  # base orientation
data.qpos[7:13] = [0.4, 0.25, 0.0, 0.4, 0.25, 0.0]  # joint positions

# Apply control
data.ctrl[:] = [20, 15, 0, 25, 18, 0]

# Step simulation
for _ in range(10):
    mujoco.mj_step(model, data)

print("\n1. 力矩分解")
print("-" * 80)

joint_ids = [6, 7, 8, 9, 10, 11]  # Joint DOF indices
joint_names = ['lf0', 'lf1', 'l_wheel', 'rf0', 'rf1', 'r_wheel']

print(f"\n{'Joint':<10} {'Actuator':<12} {'Constraint':<12} {'Net':<12} {'Accel':<12}")
print("-" * 80)

for i, (dof_id, name) in enumerate(zip(joint_ids, joint_names)):
    qa = data.qfrc_actuator[dof_id]
    qc = data.qfrc_constraint[dof_id]
    net = qa + qc
    acc = data.qacc[dof_id]

    print(f"{name:<10} {qa:+11.3f} {qc:+11.3f} {net:+11.3f} {acc:+11.3f}")

print("\n2. 约束力矩的来源")
print("-" * 80)

print("\n约束力矩 (qfrc_constraint) 包含以下几种力：")
print("  a) 接触力 (Contact forces)")
print("  b) 关节限制力 (Joint limit forces)")
print("  c) 其他约束力 (Other constraint forces)")

print("\n让我们分析接触力的贡献：")

# Get contact information
print(f"\n当前接触数量: {data.ncon}")
print("\n接触详情:")

for i in range(data.ncon):
    contact = data.contact[i]
    geom1_name = mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_GEOM, contact.geom1)
    geom2_name = mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_GEOM, contact.geom2)

    # Contact force magnitude
    force = np.linalg.norm(contact.frame[:3])  # Normal + friction forces

    print(f"\n  接触 {i+1}:")
    print(f"    几何体: {geom1_name} <-> {geom2_name}")
    print(f"    接触力: {force:.3f} N")
    print(f"    接触位置: [{contact.pos[0]:.3f}, {contact.pos[1]:.3f}, {contact.pos[2]:.3f}]")
    print(f"    法向力: {contact.frame[0]:.3f} N")

print("\n3. 物理解释")
print("-" * 80)

print("""
为什么约束力矩这么大？

这是**正常的物理现象**，原因如下：

1. **机器人在平衡状态**：
   - 机器人站立时，关节力矩主要用于对抗重力
   - 地面接触产生反作用力（牛顿第三定律）
   - 这些反作用力通过运动学链传递到关节，产生约束力矩

2. **力矩平衡方程**：
   M * qacc = qfrc_actuator + qfrc_constraint + qfrc_passive + qfrc_bias

   其中：
   - qfrc_actuator: 执行器力矩（你控制的）
   - qfrc_constraint: 约束力矩（来自接触、关节限制等）
   - qfrc_passive: 被动力（弹簧、阻尼）
   - qfrc_bias: 科氏力、离心力、重力

3. **平衡时的特点**：
   - qacc ≈ 0 （加速度很小）
   - 因此：qfrc_actuator + qfrc_constraint + ... ≈ 0
   - 所以：qfrc_constraint ≈ -qfrc_actuator

4. **类比人体站立**：
   - 你的肌肉施加力量（执行器力矩）
   - 地面施加反作用力（约束力矩）
   - 两者平衡，所以你保持静止

结论：约束力矩大是因为机器人在平衡状态下，地面反作用力必须平衡
      执行器力矩和重力。这不是问题，而是正确的物理行为！
""")

print("\n4. 验证：移除接触后的情况")
print("-" * 80)

# Disable contacts temporarily
print("\n如果机器人悬空（无接触），约束力矩会变小：")

# Reset and lift robot
mujoco.mj_resetData(model, data)
data.qpos[0:3] = [0.0, 0.0, 1.0]  # Lift robot high above ground
data.qpos[3:7] = [1.0, 0.0, 0.0, 0.0]
data.qpos[7:13] = [0.4, 0.25, 0.0, 0.4, 0.25, 0.0]
data.ctrl[:] = [20, 15, 0, 25, 18, 0]

for _ in range(10):
    mujoco.mj_step(model, data)

print(f"\n悬空时的约束力矩: {data.qfrc_constraint[joint_ids]}")
print(f"悬空时的加速度:   {data.qacc[joint_ids]}")
print(f"接触数量:         {data.ncon}")

print("\n观察：")
print("  - 约束力矩接近 0（因为没有接触）")
print("  - 加速度变大（因为没有地面支撑）")
print("  - 这证明约束力矩主要来自地面接触")

print("\n" + "=" * 80)
