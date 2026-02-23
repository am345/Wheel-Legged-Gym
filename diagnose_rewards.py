#!/usr/bin/env python3
"""诊断 Balance 任务的奖励函数问题"""

import torch

# 模拟奖励函数
def reward_base_height(base_height, command_height, scale):
    """base_height 奖励 - 正向奖励"""
    if scale < 0:
        return scale * torch.abs(base_height - command_height)
    else:
        base_height_error = torch.square(base_height - command_height)
        return scale * torch.exp(-base_height_error / 0.01)

def reward_orientation(projected_gravity, scale):
    """orientation 奖励 - 负向惩罚"""
    return scale * torch.sum(torch.square(projected_gravity[:2]))

def reward_upright_bonus(projected_gravity, base_lin_vel, base_ang_vel, scale):
    """upright_bonus 奖励 - 正向奖励"""
    grav_ok = (torch.abs(projected_gravity[0]) < 0.05) & (torch.abs(projected_gravity[1]) < 0.05)
    lin_ok = torch.norm(base_lin_vel) < 0.3
    ang_ok = torch.norm(base_ang_vel) < 0.3
    return scale * float(grav_ok & lin_ok & ang_ok)

# 配置参数
base_height_scale = 40.0
orientation_scale = -40.0
upright_bonus_scale = 20.0
target_height = 0.25

print("=" * 60)
print("Balance 任务奖励诊断")
print("=" * 60)

# 场景 1: 完美平衡
print("\n场景 1: 完美平衡 (理想状态)")
print("-" * 60)
base_height = torch.tensor(0.25)
projected_gravity = torch.tensor([0.0, 0.0, -1.0])
base_lin_vel = torch.tensor([0.0, 0.0, 0.0])
base_ang_vel = torch.tensor([0.0, 0.0, 0.0])

r1 = reward_base_height(base_height, target_height, base_height_scale)
r2 = reward_orientation(projected_gravity, orientation_scale)
r3 = reward_upright_bonus(projected_gravity, base_lin_vel, base_ang_vel, upright_bonus_scale)
total = r1 + r2 + r3

print(f"base_height:    {r1.item():8.2f}  (scale={base_height_scale})")
print(f"orientation:    {r2.item():8.2f}  (scale={orientation_scale})")
print(f"upright_bonus:  {r3:8.2f}  (scale={upright_bonus_scale})")
print(f"TOTAL:          {total.item():8.2f}")

# 场景 2: 倒地 (但高度正确)
print("\n场景 2: 倒地 (projected_gravity.z > -0.1, 触发终止)")
print("-" * 60)
base_height = torch.tensor(0.25)
projected_gravity = torch.tensor([0.9, 0.0, -0.05])  # 几乎水平
base_lin_vel = torch.tensor([0.0, 0.0, 0.0])
base_ang_vel = torch.tensor([0.0, 0.0, 0.0])

r1 = reward_base_height(base_height, target_height, base_height_scale)
r2 = reward_orientation(projected_gravity, orientation_scale)
r3 = reward_upright_bonus(projected_gravity, base_lin_vel, base_ang_vel, upright_bonus_scale)
total = r1 + r2 + r3

print(f"base_height:    {r1.item():8.2f}")
print(f"orientation:    {r2.item():8.2f}  (巨大惩罚!)")
print(f"upright_bonus:  {r3:8.2f}  (不满足条件)")
print(f"TOTAL:          {total.item():8.2f}")
print("⚠️  但是 episode 会立即终止 (projected_gravity.z > -0.1)")

# 场景 3: 轻微倾斜
print("\n场景 3: 轻微倾斜 (10度)")
print("-" * 60)
import math
angle = math.radians(10)
base_height = torch.tensor(0.25)
projected_gravity = torch.tensor([math.sin(angle), 0.0, -math.cos(angle)])
base_lin_vel = torch.tensor([0.0, 0.0, 0.0])
base_ang_vel = torch.tensor([0.0, 0.0, 0.0])

r1 = reward_base_height(base_height, target_height, base_height_scale)
r2 = reward_orientation(projected_gravity, orientation_scale)
r3 = reward_upright_bonus(projected_gravity, base_lin_vel, base_ang_vel, upright_bonus_scale)
total = r1 + r2 + r3

print(f"base_height:    {r1.item():8.2f}")
print(f"orientation:    {r2.item():8.2f}")
print(f"upright_bonus:  {r3:8.2f}  (不满足 < 0.05 条件)")
print(f"TOTAL:          {total.item():8.2f}")

# 场景 4: 高度错误但直立
print("\n场景 4: 高度错误 (0.15m) 但完全直立")
print("-" * 60)
base_height = torch.tensor(0.15)
projected_gravity = torch.tensor([0.0, 0.0, -1.0])
base_lin_vel = torch.tensor([0.0, 0.0, 0.0])
base_ang_vel = torch.tensor([0.0, 0.0, 0.0])

r1 = reward_base_height(base_height, target_height, base_height_scale)
r2 = reward_orientation(projected_gravity, orientation_scale)
r3 = reward_upright_bonus(projected_gravity, base_lin_vel, base_ang_vel, upright_bonus_scale)
total = r1 + r2 + r3

print(f"base_height:    {r1.item():8.2f}  (高度差 0.1m)")
print(f"orientation:    {r2.item():8.2f}")
print(f"upright_bonus:  {r3:8.2f}")
print(f"TOTAL:          {total.item():8.2f}")

print("\n" + "=" * 60)
print("关键发现:")
print("=" * 60)
print("1. base_height 使用 exp(-error^2/0.01)，在 scale=40 时:")
print("   - 完美高度: 40.0")
print("   - 差 0.1m:  ~0.6")
print("   - 差 0.2m:  ~0.0")
print("   → 高度奖励衰减非常快!")
print()
print("2. orientation 使用 -40 * gravity_xy^2:")
print("   - 10度倾斜: ~-1.2")
print("   - 30度倾斜: ~-10.0")
print("   - 60度倾斜: ~-30.0")
print("   → 倾斜惩罚相对较小")
print()
print("3. upright_bonus 条件非常严格:")
print("   - gravity_xy < 0.05 (约 3 度)")
print("   - lin_vel < 0.3 m/s")
print("   - ang_vel < 0.3 rad/s")
print("   → 很难同时满足所有条件")
print()
print("4. 终止条件: projected_gravity.z > -0.1 (约 84 度)")
print("   → 机器人倒地前就会终止 episode")
print()
print("=" * 60)
print("可能的问题:")
print("=" * 60)
print("❌ 奖励可能来自「倒地前的短暂时刻」而非真正平衡")
print("❌ base_height 奖励衰减太快，机器人可能学会「蹲下」")
print("❌ orientation 惩罚太弱，机器人可能学会「倾斜但不倒」")
print("❌ upright_bonus 条件太严格，很少能获得")
print("=" * 60)
