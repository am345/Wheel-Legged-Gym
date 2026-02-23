# Play Balance - 使用指南

## 功能特性

重写后的 `play_balance.py` 提供了以下增强功能：

### 🎮 交互控制
- **C**: 启动控制（激活策略）
- **R**: 重置环境
- **S**: 切换统计信息显示
- **P**: 暂停/恢复
- **ESC**: 退出

### 📊 实时监控

#### 当前状态
- **高度**: 实时显示机器人高度、目标高度和误差
- **姿态**: Roll、Pitch 角度和总倾斜角
- **速度**: 线速度和角速度（3D 向量）
- **状态评估**: EXCELLENT / GOOD / UNSTABLE / FALLING

#### Episode 统计
- **当前 Episode**: 时间和步数
- **最长 Episode**: 历史最长平衡时间
- **成功率**: 完成 Episode 的比例
- **平均长度**: 所有 Episode 的平均时长

### 🎯 状态评估标准

| 状态 | 倾斜角度 | 高度误差 | 说明 |
|------|----------|----------|------|
| EXCELLENT | < 5° | < 0.05m | 完美平衡 |
| GOOD | < 10° | < 0.1m | 良好平衡 |
| UNSTABLE | < 20° | - | 不稳定 |
| FALLING | > 20° | - | 即将倒地 |

## 使用方法

### 方式 1: 使用快速脚本（推荐）

```bash
cd /home/am345/Wheel-Legged-Gym
./test_balance.sh
```

### 方式 2: 手动运行

```bash
cd /home/am345/Wheel-Legged-Gym
conda activate gym_env
python wheel_legged_gym/scripts/play_balance.py --task=wheel_legged_vmc_balance
```

### 方式 3: 指定模型路径

```bash
python wheel_legged_gym/scripts/play_balance.py \
    --task=wheel_legged_vmc_balance \
    --load_run=logs/wheel_legged_vmc_balance/Feb23_15-34-41_
```

## 测试流程

### 1. 启动程序
```bash
./test_balance.sh
```

程序会自动加载最新的训练模型。

### 2. 观察初始状态
- 机器人处于"断电"状态（零动作）
- 统计信息显示 "STOPPED"

### 3. 启动控制
按 **C** 键启动策略控制：
- 机器人开始执行学习到的策略
- 观察机器人是否能保持平衡

### 4. 观察指标

**成功的标志**:
- ✅ 倾斜角度 < 10°
- ✅ 高度误差 < 0.05m
- ✅ Episode 长度接近 30s
- ✅ 状态显示 EXCELLENT 或 GOOD

**失败的标志**:
- ❌ 倾斜角度 > 20°
- ❌ 机器人快速倒地
- ❌ Episode 长度 < 5s
- ❌ 状态显示 FALLING

### 5. 多次测试
按 **R** 重置环境，重复测试多次：
- 观察成功率
- 记录最长平衡时间
- 检查平均 Episode 长度

## 输出示例

```
============================================================
  Balance Monitor - EXCELLENT
============================================================

Current State:
  Height:      0.248 m  (target: 0.250 m, error: 0.002 m)
  Roll:        +2.34 deg
  Pitch:       -1.87 deg
  Tilt Angle:   2.99 deg  OK

Velocities:
  Linear:  [-0.012, +0.008, -0.003] m/s
  Angular: [+0.045, -0.032, +0.001] rad/s

Episode Stats:
  Time:        15.3 s  (steps: 765)
  Max Length:  28.7 s  (1435 steps)

Overall Stats:
  Episodes:    5
  Success:     4 (80.0%)
  Avg Length:  22.4 s  (1120 steps)
  Total Steps: 5600

Controls:
  [C] Start  [R] Reset  [S] Toggle Stats  [P] Pause  [ESC] Exit
============================================================
```

## 评估标准

### 训练成功的标志

| 指标 | 目标值 | 说明 |
|------|--------|------|
| 成功率 | > 80% | 大部分 Episode 能完成 |
| 平均长度 | > 20s | 能稳定平衡较长时间 |
| 最长长度 | > 25s | 至少有一次接近最大长度 |
| 倾斜角度 | < 10° | 姿态控制良好 |
| 高度误差 | < 0.05m | 高度控制精确 |

### 训练失败的标志

| 现象 | 可能原因 | 解决方案 |
|------|----------|----------|
| 立即倒地 | 策略未学会平衡 | 检查奖励函数、继续训练 |
| 持续倾斜 | orientation 惩罚不足 | 增加 orientation 权重 |
| 高度不对 | base_height 奖励问题 | 检查目标高度设置 |
| 剧烈晃动 | PD 增益过高 | 降低 kp/kd 参数 |
| 成功率 < 50% | 训练不充分 | 继续训练更多 iterations |

## 调试技巧

### 1. 查看详细输出
修改 `print_stats` 中的更新频率：
```python
if not self.show_stats or self.total_steps % 10 != 0:  # 从 50 改为 10
```

### 2. 记录数据
在 `update()` 方法中添加日志：
```python
if self.total_steps % 100 == 0:
    with open('balance_log.txt', 'a') as f:
        f.write(f"{self.total_steps},{height},{tilt_angle}\n")
```

### 3. 可视化轨迹
使用 matplotlib 绘制历史数据：
```python
import matplotlib.pyplot as plt
plt.plot(monitor.height_history)
plt.show()
```

### 4. 慢动作播放
在主循环中添加延迟：
```python
time.sleep(0.05)  # 20 FPS
```

## 常见问题

### Q1: 提示 "Failed to load policy"
**A**: 确保已经训练过模型：
```bash
python wheel_legged_gym/scripts/train.py --task=wheel_legged_vmc_balance --num_envs=4096
```

### Q2: 机器人不动
**A**: 按 **C** 键启动控制，确保看到 "Control ENABLED" 消息。

### Q3: 统计信息不显示
**A**: 按 **S** 键切换显示，或检查终端是否支持 ANSI 清屏。

### Q4: 机器人立即倒地
**A**: 这说明策略还没学好，需要：
1. 检查训练日志中的 mean_reward 和 episode_length
2. 确认奖励函数修复已应用
3. 继续训练更多 iterations

### Q5: 如何录制视频
**A**: 使用屏幕录制工具，或修改代码保存帧：
```python
# 在 step 循环中
if started and self.total_steps % 10 == 0:
    env.gym.write_viewer_image_to_file(env.viewer, f"frame_{self.total_steps:06d}.png")
```

## 性能优化

### 提高帧率
```python
env_cfg.viewer.sync_frame_time = False  # 不同步帧时间
```

### 多环境测试
```python
env_cfg.env.num_envs = 4  # 同时测试 4 个机器人
```

### Headless 模式
```bash
python wheel_legged_gym/scripts/play_balance.py --task=wheel_legged_vmc_balance --headless
```

## 下一步

### 如果测试成功
1. 尝试添加扰动测试鲁棒性
2. 测试不同初始姿态
3. 导出策略用于实际机器人

### 如果测试失败
1. 查看 [REWARD_FIX_PLAN.md](REWARD_FIX_PLAN.md) 进一步调整
2. 检查训练日志确认收敛情况
3. 考虑降低任务难度（更宽松的终止条件）

## 相关文档

- [READY_TO_TRAIN.md](READY_TO_TRAIN.md) - 训练指南
- [REWARD_FIX_PLAN.md](REWARD_FIX_PLAN.md) - 奖励函数修复方案
- [CHANGES_APPLIED.md](CHANGES_APPLIED.md) - 已应用的修改

---

**更新时间**: 2026-02-23
**版本**: 2.0 (Enhanced)
