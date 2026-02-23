# Pitch/Roll 分离修改总结

## ✅ 修改完成

已成功将 `orientation` 奖励拆分为独立的 `orientation_pitch` 和 `orientation_roll`。

## 📝 修改内容

### 1. 新增奖励函数

**文件**: `wheel_legged_gym/envs/wheel_legged_vmc_balance/wheel_legged_vmc_balance.py`

```python
def _reward_orientation_pitch(self):
    """Penalize pitch (forward/backward tilt)"""
    return torch.square(self.projected_gravity[:, 1])

def _reward_orientation_roll(self):
    """Penalize roll (left/right tilt)"""
    return torch.square(self.projected_gravity[:, 0])
```

### 2. 配置修改

**文件**: `wheel_legged_gym/envs/wheel_legged_vmc_balance/wheel_legged_vmc_balance_config.py`

```python
# 修改前
orientation = -80.0  # 同时惩罚 pitch 和 roll

# 修改后
orientation = 0.0              # 禁用
orientation_pitch = -80.0      # 独立的 pitch 惩罚
orientation_roll = -80.0       # 独立的 roll 惩罚
```

## ✅ 验证结果

```
✅ Config loaded successfully
orientation: 0.0
orientation_pitch: -80.0
orientation_roll: -80.0
base_height: 40.0
upright_bonus: 20.0
```

## 🎯 优势

### 1. 独立调整
```python
# 如果前后倾斜更危险
orientation_pitch = -100.0
orientation_roll = -60.0

# 如果左右倾斜更危险
orientation_pitch = -60.0
orientation_roll = -100.0
```

### 2. 分别监控
在 TensorBoard 中可以看到：
- `Train/rew_orientation_pitch` - 前后倾斜惩罚
- `Train/rew_orientation_roll` - 左右倾斜惩罚

### 3. 更容易调试
可以快速识别是哪个方向的平衡有问题。

## 📊 对比

| 特性 | 修改前 | 修改后 |
|------|--------|--------|
| 奖励函数 | 1 个 (orientation) | 2 个 (pitch + roll) |
| 权重调整 | 统一 | 独立 |
| 监控指标 | 1 个 | 2 个 |
| 调试难度 | 困难 | 容易 |
| 灵活性 | 低 | 高 |

## 🔧 使用示例

### 标准配置（默认）
```python
orientation_pitch = -80.0
orientation_roll = -80.0
```
两个方向同等重要。

### 强调前后平衡
```python
orientation_pitch = -120.0  # 更严格
orientation_roll = -60.0    # 较宽松
```

### 强调左右平衡
```python
orientation_pitch = -60.0   # 较宽松
orientation_roll = -120.0   # 更严格
```

### 渐进式训练
```python
# 阶段 1: 先学左右平衡
orientation_pitch = -40.0
orientation_roll = -120.0

# 阶段 2: 再学前后平衡
orientation_pitch = -120.0
orientation_roll = -120.0
```

## 📈 监控指标

### 理想值
| 倾斜角度 | Pitch 惩罚 | Roll 惩罚 |
|----------|------------|-----------|
| 0° | 0.0 | 0.0 |
| 5° | -0.6 | -0.6 |
| 10° | -2.4 | -2.4 |
| 20° | -9.5 | -9.5 |

### 诊断
- **Pitch 惩罚很负**: 前后倾斜严重，调整轮速控制
- **Roll 惩罚很负**: 左右倾斜严重，调整腿部角度

## 🚀 下一步

### 1. 重新训练
```bash
python wheel_legged_gym/scripts/train.py --task=wheel_legged_vmc_balance --num_envs=4096
```

### 2. 监控新指标
打开 TensorBoard 查看：
```bash
tensorboard --logdir=logs/wheel_legged_vmc_balance
```

关注：
- `Train/rew_orientation_pitch`
- `Train/rew_orientation_roll`

### 3. 根据结果调整
如果某个方向倾斜更严重，增加对应权重。

## 📚 相关文档

- [PITCH_ROLL_SEPARATE.md](PITCH_ROLL_SEPARATE.md) - 详细说明
- [READY_TO_TRAIN.md](READY_TO_TRAIN.md) - 训练指南
- [PLAY_BALANCE_GUIDE.md](PLAY_BALANCE_GUIDE.md) - 测试指南

---

**修改时间**: 2026-02-23
**状态**: ✅ 已完成并验证
**优势**: 更精细的姿态控制
