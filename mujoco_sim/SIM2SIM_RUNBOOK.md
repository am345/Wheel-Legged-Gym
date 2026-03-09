# MuJoCo Sim2Sim 运行手册（`wheel_legged_vmc_balance`）

这份文档是给当前仓库的实际运行流程准备的“照着跑”手册，覆盖：

1. `MJCF` 模型检查
2. `vmc_balance_exact` 控制链路 sanity（不上网络）
3. 上最新训练 checkpoint 做真实 `sim2sim`（短步预检查 + 可视化 + 长时评估）
4. 结果查看与常见问题排查

适用任务：`wheel_legged_vmc_balance`

## 0. 当前默认对象（本地已验证）

截至本地目录检查（本次会话）：

- 最新 run：`logs/wheel_legged_vmc_balance/Feb26_13-20-50_`
- 最高编号 checkpoint：`logs/wheel_legged_vmc_balance/Feb26_13-20-50_/model_950.pt`

本手册默认使用：

- `checkpoint = logs/wheel_legged_vmc_balance/Feb26_13-20-50_/model_950.pt`
- `model = resources/robots/serialleg/mjcf/serialleg_fidelity.xml`
- `controller_mode = vmc_balance_exact`

## 1. 环境准备

在仓库根目录执行：

```bash
source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate gym_env
cd /home/am345/Wheel-Legged-Gym
```

如果你不确定依赖是否齐全，先检查版本：

```bash
python - <<'PY'
import torch, mujoco, scipy
print("torch =", torch.__version__)
print("mujoco =", mujoco.__version__)
print("scipy =", scipy.__version__)
PY
```

## 2. （可选）重新生成并验证 fidelity MJCF

如果你刚修改过 `URDF` 或 mesh，先重新生成：

```bash
python tools/build_serialleg_fidelity_mjcf.py --validate-load
```

说明：

- 默认会使用未简化 `STL` 作为碰撞体（必要时自动无损分块）
- 输出模型：`resources/robots/serialleg/mjcf/serialleg_fidelity.xml`

## 3. 先在 MuJoCo 中单独打开机器人模型（不加载网络）

静态查看（建议先用这个）：

```bash
python tools/view_mjcf_model.py \
  --model resources/robots/serialleg/mjcf/serialleg_fidelity.xml \
  --zero-joints
```

带重力模拟查看（纯物理，不上网络）：

```bash
python tools/view_mjcf_model.py \
  --model resources/robots/serialleg/mjcf/serialleg_fidelity.xml \
  --simulate \
  --gravity on \
  --real-time
```

## 4. `vmc_balance_exact` 控制链路 Sanity（不上网络）

这是上 checkpoint 前的最后一步，确认控制/观测/状态链路正常。

### 4.1 无渲染数值检查（推荐先跑）

```bash
python tools/check_mujoco_vmc_sanity.py \
  --gravity off \
  --steps 20 \
  --print-every 10 \
  --json-output /tmp/mujoco_vmc_sanity.json
```

通过标准（脚本会打印）：

- `obs.shape=(27,)`
- `wheel_collision: mode=mesh`
- `wheel_mesh_src: visual_stl`（或 `visual_stl_chunked`）
- `Sanity check PASSED.`

### 4.2 可视化 sanity（重力关闭，优先看数值是否炸）

```bash
python tools/check_mujoco_vmc_sanity.py \
  --render \
  --gravity off \
  --steps 200 \
  --print-every 20
```

### 4.3 可视化 sanity（重力开启，观察物理行为）

```bash
python tools/check_mujoco_vmc_sanity.py \
  --render \
  --gravity on \
  --steps 200 \
  --print-every 20
```

说明：

- 这一步不追求站稳（没有策略）
- 允许跌倒，但不允许出现 `NaN/Inf`、瞬间爆炸、拉丝、散架

## 4.5 像 `play_balance` 一样手动控制机器人（MuJoCo，不卡策略）

如果你想像 `play_balance.py` 一样在 viewer 里手动控制（键盘交互），用这个脚本：

- `tools/play_mujoco_manual_balance.py`

### 启动（推荐默认）

```bash
python tools/play_mujoco_manual_balance.py \
  --model resources/robots/serialleg/mjcf/serialleg_fidelity.xml \
  --controller-mode vmc_balance_exact \
  --domain-rand-mode off \
  --gravity on
```

### 常用键位（脚本启动后也会打印）

- `C`：启用/关闭手动控制（类似 `play_balance` 的 Start）
- `R`：重置环境（并关闭控制）
- `P`：暂停/继续
- `M`：开关终端状态打印
- `G`：切换重力开/关
- `Z`：动作清零
- `N`：立即打印一份详细 debug 快照
- `H`：打印帮助
- `ESC`：退出

### 手动动作映射（`vmc_balance_exact` 动作空间）

- `J/L`：同时减/增左右腿 `theta0` 动作分量
- `I/K`：同时增/减左右腿 `l0` 动作分量
- `W/S`：同时增/减左右轮速度动作分量
- `A/D`：左右轮差速（转向）
- `Q/E`：左右腿 `theta0` 差分微调

说明：

- 这是直接向 MuJoCo 环境输入 `6` 维动作，不加载策略网络
- 适合做控制方向、符号、接触行为、VMC 响应的交互调试

## 4.6 像 `play_balance` 一样按 `C` 启动策略恢复姿态（MuJoCo，加载网络）

如果你的目标是：

- 先随机 reset
- 然后按 `C` 才启动策略
- 看网络在 MuJoCo 中恢复姿态

用这个脚本（MuJoCo 版 `play_balance` 风格）：

- `tools/play_mujoco_policy_balance.py`

### 启动（默认自动发现最新 checkpoint）

```bash
python tools/play_mujoco_policy_balance.py \
  --model resources/robots/serialleg/mjcf/serialleg_fidelity.xml \
  --controller-mode vmc_balance_exact \
  --domain-rand-mode off \
<<<<<<< HEAD
  --wait-mode zero_action \
  --reset-profile nominal_demo \
=======
>>>>>>> 310d9402ea53126106695598c1daedb2f6e66e6e
  --gravity on
```

说明：

- 默认会自动扫描 `logs/wheel_legged_vmc_balance/`，选择“最新 run + 最高编号 `model_*.pt`”
- 如果你希望手动指定 checkpoint，可以加 `--checkpoint <path>`
<<<<<<< HEAD
- `--wait-mode zero_action` 更接近 Isaac `play_balance` 体感（未按 `C` 时用零 action 继续走控制链路）
- `--reset-profile nominal_demo` 用于演示恢复姿态；`random_balance` 更偏压力测试
=======
>>>>>>> 310d9402ea53126106695598c1daedb2f6e66e6e

### 核心键位（对齐 `play_balance` 使用习惯）

- `C`：启动策略控制（START）
- `R`：重置环境（默认随机 reset），并关闭策略控制
- `P`：暂停/继续
- `S`：开关终端状态打印
- `ESC`：退出

### 常用扩展键（调试用）

- `N`：打印详细 debug 快照（`L0/theta0/ctrl/projected_gravity/control_debug`）
- `G`：重力开/关切换
- `H`：打印帮助

### 默认行为（重要）

- 启动时策略已加载，但默认 **不启用**
- 每次按 `C` 启动策略时，会重置策略历史（避免 sequence policy 历史污染）
- 每次按 `R` 会 reset 环境并停控，等待你再次按 `C`
- 默认 `R` 使用：
  - `randomize_reset=True`
  - `domain_randomize_reset=False`

### 可选参数（常用）

固定 reset（关闭随机 reset）：

```bash
python tools/play_mujoco_policy_balance.py --fixed-reset
```

显示更少 MuJoCo UI（只保留画面）：

```bash
python tools/play_mujoco_policy_balance.py --hide-left-ui --hide-right-ui
```

显式指定 checkpoint：

```bash
python tools/play_mujoco_policy_balance.py \
<<<<<<< HEAD
  --checkpoint logs/wheel_legged_vmc_balance/Feb26_13-20-50_/model_2000.pt
```

演示优化（MuJoCo-only 轻微调参，不改训练端）：

```bash
python tools/play_mujoco_policy_balance.py \
  --mujoco-tuning-profile demo_tuned \
  --wait-mode zero_action \
  --reset-profile nominal_demo
```

排查“按 C 一瞬间就没”（禁用脚本层跌倒停控）：

```bash
python tools/play_mujoco_policy_balance.py \
  --wait-mode zero_action \
  --reset-profile nominal_demo \
  --no-script-fall-stop
=======
  --checkpoint logs/wheel_legged_vmc_balance/Feb26_13-20-50_/model_950.pt
>>>>>>> 310d9402ea53126106695598c1daedb2f6e66e6e
```

## 5. 上最新 checkpoint 做真实 Sim2Sim（推荐流程）

本节就是“正式开始上网络”。

### 5.1 短步预检查（无渲染，先确认 preflight 和策略加载）

```bash
python wheel_legged_gym/scripts/verify_sim2sim_mujoco.py \
  --task wheel_legged_vmc_balance \
  --checkpoint logs/wheel_legged_vmc_balance/Feb26_13-20-50_/model_950.pt \
  --model resources/robots/serialleg/mjcf/serialleg_fidelity.xml \
  --controller-mode vmc_balance_exact \
  --domain-rand-mode off \
  --episodes 1 \
  --max_steps 200 \
  --device cpu \
  --seed 0 \
  --no-randomize-reset \
  --output mujoco_sim2sim_ckpt950_smoke.json
```

你应该看到：

- `Preflight 检查通过`
- `wheel_collision: mode=mesh`
- `wheel_mesh_src: visual_stl`（或 `visual_stl_chunked`）
- 输出文件：`mujoco_sim2sim_ckpt950_smoke.json`

### 5.2 可视化短步（肉眼确认策略行为）

```bash
python wheel_legged_gym/scripts/verify_sim2sim_mujoco.py \
  --task wheel_legged_vmc_balance \
  --checkpoint logs/wheel_legged_vmc_balance/Feb26_13-20-50_/model_950.pt \
  --model resources/robots/serialleg/mjcf/serialleg_fidelity.xml \
  --controller-mode vmc_balance_exact \
  --domain-rand-mode off \
  --episodes 1 \
  --max_steps 500 \
  --device cpu \
  --seed 0 \
  --no-randomize-reset \
  --render \
  --output mujoco_sim2sim_ckpt950_render_probe.json
```

观察点：

- 模型不拉丝、不爆炸
- 动作不是完全静止，也不是高频乱抖
- 跌倒（如果发生）应是物理合理跌倒

### 5.3 长时主评估（随机化鲁棒性）

```bash
python wheel_legged_gym/scripts/verify_sim2sim_mujoco.py \
  --task wheel_legged_vmc_balance \
  --checkpoint logs/wheel_legged_vmc_balance/Feb26_13-20-50_/model_950.pt \
  --model resources/robots/serialleg/mjcf/serialleg_fidelity.xml \
  --controller-mode vmc_balance_exact \
  --domain-rand-mode train_ranges \
  --episodes 10 \
  --max_steps 3000 \
  --device cpu \
  --seed 0 \
  --randomize-reset \
  --output mujoco_sim2sim_ckpt950_real_eval.json \
  --label latest_Feb26_13-20-50_model_950
```

输出文件：

- `mujoco_sim2sim_ckpt950_real_eval.json`

## 6. 查看结果（重点看 `aggregate`）

快速打印关键元信息与汇总指标：

```bash
python - <<'PY'
import json
d = json.load(open("mujoco_sim2sim_ckpt950_real_eval.json", "r"))
print("checkpoint_path =", d["metadata"]["checkpoint_path"])
print("controller_mode =", d["metadata"]["controller_mode"])
print("fidelity_level =", d["metadata"]["fidelity_level"])
print("domain_rand_mode =", d["metadata"]["domain_rand_mode"])
print("collision_representation =", d["metadata"].get("collision_representation"))
print("aggregate =", d["aggregate"])
PY
```

重点字段解释（`aggregate`）：

- `mean_steps`: 平均存活步数
- `survival_time_seconds`: 平均存活时间（秒）
- `mean_upright_ratio`: 直立占比（越高越好）
- `mean_tilt`: 平均倾角（越低越好）
- `mean_max_tilt`: 最大倾角平均值（越低越好）
- `fall_rate`: 跌倒率（越低越好）
- `failure_modes`: 终止原因统计（比如 `fall_tilt`, `timeout`）

## 7. （可选）长时名义评估基线（不加随机化）

为了和 `train_ranges` 对比，建议再跑一轮 `domain_rand_mode=off` 的长时评估：

```bash
python wheel_legged_gym/scripts/verify_sim2sim_mujoco.py \
  --task wheel_legged_vmc_balance \
  --checkpoint logs/wheel_legged_vmc_balance/Feb26_13-20-50_/model_950.pt \
  --model resources/robots/serialleg/mjcf/serialleg_fidelity.xml \
  --controller-mode vmc_balance_exact \
  --domain-rand-mode off \
  --episodes 10 \
  --max_steps 3000 \
  --device cpu \
  --seed 0 \
  --randomize-reset \
  --output mujoco_sim2sim_ckpt950_nominal_eval.json \
  --label latest_Feb26_13-20-50_model_950_nominal
```

这样你可以直接比较：

- `off`（名义）
- `train_ranges`（随机化鲁棒性）

## 8. （可选）对比同一 run 的多个 checkpoint

如果 `model_950.pt` 表现不理想，优先对比同一 run 中的 `model_900.pt` / `model_850.pt`：

```bash
cat > checkpoint_list.txt <<'EOF'
logs/wheel_legged_vmc_balance/Feb26_13-20-50_/model_850.pt
logs/wheel_legged_vmc_balance/Feb26_13-20-50_/model_900.pt
logs/wheel_legged_vmc_balance/Feb26_13-20-50_/model_950.pt
EOF

python wheel_legged_gym/scripts/verify_sim2sim_mujoco.py \
  --task wheel_legged_vmc_balance \
  --checkpoint-list checkpoint_list.txt \
  --model resources/robots/serialleg/mjcf/serialleg_fidelity.xml \
  --controller-mode vmc_balance_exact \
  --domain-rand-mode train_ranges \
  --episodes 10 \
  --max_steps 3000 \
  --seed 0 \
  --output mujoco_sim2sim_ckpt_sweep_feb26.json
```

## 9. 常见问题排查

### 9.1 `ModuleNotFoundError: No module named 'torch'`

说明你没有进入训练环境，重新执行：

```bash
source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate gym_env
```

### 9.2 打开 viewer 后模型拉丝/几何异常

优先检查：

1. 你是否使用了最新生成的 `serialleg_fidelity.xml`
2. 是否重新生成过 `base_link` 的分块 STL（`_mjc_generated_collision`）

重新生成并验证：

```bash
python tools/build_serialleg_fidelity_mjcf.py --validate-load
```

### 9.3 `Preflight checks failed`

常见方向：

- `checkpoint` 结构与当前 `PolicyLoader` 不兼容
- `MJCF` 关节/执行器顺序被改坏
- `obs/action/ctrl` 出现 NaN/Inf

建议先回退跑：

```bash
python tools/check_mujoco_vmc_sanity.py --steps 20 --gravity off
```

如果 sanity 通过，问题更可能在 checkpoint/策略侧。

### 9.4 `wheel_collision_mode_detected != mesh`

说明你当前 `serialleg_fidelity.xml` 不是预期版本（或被回退成简化碰撞表示）。

重新生成（默认即 `visual_stl + wheel mesh`）：

```bash
python tools/build_serialleg_fidelity_mjcf.py --validate-load
```

## 10. 结果解读注意事项（很重要）

1. 即使使用 `vmc_balance_exact + serialleg_fidelity.xml`，MuJoCo 与 IsaacGym/PhysX 的接触模型仍不同。
2. 因此推荐同时看：
   - 短步/可视化行为是否合理
   - 长时随机化统计（`aggregate`）
   - （后续）Isaac 对齐短序列 RMSE
3. 本轮如果 `domain_rand=train_ranges` 下跌倒率高，不代表链路错，可能是策略对随机化不够鲁棒或 MuJoCo 接触差异放大。

## 11. 本次会话已跑通的参考结果（供对照）

以 `model_950.pt` 为例（本地本次运行）：

- `smoke (1x200, off)`: `upright_ratio ≈ 97.5%`
- `render probe (1x500, off)`: `upright_ratio ≈ 99.0%`
- `real eval (10x3000, train_ranges)`:
  - `mean_steps ≈ 600.8`
  - `fall_rate = 80%`
  - `failure_modes = {'fall_tilt': 8, 'timeout': 2}`

这组结果说明：

- 链路是通的（preflight/短步/长时都能完成）
- 随机化下鲁棒性还有明显提升空间（建议做 checkpoint 对比 + 名义评估基线）
