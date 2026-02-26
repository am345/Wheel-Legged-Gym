# MuJoCo Sim2Sim 验证

这个模块用于在 MuJoCo 中验证 `wheel_legged_vmc_balance` 强化学习策略，并支持两条链路：

1. 真实 sim2sim 长时评估：`vmc_balance_exact + serialleg_fidelity.xml + domain randomization`
2. IsaacGym vs MuJoCo 工程对齐：Isaac 参考轨迹采集 + MuJoCo 重放对齐 RMSE

## 当前能力（已实现）

- `MuJoCoBalanceEnv` 支持：
  - `simplified_joint_pd`（旧基线）
  - `vmc_balance_exact`（训练一致 VMC 控制链路，保留 balance pitch 提示扭矩）
- `serialleg_fidelity.xml` 高保真 MJCF（由 URDF 生成，轮子碰撞使用解析圆柱）
- MuJoCo 侧 episode 级 domain randomization（对齐训练范围）
- 主验证脚本支持 checkpoint sweep、随机化评估、结构化 JSON 输出
- Isaac 参考轨迹采集与 MuJoCo 对齐回放脚本

## 文件结构

```text
mujoco_sim/
├── mujoco_balance_env.py         # MuJoCo环境（baseline + VMC exact）
├── observation_computer.py       # 27维观测计算
├── vmc_kinematics.py             # VMC运动学/Jacobian/力矩映射
├── policy_loader.py              # 直接加载 checkpoint(.pt)
├── control_config.py             # MuJoCo侧控制/评估参数镜像
├── domain_randomizer.py          # MuJoCo侧域随机化
├── alignment_utils.py            # Isaac参考轨迹回放与对齐指标
└── README.md

resources/robots/serialleg/mjcf/
├── serialleg_simple.xml          # 简化模型（旧基线）
├── serialleg_fidelity.xml        # 高保真模型（推荐）
└── README_fidelity.md            # fidelity模型说明

tools/
└── build_serialleg_fidelity_mjcf.py   # URDF -> fidelity MJCF 生成工具

wheel_legged_gym/scripts/
├── verify_sim2sim_mujoco.py           # 主验证脚本（长时评估 + 可选对齐）
├── collect_isaac_reference_rollout.py # Isaac参考轨迹采集
├── replay_mujoco_reference_rollout.py # MuJoCo重放对齐
└── compare_sim2sim_alignment.py       # 对齐报告汇总（可选）
```

## 环境准备

```bash
source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate gym_env
pip install mujoco scipy
```

## 1. 真实 Sim2Sim 长时评估（推荐）

### 烟雾测试（真实模式）

```bash
python wheel_legged_gym/scripts/verify_sim2sim_mujoco.py \
  --task wheel_legged_vmc_balance \
  --checkpoint logs/wheel_legged_vmc_balance/Feb23_15-01-26_/model_900.pt \
  --model resources/robots/serialleg/mjcf/serialleg_fidelity.xml \
  --controller-mode vmc_balance_exact \
  --domain-rand-mode train_ranges \
  --episodes 1 \
  --max_steps 200 \
  --device cpu \
  --seed 0 \
  --output mujoco_sim2sim_smoke_real.json
```

### 主量化评估（随机化鲁棒性）

```bash
python wheel_legged_gym/scripts/verify_sim2sim_mujoco.py \
  --task wheel_legged_vmc_balance \
  --checkpoint logs/wheel_legged_vmc_balance/Feb23_15-01-26_/model_900.pt \
  --model resources/robots/serialleg/mjcf/serialleg_fidelity.xml \
  --controller-mode vmc_balance_exact \
  --domain-rand-mode train_ranges \
  --episodes 10 \
  --max_steps 3000 \
  --device cpu \
  --seed 0 \
  --output mujoco_sim2sim_results_real.json
```

### Checkpoint Sweep（横向比较）

`checkpoint_list.txt` 每行一个 checkpoint：

```text
logs/wheel_legged_vmc_balance/.../model_300.pt
logs/wheel_legged_vmc_balance/.../model_600.pt
logs/wheel_legged_vmc_balance/.../model_900.pt
```

运行：

```bash
python wheel_legged_gym/scripts/verify_sim2sim_mujoco.py \
  --task wheel_legged_vmc_balance \
  --checkpoint-list checkpoint_list.txt \
  --model resources/robots/serialleg/mjcf/serialleg_fidelity.xml \
  --controller-mode vmc_balance_exact \
  --domain-rand-mode train_ranges \
  --episodes 10 \
  --max_steps 3000 \
  --seed 0 \
  --output mujoco_sim2sim_sweep.json
```

## 2. IsaacGym vs MuJoCo 对齐（短序列）

### 采集 Isaac 参考轨迹

```bash
python wheel_legged_gym/scripts/collect_isaac_reference_rollout.py \
  --task wheel_legged_vmc_balance \
  --checkpoint logs/wheel_legged_vmc_balance/Feb23_15-01-26_/model_900.pt \
  --steps 500 \
  --seed 0 \
  --scenario deterministic_nominal \
  --domain-rand-mode off \
  --output isaac_reference_rollout.npz
```

### 在 MuJoCo 中重放参考轨迹并输出对齐指标

```bash
python wheel_legged_gym/scripts/replay_mujoco_reference_rollout.py \
  --reference-rollout isaac_reference_rollout.npz \
  --model resources/robots/serialleg/mjcf/serialleg_fidelity.xml \
  --controller-mode vmc_balance_exact \
  --domain-rand-mode off \
  --seed 0 \
  --output mujoco_alignment_report.json
```

### 在主评估脚本中附带对齐结果（可选）

```bash
python wheel_legged_gym/scripts/verify_sim2sim_mujoco.py \
  --task wheel_legged_vmc_balance \
  --checkpoint logs/wheel_legged_vmc_balance/Feb23_15-01-26_/model_900.pt \
  --model resources/robots/serialleg/mjcf/serialleg_fidelity.xml \
  --controller-mode vmc_balance_exact \
  --domain-rand-mode off \
  --episodes 1 \
  --max_steps 500 \
  --seed 0 \
  --compare-isaac \
  --reference-rollout isaac_reference_rollout.npz \
  --output mujoco_eval_with_alignment.json
```

## 常用参数说明（主脚本）

- `--controller-mode {simplified_joint_pd,vmc_balance_exact}`：控制模式，真实评估建议 `vmc_balance_exact`
- `--domain-rand-mode {off,train_ranges}`：MuJoCo 侧域随机化，真实评估建议 `train_ranges`
- `--randomize-reset/--no-randomize-reset`：balance 初始化随机化开关
- `--eval-fall-tilt-deg`：脚本层跌倒判定阈值（度），用于统计 `fall_rate`
- `--checkpoint-list`：批量评估多个 checkpoint
- `--compare-isaac` + `--reference-rollout`：追加对齐回放结果到输出 JSON
- `--record-episode-params/--no-record-episode-params`：是否记录每个 episode 的随机化样本

## 输出结果（JSON）

主脚本默认输出：

- `metadata`：模型/控制器/随机化/seed/config snapshot/preflight 等元信息
- `aggregate`：汇总指标
  - `mean_steps`
  - `survival_time_seconds`
  - `mean_upright_ratio`
  - `mean_tilt`
  - `mean_max_tilt`
  - `fall_rate`
  - `mean_torque_saturation_rate`
  - `mean_action_clip_rate`
  - `mean_prompt_torque_trigger_ratio`
  - `failure_modes`
  - `episode_params_summary`
- `episodes`：每回合详细指标与随机化参数样本（可选）
- `alignment`：仅在 `--compare-isaac` 时存在

## 注意事项（结果解读）

1. 即使使用 `vmc_balance_exact + serialleg_fidelity.xml`，MuJoCo 与 IsaacGym/PhysX 的接触模型仍然不同。
2. 因此建议同时看：
   - 工程对齐指标（短序列 RMSE）
   - 长时 rollout 鲁棒性指标（随机化分布下）
3. `serialleg_fidelity.xml` 对轮子碰撞使用了解析圆柱（有意设计，用于稳定滚动接触）。
4. `collect_isaac_reference_rollout.py` 中必须先导入 `isaacgym` 再导入 `torch`（脚本已处理）。
