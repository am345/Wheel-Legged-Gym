"""
策略加载模块
加载训练好的PyTorch策略进行推理（当前支持 ActorCriticSequence）
"""

import os
import sys
from dataclasses import asdict, dataclass, replace
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np
import torch

# 添加项目路径
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from wheel_legged_gym.rsl_rl.modules.actor_critic_sequence import ActorCriticSequence


@dataclass(frozen=True)
class PolicySpec:
    """轻量策略结构定义（用于MuJoCo sim2sim推理侧）"""

    num_obs: int = 27
    num_actions: int = 6
    obs_history_length: int = 5
    latent_dim: int = 3
    encoder_hidden_dims: Tuple[int, ...] = (128, 64)
    actor_hidden_dims: Tuple[int, ...] = (128, 64, 32)
    critic_hidden_dims: Tuple[int, ...] = (256, 128, 64)
    activation: str = "elu"
    init_noise_std: float = 1.0
    num_critic_obs: Optional[int] = None

    @property
    def num_encoder_obs(self) -> int:
        return self.num_obs * self.obs_history_length

    def to_dict(self) -> Dict[str, object]:
        return asdict(self)


def get_default_policy_spec(task: str = "wheel_legged_vmc_balance") -> PolicySpec:
    """返回当前支持任务的默认策略结构"""
    if task != "wheel_legged_vmc_balance":
        raise ValueError(
            f"Unsupported task '{task}' for MuJoCo sim2sim PolicyLoader. "
            "Current implementation only supports 'wheel_legged_vmc_balance'."
        )
    return PolicySpec()


class PolicyLoader:
    """加载训练好的策略（checkpoint -> ActorCriticSequence inference）"""

    def __init__(
        self,
        checkpoint_path: str,
        device: str = "cpu",
        policy_spec: Optional[PolicySpec] = None,
        task: str = "wheel_legged_vmc_balance",
    ):
        """
        初始化策略加载器

        Args:
            checkpoint_path: checkpoint文件路径
            device: 运行设备 ('cpu' 或 'cuda')
            policy_spec: 策略结构定义，默认使用balance任务预设
            task: 任务名（用于选择默认spec）
        """
        self.device = torch.device(device)
        self.checkpoint_path = checkpoint_path
        self.task = task

        print(f"Loading checkpoint from: {checkpoint_path}")
        self.checkpoint = torch.load(checkpoint_path, map_location=self.device)
        if "model_state_dict" not in self.checkpoint:
            raise KeyError(
                "Checkpoint missing 'model_state_dict'. "
                "Expected training checkpoint saved by OnPolicyRunner.save()."
            )

        self.state_dict = self.checkpoint["model_state_dict"]
        base_spec = policy_spec if policy_spec is not None else get_default_policy_spec(task)
        self.spec = self._resolve_and_validate_policy_spec(base_spec, self.state_dict)

        # 创建ActorCriticSequence网络（当前sim2sim验证仅支持sequence策略）
        self.policy = ActorCriticSequence(
            num_obs=self.spec.num_obs,
            num_critic_obs=self.spec.num_critic_obs,
            num_actions=self.spec.num_actions,
            num_encoder_obs=self.spec.num_encoder_obs,
            latent_dim=self.spec.latent_dim,
            encoder_hidden_dims=list(self.spec.encoder_hidden_dims),
            actor_hidden_dims=list(self.spec.actor_hidden_dims),
            critic_hidden_dims=list(self.spec.critic_hidden_dims),
            activation=self.spec.activation,
            init_noise_std=self.spec.init_noise_std,
        ).to(self.device)

        # 加载权重
        self.policy.load_state_dict(self.state_dict, strict=True)
        self.policy.eval()

        self.checkpoint_iter = self.checkpoint.get("iter", "unknown")
        print(
            "Policy loaded successfully "
            f"(iteration {self.checkpoint_iter}, task={self.task}, device={self.device})"
        )

        # 观测历史缓冲
        self.obs_history = np.zeros((self.spec.num_encoder_obs,), dtype=np.float32)

    def _resolve_and_validate_policy_spec(
        self, policy_spec: PolicySpec, state_dict: Dict[str, torch.Tensor]
    ) -> PolicySpec:
        """根据checkpoint信息补全spec，并执行严格形状校验。"""
        critic_layers = self._extract_linear_weight_shapes(state_dict, "critic")
        if not critic_layers:
            raise ValueError(
                "Checkpoint does not look like ActorCriticSequence: no critic linear weights found."
            )
        inferred_num_critic_obs = critic_layers[0][1][1]  # (out, in)
        if policy_spec.num_critic_obs is None:
            resolved_spec = replace(policy_spec, num_critic_obs=inferred_num_critic_obs)
        else:
            resolved_spec = policy_spec
            if int(policy_spec.num_critic_obs) != int(inferred_num_critic_obs):
                raise ValueError(
                    "PolicySpec.num_critic_obs mismatch with checkpoint: "
                    f"spec={policy_spec.num_critic_obs}, checkpoint={inferred_num_critic_obs}"
                )

        self._validate_checkpoint_shapes(state_dict, resolved_spec)
        return resolved_spec

    @staticmethod
    def _extract_linear_weight_shapes(
        state_dict: Dict[str, torch.Tensor], prefix: str
    ) -> List[Tuple[int, Tuple[int, int]]]:
        """提取Sequential中线性层权重形状，返回[(module_idx, (out, in)), ...]。"""
        layers: List[Tuple[int, Tuple[int, int]]] = []
        for key, tensor in state_dict.items():
            if not key.startswith(prefix + ".") or not key.endswith(".weight"):
                continue
            parts = key.split(".")
            if len(parts) != 3:
                continue
            module_idx_str = parts[1]
            if not module_idx_str.isdigit():
                continue
            if tensor.ndim != 2:
                continue
            module_idx = int(module_idx_str)
            layers.append((module_idx, (int(tensor.shape[0]), int(tensor.shape[1]))))
        layers.sort(key=lambda x: x[0])
        return layers

    @staticmethod
    def _expected_linear_shapes(dims: Sequence[int]) -> List[Tuple[int, int]]:
        return [(int(dims[i + 1]), int(dims[i])) for i in range(len(dims) - 1)]

    def _validate_checkpoint_shapes(
        self, state_dict: Dict[str, torch.Tensor], spec: PolicySpec
    ) -> None:
        """校验checkpoint与PolicySpec关键结构是否一致。"""
        encoder_layers = self._extract_linear_weight_shapes(state_dict, "encoder")
        actor_layers = self._extract_linear_weight_shapes(state_dict, "actor")
        critic_layers = self._extract_linear_weight_shapes(state_dict, "critic")

        encoder_actual = [shape for _, shape in encoder_layers]
        actor_actual = [shape for _, shape in actor_layers]
        critic_actual = [shape for _, shape in critic_layers]

        encoder_expected = self._expected_linear_shapes(
            [spec.num_encoder_obs] + list(spec.encoder_hidden_dims) + [spec.latent_dim]
        )
        actor_expected = self._expected_linear_shapes(
            [spec.num_obs + spec.latent_dim] + list(spec.actor_hidden_dims) + [spec.num_actions]
        )
        critic_expected = self._expected_linear_shapes(
            [spec.num_critic_obs] + list(spec.critic_hidden_dims) + [1]
        )

        errors: List[str] = []
        if encoder_actual != encoder_expected:
            errors.append(
                "Encoder shape mismatch: "
                f"expected {encoder_expected}, got {encoder_actual}"
            )
        if actor_actual != actor_expected:
            errors.append(
                "Actor shape mismatch: "
                f"expected {actor_expected}, got {actor_actual}"
            )
        if critic_actual != critic_expected:
            errors.append(
                "Critic shape mismatch: "
                f"expected {critic_expected}, got {critic_actual}"
            )

        # 关键显式校验（便于用户快速定位）
        if encoder_actual:
            encoder_in_dim = encoder_actual[0][1]
            if encoder_in_dim != spec.num_encoder_obs:
                errors.append(
                    "Encoder input dim mismatch "
                    f"(num_obs * obs_history_length): expected {spec.num_encoder_obs}, got {encoder_in_dim}"
                )
        if actor_actual:
            actor_in_dim = actor_actual[0][1]
            actor_out_dim = actor_actual[-1][0]
            expected_actor_in = spec.num_obs + spec.latent_dim
            if actor_in_dim != expected_actor_in:
                errors.append(
                    "Actor first-layer input dim mismatch "
                    f"(num_obs + latent_dim): expected {expected_actor_in}, got {actor_in_dim}"
                )
            if actor_out_dim != spec.num_actions:
                errors.append(
                    "Actor output dim mismatch "
                    f"(num_actions): expected {spec.num_actions}, got {actor_out_dim}"
                )

        if errors:
            joined = "\n  - ".join(errors)
            raise ValueError(
                "Checkpoint is incompatible with the provided/default PolicySpec.\n"
                f"  - {joined}\n"
                "Tip: verify this checkpoint comes from wheel_legged_vmc_balance "
                "and uses ActorCriticSequence with 27-dim obs and 5-step history."
            )

    def get_action(self, obs: np.ndarray) -> np.ndarray:
        """
        获取确定性动作

        Args:
            obs: 当前观测 (num_obs,) numpy数组

        Returns:
            action: 动作 (num_actions,) numpy数组
        """
        obs = np.asarray(obs, dtype=np.float32)
        if obs.shape != (self.spec.num_obs,):
            raise ValueError(
                f"Observation shape mismatch: expected {(self.spec.num_obs,)}, got {obs.shape}"
            )

        # 更新观测历史（FIFO）
        self.obs_history = np.concatenate([self.obs_history[self.spec.num_obs :], obs])

        with torch.no_grad():
            obs_tensor = torch.from_numpy(obs).float().unsqueeze(0).to(self.device)
            obs_history_tensor = (
                torch.from_numpy(self.obs_history).float().unsqueeze(0).to(self.device)
            )
            action_tensor, _latent = self.policy.act_inference(obs_tensor, obs_history_tensor)
            action = action_tensor.cpu().numpy().squeeze()

        action = np.asarray(action, dtype=np.float32)
        if action.shape != (self.spec.num_actions,):
            raise RuntimeError(
                f"Policy output shape mismatch: expected {(self.spec.num_actions,)}, got {action.shape}"
            )
        return action

    def reset(self) -> None:
        """重置观测历史"""
        self.obs_history = np.zeros((self.spec.num_encoder_obs,), dtype=np.float32)
