"""Policy loader for MuJoCo sim2sim inference (ActorCriticSequence checkpoints)."""

import os
import sys
from dataclasses import asdict, dataclass, replace
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np
import torch

# Add project root for local imports.
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from wheel_legged_gym.rsl_rl.modules.actor_critic_sequence import ActorCriticSequence


SUPPORTED_TASKS = ("wheel_legged_vmc_balance", "wheel_legged_fzqver")


@dataclass(frozen=True)
class PolicySpec:
    """Compact policy architecture spec used for MuJoCo-side inference."""

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
    if task == "wheel_legged_vmc_balance":
        return PolicySpec()
    if task == "wheel_legged_fzqver":
        return PolicySpec(
            actor_hidden_dims=(512, 256, 128),
            critic_hidden_dims=(512, 256, 128),
        )
    raise ValueError(f"Unsupported task '{task}'. Supported tasks: {SUPPORTED_TASKS}")


class PolicyLoader:
    """Load a training checkpoint and expose deterministic numpy action inference."""

    def __init__(
        self,
        checkpoint_path: str,
        device: str = "cpu",
        policy_spec: Optional[PolicySpec] = None,
        task: str = "wheel_legged_vmc_balance",
    ):
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

        if policy_spec is None:
            inferred = self._infer_policy_spec_from_state_dict(self.state_dict)
            base_spec = inferred
            print("Using inferred PolicySpec from checkpoint:", base_spec.to_dict())
        else:
            base_spec = policy_spec

        # Fill critic input dim if needed and validate all key shapes strictly.
        self.spec = self._resolve_and_validate_policy_spec(base_spec, self.state_dict)

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

        self.policy.load_state_dict(self.state_dict, strict=True)
        self.policy.eval()

        self.checkpoint_iter = self.checkpoint.get("iter", "unknown")
        print(
            "Policy loaded successfully "
            f"(iteration {self.checkpoint_iter}, task={self.task}, device={self.device})"
        )

        self.obs_history = np.zeros((self.spec.num_encoder_obs,), dtype=np.float32)

    @staticmethod
    def _extract_linear_weight_shapes(
        state_dict: Dict[str, torch.Tensor], prefix: str
    ) -> List[Tuple[int, Tuple[int, int]]]:
        """Extract linear-layer weight shapes from a Sequential-like module."""
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

    @staticmethod
    def _validate_linear_chain(prefix: str, layers: List[Tuple[int, Tuple[int, int]]]) -> None:
        if not layers:
            raise ValueError(f"Checkpoint has no linear layers for '{prefix}'.")
        shapes = [shape for _, shape in layers]
        for i in range(len(shapes) - 1):
            out_i, _in_i = shapes[i]
            _out_n, in_n = shapes[i + 1]
            if out_i != in_n:
                raise ValueError(
                    f"Non-chain linear shapes under {prefix}: {shapes}. "
                    f"Layer {i} out={out_i} but layer {i+1} in={in_n}."
                )

    def _infer_policy_spec_from_state_dict(self, state_dict: Dict[str, torch.Tensor]) -> PolicySpec:
        """Infer PolicySpec from checkpoint weight tensors."""
        encoder_layers = self._extract_linear_weight_shapes(state_dict, "encoder")
        actor_layers = self._extract_linear_weight_shapes(state_dict, "actor")
        critic_layers = self._extract_linear_weight_shapes(state_dict, "critic")

        self._validate_linear_chain("encoder", encoder_layers)
        self._validate_linear_chain("actor", actor_layers)
        self._validate_linear_chain("critic", critic_layers)

        encoder_shapes = [shape for _, shape in encoder_layers]
        actor_shapes = [shape for _, shape in actor_layers]
        critic_shapes = [shape for _, shape in critic_layers]

        num_encoder_obs = int(encoder_shapes[0][1])
        latent_dim = int(encoder_shapes[-1][0])

        actor_in = int(actor_shapes[0][1])
        num_actions = int(actor_shapes[-1][0])
        num_obs = actor_in - latent_dim
        if num_obs <= 0:
            raise ValueError(
                "Failed to infer num_obs: actor first-layer input minus latent_dim is non-positive. "
                f"actor_in={actor_in}, latent_dim={latent_dim}"
            )

        if num_encoder_obs % num_obs != 0:
            raise ValueError(
                "Failed to infer obs_history_length: encoder input is not divisible by inferred num_obs. "
                f"num_encoder_obs={num_encoder_obs}, num_obs={num_obs}"
            )
        obs_history_length = int(num_encoder_obs // num_obs)

        num_critic_obs = int(critic_shapes[0][1])
        encoder_hidden_dims = tuple(int(out) for out, _in in encoder_shapes[:-1])
        actor_hidden_dims = tuple(int(out) for out, _in in actor_shapes[:-1])
        critic_hidden_dims = tuple(int(out) for out, _in in critic_shapes[:-1])

        default_for_task = get_default_policy_spec(self.task) if self.task in SUPPORTED_TASKS else PolicySpec()

        return PolicySpec(
            num_obs=num_obs,
            num_actions=num_actions,
            obs_history_length=obs_history_length,
            latent_dim=latent_dim,
            encoder_hidden_dims=encoder_hidden_dims,
            actor_hidden_dims=actor_hidden_dims,
            critic_hidden_dims=critic_hidden_dims,
            activation=default_for_task.activation,
            init_noise_std=default_for_task.init_noise_std,
            num_critic_obs=num_critic_obs,
        )

    def _resolve_and_validate_policy_spec(
        self, policy_spec: PolicySpec, state_dict: Dict[str, torch.Tensor]
    ) -> PolicySpec:
        critic_layers = self._extract_linear_weight_shapes(state_dict, "critic")
        if not critic_layers:
            raise ValueError(
                "Checkpoint does not look like ActorCriticSequence: no critic linear weights found."
            )
        inferred_num_critic_obs = critic_layers[0][1][1]
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

    def _validate_checkpoint_shapes(
        self, state_dict: Dict[str, torch.Tensor], spec: PolicySpec
    ) -> None:
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

        if encoder_actual:
            encoder_in_dim = encoder_actual[0][1]
            if encoder_in_dim != spec.num_encoder_obs:
                errors.append(
                    "Encoder input dim mismatch (num_obs * obs_history_length): "
                    f"expected {spec.num_encoder_obs}, got {encoder_in_dim}"
                )
        if actor_actual:
            actor_in_dim = actor_actual[0][1]
            actor_out_dim = actor_actual[-1][0]
            expected_actor_in = spec.num_obs + spec.latent_dim
            if actor_in_dim != expected_actor_in:
                errors.append(
                    "Actor first-layer input dim mismatch (num_obs + latent_dim): "
                    f"expected {expected_actor_in}, got {actor_in_dim}"
                )
            if actor_out_dim != spec.num_actions:
                errors.append(
                    "Actor output dim mismatch (num_actions): "
                    f"expected {spec.num_actions}, got {actor_out_dim}"
                )

        if errors:
            joined = "\n  - ".join(errors)
            raise ValueError(
                "Checkpoint is incompatible with the provided/inferred PolicySpec.\n"
                f"  - {joined}"
            )

    def get_action(self, obs: np.ndarray) -> np.ndarray:
        obs = np.asarray(obs, dtype=np.float32)
        if obs.shape != (self.spec.num_obs,):
            raise ValueError(
                f"Observation shape mismatch: expected {(self.spec.num_obs,)}, got {obs.shape}"
            )

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
        self.obs_history = np.zeros((self.spec.num_encoder_obs,), dtype=np.float32)
