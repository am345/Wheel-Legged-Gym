# SPDX-FileCopyrightText: Copyright (c) 2021 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: BSD-3-Clause
# ... (版权声明省略)

from wheel_legged_gym import WHEEL_LEGGED_GYM_ROOT_DIR
import os

import isaacgym
from isaacgym import gymapi  # <--- 新增: 导入 gymapi 以使用键盘常量
from isaacgym.torch_utils import *
from wheel_legged_gym.envs import *
from wheel_legged_gym.utils import get_args, export_policy_as_jit, task_registry, Logger

import numpy as np
import torch


class KeyboardController:
    def __init__(self, env):
        self.gym = env.gym
        self.viewer = env.viewer
        self.env = env
        
        self.subscribe_keyboard_events()
        
        # 初始指令: [Lin_Vel_X, Ang_Vel_Yaw, Height]
        self.command = np.array([0.0, 0.0, 0.25])
        
        # 控制步长（普通模式）
        self.lin_vel_step = 0.02
        self.ang_vel_step = 0.05
        self.height_step = 0.002
        
        # --- 新增: 自旋速度设定 ---
        self.spin_velocity = 2.5 # 按住 Shift 时的自转速度 (rad/s)
        
        # 按键状态记录
        self.key_states = {
            "FORWARD": False,
            "BACKWARD": False,
            "LEFT": False,
            "RIGHT": False,
            "UP": False,
            "DOWN": False,
            "SPIN_MODE": False # 新增 Shift 状态
        }
        
        print("\n" + "="*30)
        print("🎮 键盘控制 (增强版)")
        print("------------------------------")
        print("   W/S     : 持续加速/减速")
        print("   A/D     : 持续左转/右转")
        print("   Q/E     : 持续升高/降低")
        print("   L_SHIFT : [原地自旋] (速度归零+高速旋转)")
        print("   SPACE   : 急停 (归零)")
        print("   R       : 重置环境")
        print("="*30 + "\n")

    def subscribe_keyboard_events(self):
        self.gym.subscribe_viewer_keyboard_event(self.viewer, gymapi.KEY_W, "FORWARD")
        self.gym.subscribe_viewer_keyboard_event(self.viewer, gymapi.KEY_S, "BACKWARD")
        self.gym.subscribe_viewer_keyboard_event(self.viewer, gymapi.KEY_A, "LEFT")
        self.gym.subscribe_viewer_keyboard_event(self.viewer, gymapi.KEY_D, "RIGHT")
        self.gym.subscribe_viewer_keyboard_event(self.viewer, gymapi.KEY_Q, "UP")
        self.gym.subscribe_viewer_keyboard_event(self.viewer, gymapi.KEY_E, "DOWN")
        self.gym.subscribe_viewer_keyboard_event(self.viewer, gymapi.KEY_SPACE, "STOP")
        self.gym.subscribe_viewer_keyboard_event(self.viewer, gymapi.KEY_R, "RESET")
        
        # --- 新增: 监听左 Shift 键 ---
        self.gym.subscribe_viewer_keyboard_event(self.viewer, gymapi.KEY_LEFT_SHIFT, "SPIN_MODE")

    def get_command(self):
        # 1. 更新按键状态
        for evt in self.gym.query_viewer_action_events(self.viewer):
            if evt.action == "STOP" and evt.value > 0:
                self.command[0] = 0.0
                self.command[1] = 0.0
                continue
            elif evt.action == "RESET" and evt.value > 0:
                self.env.reset_idx(torch.arange(self.env.num_envs, device=self.env.device))
                continue
            
            if evt.action in self.key_states:
                self.key_states[evt.action] = (evt.value > 0)

        # 2. 核心逻辑：优先判断 Shift 模式
        if self.key_states["SPIN_MODE"]:
            # --- Shift 模式逻辑 ---
            self.command[0] = 0.0  # 强制线速度为 0 (原地)
            self.command[1] = self.spin_velocity # 强制设定为固定的自转角速度
            # 如果你想反向转，可以改成 -self.spin_velocity
            
        else:
            # --- 普通模式逻辑 (只有没按 Shift 时才生效) ---
            if self.key_states["FORWARD"]:
                self.command[0] += self.lin_vel_step
            if self.key_states["BACKWARD"]:
                self.command[0] -= self.lin_vel_step
                
            if self.key_states["LEFT"]:
                self.command[1] += self.ang_vel_step
            if self.key_states["RIGHT"]:
                self.command[1] -= self.ang_vel_step

        # 始终允许高度调整
        if self.key_states["UP"]:
            self.command[2] += self.height_step
        if self.key_states["DOWN"]:
            self.command[2] -= self.height_step

        # 3. 限制范围
        self.command[0] = np.clip(self.command[0], -2.5, 2.5)
        self.command[1] = np.clip(self.command[1], -5.0, 5.0) # 稍微放宽角速度上限以允许快速自转
        self.command[2] = np.clip(self.command[2], 0.1, 0.5)
        
        return self.command

def play(args):
    env_cfg, train_cfg = task_registry.get_cfgs(name=args.task)
    # override some parameters for testing
    env_cfg.env.episode_length_s = 20
    env_cfg.env.fail_to_terminal_time_s = 3
    
    # --- 修改: 键盘控制时通常只需要观察一个机器人 ---
    env_cfg.env.num_envs = 1 
    # ----------------------------------------
    
    env_cfg.terrain.num_rows = 5
    env_cfg.terrain.num_cols = 10
    env_cfg.terrain.max_init_terrain_level = env_cfg.terrain.num_rows - 1
    env_cfg.terrain.curriculum = True
    env_cfg.noise.add_noise = False
    env_cfg.domain_rand.randomize_friction = False
    env_cfg.domain_rand.friction_range = [0.1, 0.2]
    env_cfg.domain_rand.randomize_restitution = False
    env_cfg.domain_rand.randomize_base_com = False
    env_cfg.domain_rand.push_robots = False
    env_cfg.domain_rand.push_interval_s = 2
    env_cfg.domain_rand.max_push_vel_xy = 3
    env_cfg.domain_rand.randomize_Kp = False
    env_cfg.domain_rand.randomize_Kd = False
    env_cfg.domain_rand.randomize_motor_torque = False
    env_cfg.domain_rand.randomize_default_dof_pos = False
    env_cfg.domain_rand.randomize_action_delay = False

    # --- 新增: 禁用自动指令重采样和朝向控制 ---
    env_cfg.commands.resampling_time = 999999  # 防止环境自动改变指令
    env_cfg.commands.heading_command = False   # 关闭自动朝向计算，直接控制 yaw 速度
    # ----------------------------------------

    # prepare environment
    env, _ = task_registry.make_env(name=args.task, args=args, env_cfg=env_cfg)
    obs, obs_history = env.get_observations()
    # load policy
    train_cfg.runner.resume = True
    ppo_runner, train_cfg = task_registry.make_alg_runner(
        env=env, name=args.task, args=args, train_cfg=train_cfg
    )
    policy = ppo_runner.get_inference_policy(device=env.device)

    # export policy as a jit module (used to run it from C++)
    if EXPORT_POLICY:
        path = os.path.join(
            WHEEL_LEGGED_GYM_ROOT_DIR,
            "logs",
            train_cfg.runner.experiment_name,
            "exported",
            "policies",
        )
        export_policy_as_jit(ppo_runner.alg.actor_critic, path)
        print("Exported policy as jit script to: ", path)

    logger = Logger(env.dt)
    robot_index = 0  # --- 修改: 因为 num_envs=1，这里改为0 ---
    joint_index = 1  
    stop_state_log = 1000  
    stop_rew_log = (
        env.max_episode_length + 1
    )  
    
    img_idx = 0
    latent = None

    # --- 修改: 禁用原有的 CoM 补偿逻辑，因为我们要用键盘控制 ---
    CoM_offset_compensate = False 
    # ----------------------------------------------------

    # 初始化键盘控制器
    controller = KeyboardController(env)

    for i in range(1000 * int(env.max_episode_length)):
        if ppo_runner.alg.actor_critic.is_sequence:
            actions, latent = policy(obs, obs_history)
        else:
            actions = policy(obs.detach())

        # --- 修改: 使用键盘指令覆盖环境指令 ---
        kb_cmd = controller.get_command()
        
        # 将键盘指令应用到所有环境（这里只有1个）
        env.commands[:, 0] = kb_cmd[0] # Lin Vel X
        env.commands[:, 1] = kb_cmd[1] # Ang Vel Yaw (因为 heading_command=False)
        env.commands[:, 2] = kb_cmd[2] # Height
        env.commands[:, 3] = 0.0       # Heading (未使用)
        
        # 打印当前指令以便调试
        # print(f"\rCmd: V={kb_cmd[0]:.2f}, W={kb_cmd[1]:.2f}, H={kb_cmd[2]:.2f}", end="")
        # -----------------------------------

        # 原有的硬编码指令逻辑已移除/注释
        # env.commands[:, 0] = 2.5
        # env.commands[:, 2] = 0.18
        # env.commands[:, 3] = 0

        # if CoM_offset_compensate: ... (已通过设置 False 禁用)

        obs, _, rews, dones, infos, obs_history = env.step(actions)
        
        if RECORD_FRAMES:
            if i % 2:
                filename = os.path.join(
                    WHEEL_LEGGED_GYM_ROOT_DIR,
                    "logs",
                    train_cfg.runner.experiment_name,
                    "exported",
                    "frames",
                    f"{img_idx}.png",
                )
                env.gym.write_viewer_image_to_file(env.viewer, filename)
                img_idx += 1
        if MOVE_CAMERA:
            camera_offset = np.array(env_cfg.viewer.pos)
            target_position = np.array(
                env.base_position[robot_index, :].to(device="cpu")
            )
            camera_position = target_position + camera_offset
            env.set_camera(camera_position, target_position)

        if i < stop_state_log:
            logger.log_states(
                {
                    "dof_pos_target": actions[robot_index, joint_index].item()
                    * env.cfg.control.action_scale
                    + env.default_dof_pos[robot_index, joint_index].item(),
                    "dof_pos": env.dof_pos[robot_index, joint_index].item(),
                    "dof_vel": env.dof_vel[robot_index, joint_index].item(),
                    "dof_torque": env.torques[robot_index, joint_index].item(),
                    "command_yaw": env.commands[robot_index, 1].item(),
                    "command_height": env.commands[robot_index, 2].item(),
                    "base_height": env.base_height[robot_index].item(),
                    "base_vel_x": env.base_lin_vel[robot_index, 0].item(),
                    "base_vel_y": env.base_lin_vel[robot_index, 1].item(),
                    "base_vel_z": env.base_lin_vel[robot_index, 2].item(),
                    "base_vel_yaw": env.base_ang_vel[robot_index, 2].item(),
                    "contact_forces_z": env.contact_forces[
                        robot_index, env.feet_indices, 2
                    ]
                    .cpu()
                    .numpy(),
                }
            )
            logger.log_states({"command_x": env.commands[robot_index, 0].item()})

            if latent is not None:
                # Log latent states if available
                pass # 保持原有逻辑不变

        elif i == stop_state_log:
            logger.plot_states()
        if 0 < i < stop_rew_log:
            if infos["episode"]:
                num_episodes = torch.sum(env.reset_buf).item()
                if num_episodes > 0:
                    logger.log_rewards(infos["episode"], num_episodes)
        elif i == stop_rew_log:
            logger.print_rewards()


if __name__ == "__main__":
    EXPORT_POLICY = True
    RECORD_FRAMES = False
    MOVE_CAMERA = False
    args = get_args()
    play(args)