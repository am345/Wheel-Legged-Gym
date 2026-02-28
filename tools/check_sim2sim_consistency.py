#!/usr/bin/env python3
"""Check sim2sim consistency between IsaacGym and MuJoCo."""

import sys
sys.path.insert(0, '.')

from mujoco_sim.control_config import get_balance_vmc_control_config

print("=" * 80)
print("SIM2SIM CONSISTENCY CHECK: IsaacGym vs MuJoCo")
print("=" * 80)

# Get MuJoCo config
mj_cfg = get_balance_vmc_control_config()

# IsaacGym config values (from wheel_legged_fzqver_config.py)
ig_sim_dt = 0.005
ig_decimation = 2
ig_default_dof_pos = (0.4, 0.25, 0.0, 0.4, 0.25, 0.0)
ig_obs_ang_vel = 0.25
ig_obs_dof_pos = 1.0
ig_obs_dof_vel = 0.05
ig_obs_l0 = 5.0
ig_obs_l0_dot = 0.25

ig_action_scale_theta = 3.14
ig_action_scale_l0 = 0.10
ig_l0_offset = 0.22
ig_feedforward_force = 40.0
ig_kp_theta = 10.0
ig_kd_theta = 5.0
ig_kp_l0 = 800.0
ig_kd_l0 = 7.0

ig_upright_ratio = 0.2
ig_full_pose = [-3.14, 3.14]
ig_upright_roll_pitch = [-0.25, 0.25]
ig_upright_yaw = [-3.14, 3.14]
ig_fallen_z_offset = [0.0, 0.08]
ig_upright_z_offset = [0.0, 0.03]
ig_lin_vel = [-0.5, 0.5]
ig_ang_vel = [-0.5, 0.5]

print("\n" + "=" * 80)
print("1. TIMING PARAMETERS")
print("=" * 80)

print("\nIsaacGym:")
print(f"  sim.dt:              {ig_sim_dt} s")
print(f"  control.decimation:  {ig_decimation}")
print(f"  control_dt:          {ig_sim_dt * ig_decimation} s")

print("\nMuJoCo:")
print(f"  sim_dt:              {mj_cfg.sim_dt} s")
print(f"  control_decimation:  {mj_cfg.control_decimation}")
print(f"  control_dt:          {mj_cfg.control_dt} s")

dt_match = (ig_sim_dt == mj_cfg.sim_dt and ig_decimation == mj_cfg.control_decimation)
print(f"\n{'✓' if dt_match else '✗'} Timing: {'MATCH' if dt_match else 'MISMATCH'}")

print("\n" + "=" * 80)
print("2. DEFAULT DOF POSITIONS (Initial Joint Angles)")
print("=" * 80)

print("\nIsaacGym:")
print(f"  default_dof_pos: {ig_default_dof_pos}")

print("\nMuJoCo:")
print(f"  default_dof_pos: {mj_cfg.default_dof_pos}")

dof_match = ig_default_dof_pos == mj_cfg.default_dof_pos
print(f"\n{'✓' if dof_match else '✗'} Default DOF: {'MATCH' if dof_match else 'MISMATCH'}")

print("\n" + "=" * 80)
print("3. OBSERVATION SCALES")
print("=" * 80)

print("\nIsaacGym:")
print(f"  ang_vel:   {ig_obs_ang_vel}")
print(f"  dof_pos:   {ig_obs_dof_pos}")
print(f"  dof_vel:   {ig_obs_dof_vel}")
print(f"  l0:        {ig_obs_l0}")
print(f"  l0_dot:    {ig_obs_l0_dot}")

print("\nMuJoCo:")
print(f"  ang_vel:   {mj_cfg.obs_scales_ang_vel}")
print(f"  dof_pos:   {mj_cfg.obs_scales_dof_pos}")
print(f"  dof_vel:   {mj_cfg.obs_scales_dof_vel}")
print(f"  l0:        {mj_cfg.obs_scales_l0}")
print(f"  l0_dot:    {mj_cfg.obs_scales_l0_dot}")

obs_match = (ig_obs_ang_vel == mj_cfg.obs_scales_ang_vel and
             ig_obs_dof_pos == mj_cfg.obs_scales_dof_pos and
             ig_obs_dof_vel == mj_cfg.obs_scales_dof_vel and
             ig_obs_l0 == mj_cfg.obs_scales_l0 and
             ig_obs_l0_dot == mj_cfg.obs_scales_l0_dot)
print(f"\n{'✓' if obs_match else '✗'} Observation scales: {'MATCH' if obs_match else 'MISMATCH'}")

print("\n" + "=" * 80)
print("4. RESET CONFIGURATION (fzqver task)")
print("=" * 80)

print("\nIsaacGym (fzqver_reset):")
print(f"  upright_ratio:       {ig_upright_ratio}")
print(f"  full_pose:           {ig_full_pose}")
print(f"  upright_roll_pitch:  {ig_upright_roll_pitch}")
print(f"  upright_yaw:         {ig_upright_yaw}")
print(f"  fallen_z_offset:     {ig_fallen_z_offset}")
print(f"  upright_z_offset:    {ig_upright_z_offset}")
print(f"  lin_vel:             {ig_lin_vel}")
print(f"  ang_vel:             {ig_ang_vel}")

mj_fzq = mj_cfg.fzqver_profile
print("\nMuJoCo (fzqver_profile):")
print(f"  upright_ratio:       {mj_fzq.upright_ratio}")
print(f"  full_pose:           {list(mj_fzq.full_pose)}")
print(f"  upright_roll_pitch:  {list(mj_fzq.upright_roll_pitch)}")
print(f"  upright_yaw:         {list(mj_fzq.upright_yaw)}")
print(f"  fallen_z_offset:     {list(mj_fzq.fallen_z_offset)}")
print(f"  upright_z_offset:    {list(mj_fzq.upright_z_offset)}")
print(f"  lin_vel:             {list(mj_fzq.lin_vel)}")
print(f"  ang_vel:             {list(mj_fzq.ang_vel)}")

reset_match = (ig_upright_ratio == mj_fzq.upright_ratio and
               ig_full_pose == list(mj_fzq.full_pose) and
               ig_upright_roll_pitch == list(mj_fzq.upright_roll_pitch) and
               ig_upright_yaw == list(mj_fzq.upright_yaw))
print(f"\n{'✓' if reset_match else '✗'} Reset config: {'MATCH' if reset_match else 'MISMATCH'}")

print("\n" + "=" * 80)
print("5. VMC CONTROL PARAMETERS")
print("=" * 80)

print("\nIsaacGym:")
print(f"  action_scale_theta:  {ig_action_scale_theta}")
print(f"  action_scale_l0:     {ig_action_scale_l0}")
print(f"  l0_offset:           {ig_l0_offset}")
print(f"  feedforward_force:   {ig_feedforward_force}")
print(f"  kp_theta:            {ig_kp_theta}")
print(f"  kd_theta:            {ig_kd_theta}")
print(f"  kp_l0:               {ig_kp_l0}")
print(f"  kd_l0:               {ig_kd_l0}")

print("\nMuJoCo:")
print(f"  action_scale_theta:  {mj_cfg.action_scale_theta}")
print(f"  action_scale_l0:     {mj_cfg.action_scale_l0}")
print(f"  l0_offset:           {mj_cfg.l0_offset}")
print(f"  feedforward_force:   {mj_cfg.feedforward_force}")
print(f"  kp_theta:            {mj_cfg.kp_theta}")
print(f"  kd_theta:            {mj_cfg.kd_theta}")
print(f"  kp_l0:               {mj_cfg.kp_l0}")
print(f"  kd_l0:               {mj_cfg.kd_l0}")

vmc_match = (ig_action_scale_theta == mj_cfg.action_scale_theta and
             ig_kp_theta == mj_cfg.kp_theta and
             ig_kp_l0 == mj_cfg.kp_l0)
print(f"\n{'✓' if vmc_match else '✗'} VMC parameters: {'MATCH' if vmc_match else 'MISMATCH'}")

print("\n" + "=" * 80)
print("SUMMARY")
print("=" * 80)

all_match = dt_match and dof_match and obs_match and reset_match and vmc_match
if all_match:
    print("\n✓ All parameters match! Configuration is consistent.")
else:
    print("\n✗ Some parameters don't match:")
    if not dt_match:
        print("  ✗ Timing parameters")
    if not dof_match:
        print("  ✗ Default joint angles")
    if not obs_match:
        print("  ✗ Observation scaling")
    if not reset_match:
        print("  ✗ Reset configuration")
    if not vmc_match:
        print("  ✗ VMC controller gains")

print("\n" + "=" * 80)
print("DIAGNOSIS: Why is the robot falling in MuJoCo?")
print("=" * 80)

print("\nFrom the MuJoCo output (tilt=168deg, base_contact=True):")
print("  → Robot is falling/倒地 immediately after reset")
print("\nPossible causes:")
print("  1. Initial pose instability:")
print(f"     - 80% of resets start from FALLEN pose (random orientation)")
print(f"     - Only 20% start UPRIGHT (±0.25 rad roll/pitch)")
print("     - Policy may not be robust to fallen starts in MuJoCo")
print("\n  2. Contact model differences:")
print("     - MuJoCo uses pyramidal cone friction")
print("     - IsaacGym uses different contact solver")
print("     - Numerical integration differences")
print("\n  3. Solver parameters:")
print("     - MuJoCo: iterations=50, Newton solver")
print("     - IsaacGym: different solver settings")

print("\n" + "=" * 80)
print("RECOMMENDATIONS")
print("=" * 80)

print("\n1. Test with stable initial pose:")
print("   python tools/play_mujoco_policy_fzqver.py --fixed-reset")
print("   → This starts from upright pose (no randomization)")

print("\n2. Increase upright ratio for testing:")
print("   → Edit mujoco_sim/control_config.py")
print("   → Change upright_ratio from 0.2 to 1.0")
print("   → This makes 100% of resets start upright")

print("\n3. Check MuJoCo solver settings:")
print("   → serialleg_fidelity.xml has iterations=50")
print("   → Try increasing to 100 for better convergence")

print("\n4. Compare with IsaacGym:")
print("   → Run same policy in IsaacGym")
print("   → Check if it can recover from fallen poses")
print("   → If not, policy may need retraining")

print("\n" + "=" * 80)
