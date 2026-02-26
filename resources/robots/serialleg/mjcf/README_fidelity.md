# `serialleg_fidelity.xml`

High-fidelity MuJoCo MJCF model for sim2sim evaluation of `wheel_legged_vmc_balance`.

## Source

Generated from:
- `resources/robots/serialleg/urdf/serialleg.urdf`
- mesh assets under `resources/robots/serialleg/meshes/`

Generator script:
- `tools/build_serialleg_fidelity_mjcf.py`

## Key conversion rules

- Preserves joint topology and joint names used by training / sim2sim:
  - `lf0_Joint`, `lf1_Joint`, `l_wheel_Joint`, `rf0_Joint`, `rf1_Joint`, `r_wheel_Joint`
- Preserves URDF inertial mass and full inertia tensors.
- Converts full inertia tensors to MuJoCo-friendly principal-axis form (`diaginertia + quat`) via eigendecomposition.
- Uses direct torque `motor` actuators in training-aligned order.

## Contact modeling choices (important)

- Non-wheel collision geoms (`base/thigh/calf`) use collision meshes from URDF.
- Wheel collision uses analytic `cylinder` geoms for stable rolling contact.
- Wheel visual still uses mesh (collision and visual can differ by design).

This is intentional: wheel mesh collisions often degrade contact stability and rolling behavior in MuJoCo.

## Known preprocessing applied internally by generator

- Fixes malformed URDF attribute: `velocity="49.1.0" -> "49.1"`
- Normalizes mesh paths for MuJoCo resolution.

## Regenerate / validate

```bash
source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate gym_env
python tools/build_serialleg_fidelity_mjcf.py --validate-load
```

## Notes on sim2sim interpretation

Even with fidelity MJCF + `vmc_balance_exact`, MuJoCo and IsaacGym/PhysX contact models differ.
Use both:
- short-sequence alignment metrics (Isaac reference replay)
- long rollout robustness metrics (randomized MuJoCo evaluation)
