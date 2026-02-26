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

- Collision geoms use **unsimplified visual STL meshes by default** (`--collision-mesh-source visual_stl`).
- This applies to all links, including wheels.
- Oversized STL meshes are **automatically split into multiple binary STL chunks without simplification**
  to satisfy MuJoCo's per-mesh face limit (e.g. `base_link.STL`).
- Generator supports fallbacks for comparison/debug:
  - `--collision-mesh-source urdf_collision` (use URDF collision OBJ meshes, often simplified/split)
  - `--wheel-collision-mode cylinder` (wheel-only cylinder proxy; non-wheel links still follow `--collision-mesh-source`)
- Wheel visual still uses mesh (visual proxy).

Using visual STL meshes for collision is closer to the full geometry, but may reduce contact stability
or increase simulation cost in MuJoCo compared with simplified collision meshes / cylinder fallback.

## Known preprocessing applied internally by generator

- Fixes malformed URDF attribute: `velocity="49.1.0" -> "49.1"`
- Normalizes mesh paths for MuJoCo resolution.
- Generates chunked STL files under `resources/robots/serialleg/meshes/_mjc_generated_collision/`
  when a visual STL exceeds MuJoCo's mesh face limit.

## Regenerate / validate

```bash
source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate gym_env
python tools/build_serialleg_fidelity_mjcf.py --validate-load

# optional fallback for comparison (not default)
python tools/build_serialleg_fidelity_mjcf.py --collision-mesh-source urdf_collision --validate-load
python tools/build_serialleg_fidelity_mjcf.py --wheel-collision-mode cylinder --validate-load
```

## Notes on sim2sim interpretation

Even with fidelity MJCF + `vmc_balance_exact`, MuJoCo and IsaacGym/PhysX contact models differ.
Use both:
- short-sequence alignment metrics (Isaac reference replay)
- long rollout robustness metrics (randomized MuJoCo evaluation)
