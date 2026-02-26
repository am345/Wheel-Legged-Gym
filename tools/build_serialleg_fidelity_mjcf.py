#!/usr/bin/env python3
"""
Generate a higher-fidelity MuJoCo MJCF from serialleg URDF.

This script is intentionally robot-specific (serialleg) but keeps the parsing/generation
pipeline explicit so the generated MJCF is reproducible and auditable.

Key decisions:
- Preserve URDF topology, joints, inertials, visuals, collision meshes for base/thigh/calf
- Replace wheel collision meshes with analytical cylinders for more stable rolling contact
- Use motor actuators (direct torque control), not MJCF-embedded PD servos
- Keep control order aligned with training/MuJoCo sim2sim env expectations
"""

from __future__ import annotations

import argparse
import math
import os
import re
import sys
import xml.etree.ElementTree as ET
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
from scipy.spatial.transform import Rotation


WHEEL_COLLISION_RADIUS = 0.0675
WHEEL_COLLISION_HALF_WIDTH = 0.025
WHEEL_CYLINDER_QUAT_WXYZ = (0.70710678, -0.70710678, 0.0, 0.0)  # rotate z-axis to y-axis

EXPECTED_ACTUATOR_ORDER = [
    "lf0_Joint",
    "lf1_Joint",
    "l_wheel_Joint",
    "rf0_Joint",
    "rf1_Joint",
    "r_wheel_Joint",
]


@dataclass
class Origin:
    xyz: np.ndarray = field(default_factory=lambda: np.zeros(3, dtype=np.float64))
    rpy: np.ndarray = field(default_factory=lambda: np.zeros(3, dtype=np.float64))


@dataclass
class Inertial:
    origin: Origin
    mass: float
    inertia_matrix_local: np.ndarray  # expressed in inertial origin frame (URDF convention)


@dataclass
class MeshGeom:
    origin: Origin
    filename: str
    geom_type: str  # visual/collision


@dataclass
class LinkData:
    name: str
    inertial: Optional[Inertial] = None
    visuals: List[MeshGeom] = field(default_factory=list)
    collisions: List[MeshGeom] = field(default_factory=list)


@dataclass
class JointData:
    name: str
    joint_type: str
    parent: str
    child: str
    origin: Origin
    axis: np.ndarray
    lower: Optional[float]
    upper: Optional[float]
    effort: Optional[float]
    velocity: Optional[float]
    damping: Optional[float]
    friction: Optional[float]


def parse_xyz(s: Optional[str]) -> np.ndarray:
    if not s:
        return np.zeros(3, dtype=np.float64)
    vals = [float(x) for x in s.strip().split()]
    return np.array(vals, dtype=np.float64)


def parse_origin(elem: Optional[ET.Element]) -> Origin:
    if elem is None:
        return Origin()
    return Origin(
        xyz=parse_xyz(elem.attrib.get("xyz")),
        rpy=parse_xyz(elem.attrib.get("rpy")),
    )


def urdf_to_rotation(rpy: np.ndarray) -> Rotation:
    return Rotation.from_euler("xyz", rpy)


def quat_xyzw_to_wxyz(q: np.ndarray) -> np.ndarray:
    return np.array([q[3], q[0], q[1], q[2]], dtype=np.float64)


def rotation_to_mjcf_quat_wxyz(rot: Rotation) -> np.ndarray:
    return quat_xyzw_to_wxyz(rot.as_quat())


def fmt_floats(vals) -> str:
    arr = np.asarray(vals, dtype=np.float64).reshape(-1)
    return " ".join(f"{x:.9g}" for x in arr)


def sanitize_urdf_text(text: str) -> str:
    # Fix known malformed velocity field in left wheel joint
    text = text.replace('velocity="49.1.0"', 'velocity="49.1"')
    return text


def parse_urdf(urdf_path: Path) -> Tuple[Dict[str, LinkData], List[JointData], str]:
    text = sanitize_urdf_text(urdf_path.read_text(encoding="utf-8"))
    root = ET.fromstring(text)

    links: Dict[str, LinkData] = {}
    joints: List[JointData] = []

    for link_elem in root.findall("link"):
        name = link_elem.attrib["name"]
        link = LinkData(name=name)

        inertial_elem = link_elem.find("inertial")
        if inertial_elem is not None:
            origin = parse_origin(inertial_elem.find("origin"))
            mass_elem = inertial_elem.find("mass")
            inertia_elem = inertial_elem.find("inertia")
            if mass_elem is not None and inertia_elem is not None:
                mass = float(mass_elem.attrib["value"])
                ixx = float(inertia_elem.attrib["ixx"])
                ixy = float(inertia_elem.attrib.get("ixy", 0.0))
                ixz = float(inertia_elem.attrib.get("ixz", 0.0))
                iyy = float(inertia_elem.attrib["iyy"])
                iyz = float(inertia_elem.attrib.get("iyz", 0.0))
                izz = float(inertia_elem.attrib["izz"])
                I = np.array(
                    [[ixx, ixy, ixz], [ixy, iyy, iyz], [ixz, iyz, izz]], dtype=np.float64
                )
                link.inertial = Inertial(origin=origin, mass=mass, inertia_matrix_local=I)

        for visual_elem in link_elem.findall("visual"):
            origin = parse_origin(visual_elem.find("origin"))
            geom_elem = visual_elem.find("geometry")
            mesh_elem = None if geom_elem is None else geom_elem.find("mesh")
            if mesh_elem is not None:
                link.visuals.append(
                    MeshGeom(origin=origin, filename=mesh_elem.attrib["filename"], geom_type="visual")
                )

        for collision_elem in link_elem.findall("collision"):
            origin = parse_origin(collision_elem.find("origin"))
            geom_elem = collision_elem.find("geometry")
            mesh_elem = None if geom_elem is None else geom_elem.find("mesh")
            if mesh_elem is not None:
                link.collisions.append(
                    MeshGeom(origin=origin, filename=mesh_elem.attrib["filename"], geom_type="collision")
                )

        links[name] = link

    for joint_elem in root.findall("joint"):
        name = joint_elem.attrib["name"]
        joint_type = joint_elem.attrib["type"]
        parent = joint_elem.find("parent").attrib["link"]
        child = joint_elem.find("child").attrib["link"]
        origin = parse_origin(joint_elem.find("origin"))
        axis_elem = joint_elem.find("axis")
        axis = parse_xyz(axis_elem.attrib.get("xyz") if axis_elem is not None else None)
        if np.linalg.norm(axis) > 0:
            axis = axis / np.linalg.norm(axis)
        else:
            axis = np.array([0.0, 0.0, 1.0], dtype=np.float64)

        limit_elem = joint_elem.find("limit")
        lower = float(limit_elem.attrib["lower"]) if limit_elem is not None and "lower" in limit_elem.attrib else None
        upper = float(limit_elem.attrib["upper"]) if limit_elem is not None and "upper" in limit_elem.attrib else None
        effort = float(limit_elem.attrib["effort"]) if limit_elem is not None and "effort" in limit_elem.attrib else None
        velocity = float(limit_elem.attrib["velocity"]) if limit_elem is not None and "velocity" in limit_elem.attrib else None

        dyn_elem = joint_elem.find("dynamics")
        damping = float(dyn_elem.attrib["damping"]) if dyn_elem is not None and "damping" in dyn_elem.attrib else None
        friction = float(dyn_elem.attrib["friction"]) if dyn_elem is not None and "friction" in dyn_elem.attrib else None

        joints.append(
            JointData(
                name=name,
                joint_type=joint_type,
                parent=parent,
                child=child,
                origin=origin,
                axis=axis,
                lower=lower,
                upper=upper,
                effort=effort,
                velocity=velocity,
                damping=damping,
                friction=friction,
            )
        )

    parent_links = {j.child for j in joints}
    roots = [name for name in links if name not in parent_links]
    if len(roots) != 1:
        raise RuntimeError(f"Expected exactly one URDF root link, got {roots}")
    return links, joints, roots[0]


def principal_inertial_from_urdf(inertial: Inertial) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Return (pos_xyz, quat_wxyz, diaginertia) in body frame for MuJoCo inertial tag."""
    pos = inertial.origin.xyz
    R_origin = urdf_to_rotation(inertial.origin.rpy).as_matrix()
    I_body = R_origin @ inertial.inertia_matrix_local @ R_origin.T

    # Symmetrize for numerical safety
    I_body = 0.5 * (I_body + I_body.T)
    vals, vecs = np.linalg.eigh(I_body)
    vals = np.maximum(vals, 1e-12)

    # Ensure right-handed basis
    if np.linalg.det(vecs) < 0:
        vecs[:, 2] *= -1.0

    quat = rotation_to_mjcf_quat_wxyz(Rotation.from_matrix(vecs))
    return pos, quat, vals


def mesh_basename(filename: str) -> str:
    return Path(filename.replace("\\", "/")).name


def is_wheel_link(name: str) -> bool:
    return name in {"l_wheel_Link", "r_wheel_Link"}


def is_visual_only_geom_for_link(link_name: str) -> bool:
    # keep visuals for all links; contact control done via collision geoms and wheel cylinders
    return False


def build_asset_mesh_tables(links: Dict[str, LinkData]) -> Dict[Tuple[str, str], str]:
    """Map (filename, visual|collision) -> unique asset name."""
    used: Dict[Tuple[str, str], str] = {}
    seen_names = set()

    def add(filename: str, kind: str):
        key = (filename, kind)
        if key in used:
            return
        stem = Path(mesh_basename(filename)).stem
        base_name = f"{kind}_{re.sub(r'[^a-zA-Z0-9_]+', '_', stem)}"
        name = base_name
        i = 1
        while name in seen_names:
            i += 1
            name = f"{base_name}_{i}"
        used[key] = name
        seen_names.add(name)

    for link in links.values():
        for c in link.collisions:
            add(c.filename, "collision")
    return used


def indent(elem: ET.Element, level: int = 0):
    i = "\n" + level * "  "
    if len(elem):
        if not elem.text or not elem.text.strip():
            elem.text = i + "  "
        for child in elem:
            indent(child, level + 1)
        if not child.tail or not child.tail.strip():
            child.tail = i
    if level and (not elem.tail or not elem.tail.strip()):
        elem.tail = i


def add_geom_mesh(
    body_elem: ET.Element,
    mesh_asset_name: str,
    origin: Origin,
    is_collision: bool,
    rgba: Optional[str] = None,
    geom_name: Optional[str] = None,
):
    attrs = {
        "type": "mesh",
        "mesh": mesh_asset_name,
        "pos": fmt_floats(origin.xyz),
    }
    if np.linalg.norm(origin.rpy) > 0:
        quat = rotation_to_mjcf_quat_wxyz(urdf_to_rotation(origin.rpy))
        attrs["quat"] = fmt_floats(quat)
    if geom_name:
        attrs["name"] = geom_name
    if is_collision:
        attrs["contype"] = "1"
        attrs["conaffinity"] = "1"
        attrs["group"] = "0"
    else:
        attrs["contype"] = "0"
        attrs["conaffinity"] = "0"
        attrs["group"] = "1"
        if rgba:
            attrs["rgba"] = rgba
    ET.SubElement(body_elem, "geom", attrs)


def add_wheel_collision_cylinder(body_elem: ET.Element, side: str):
    # `side` currently unused; kept for future side-specific tuning.
    ET.SubElement(
        body_elem,
        "geom",
        {
            "name": f"{side}_wheel_collision",
            "type": "cylinder",
            "size": fmt_floats([WHEEL_COLLISION_RADIUS, WHEEL_COLLISION_HALF_WIDTH]),
            "quat": fmt_floats(WHEEL_CYLINDER_QUAT_WXYZ),
            "friction": "0.8 0.005 0.0001",
            "group": "0",
            "contype": "1",
            "conaffinity": "1",
            "rgba": "0.1 0.1 0.1 1",
        },
    )


def build_mjcf_tree(
    links: Dict[str, LinkData],
    joints: List[JointData],
    root_link: str,
    meshdir_rel: str,
) -> ET.Element:
    joint_by_parent: Dict[str, List[JointData]] = {}
    for j in joints:
        joint_by_parent.setdefault(j.parent, []).append(j)

    model = ET.Element("mujoco", {"model": "serialleg_fidelity"})
    ET.SubElement(
        model,
        "compiler",
        {
            "angle": "radian",
            "meshdir": meshdir_rel,
            "autolimits": "true",
            "strippath": "false",
        },
    )
    ET.SubElement(
        model,
        "option",
        {
            "timestep": "0.005",
            "iterations": "50",
            "solver": "Newton",
            "cone": "pyramidal",
        },
    )

    asset_elem = ET.SubElement(model, "asset")
    ET.SubElement(
        asset_elem,
        "texture",
        {
            "name": "grid",
            "type": "2d",
            "builtin": "checker",
            "width": "512",
            "height": "512",
            "rgb1": ".1 .2 .3",
            "rgb2": ".2 .3 .4",
        },
    )
    ET.SubElement(
        asset_elem,
        "material",
        {
            "name": "grid",
            "texture": "grid",
            "texrepeat": "1 1",
            "texuniform": "true",
            "reflectance": ".2",
        },
    )

    mesh_assets = build_asset_mesh_tables(links)
    for (filename, kind), asset_name in sorted(mesh_assets.items(), key=lambda kv: kv[1]):
        ET.SubElement(
            asset_elem,
            "mesh",
            {
                "name": asset_name,
                "file": mesh_basename(filename),
            },
        )

    world = ET.SubElement(model, "worldbody")
    ET.SubElement(
        world,
        "geom",
        {
            "name": "floor",
            "type": "plane",
            "size": "0 0 .05",
            "material": "grid",
            # Training nominal friction is 0.5; keep auxiliary terms small and stable for MuJoCo
            "friction": "0.5 0.005 0.0001",
        },
    )
    ET.SubElement(
        world,
        "light",
        {
            "name": "spotlight",
            "mode": "targetbodycom",
            "target": "base_link",
            "diffuse": ".8 .8 .8",
            "specular": "0.3 0.3 0.3",
            "pos": "0 -6 4",
            "cutoff": "30",
        },
    )

    def emit_link_body(parent_elem: ET.Element, link_name: str, incoming_joint: Optional[JointData]):
        link = links[link_name]
        body_attrs = {"name": link_name}
        if incoming_joint is None:
            body_attrs["pos"] = "0 0 0.30"
        else:
            body_attrs["pos"] = fmt_floats(incoming_joint.origin.xyz)
            if np.linalg.norm(incoming_joint.origin.rpy) > 0:
                body_attrs["quat"] = fmt_floats(
                    rotation_to_mjcf_quat_wxyz(urdf_to_rotation(incoming_joint.origin.rpy))
                )
        body_elem = ET.SubElement(parent_elem, "body", body_attrs)

        if incoming_joint is None:
            ET.SubElement(body_elem, "freejoint")
        else:
            if incoming_joint.joint_type in ("revolute", "continuous"):
                jattrs = {
                    "name": incoming_joint.name,
                    "type": "hinge",
                    "axis": fmt_floats(incoming_joint.axis),
                }
                if incoming_joint.lower is not None and incoming_joint.upper is not None:
                    jattrs["range"] = fmt_floats([incoming_joint.lower, incoming_joint.upper])
                if incoming_joint.damping is not None:
                    jattrs["damping"] = f"{incoming_joint.damping:.9g}"
                ET.SubElement(body_elem, "joint", jattrs)
            elif incoming_joint.joint_type == "fixed":
                pass  # keep body nesting without joint
            else:
                raise ValueError(
                    f"Unsupported joint type '{incoming_joint.joint_type}' for {incoming_joint.name}"
                )

        # Inertial
        if link.inertial is not None:
            ipos, iquat, idiag = principal_inertial_from_urdf(link.inertial)
            iattrs = {
                "pos": fmt_floats(ipos),
                "mass": f"{link.inertial.mass:.9g}",
                "diaginertia": fmt_floats(idiag),
            }
            if not np.allclose(iquat, np.array([1.0, 0.0, 0.0, 0.0])):
                iattrs["quat"] = fmt_floats(iquat)
            ET.SubElement(body_elem, "inertial", iattrs)

        # Collision geoms
        if is_wheel_link(link_name):
            side = "l" if link_name.startswith("l_") else "r"
            # Keep a mesh visual for wheels using the first collision OBJ (STL visuals are skipped)
            if link.collisions:
                first_coll = link.collisions[0]
                mesh_asset_name = mesh_assets.get((first_coll.filename, "collision"))
                if mesh_asset_name is not None:
                    add_geom_mesh(
                        body_elem,
                        mesh_asset_name=mesh_asset_name,
                        origin=first_coll.origin,
                        is_collision=False,
                        geom_name=f"{link_name}_visual_proxy",
                    )
            add_wheel_collision_cylinder(body_elem, side)
        else:
            for idx, coll in enumerate(link.collisions):
                asset_name = mesh_assets.get((coll.filename, "collision"))
                if asset_name is None:
                    continue
                add_geom_mesh(
                    body_elem,
                    mesh_asset_name=asset_name,
                    origin=coll.origin,
                    is_collision=True,
                    geom_name=f"{link_name}_collision_{idx}",
                )

        for child_joint in joint_by_parent.get(link_name, []):
            emit_link_body(body_elem, child_joint.child, child_joint)

        return body_elem

    emit_link_body(world, root_link, incoming_joint=None)

    actuator_elem = ET.SubElement(model, "actuator")
    joint_map = {j.name: j for j in joints}
    for joint_name in EXPECTED_ACTUATOR_ORDER:
        j = joint_map[joint_name]
        effort = abs(float(j.effort)) if j.effort is not None else (2.0 if "wheel" in joint_name else 30.0)
        act_name = joint_name.replace("_Joint", "_act")
        ET.SubElement(
            actuator_elem,
            "motor",
            {
                "name": act_name,
                "joint": joint_name,
                "gear": "1",
                "ctrllimited": "true",
                "ctrlrange": fmt_floats([-effort, effort]),
            },
        )

    return model


def generate_serialleg_fidelity_mjcf(
    urdf_path: Path,
    output_path: Path,
    mesh_dir: Optional[Path] = None,
    validate_load: bool = False,
) -> Path:
    links, joints, root_link = parse_urdf(urdf_path)

    # meshdir should be relative to output xml directory
    if mesh_dir is None:
        mesh_dir = (urdf_path.parent.parent / "meshes").resolve()
    output_dir = output_path.parent.resolve()
    # robust relative path that preserves '..' when output and meshdir are not nested
    meshdir_rel = Path(os.path.relpath(str(mesh_dir), str(output_dir))).as_posix()

    mjcf = build_mjcf_tree(links, joints, root_link, meshdir_rel)
    indent(mjcf)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    ET.ElementTree(mjcf).write(output_path, encoding="utf-8", xml_declaration=False)

    if validate_load:
        import mujoco

        _ = mujoco.MjModel.from_xml_path(str(output_path))

    return output_path


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Build serialleg fidelity MJCF from URDF")
    p.add_argument(
        "--urdf",
        type=str,
        default="resources/robots/serialleg/urdf/serialleg.urdf",
        help="Input URDF path",
    )
    p.add_argument(
        "--output",
        type=str,
        default="resources/robots/serialleg/mjcf/serialleg_fidelity.xml",
        help="Output MJCF path",
    )
    p.add_argument(
        "--mesh-dir",
        type=str,
        default=None,
        help="Mesh directory (defaults to URDF sibling ../meshes)",
    )
    p.add_argument("--validate-load", action="store_true", help="Load generated MJCF with MuJoCo")
    return p.parse_args()


def main() -> None:
    args = parse_args()
    urdf_path = Path(args.urdf).resolve()
    output_path = Path(args.output).resolve()
    mesh_dir = Path(args.mesh_dir).resolve() if args.mesh_dir else None
    out = generate_serialleg_fidelity_mjcf(
        urdf_path=urdf_path,
        output_path=output_path,
        mesh_dir=mesh_dir,
        validate_load=args.validate_load,
    )
    print(f"Generated fidelity MJCF: {out}")


if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        print(f"[ERROR] {type(e).__name__}: {e}", file=sys.stderr)
        sys.exit(1)
