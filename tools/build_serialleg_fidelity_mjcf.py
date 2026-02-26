#!/usr/bin/env python3
"""
Generate a higher-fidelity MuJoCo MJCF from serialleg URDF.

This script is intentionally robot-specific (serialleg) but keeps the parsing/generation
pipeline explicit so the generated MJCF is reproducible and auditable.

Key decisions:
- Preserve URDF topology, joints, inertials, visuals, and collision meshes
- Default to unsimplified STL meshes (URDF visual meshes) as collision meshes
- Optionally allow URDF collision OBJ meshes or wheel cylinder fallback for comparison
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
WHEEL_COLLISION_FRICTION = "0.8 0.005 0.0001"
MUJOCO_MAX_STL_FACES = 200000
GENERATED_COLLISION_SUBDIR = "_mjc_generated_collision"

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


def mesh_file_attr_path(filename: str, kind: str) -> str:
    """Path written into MJCF <mesh file=...>. Keep subpaths for generated assets."""
    if kind == "generated":
        return filename.replace("\\", "/")
    return mesh_basename(filename)


def is_wheel_link(name: str) -> bool:
    return name in {"l_wheel_Link", "r_wheel_Link"}


def is_visual_only_geom_for_link(link_name: str) -> bool:
    # keep visuals for all links; contact control done via collision geoms and wheel cylinders
    return False


def build_asset_mesh_tables(
    links: Dict[str, LinkData],
    include_visuals: bool = False,
    extra_mesh_files: Optional[Dict[Tuple[str, int], str]] = None,
    skip_visual_files: Optional[set] = None,
) -> Dict[Tuple[str, str], str]:
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

    if extra_mesh_files is None:
        extra_mesh_files = {}
    if skip_visual_files is None:
        skip_visual_files = set()

    for link in links.values():
        if include_visuals:
            for v in link.visuals:
                if v.filename in skip_visual_files:
                    continue
                add(v.filename, "visual")
        for c in link.collisions:
            add(c.filename, "collision")
    for key in sorted(extra_mesh_files.keys()):
        add(extra_mesh_files[key], "generated")
    return used


def _is_binary_stl(stl_path: Path) -> Tuple[bool, int]:
    size = stl_path.stat().st_size
    if size < 84:
        return False, 0
    with open(stl_path, "rb") as f:
        hdr = f.read(84)
    tri_count = int(np.frombuffer(hdr[80:84], dtype=np.uint32)[0])
    expected_size = 84 + 50 * tri_count
    return (expected_size == size), tri_count


def split_binary_stl_for_mujoco(
    src_stl: Path,
    out_dir: Path,
    file_prefix: str,
    max_faces_per_chunk: int = MUJOCO_MAX_STL_FACES,
) -> List[Path]:
    """
    Split a binary STL into multiple binary STL chunks without changing triangle data.
    This preserves geometry exactly while satisfying MuJoCo's per-mesh face limit.
    """
    is_binary, tri_count = _is_binary_stl(src_stl)
    if not is_binary:
        raise ValueError(
            f"STL is not recognized as binary or size is inconsistent: {src_stl}. "
            "ASCII STL is currently not supported by this splitter."
        )
    if tri_count <= max_faces_per_chunk:
        return []

    out_dir.mkdir(parents=True, exist_ok=True)
    outputs: List[Path] = []
    with open(src_stl, "rb") as f:
        header = f.read(80)
        count_bytes = f.read(4)
        if len(count_bytes) != 4:
            raise IOError(f"Failed to read triangle count field from {src_stl}")
        file_tri_count = int(np.frombuffer(count_bytes, dtype=np.uint32)[0])
        if file_tri_count != tri_count:
            raise ValueError(
                f"Triangle count mismatch while splitting {src_stl}: "
                f"header says {file_tri_count}, expected {tri_count}"
            )
        remaining = tri_count
        chunk_idx = 0
        while remaining > 0:
            chunk_faces = min(max_faces_per_chunk, remaining)
            chunk_bytes = f.read(50 * chunk_faces)
            if len(chunk_bytes) != 50 * chunk_faces:
                raise IOError(
                    f"Unexpected EOF while splitting {src_stl}; "
                    f"expected {50 * chunk_faces} bytes, got {len(chunk_bytes)}"
                )
            out_path = out_dir / f"{file_prefix}_part{chunk_idx:03d}.stl"
            with open(out_path, "wb") as wf:
                # Keep original header bytes as prefix, annotate chunk index in-place for readability
                hdr = bytearray(header[:80])
                label = f"mjc_chunk {chunk_idx}".encode("ascii", errors="ignore")
                hdr[: len(label)] = label[:80]
                wf.write(hdr)
                wf.write(np.uint32(chunk_faces).tobytes())
                wf.write(chunk_bytes)
            outputs.append(out_path)
            remaining -= chunk_faces
            chunk_idx += 1
    return outputs


def prepare_visual_collision_mesh_overrides(
    links: Dict[str, LinkData],
    mesh_dir: Path,
) -> Dict[str, List[str]]:
    """
    For visual STL collision mode, pre-split oversized binary STLs into generated chunk STLs
    to satisfy MuJoCo's mesh face limit without simplifying geometry.

    Returns:
      map: link_name -> list of mesh filenames (relative to mesh_dir) to use as collision geoms
            only present when a visual STL was split into multiple chunk files.
    """
    overrides: Dict[str, List[str]] = {}
    generated_dir = mesh_dir / GENERATED_COLLISION_SUBDIR
    for link_name, link in links.items():
        if not link.visuals:
            continue
        # serialleg links use one visual STL per link; if multiple visuals exist, only split the first if needed.
        v = link.visuals[0]
        src = (mesh_dir / mesh_basename(v.filename)).resolve()
        if src.suffix.lower() != ".stl" or not src.exists():
            continue
        is_binary, tri_count = _is_binary_stl(src)
        if not is_binary:
            continue
        if tri_count <= MUJOCO_MAX_STL_FACES:
            continue
        safe_prefix = re.sub(r"[^a-zA-Z0-9_]+", "_", Path(src.name).stem)
        chunk_paths = split_binary_stl_for_mujoco(
            src_stl=src,
            out_dir=generated_dir,
            file_prefix=safe_prefix,
            max_faces_per_chunk=MUJOCO_MAX_STL_FACES,
        )
        rels = [str(p.relative_to(mesh_dir).as_posix()) for p in chunk_paths]
        overrides[link_name] = rels
    return overrides


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
    friction: Optional[str] = None,
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
        if friction:
            attrs["friction"] = friction
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
            "friction": WHEEL_COLLISION_FRICTION,
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
    wheel_collision_mode: str = "mesh",
    collision_mesh_source: str = "visual_stl",
    visual_collision_overrides: Optional[Dict[str, List[str]]] = None,
) -> ET.Element:
    if wheel_collision_mode not in {"mesh", "cylinder"}:
        raise ValueError(
            f"Unsupported wheel_collision_mode={wheel_collision_mode}; expected 'mesh' or 'cylinder'"
        )
    if collision_mesh_source not in {"visual_stl", "urdf_collision"}:
        raise ValueError(
            "Unsupported collision_mesh_source="
            f"{collision_mesh_source}; expected 'visual_stl' or 'urdf_collision'"
        )
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

    skip_visual_files = set()
    if collision_mesh_source == "visual_stl":
        for link_name in visual_collision_overrides.keys():
            if links[link_name].visuals:
                skip_visual_files.add(links[link_name].visuals[0].filename)

    mesh_assets = build_asset_mesh_tables(
        links,
        include_visuals=(collision_mesh_source == "visual_stl"),
        extra_mesh_files={
            (link_name, i): rel
            for link_name, rels in visual_collision_overrides.items()
            for i, rel in enumerate(rels)
        },
        skip_visual_files=skip_visual_files,
    )
    for (filename, kind), asset_name in sorted(mesh_assets.items(), key=lambda kv: kv[1]):
        ET.SubElement(
            asset_elem,
            "mesh",
            {
                "name": asset_name,
                "file": mesh_file_attr_path(filename, kind),
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

        # Collision source selection:
        # - visual_stl: use URDF visual meshes as (typically unsimplified) collision meshes
        # - urdf_collision: use URDF collision meshes (often simplified/split)
        collision_entries = []
        if collision_mesh_source == "visual_stl" and len(link.visuals) > 0:
            if link_name in visual_collision_overrides:
                # Split chunks preserve the first visual mesh origin.
                v0 = link.visuals[0]
                for rel_file in visual_collision_overrides[link_name]:
                    collision_entries.append((rel_file, "generated", v0.origin))
            else:
                for v in link.visuals:
                    collision_entries.append((v.filename, "visual", v.origin))
        else:
            for c in link.collisions:
                collision_entries.append((c.filename, "collision", c.origin))

        # Collision geoms
        if is_wheel_link(link_name):
            side = "l" if link_name.startswith("l_") else "r"
            # Keep a wheel visual proxy. Prefer visual STL (if included), then fallback to collision OBJ.
            proxy_candidates = []
            if link.visuals:
                proxy_candidates.append((link.visuals[0], "visual"))
            if link.collisions:
                proxy_candidates.append((link.collisions[0], "collision"))
            for proxy_geom, proxy_kind in proxy_candidates:
                mesh_asset_name = mesh_assets.get((proxy_geom.filename, proxy_kind))
                if mesh_asset_name is None:
                    continue
                add_geom_mesh(
                    body_elem,
                    mesh_asset_name=mesh_asset_name,
                    origin=proxy_geom.origin,
                    is_collision=False,
                    geom_name=f"{link_name}_visual_proxy",
                )
                break
            if wheel_collision_mode == "cylinder":
                add_wheel_collision_cylinder(body_elem, side)
            else:
                for idx, (mesh_file, mesh_kind, mesh_origin) in enumerate(collision_entries):
                    asset_name = mesh_assets.get((mesh_file, mesh_kind))
                    if asset_name is None:
                        continue
                    add_geom_mesh(
                        body_elem,
                        mesh_asset_name=asset_name,
                        origin=mesh_origin,
                        is_collision=True,
                        geom_name=f"{link_name}_collision_{idx}",
                        friction=WHEEL_COLLISION_FRICTION,
                    )
        else:
            for idx, (mesh_file, mesh_kind, mesh_origin) in enumerate(collision_entries):
                asset_name = mesh_assets.get((mesh_file, mesh_kind))
                if asset_name is None:
                    continue
                add_geom_mesh(
                    body_elem,
                    mesh_asset_name=asset_name,
                    origin=mesh_origin,
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
    wheel_collision_mode: str = "mesh",
    collision_mesh_source: str = "visual_stl",
) -> Path:
    links, joints, root_link = parse_urdf(urdf_path)

    # meshdir should be relative to output xml directory
    if mesh_dir is None:
        mesh_dir = (urdf_path.parent.parent / "meshes").resolve()
    output_dir = output_path.parent.resolve()
    # robust relative path that preserves '..' when output and meshdir are not nested
    meshdir_rel = Path(os.path.relpath(str(mesh_dir), str(output_dir))).as_posix()
    visual_collision_overrides = (
        prepare_visual_collision_mesh_overrides(links, mesh_dir)
        if collision_mesh_source == "visual_stl"
        else {}
    )

    mjcf = build_mjcf_tree(
        links,
        joints,
        root_link,
        meshdir_rel,
        wheel_collision_mode=wheel_collision_mode,
        collision_mesh_source=collision_mesh_source,
        visual_collision_overrides=visual_collision_overrides,
    )
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
    p.add_argument(
        "--wheel-collision-mode",
        type=str,
        choices=["mesh", "cylinder"],
        default="mesh",
        help="Wheel collision geometry representation in generated MJCF",
    )
    p.add_argument(
        "--collision-mesh-source",
        type=str,
        choices=["visual_stl", "urdf_collision"],
        default="visual_stl",
        help="Collision mesh source for link geoms (default uses unsimplified visual STL meshes)",
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
        wheel_collision_mode=args.wheel_collision_mode,
        collision_mesh_source=args.collision_mesh_source,
    )
    print(f"Generated fidelity MJCF: {out}")


if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        print(f"[ERROR] {type(e).__name__}: {e}", file=sys.stderr)
        sys.exit(1)
