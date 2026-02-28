#!/usr/bin/env python3
"""Compare inertia properties between URDF and MJCF files."""

import xml.etree.ElementTree as ET
import numpy as np
from scipy.spatial.transform import Rotation

def parse_urdf_inertia(urdf_path):
    """Extract inertia information from URDF."""
    tree = ET.parse(urdf_path)
    root = tree.getroot()

    inertias = {}
    for link in root.findall('link'):
        link_name = link.get('name')
        inertial = link.find('inertial')
        if inertial is not None:
            mass_elem = inertial.find('mass')
            inertia_elem = inertial.find('inertia')
            origin_elem = inertial.find('origin')

            if mass_elem is not None and inertia_elem is not None:
                mass = float(mass_elem.get('value'))

                # Full inertia matrix
                ixx = float(inertia_elem.get('ixx'))
                ixy = float(inertia_elem.get('ixy'))
                ixz = float(inertia_elem.get('ixz'))
                iyy = float(inertia_elem.get('iyy'))
                iyz = float(inertia_elem.get('iyz'))
                izz = float(inertia_elem.get('izz'))

                inertia_matrix = np.array([
                    [ixx, ixy, ixz],
                    [ixy, iyy, iyz],
                    [ixz, iyz, izz]
                ])

                # COM position
                if origin_elem is not None:
                    xyz = origin_elem.get('xyz', '0 0 0').split()
                    com = np.array([float(x) for x in xyz])
                else:
                    com = np.zeros(3)

                inertias[link_name] = {
                    'mass': mass,
                    'inertia_matrix': inertia_matrix,
                    'com': com
                }

    return inertias

def parse_mjcf_inertia(mjcf_path):
    """Extract inertia information from MJCF."""
    tree = ET.parse(mjcf_path)
    root = tree.getroot()

    inertias = {}
    for body in root.iter('body'):
        body_name = body.get('name')
        inertial = body.find('inertial')
        if inertial is not None:
            mass = float(inertial.get('mass'))

            # Diagonal inertia in principal axes
            diag_str = inertial.get('diaginertia')
            diag = np.array([float(x) for x in diag_str.split()])

            # Quaternion for principal axes orientation
            quat_str = inertial.get('quat', '1 0 0 0')
            quat_xyzw = np.array([float(x) for x in quat_str.split()])
            # MuJoCo uses w,x,y,z order
            quat_wxyz = np.array([quat_xyzw[0], quat_xyzw[1], quat_xyzw[2], quat_xyzw[3]])

            # COM position
            pos_str = inertial.get('pos')
            com = np.array([float(x) for x in pos_str.split()])

            # Reconstruct full inertia matrix
            # I = R * diag(d1, d2, d3) * R^T
            rot = Rotation.from_quat([quat_wxyz[1], quat_wxyz[2], quat_wxyz[3], quat_wxyz[0]])  # scipy uses xyzw
            R = rot.as_matrix()
            inertia_matrix = R @ np.diag(diag) @ R.T

            inertias[body_name] = {
                'mass': mass,
                'inertia_matrix': inertia_matrix,
                'diaginertia': diag,
                'quat': quat_wxyz,
                'com': com
            }

    return inertias

def compare_inertias(urdf_path, mjcf_path):
    """Compare inertia properties between URDF and MJCF."""
    urdf_inertias = parse_urdf_inertia(urdf_path)
    mjcf_inertias = parse_mjcf_inertia(mjcf_path)

    print("=" * 80)
    print("INERTIA COMPARISON: URDF vs MJCF")
    print("=" * 80)

    for link_name in urdf_inertias:
        if link_name not in mjcf_inertias:
            print(f"\n⚠️  Link '{link_name}' not found in MJCF")
            continue

        urdf = urdf_inertias[link_name]
        mjcf = mjcf_inertias[link_name]

        print(f"\n{'='*80}")
        print(f"Link: {link_name}")
        print(f"{'='*80}")

        # Compare mass
        mass_diff = abs(urdf['mass'] - mjcf['mass'])
        mass_rel_err = mass_diff / urdf['mass'] * 100 if urdf['mass'] > 0 else 0
        print(f"\nMass:")
        print(f"  URDF: {urdf['mass']:.6f} kg")
        print(f"  MJCF: {mjcf['mass']:.6f} kg")
        print(f"  Diff: {mass_diff:.6e} ({mass_rel_err:.3f}%)")

        # Compare COM
        com_diff = np.linalg.norm(urdf['com'] - mjcf['com'])
        print(f"\nCenter of Mass:")
        print(f"  URDF: [{urdf['com'][0]:+.6f}, {urdf['com'][1]:+.6f}, {urdf['com'][2]:+.6f}]")
        print(f"  MJCF: [{mjcf['com'][0]:+.6f}, {mjcf['com'][1]:+.6f}, {mjcf['com'][2]:+.6f}]")
        print(f"  Diff: {com_diff:.6e} m")

        # Compare inertia matrices
        inertia_diff = np.linalg.norm(urdf['inertia_matrix'] - mjcf['inertia_matrix'], 'fro')
        inertia_rel_err = inertia_diff / np.linalg.norm(urdf['inertia_matrix'], 'fro') * 100

        print(f"\nInertia Matrix:")
        print(f"  URDF:")
        for row in urdf['inertia_matrix']:
            print(f"    [{row[0]:+.6e}, {row[1]:+.6e}, {row[2]:+.6e}]")

        print(f"  MJCF (reconstructed from diag + quat):")
        for row in mjcf['inertia_matrix']:
            print(f"    [{row[0]:+.6e}, {row[1]:+.6e}, {row[2]:+.6e}]")

        print(f"  Frobenius norm diff: {inertia_diff:.6e} ({inertia_rel_err:.3f}%)")

        # Show MJCF principal axes representation
        print(f"\n  MJCF principal axes:")
        print(f"    diaginertia: [{mjcf['diaginertia'][0]:.6e}, {mjcf['diaginertia'][1]:.6e}, {mjcf['diaginertia'][2]:.6e}]")
        print(f"    quat (wxyz): [{mjcf['quat'][0]:+.6f}, {mjcf['quat'][1]:+.6f}, {mjcf['quat'][2]:+.6f}, {mjcf['quat'][3]:+.6f}]")

        # Check if difference is significant
        if inertia_rel_err > 1.0:  # More than 1% error
            print(f"\n  ⚠️  WARNING: Inertia mismatch > 1%!")

if __name__ == "__main__":
    urdf_path = "/home/am345/Wheel-Legged-Gym/resources/robots/serialleg/urdf/serialleg.urdf"
    mjcf_path = "/home/am345/Wheel-Legged-Gym/resources/robots/serialleg/mjcf/serialleg_fidelity.xml"

    compare_inertias(urdf_path, mjcf_path)
