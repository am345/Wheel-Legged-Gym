#!/usr/bin/env python3
"""检查checkpoint文件内容，特别是观测归一化器"""

import sys
import torch
from pathlib import Path

if len(sys.argv) < 2:
    print("Usage: python check_checkpoint.py <checkpoint_path>")
    sys.exit(1)

checkpoint_path = Path(sys.argv[1])
if not checkpoint_path.exists():
    print(f"Error: Checkpoint not found: {checkpoint_path}")
    sys.exit(1)

print(f"Loading checkpoint: {checkpoint_path}")
checkpoint = torch.load(checkpoint_path, map_location='cpu')

print("\n" + "="*80)
print("Checkpoint Keys:")
print("="*80)
for key in sorted(checkpoint.keys()):
    value = checkpoint[key]
    if isinstance(value, dict):
        print(f"  {key}: dict with {len(value)} items")
        if len(value) <= 10:
            for subkey in sorted(value.keys()):
                subvalue = value[subkey]
                if hasattr(subvalue, 'shape'):
                    print(f"    - {subkey}: tensor {subvalue.shape}")
                else:
                    print(f"    - {subkey}: {type(subvalue).__name__}")
    elif hasattr(value, 'shape'):
        print(f"  {key}: tensor {value.shape}")
    else:
        print(f"  {key}: {type(value).__name__} = {value}")

print("\n" + "="*80)
print("Observation Normalizer Check:")
print("="*80)

# 检查各种可能的归一化器键名
normalizer_keys = ['obs_normalizer', 'normalizer', 'obs_mean', 'obs_std',
                   'observation_normalizer', 'empirical_normalization']

found_normalizer = False
for key in normalizer_keys:
    if key in checkpoint:
        found_normalizer = True
        print(f"✓ Found: {key}")
        value = checkpoint[key]
        if isinstance(value, dict):
            print(f"  Type: dict with keys: {list(value.keys())}")
            if '_mean' in value and '_std' in value:
                print(f"  Mean shape: {value['_mean'].shape}")
                print(f"  Std shape: {value['_std'].shape}")
                print(f"  Mean sample: {value['_mean'].squeeze()[:5]}")
                print(f"  Std sample: {value['_std'].squeeze()[:5]}")
        elif hasattr(value, 'shape'):
            print(f"  Shape: {value.shape}")

if not found_normalizer:
    print("✗ No observation normalizer found in checkpoint")
    print("\nThis means the policy was trained WITHOUT observation normalization,")
    print("or the normalizer was not saved to the checkpoint.")
