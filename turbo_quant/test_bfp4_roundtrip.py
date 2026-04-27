#!/usr/bin/env python3
"""Verify BFP4 round-trip for integer values 0-7 (TQ index encoding).

If BFP4 doesn't preserve integer values 0-7 exactly, the kernel's
unary_ge_tile comparisons against integer thresholds would gather wrong
centroids, leading to corrupt output.
"""
import sys

import torch
import ttnn

sys.path.insert(0, "/localdev/mtairum/tt-metal")


def main():
    device = ttnn.open_device(device_id=0)
    try:
        # Test 1: simple block of mixed integer values 0-7.
        x = torch.zeros(1, 1, 32, 32, dtype=torch.bfloat16)
        for r in range(32):
            x[0, 0, r] = torch.tensor([0.0, 1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0] * 4)

        # Round-trip: bf16 -> bfp4 -> bf16 (via .float()).
        x_dev = ttnn.from_torch(x, dtype=ttnn.bfloat4_b, layout=ttnn.TILE_LAYOUT, device=device)
        x_back = ttnn.to_torch(x_dev).float()
        ttnn.deallocate(x_dev)

        diff = (x_back - x.float()).abs()
        print(f"Test 1 (sequential 0-7 repeated):")
        print(f"  max abs diff: {diff.max().item():.6f}")
        print(f"  mean abs diff: {diff.mean().item():.6f}")
        print(f"  unique original values: {sorted(x.float().unique().tolist())}")
        print(f"  unique recovered values: {sorted(x_back.unique().tolist())}")
        print()

        # Test 2: mixed integers including 0s (mimics partially-filled cache).
        torch.manual_seed(0)
        x = torch.zeros(1, 1, 32, 32, dtype=torch.bfloat16)
        # Fill first 4 rows with random 0-7 integers (real data).
        x[:, :, :4] = torch.randint(0, 8, (1, 1, 4, 32)).to(torch.bfloat16)

        x_dev = ttnn.from_torch(x, dtype=ttnn.bfloat4_b, layout=ttnn.TILE_LAYOUT, device=device)
        x_back = ttnn.to_torch(x_dev).float()
        ttnn.deallocate(x_dev)

        diff = (x_back - x.float()).abs()
        print(f"Test 2 (4 rows random 0-7, 28 rows zero):")
        print(f"  max abs diff: {diff.max().item():.6f}")
        print(f"  mean abs diff: {diff.mean().item():.6f}")
        print(f"  unique recovered values (filled rows): {sorted(x_back[:, :, :4].unique().tolist())}")
        print(f"  unique recovered values (zero rows): {sorted(x_back[:, :, 4:].unique().tolist())}")
        print()

        # Test 3: single non-zero value at row 4 (mimics writing position 4).
        x = torch.zeros(1, 1, 32, 32, dtype=torch.bfloat16)
        x[:, :, 4, :] = torch.tensor([0.0, 1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0] * 4).to(torch.bfloat16)

        x_dev = ttnn.from_torch(x, dtype=ttnn.bfloat4_b, layout=ttnn.TILE_LAYOUT, device=device)
        x_back = ttnn.to_torch(x_dev).float()
        ttnn.deallocate(x_dev)

        diff = (x_back - x.float()).abs()
        print(f"Test 3 (only row 4 has data 0-7):")
        print(f"  max abs diff: {diff.max().item():.6f}")
        print(f"  row 4 original: {x[0, 0, 4, :8].tolist()}")
        print(f"  row 4 recovered: {x_back[0, 0, 4, :8].tolist()}")
        print(f"  row 0 (zero) recovered: {x_back[0, 0, 0, :8].tolist()}")
        print()
    finally:
        ttnn.close_device(device)


if __name__ == "__main__":
    main()
