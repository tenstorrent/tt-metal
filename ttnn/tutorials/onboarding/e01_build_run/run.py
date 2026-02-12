# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC

# SPDX-License-Identifier: Apache-2.0

"""
E01: Build and Run

Learn to build tt-metal and run a simple ttnn operation.

Usage:
    python e01_build_run/run.py
"""

import torch
import ttnn


def main():
    # Open device
    device = ttnn.open_device(device_id=0)

    # Create input tensors
    a = torch.tensor([[1, 2], [3, 4]], dtype=torch.float32)
    b = torch.tensor([[10, 20], [30, 40]], dtype=torch.float32)

    print(f"a = {a.tolist()}")
    print(f"b = {b.tolist()}")

    # Convert to ttnn tensors
    tt_a = ttnn.from_torch(a, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=device)
    tt_b = ttnn.from_torch(b, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=device)

    # Run add on device
    tt_result = ttnn.add(tt_a, tt_b)

    # Convert back to torch
    result = ttnn.to_torch(tt_result)

    print(f"a + b = {result.tolist()}")

    # Close device
    ttnn.close_device(device)

    print("Success!")


if __name__ == "__main__":
    main()
