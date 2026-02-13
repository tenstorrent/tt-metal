#!/usr/bin/env python3
# SPDX-FileCopyrightText: © 2026 Tenstorrent Inc.
# SPDX-License-Identifier: Apache-2.0
#
# Quick Python test to trigger PACKER_L1_ACC race via ttnn.matmul + bias.
#
# The matmul with bias fusion uses the FUSE_BIAS + PACKER_L1_ACC code path
# in bmm_large_block_zm_fused_bias_activation.cpp, which has the missing
# TTI_STALLWAIT(STALL_PACK, TRISC_CFG) in reconfigure_packer_l1_acc().
#
# Usage:
#   TT_METAL_OPERATION_TIMEOUT_SECONDS=3 python vit_n300/test_l1acc_race_matmul.py
#
# Detection methods:
#   1. Device timeout (hang) = packer FSM entered invalid state from race
#   2. PCC < 0.999 = wrong L1_ACC mode produced incorrect accumulation
#
# The race is extremely narrow (~1 in millions), so this test may need
# thousands of iterations. The custom C++ kernel test (test_packer_l1acc_race.cpp)
# is far more targeted.

import os
import sys
import time
import torch
import ttnn

# Short timeout to catch hangs quickly
os.environ.setdefault("TT_METAL_OPERATION_TIMEOUT_SECONDS", "3")


def compute_pcc(golden, actual):
    """Compute Pearson correlation coefficient."""
    g = golden.flatten().float()
    a = actual.flatten().float()
    if g.std() == 0 or a.std() == 0:
        return 1.0 if torch.allclose(g, a) else 0.0
    return torch.corrcoef(torch.stack([g, a]))[0, 1].item()


def run_matmul_l1acc_test(device, num_iters=1000):
    """
    Run matmul+bias with shapes that force PACKER_L1_ACC usage.

    For PACKER_L1_ACC to be triggered, we need:
    - Multi-core matmul with multicast (block sharded)
    - Inner dimension K split into multiple blocks
    - Bias fusion enabled
    """
    # Shapes chosen to force multiple inner dim blocks:
    # M=1024, K=1024, N=1024 on 8x8 grid
    # Per-core: ~128 rows, ~128 cols, K split into blocks
    batch = 1
    M, K, N = 1024, 1024, 1024

    # Use random inputs so PCC is meaningful
    torch.manual_seed(42)
    torch_a = torch.randn(batch, 1, M, K, dtype=torch.bfloat16)
    torch_b = torch.randn(batch, 1, K, N, dtype=torch.bfloat16)
    torch_bias = torch.randn(1, 1, 1, N, dtype=torch.bfloat16)

    # Golden: compute in float32 for reference
    golden = (torch_a.float() @ torch_b.float()) + torch_bias.float()

    # Establish baseline: run once to get the "device golden" PCC
    # (accounts for normal bfloat16 precision loss)
    a_tt = ttnn.from_torch(torch_a, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=device)
    b_tt = ttnn.from_torch(torch_b, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=device)
    bias_tt = ttnn.from_torch(torch_bias, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=device)
    baseline_result = ttnn.to_torch(ttnn.linear(a_tt, b_tt, bias=bias_tt))
    baseline_pcc = compute_pcc(golden, baseline_result)
    ttnn.deallocate(a_tt)
    ttnn.deallocate(b_tt)
    ttnn.deallocate(bias_tt)

    print(f"Running {num_iters} matmul+bias iterations (M={M}, K={K}, N={N})")
    print(f"Baseline PCC (device vs float32 golden): {baseline_pcc:.6f}")
    print(f"Golden output range: [{golden.min():.4f}, {golden.max():.4f}]")
    print(f"Detection: PCC drop > 0.01 from baseline, or device timeout")
    print()

    pcc_threshold = baseline_pcc - 0.01  # Allow small PCC variation
    min_pcc = 1.0
    errors = 0

    for i in range(num_iters):
        try:
            a_tt = ttnn.from_torch(torch_a, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=device)
            b_tt = ttnn.from_torch(torch_b, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=device)
            bias_tt = ttnn.from_torch(torch_bias, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=device)

            result = ttnn.linear(a_tt, b_tt, bias=bias_tt)
            result_torch = ttnn.to_torch(result)

            pcc = compute_pcc(golden, result_torch)
            if pcc < min_pcc:
                min_pcc = pcc

            if pcc < pcc_threshold:
                print(f"  *** ANOMALY at iteration {i}! PCC = {pcc:.6f} (baseline={baseline_pcc:.6f}) ***")
                max_err = (result_torch.float() - baseline_result.float()).abs().max().item()
                print(f"  Max deviation from baseline: {max_err:.4f}")
                errors += 1
                if errors >= 5:
                    print("  Stopping after 5 anomalies.")
                    break

            if (i + 1) % 100 == 0:
                print(f"  Iteration {i+1}/{num_iters}: min_pcc={min_pcc:.6f} (baseline={baseline_pcc:.6f})")

            # Free device tensors
            ttnn.deallocate(a_tt)
            ttnn.deallocate(b_tt)
            ttnn.deallocate(bias_tt)
            ttnn.deallocate(result)

        except Exception as e:
            print(f"\n  *** DEVICE ERROR at iteration {i}: {e} ***")
            print("  This may indicate a packer FSM hang from the L1_ACC race!")
            errors += 1
            break

    print(f"\nResults: {errors} anomalies in {num_iters} iterations. Min PCC: {min_pcc:.6f}")

    if errors == 0:
        print("No race detected. The race window is very narrow (~1 in millions).")
        print("For better coverage, use the custom C++ kernel test (test_packer_l1acc_race.cpp)")
        print("which hammers the reconfig->pack transition directly.")

    return errors == 0


def main():
    device = ttnn.open_device(device_id=0)

    try:
        num_iters = int(sys.argv[1]) if len(sys.argv) > 1 else 1000
        success = run_matmul_l1acc_test(device, num_iters)
        sys.exit(0 if success else 1)
    finally:
        ttnn.close_device(device)


if __name__ == "__main__":
    main()
