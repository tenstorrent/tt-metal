# SPDX-FileCopyrightText: 2026 Tenstorrent Inc.
# SPDX-License-Identifier: Apache-2.0

"""w_stream micro: aggregate W DRAM read rate for the N-group senders' pattern, by NoC split.
MICRO_CASES: comma list of case names (see CASES)."""

import os
from pathlib import Path

import pytest
import ttnn

KERNEL = str(Path(__file__).parent / "kernels/wread.cpp")
K_BLOCK, N_TILES, WIDTH, ROUNDS = 4, 8, 72, 6
TILE_BYTES = 2048
BLOCK = K_BLOCK * N_TILES


def diag(n):  # transposed-fold W senders (n, n)
    return (n, n)


def shift(n):  # second sender of the same N-group (row y = n), 6 cores along
    return ((n + 6) % 11, n)


# case -> list of (core, group, risc, lo, hi)
CASES = {
    "n1_full": [(diag(g), g, "w", 0, BLOCK) for g in range(9)],
    "n0_full": [(diag(g), g, "r", 0, BLOCK) for g in range(9)],
    "dual_same": [(diag(g), g, "w", 0, BLOCK // 2) for g in range(9)]
    + [(diag(g), g, "r", BLOCK // 2, BLOCK) for g in range(9)],
    "n1_two": [(diag(g), g, "w", 0, BLOCK // 2) for g in range(9)]
    + [(shift(g), g, "w", BLOCK // 2, BLOCK) for g in range(9)],
    "dual_two": [(diag(g), g, "w", 0, BLOCK // 2) for g in range(9)]
    + [(shift(g), g, "r", BLOCK // 2, BLOCK) for g in range(9)],
    "n0_two": [(diag(g), g, "r", 0, BLOCK // 2) for g in range(9)]
    + [(shift(g), g, "r", BLOCK // 2, BLOCK) for g in range(9)],
    "row9_n0": [((g, 9), g, "r", 0, BLOCK) for g in range(9)],
    "row9_n1": [((g, 9), g, "w", 0, BLOCK) for g in range(9)],
    "row9_dual": [((g, 9), g, "r", 0, BLOCK // 2) for g in range(9)]
    + [((g, 9), g, "w", BLOCK // 2, BLOCK) for g in range(9)],
    "shift1_n0": [(((g + 1) % 11, g), g, "r", 0, BLOCK) for g in range(9)],
    "shift1_n1": [(((g + 1) % 11, g), g, "w", 0, BLOCK) for g in range(9)],
}
NAMES = os.environ.get("MICRO_CASES", ",".join(CASES)).split(",")


@pytest.mark.parametrize("case", NAMES)
def test_micro(device, case):
    import torch

    w = ttnn.from_torch(
        torch.randn(768, 2304),
        dtype=ttnn.bfloat16,
        layout=ttnn.TILE_LAYOUT,
        device=device,
        memory_config=ttnn.DRAM_MEMORY_CONFIG,
    )
    out = ttnn.from_torch(
        torch.zeros(32, 32),
        dtype=ttnn.bfloat16,
        layout=ttnn.TILE_LAYOUT,
        device=device,
        memory_config=ttnn.DRAM_MEMORY_CONFIG,
    )
    entries = CASES[case]
    cores = sorted({c for c, *_ in entries})
    crs = ttnn.CoreRangeSet([ttnn.CoreRange(ttnn.CoreCoord(*c), ttnn.CoreCoord(*c)) for c in cores])
    cb = ttnn.CBDescriptor(
        total_size=BLOCK * TILE_BYTES,
        core_ranges=crs,
        format_descriptors=[ttnn.CBFormatDescriptor(buffer_index=0, data_format=ttnn.bfloat16, page_size=TILE_BYTES)],
    )
    ct = [0, K_BLOCK, N_TILES, WIDTH, TILE_BYTES] + list(ttnn.TensorAccessorArgs(w).get_compile_time_args())
    kernels = []
    for risc, config in (("w", ttnn.WriterConfigDescriptor()), ("r", ttnn.ReaderConfigDescriptor())):
        mine = [e for e in entries if e[2] == risc]
        if not mine:
            continue
        rt = ttnn.RuntimeArgs()
        for core, group, _, lo, hi in mine:
            rt[core[0]][core[1]] = [w.buffer_address(), 8 * group, lo, hi, ROUNDS]
        kernels.append(
            ttnn.KernelDescriptor(
                kernel_source=KERNEL,
                core_ranges=ttnn.CoreRangeSet(
                    [ttnn.CoreRange(ttnn.CoreCoord(*e[0]), ttnn.CoreCoord(*e[0])) for e in mine]
                ),
                compile_time_args=ct,
                runtime_args=rt,
                config=config,
            )
        )
    ttnn.generic_op([w, out], ttnn.ProgramDescriptor(kernels=kernels, cbs=[cb]))
    ttnn.synchronize_device(device)
    print(f"MICRO {case}")
