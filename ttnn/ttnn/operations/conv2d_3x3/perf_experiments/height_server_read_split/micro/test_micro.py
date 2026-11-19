# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0

"""height_server_read_split micro-bench: how fast do one-tile DRAM reads of the res3.x prepared weights (bf8_b, 1088 B
pages, interleaved DRAM) land with one engine vs several (two RISCs of one core on both NoCs, or several cores)?
One program per config; correctness = the read pages copied bit-exactly to an output tensor. Timing: the per-engine
zone "mb_read" and DEVICE KERNEL DURATION, via scripts/run_safe_pytest.sh --profile.

Engines are (core, risc, page_start, page_step, count, l1_tile0); "N" = NCRISC (NoC0), "B" = BRISC (NoC1).
Cores sit where the op puts its weight servers / spare cores (logical row 7 cols 9-10, row 8)."""

from pathlib import Path

import pytest
import ttnn

KDIR = Path(__file__).resolve().parent / "kernels"
TILES = 144  # res3.x weights: K = 9 * 128, N = 128 -> 36 x 4 tiles
S, H1, H2, H3 = (10, 7), (10, 8), (9, 8), (8, 8)


def _split(n, k):
    base, rem = divmod(n, k)
    out, start = [], 0
    for i in range(k):
        c = base + (1 if i < rem else 0)
        out.append((start, c))
        start += c
    return out


def _engines(name):
    if name == "1core_N_48":
        return [(S, "N", 0, 1, 48, 0)], 48
    if name == "1core_B_48":
        return [(S, "B", 0, 1, 48, 0)], 48
    if name == "1core_N_144":
        return [(S, "N", 0, 1, 144, 0)], 144
    if name == "1core_NB_24_24":
        return [(S, "N", 0, 1, 24, 0), (S, "B", 24, 1, 24, 24)], 48
    if name == "1core_NB_interleaved":
        return [(S, "N", 0, 2, 24, 0), (S, "B", 1, 2, 24, 24)], 48
    if name == "2core_N_24_24":
        return [(S, "N", 0, 1, 24, 0), (H1, "N", 24, 1, 24, 0)], 48
    if name == "2core_N_same48":
        return [(S, "N", 0, 1, 48, 0), (H1, "N", 0, 1, 48, 0)], 48
    if name == "3core_N_16":
        return [(c, "N", s, 1, n, 0) for c, (s, n) in zip((S, H1, H2), _split(48, 3))], 48
    if name == "4core_N_12":
        return [(c, "N", s, 1, n, 0) for c, (s, n) in zip((S, H1, H2, H3), _split(48, 4))], 48
    if name == "2core_NB_12x4":
        return [
            (S, "N", 0, 1, 12, 0),
            (S, "B", 12, 1, 12, 12),
            (H1, "N", 24, 1, 12, 0),
            (H1, "B", 36, 1, 12, 12),
        ], 48
    raise ValueError(name)


CONFIGS = [
    "1core_N_48",
    "1core_B_48",
    "1core_N_144",
    "1core_NB_24_24",
    "1core_NB_interleaved",
    "2core_N_24_24",
    "2core_N_same48",
    "3core_N_16",
    "4core_N_12",
    "2core_NB_12x4",
]


@pytest.mark.parametrize("chunk", [16, 8])
@pytest.mark.parametrize("config", CONFIGS)
def test_micro(config, chunk, device):
    import torch

    torch.manual_seed(0)
    x = torch.randn(1, 1, TILES // 4 * 32, 128).to(torch.bfloat16)
    dram = ttnn.DRAM_MEMORY_CONFIG
    tin = ttnn.from_torch(x, dtype=ttnn.bfloat8_b, layout=ttnn.TILE_LAYOUT, device=device, memory_config=dram)
    tout = ttnn.from_torch(
        torch.zeros_like(x), dtype=ttnn.bfloat8_b, layout=ttnn.TILE_LAYOUT, device=device, memory_config=dram
    )
    engines, covered = _engines(config)
    tile_bytes = tin.buffer_aligned_page_size()
    cores = sorted({c for c, *_ in engines})
    crs = ttnn.CoreRangeSet([ttnn.CoreRange(ttnn.CoreCoord(*c), ttnn.CoreCoord(*c)) for c in cores])
    cb = ttnn.CBDescriptor(
        total_size=TILES * tile_bytes,
        core_ranges=crs,
        format_descriptors=[ttnn.CBFormatDescriptor(buffer_index=0, data_format=ttnn.bfloat8_b, page_size=tile_bytes)],
    )
    ct = [0, chunk, 0] + list(ttnn.TensorAccessorArgs(tin).get_compile_time_args())
    ct += list(ttnn.TensorAccessorArgs(tout).get_compile_time_args())
    kernels = []
    for risc, cfg in (("N", ttnn.ReaderConfigDescriptor()), ("B", ttnn.WriterConfigDescriptor())):
        rt = ttnn.RuntimeArgs()
        for c in cores:
            rt[c[0]][c[1]] = [tin.buffer_address(), tout.buffer_address(), 0, 0, 1, 0]
        for c, r, start, step, count, l1t in engines:
            if r == risc:
                rt[c[0]][c[1]] = [tin.buffer_address(), tout.buffer_address(), count, start, step, l1t]
        kernels.append(
            ttnn.KernelDescriptor(
                kernel_source=str(KDIR / "mb_read.cpp"),
                core_ranges=crs,
                compile_time_args=ct,
                runtime_args=rt,
                config=cfg,
            )
        )
    desc = ttnn.ProgramDescriptor(kernels=kernels, semaphores=[], cbs=[cb])
    ttnn.generic_op([tin, tout], desc)
    got = ttnn.to_torch(tout)
    ref = ttnn.to_torch(tin)
    rows = covered // 4 * 32
    assert torch.equal(got[..., :rows, :], ref[..., :rows, :]), f"{config}: copy mismatch"
