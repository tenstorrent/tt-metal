# SPDX-FileCopyrightText: © 2026 Tenstorrent Inc.
# SPDX-License-Identifier: Apache-2.0
"""Bake-off candidate: bf16 ROW_MAJOR -> TILE with the reorder on the Tensix (unpack-tilize -> MOVA2D
-> pack). Bit-exact only with the two CBs declared UInt16 ("-u16"): a Float16_b path flushes exponent-0
patterns to +0 and NaN payloads to +-inf even with the Src zero flag pinned to preserve (see
kernels/tt_tilize_compute.cpp). Self-contained: it plugs into the op only through monkeypatching
(install()), never edits it.

Recommended: cand-fast-none-w0-b16-d2-u16 (== cand-opk-none-w0-b16-d2-u16, the op's own R6 compute
kernel): R6's reader schedule without the bfp8 conditioning, compute_kernel_lib::tilize (standard tilize
on UInt16), the op's tile writer with split_reorder off, block = one tile-row per handoff.

Plan: the op's own compute_plan with the bf16 RM->TILE regime forced to the packer_tilize plan (R6's
walk, L1 budget and CB sizes, with a 2048-B bf16 output tile). Program: kernels/ of this dir
(tt_row_reader.cpp, tt_tilize_compute.cpp) plus the op's tile writer (copied, split_reorder off).

Variant name: cand-<impl>-<zero>-w<read_window>-b<blocks_per_core>-d<depth>
    impl   fast (compute_kernel_lib::tilize: BH fast tilize on Float16_b CBs, standard tilize on UInt16)
           | slow (compute API tilize) | fastapi (compute API fast tilize) | copydiag | live | tile | opk
    zero   raw (ckernel::math::_configure_preserve_zero_flag_state_) | copy (copy_init side effect) |
           late (raw, after the tilize init) | none
    u16    both CBs UInt16 (bit-exact); without it Float16_b (not bit-exact)
    a<n>   ablation bits (1: no reads, 2: no writes); g1: writer waits / writes tile by tile
    w      0: R6 schedule (barrier per block); n >= 1: trid ring, n tile-rows in flight, push per tile-row
    b      PIPELINE_BLOCKS_PER_CORE of the plan (sets block_rows_tiles)
    d      TILE_BLOCK_DEPTH (blocks resident in cb_row_stage)
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import ttnn
import ttnn.operations.g6_to_layout  # noqa: F401  (registers the package)
import ttnn.operations.g6_to_layout.g6_to_layout_program_descriptor as pd
import sys

op_mod = sys.modules["ttnn.operations.g6_to_layout.g6_to_layout"]

HERE = Path(__file__).resolve().parent
KERNELS = HERE / "kernels"

_ZERO = {"none": 0, "raw": 1, "copy": 2, "late": 3}
_IMPL = {"fast": 0, "slow": 1, "fastapi": 2, "copydiag": 3, "live": 4, "tile": 5, "opk": 6}


@dataclass(frozen=True)
class Cfg:
    impl: str = "fast"
    zero: str = "raw"
    read_window: int = 2
    blocks_per_core: int = pd.PIPELINE_BLOCKS_PER_CORE
    depth: int = pd.TILE_BLOCK_DEPTH
    u16: bool = False  # CBs declared UInt16: the Tensix moves raw 16-bit integers, no float semantics
    ablate: int = 0  # ablation only: bit 0 = reader skips its reads, bit 1 = writer skips its writes
    granular: int = 0  # writer waits and writes tile by tile (g1)

    @staticmethod
    def parse(name):
        # cand-fast-raw-w2-b16-d2[-u16]
        parts = name.split("-")
        assert parts[0] == "cand", name
        cfg = {"impl": parts[1], "zero": parts[2]}
        assert cfg["impl"] in _IMPL and cfg["zero"] in _ZERO, name
        for p in parts[3:]:
            if p == "u16":
                cfg["u16"] = True
                continue
            key, val = p[0], int(p[1:])
            cfg[{"w": "read_window", "b": "blocks_per_core", "d": "depth", "a": "ablate", "g": "granular"}[key]] = val
        return Cfg(**cfg)


class CandidateProgram:
    def __init__(self, plan, input_tensor, output_tensor, cfg):
        g = plan.geometry
        all_cores = plan.all_cores
        geometry_ct_args = [
            g.tensor_cols_tiles,
            g.image_rows_tiles,
            g.height,
            g.width,
            plan.block_rows_tiles,
            plan.block_cols_tiles,
            plan.tile_row_stage_rows,
        ]
        assert plan.tile_row_stage_rows == pd.TILE_HW
        stage_bytes = plan.cb_bytes[pd.CB_ROW_STAGE]
        tile_row_bytes = plan.block_cols_tiles * pd.TILE_BYTES
        assert stage_bytes % tile_row_bytes == 0
        stage_tile_rows = stage_bytes // tile_row_bytes
        read_window = min(cfg.read_window, stage_tile_rows)
        # The CB format is only how the Tensix reads the bytes: a UInt16 CB holds the same bf16 bytes.
        cb_format = ttnn.uint16 if cfg.u16 else ttnn.bfloat16
        cbs = [
            pd._cb(pd.CB_ROW_STAGE, stage_bytes, pd.TILE_BYTES, all_cores, cb_format),
            pd._cb(pd.CB_TILE_BLOCK, plan.cb_bytes[pd.CB_TILE_BLOCK], plan.out_tile_bytes, all_cores, cb_format),
        ]
        reader_ct = geometry_ct_args + [pd.CB_ROW_STAGE, read_window, stage_tile_rows]
        reader_ct += ttnn.TensorAccessorArgs(input_tensor).get_compile_time_args()
        writer_ct = geometry_ct_args + [pd.CB_TILE_BLOCK, pd.CB_ROW_STAGE, 0, pd.ROW_WINDOW_DEPTH]
        writer_ct += ttnn.TensorAccessorArgs(output_tensor).get_compile_time_args()
        compute_ct = geometry_ct_args + [pd.CB_ROW_STAGE, pd.CB_TILE_BLOCK]

        rt = ttnn.RuntimeArgs()
        for (x, y), args in plan.core_args:
            rt[x][y] = list(args)
        self._reader = ttnn.KernelDescriptor(
            kernel_source=str(KERNELS / "tt_row_reader.cpp"),
            core_ranges=all_cores,
            compile_time_args=reader_ct,
            runtime_args=rt,
            common_runtime_args=[input_tensor.buffer_address()],
            defines=[("TT_ABLATE", str(cfg.ablate))],
            config=ttnn.ReaderConfigDescriptor(),
        )
        self._writer = ttnn.KernelDescriptor(
            kernel_source=str(KERNELS / "g6_to_layout_tile_writer.cpp"),
            core_ranges=all_cores,
            compile_time_args=writer_ct,
            runtime_args=rt,
            common_runtime_args=[output_tensor.buffer_address()],
            defines=[("TT_ABLATE", str(cfg.ablate)), ("TT_WRITER_GRANULAR", str(cfg.granular))],
            config=ttnn.WriterConfigDescriptor(),
        )
        # The op's fixed compute config (HiFi4, 16-bit DEST, no approx): not a knob of this bake-off.
        self._compute = ttnn.KernelDescriptor(
            # opk: the op's own R6 tilize compute kernel, verbatim (kernels/op_tilize_compute.cpp is a copy).
            kernel_source=str(KERNELS / ("op_tilize_compute.cpp" if cfg.impl == "opk" else "tt_tilize_compute.cpp")),
            core_ranges=all_cores,
            compile_time_args=compute_ct,
            defines=[("TT_ZERO_FLAG", str(_ZERO[cfg.zero])), ("TT_TILIZE_IMPL", str(_IMPL[cfg.impl]))],
            runtime_args=rt,
            config=ttnn.ComputeConfigDescriptor(
                math_fidelity=ttnn.MathFidelity.HiFi4,
                fp32_dest_acc_en=False,
                dst_full_sync_en=False,
                math_approx_mode=False,
            ),
        )
        self._kernels = [self._reader, self._writer, self._compute]
        self._descriptor = ttnn.ProgramDescriptor(kernels=self._kernels, semaphores=[], cbs=cbs)

    def descriptor_for(self, input_tensor, output_tensor):
        self._reader.common_runtime_args = [input_tensor.buffer_address()]
        self._writer.common_runtime_args = [output_tensor.buffer_address()]
        self._descriptor.kernels = self._kernels
        return self._descriptor


def clear_caches():
    op_mod._PLAN_CACHE.clear()
    op_mod._PROGRAM_CACHE.clear()


def install(monkeypatch, cfg):
    """Route bf16 RM -> TILE calls of g6_to_layout through the candidate program (bf8b output and every
    TILE input keep the op's own path)."""
    original_select = pd.select_regime
    original_program = pd.G6ToLayoutProgram

    def select_regime(geometry, dtype, sharded, to_tile, out_dtype, out_alignment):
        if to_tile and out_dtype == ttnn.bfloat16:
            return pd.PACKER_TILIZE
        return original_select(geometry, dtype, sharded, to_tile, out_dtype, out_alignment)

    def program(plan, input_tensor, output_tensor):
        if plan.regime == pd.PACKER_TILIZE and output_tensor.dtype == ttnn.bfloat16:
            return CandidateProgram(plan, input_tensor, output_tensor, cfg)
        return original_program(plan, input_tensor, output_tensor)

    monkeypatch.setattr(pd, "select_regime", select_regime)
    monkeypatch.setattr(pd, "PIPELINE_BLOCKS_PER_CORE", cfg.blocks_per_core)
    monkeypatch.setattr(pd, "TILE_BLOCK_DEPTH", cfg.depth)
    monkeypatch.setattr(op_mod, "G6ToLayoutProgram", program)
    clear_caches()
