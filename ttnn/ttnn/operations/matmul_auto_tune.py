# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
#
# SPDX-License-Identifier: Apache-2.0

"""Host-side matmul tuning helpers.

Thin Python shim — the algorithm lives in C++
(``ttnn::operations::matmul::auto_tune`` in
``ttnn/cpp/ttnn/operations/matmul/device/config/matmul_auto_tuner.{hpp,cpp}``)
and is bound via nanobind under ``ttnn._ttnn.operations.matmul.matmul_auto_tune``.
This module re-exports those bindings and adds the lone duck-typed convenience
``upgrade_subblock`` that mutates any program-config variant in place.

Example::

    cfg = ttnn.MatmulMultiCoreReuseMultiCastProgramConfig(
        compute_with_storage_grid_size=(8, 8),
        in0_block_w=2, per_core_M=4, per_core_N=8,
        out_subblock_h=1, out_subblock_w=1,
        transpose_mcast=False, fused_activation=None,
    )
    upgraded = ttnn.matmul_auto_tune.upgrade_subblock(cfg, fp32_dest_acc_en=False)
    # cfg is mutated in place.

    fp = ttnn.matmul_auto_tune.estimate_l1_per_core(
        per_core_M=cfg.per_core_M, per_core_N=cfg.per_core_N,
        in0_block_w=cfg.in0_block_w,
        tile_pack_row_major=getattr(cfg, "tile_pack_row_major", False),
    )
    assert fp["fits_bh"], f"Estimated {fp['estimated_bytes']} bytes overflows BH L1"
"""

from __future__ import annotations

from typing import Any, Optional

import ttnn

# Re-export C++ bindings — single source of truth for the algorithm.
_cpp = ttnn._ttnn.operations.matmul.matmul_auto_tune

dst_capacity = _cpp.dst_capacity
subblock_options = _cpp.subblock_options
largest_subblock = _cpp.largest_subblock
needs_row_major = _cpp.needs_row_major


# L1 budget table — mirrors the C++ constants for Python-side use.
L1_BUDGET_BYTES = {
    "wormhole": _cpp.L1_BUDGET_BYTES_WORMHOLE,
    "blackhole": _cpp.L1_BUDGET_BYTES_BLACKHOLE,
}


def estimate_l1_per_core(
    program_config: Any,
    *,
    in0_tile_bytes: int = 2048,
    in1_tile_bytes: int = 2048,
    out_tile_bytes: int = 2048,
    interm_tile_bytes: Optional[int] = None,
    fuse_bias: bool = False,
    num_buffered_blocks: int = 2,
) -> dict:
    """Estimate per-core L1 footprint for a matmul program config.

    Reads ``per_core_M`` / ``per_core_N`` / ``in0_block_w`` and (when present)
    ``tile_pack_row_major`` off the passed config object. Delegates to the C++
    binding for the byte math.
    """
    return _cpp.estimate_l1_per_core(
        per_core_M=int(program_config.per_core_M),
        per_core_N=int(program_config.per_core_N),
        in0_block_w=int(program_config.in0_block_w),
        tile_pack_row_major=bool(getattr(program_config, "tile_pack_row_major", False)),
        fuse_bias=fuse_bias,
        in0_tile_bytes=in0_tile_bytes,
        in1_tile_bytes=in1_tile_bytes,
        out_tile_bytes=out_tile_bytes,
        interm_tile_bytes=interm_tile_bytes if interm_tile_bytes is not None else 0,
        num_buffered_blocks=num_buffered_blocks,
    )


def _capacity_flags_from(compute_kernel_config: Any) -> tuple[bool, bool]:
    """Extract (fp32_dest_acc_en, dst_full_sync_en) from a ttnn compute kernel config.

    Falls back to (False, True) — the default WH/BH bf16 full-sync DST layout — when
    an attribute is absent (older config types or test stubs).
    """
    fp32 = bool(getattr(compute_kernel_config, "fp32_dest_acc_en", False))
    full_sync = bool(getattr(compute_kernel_config, "dst_full_sync_en", True))
    return fp32, full_sync


def upgrade_subblock(
    program_config: Any,
    fp32_dest_acc_en: bool = False,
    *,
    dst_full_sync_en: bool = True,
    enable_tile_pack_row_major: bool = True,
    require_legacy_writer: bool = False,
    compute_kernel_config: Optional[Any] = None,
) -> Any:
    """Mutate ``program_config`` to use the largest legal (h, w) and return it.

    Reads ``per_core_M`` / ``per_core_N`` from the config, picks the best
    subblock via the C++ binding, and writes ``out_subblock_h`` /
    ``out_subblock_w`` back. If the chosen pair requires the row-major writer
    (and ``enable_tile_pack_row_major`` is True), also flips
    ``tile_pack_row_major`` on. Pass ``require_legacy_writer=True`` to force a
    legacy-compatible pick (no rmo).

    When ``compute_kernel_config`` is supplied, its ``fp32_dest_acc_en`` and
    ``dst_full_sync_en`` attributes are used in preference to the keyword
    arguments — explicit kwargs override when both are given.

    Stays in Python because it mutates a program_config variant in place via
    duck typing — the C++ side can't generically mutate a variant from Python.

    Returns the same config object for chaining.
    """
    if compute_kernel_config is not None:
        fp32_from_ckc, full_sync_from_ckc = _capacity_flags_from(compute_kernel_config)
        if fp32_dest_acc_en is False:
            fp32_dest_acc_en = fp32_from_ckc
        if dst_full_sync_en is True:
            dst_full_sync_en = full_sync_from_ckc
    pm = int(program_config.per_core_M)
    pn = int(program_config.per_core_N)
    h, w = largest_subblock(
        pm,
        pn,
        fp32_dest_acc_en,
        dst_full_sync_en,
        require_legacy_writer,
    )
    program_config.out_subblock_h = h
    program_config.out_subblock_w = w
    if enable_tile_pack_row_major and needs_row_major(h, w, pn):
        if hasattr(program_config, "tile_pack_row_major"):
            program_config.tile_pack_row_major = True
    return program_config


__all__ = [
    "L1_BUDGET_BYTES",
    "dst_capacity",
    "subblock_options",
    "largest_subblock",
    "needs_row_major",
    "upgrade_subblock",
    "estimate_l1_per_core",
]
