# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
#
# SPDX-License-Identifier: Apache-2.0

"""Host-side matmul tuning helpers.

Mirror of the C++ auto-tuner at
``ttnn/cpp/ttnn/operations/matmul/device/config/matmul_auto_tuner.{hpp,cpp}``.
Useful for callers that already have a hand-tuned ``program_config`` and want
to:

* upgrade the subblock dims to the largest legal pair (``upgrade_subblock``),
* estimate the per-core L1 footprint before submitting (``estimate_l1_per_core``),
* enumerate which subblock pairs are legal for a given shape
  (``subblock_options``).

The C++ tuner only fires when no ``program_config`` is passed to ``ttnn.matmul``.
This module fills the gap for explicit-config callers (per-model program-config
objects) and for offline analysis.

Example::

    cfg = ttnn.MatmulMultiCoreReuseMultiCastProgramConfig(
        compute_with_storage_grid_size=(8, 8),
        in0_block_w=2, per_core_M=4, per_core_N=8,
        out_subblock_h=1, out_subblock_w=1,
        transpose_mcast=False, fused_activation=None,
    )
    upgraded = ttnn.matmul_auto_tune.upgrade_subblock(cfg, fp32_dest_acc_en=False)
    # cfg is mutated in place: out_subblock_h=1, out_subblock_w=8 (volume 1 -> 8)

    # Pre-flight L1 check
    fp = ttnn.matmul_auto_tune.estimate_l1_per_core(
        cfg, in0_tile_bytes=2048, in1_tile_bytes=2048, out_tile_bytes=2048,
    )
    assert fp["fits_bh"], f"Estimated {fp['estimated_bytes']} bytes overflows BH L1"
"""

from __future__ import annotations

from typing import Any, Optional


# Approximate per-core L1 budget after subtracting reserved regions (kernel binaries,
# semaphores, etc.). Values match the auto-tuner's adaptive-margin defaults; treat
# them as floors rather than hard limits.
L1_BUDGET_BYTES = {
    "wormhole": 1_400_000,
    "blackhole": 1_500_000,
}


def dst_capacity(fp32_dest_acc_en: bool, dst_full_sync_en: bool = True) -> int:
    """Return the number of DST tiles available for a single matmul subblock.

    Mirrors :cpp:func:`ttnn::get_dest_reg_count` for the standard 32x32 tile shape.
    Half-sync (dst_full_sync_en=False) doubles the accessible DST tile count;
    fp32 accumulation halves it. WH and BH agree on these values today.
    """
    base = 8 if dst_full_sync_en else 16
    if fp32_dest_acc_en:
        base //= 2
    return base


def subblock_options(
    per_core_M: int,
    per_core_N: int,
    fp32_dest_acc_en: bool = False,
    *,
    dst_full_sync_en: bool = True,
    require_legacy_writer: bool = False,
) -> list[tuple[int, int]]:
    """Return all (h, w) subblock pairs that are legal for this shape.

    Filters by:
      * ``h | per_core_M`` and ``w | per_core_N`` (factory divisibility constraint)
      * ``h * w <= dst_capacity`` (DST register file limit)
      * if ``require_legacy_writer``: ``h == 1`` or ``w == per_core_N`` (the legacy
        subblock-major writer's FATAL gate). Pass ``False`` for callers willing to
        opt into ``tile_pack_row_major``.

    The list is sorted by descending volume — first entry is the best pick.
    Within the same volume, ``(1, N)`` and ``(N, 1)`` come before ``(h, w)`` shapes
    where both dimensions exceed 1; this matches the helper's pack fast-path
    preference (``h == 1`` collapses row-major to subblock-major with no overhead).
    """
    cap = dst_capacity(fp32_dest_acc_en, dst_full_sync_en)
    if per_core_M <= 0 or per_core_N <= 0:
        return []
    out: list[tuple[int, int, tuple[int, int, int]]] = []
    for h in range(1, cap + 1):
        if per_core_M % h != 0:
            continue
        for w in range(1, cap + 1):
            if per_core_N % w != 0:
                continue
            if h * w > cap:
                continue
            if require_legacy_writer and not (h == 1 or w == per_core_N):
                continue
            volume = h * w
            # Sort key: high volume first, then h==1 / w==1 fast-path, then h==w mostly-square.
            fast_path = 0 if (h == 1 or w == 1) else 1
            sort_key = (-volume, fast_path, abs(h - w))
            out.append((h, w, sort_key))
    out.sort(key=lambda x: x[2])
    return [(h, w) for h, w, _ in out]


def largest_subblock(
    per_core_M: int,
    per_core_N: int,
    fp32_dest_acc_en: bool = False,
    *,
    dst_full_sync_en: bool = True,
    require_legacy_writer: bool = False,
) -> tuple[int, int]:
    """Return the largest legal (out_subblock_h, out_subblock_w) for this shape."""
    options = subblock_options(
        per_core_M,
        per_core_N,
        fp32_dest_acc_en,
        dst_full_sync_en=dst_full_sync_en,
        require_legacy_writer=require_legacy_writer,
    )
    return options[0] if options else (1, 1)


def needs_row_major(h: int, w: int, per_core_N: int) -> bool:
    """True iff this (h, w) requires ``tile_pack_row_major`` to compile.

    Legacy-compatible subblocks satisfy ``h == 1`` (single M-row group) or
    ``w == per_core_N`` (single N-subblock per row group). Anything else needs
    the row-major writer.
    """
    return h > 1 and w != per_core_N


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
    subblock, and writes ``out_subblock_h`` / ``out_subblock_w`` back. If the
    chosen pair requires the row-major writer (and ``enable_tile_pack_row_major``
    is True), also flips ``tile_pack_row_major`` on. Pass
    ``require_legacy_writer=True`` to force a legacy-compatible pick (no rmo).

    When ``compute_kernel_config`` is supplied, its ``fp32_dest_acc_en`` and
    ``dst_full_sync_en`` attributes are used in preference to the keyword
    arguments — saves the caller from re-stating them. Pass either form;
    explicit kwargs override when both are given.

    Returns the same config object for chaining.
    """
    if compute_kernel_config is not None:
        fp32_from_ckc, full_sync_from_ckc = _capacity_flags_from(compute_kernel_config)
        # Only override when the kwargs are still at their defaults — explicit values win.
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
        dst_full_sync_en=dst_full_sync_en,
        require_legacy_writer=require_legacy_writer,
    )
    program_config.out_subblock_h = h
    program_config.out_subblock_w = w
    if enable_tile_pack_row_major and needs_row_major(h, w, pn):
        # Only some config types have tile_pack_row_major (mcast 2D / 1D / non-mcast).
        # DRAM-sharded factories hardcode rmo today; others keep it default.
        if hasattr(program_config, "tile_pack_row_major"):
            program_config.tile_pack_row_major = True
    return program_config


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
    """Estimate per-core L1 footprint of a matmul ``program_config``.

    Counts:
      * Output buffer: ``per_core_M * per_core_N * out_tile_bytes``
      * Intermediate buffer (if ``fuse_bias`` or ``tile_pack_row_major`` set):
        same size as output, in fp32 if ``interm_tile_bytes`` given
      * Double-buffered in0 block: ``num_buffered_blocks * per_core_M *
        in0_block_w * in0_tile_bytes``
      * Double-buffered in1 block: ``num_buffered_blocks * per_core_N *
        in0_block_w * in1_tile_bytes``

    Defaults assume bf16 (2 KB per 32x32 tile). Pass ``interm_tile_bytes=4096``
    for fp32 partials.

    Returns a dict with byte breakdowns and ``fits_wh`` / ``fits_bh`` booleans
    against the conservative L1 budgets above. Callers can compare
    ``estimated_bytes`` directly against their own budget if needed.

    Note: this estimator is deliberately conservative on the under-side. It
    omits reader/writer scratch CBs, sync CBs, and bias/activation CBs, all of
    which add small overheads. If the estimate says you have <50 KB headroom,
    treat the config as borderline and verify by running it.
    """
    pm = int(program_config.per_core_M)
    pn = int(program_config.per_core_N)
    ibw = int(program_config.in0_block_w)
    interm_bytes = interm_tile_bytes if interm_tile_bytes is not None else out_tile_bytes

    out_bytes = pm * pn * out_tile_bytes
    rmo = bool(getattr(program_config, "tile_pack_row_major", False))
    interm_buf_bytes = pm * pn * interm_bytes if (fuse_bias or rmo) else 0
    in0_buf_bytes = num_buffered_blocks * pm * ibw * in0_tile_bytes
    in1_buf_bytes = num_buffered_blocks * pn * ibw * in1_tile_bytes

    total = out_bytes + interm_buf_bytes + in0_buf_bytes + in1_buf_bytes

    return {
        "estimated_bytes": total,
        "out_buf_bytes": out_bytes,
        "interm_buf_bytes": interm_buf_bytes,
        "in0_buf_bytes": in0_buf_bytes,
        "in1_buf_bytes": in1_buf_bytes,
        "fits_wh": total <= L1_BUDGET_BYTES["wormhole"],
        "fits_bh": total <= L1_BUDGET_BYTES["blackhole"],
        "headroom_wh": L1_BUDGET_BYTES["wormhole"] - total,
        "headroom_bh": L1_BUDGET_BYTES["blackhole"] - total,
    }


__all__ = [
    "L1_BUDGET_BYTES",
    "dst_capacity",
    "subblock_options",
    "largest_subblock",
    "needs_row_major",
    "upgrade_subblock",
    "estimate_l1_per_core",
]
