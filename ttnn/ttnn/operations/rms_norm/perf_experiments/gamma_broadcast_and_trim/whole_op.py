# SPDX-FileCopyrightText: © 2026 Tenstorrent Inc.
# SPDX-License-Identifier: Apache-2.0
#
# WHOLE-OP runner for the gamma_broadcast_and_trim bake-off.
#
# `descriptor.py` is a verbatim clone of the op's ProgramDescriptor plus two knobs
# (GAMMA_TRIM / GAMMA_MCAST) and the wiring they need; `kernels/` is a verbatim clone
# of the op's three kernels plus the same two knobs in the reader's gamma filler.
# With both knobs 0 the clone IS the op, which is what makes the delta attributable.
#
# The real op is NEVER imported or touched here.
#
# ======================================================================================
# MEASURED — blackhole p150b, 110-core grid, CHIP_FREQ 1350 MHz, 2026-08-05.
# `DEVICE KERNEL DURATION [ns]` from `run_safe_pytest.sh --profile`; one fresh launch per
# cell, 3-4 reps of the primary target and 2 of the sweep, MEDIAN reported.  Noise floor
# measured directly: the `gamma_rm` control runs the SAME kernel under three different
# option names and spread 0.985-1.007x, so +-1.5% is noise and +-3% is the call band.
# EVERY option is `torch.equal` to the baseline on 7 geometries x 3 gamma dtypes (54+42
# cells) -- these levers move bytes, not math.
# ======================================================================================
#
# (A) PRIMARY TARGET (1,1,8192,1024) INTERLEAVED bf16, 110 cores, Wt=32, HiFi2:
#         option              ns      vs base   share of gamma's marginal cost
#         baseline        104376        1.000x
#         trim_half        95184        1.097x    41%
#         trim_rows        91292        1.143x    60%
#         trim_rows64      90338        1.155x    65%
#         mcast            88935        1.174x    71%
#         mcast_trim_rows  88611        1.178x    73%
#         mcast_trim_half  88679        1.177x    73%
#      HAS_GAMMA=0 control  82785  ->  the WHOLE gamma feature costs 21591 ns, so
#      **1.261x is the hard ceiling** for any gamma idea on this shape.  Of that, the
#      `compute_gamma_mul` zone is 4700 ns (untouched by both levers), so gamma's
#      marginal READ cost is ~16.9 us and the trim captures 83% / mcast 91% of it.
#
# (B) THE COMPOSITION IS NOT MULTIPLICATIVE, and that is the headline.  Both sub-ideas
#     attack the SAME quantity -- gamma's share of reader DRAM bytes -- one by cutting
#     the multiplicity (110 readers -> 1) and one by cutting the size (2048 B -> 128 B
#     per tile).  Whichever runs first takes the budget; the second finds ~nothing.
#     Composition == mcast alone to within noise on every profile measured.
#
# (C) THE OTHER INTERLEAVED PREFILL PROFILES (median of 3):
#         W       baseline   trim_rows64   mcast      best composition
#         1024      104376     90338 1.16x  88935 1.17x  88611 1.18x
#         2304      218264    198213 1.10x 187596 1.16x 186179 1.17x
#         5120      467819    425647 1.10x 422381 1.11x 420748 1.11x
#         7168      644919    579744 1.11x 557446 1.16x 563628 1.14x
#     Narrow W at the same 110-core group (median of 2): W=128 (Wt=4) 17855 ->
#     trim 15690 1.14x / mcast 15139 1.18x;  W=512 (Wt=16) 53552 -> trim 47950 1.12x /
#     mcast 49140 1.09x.  No Wt floor for either lever.
#
# (D) SHARING-GROUP SWEEP (W=1024, ONE tile-row per core so per-core work is constant;
#     the row split makes num_cores == the gamma sharing group).  2 reps:
#         group   baseline    trim_rows          mcast
#             4     6764-6778  1.019 / 1.035x   0.975 / 0.976x
#             8     8531-8579  1.060 / 1.090x   0.970 / 1.019x
#            11    10716-10741 1.100 / 1.110x   1.008 / 1.054x
#            22    17138-17278 1.038 / 1.084x   0.977 / 0.982x
#            44    24219-25272 1.166 / 1.191x   0.995 / 1.046x
#            55    27551-28557 1.096 / 1.139x   0.996 / 1.021x
#           110    53597-54525 1.230 / 1.257x   1.231 / 1.256x
#     MECHANISM.  The injector's read is SERIAL in Wt and CONSTANT in the group size, and
#     every receiver blocks on it; the replicated read's cost grows with the number of
#     cores hammering the same gamma pages once that count passes the DRAM banks'
#     request-service rate.  So the crossover is set by concurrent-identical-readers, not
#     by any shape: below it the group pays the serial prefix without amortising it.
#     Whole-op crossover measured between 55 and 110 cores.  The TRIM has NO such
#     crossover -- it is a per-core byte reduction with no cross-core dependency, so it
#     cannot lose, and it is measured winning at every group size.
#
# (E) FOCUS-SHAPE NO-REGRESSION, (1,1,8192,1024) BLOCK_SHARDED [1024,128] (8,8) = 64c,
#     gamma group = a grid COLUMN of 8 (3 reps): baseline 64622/64701/64676; every one of
#     the 7 options within +-0.5% (0.995x .. 1.006x).  FLAT, as expected -- gamma is 3.8%
#     of that wall.  Both levers are IN the domain there; neither buys anything.
#
# (F) PER-STAGE ZONES, primary target, max ns/core (and why a stage zone lies):
#         option           read_gamma  read_x   SUM   gamma_mul
#         baseline              61806   25043  86849      4701
#         trim_half             51425   27243  78668      4699
#         trim_rows             57817   19873  77690      4761
#         trim_rows64           55892   20528  76420      4741
#         mcast                  3261   62250  65511      4729
#         mcast_trim_rows        3280   62422  65702      4705
#         HAS_GAMMA=0               -   61848  61848      4729
#     Reader DRAM is ONE shared queue, so the gamma/x split is meaningless individually --
#     only the SUM is.  The closure that proves the mechanism: with the mcast, `read_x`
#     (62250) equals the NO-GAMMA reader's `read_x` (61848), i.e. gamma's DRAM cost is
#     fully gone and all that is left is the 3.3 us injector prefix.
#
# (G) A ROW_MAJOR gamma is a different animal and needs neither lever: (1,1,8192,1024)
#     with gamma RM = 83744-84065 ns against the 82785 ns no-gamma floor -- gamma costs
#     ~1 us there, because an RM gamma is ONE W-element stick per core (2 kB) instead of
#     Wt tile pages (64 kB).  The trim is inexpressible on it (no tile padding to trim).

from __future__ import annotations

import ttnn

from . import descriptor as _desc

# The frozen user config for every variant, every launch, every shape.  NOT a lever.
FROZEN_COMPUTE_CONFIG = dict(math_fidelity=ttnn.MathFidelity.HiFi2, fp32_dest_acc_en=False, math_approx_mode=False)


def compute_config():
    return ttnn.ComputeConfigDescriptor(**FROZEN_COMPUTE_CONFIG)


def set_knobs(*, trim=0, mcast=0):
    _desc.GAMMA_TRIM = trim
    _desc.GAMMA_MCAST = mcast


def run(input_tensor, *, gamma=None, epsilon=1e-6, trim=0, mcast=0, memory_config=None):
    """One launch of the cloned op at the given (trim, mcast) setting."""
    set_knobs(trim=trim, mcast=mcast)
    device = input_tensor.device()
    out_mem = memory_config if memory_config is not None else input_tensor.memory_config()
    output_tensor = ttnn.allocate_tensor_on_device(
        ttnn.Shape(list(input_tensor.shape)),
        input_tensor.dtype,
        input_tensor.layout,
        device,
        out_mem,
    )
    pd = _desc.create_program_descriptor(
        input_tensor,
        output_tensor,
        gamma=gamma,
        epsilon=epsilon,
        compute_kernel_config=compute_config(),
    )
    tensors = [input_tensor] + ([gamma] if gamma is not None else []) + [output_tensor]
    return ttnn.generic_op(tensors, pd)


# --------------------------------------------------------------------------------------
# The option menu.  `id` is what the profiler CSV rows are matched against.
# --------------------------------------------------------------------------------------
OPTIONS = {
    # the op's current approach for this part — the honest baseline
    "baseline": dict(trim=0, mcast=0),
    # sub-idea (2) alone, three granularities
    "trim_half": dict(trim=1, mcast=0),  # faces 0+1        (1024 B / tile, 1 txn)
    "trim_rows": dict(trim=2, mcast=0),  # 2 x face row 0   (64 B / tile,  2 txn)
    "trim_rows64": dict(trim=3, mcast=0),  # 2 x 64 B         (128 B / tile, 2 txn)
    # sub-idea (1) alone
    "mcast": dict(trim=0, mcast=1),
    # (1) o (2)
    "mcast_trim_rows": dict(trim=2, mcast=1),
    "mcast_trim_half": dict(trim=1, mcast=1),
}


def gamma_dram_bytes_per_core(Wt, option, gamma_elem_bytes=2, group_size=1):
    """DRAM bytes the WHOLE PROGRAM moves for gamma, amortised per participating core.

    Trim granularity in DRAM-burst terms: the DRAM read granule is 64 B, so a 32 B
    fetch still costs 64 B of bank traffic — that is what is counted here.
    """
    tile = 32 * 32 * gamma_elem_bytes
    face = 16 * 16 * gamma_elem_bytes
    per_tile = {
        0: tile,
        1: 2 * face,
        2: 2 * 64,
        3: 2 * 64,
    }[option["trim"]]
    readers_per_group = 1 if option["mcast"] else group_size
    return Wt * per_tile * readers_per_group / group_size
