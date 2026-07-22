# SPDX-FileCopyrightText: © 2026 Tenstorrent Inc.
# SPDX-License-Identifier: Apache-2.0

"""Single-core, MATH-THREAD-ISOLATED benchmark: SFPU WORK-SCOPING (down to a single axis).

An SFPU (vector-unit) unary op — here rsqrt or reciprocal — runs over a 32x32 tile as a sequence of
32-lane VECTOR ops. Each vector op covers 4 rows x 8 columns taken with stride 2 (the even columns
0,2,..,14 OR the odd columns 1,3,..,15 — column parity is one address bit). A full tile is 4 faces
x (4 row-groups x 2 parities) = 32 vector ops. When the meaningful data lives on ONE axis after a
reduction (a per-row result in column 0, a per-column result in row 0, a scalar at [0,0]) most of
those vectors compute lanes you never read. Scoping to just the vectors that matter is the same math
on the data you keep, for a fraction of the SFPU cycles.

This measures ONLY the SFPU on the math thread: the input is copied into DEST once and packed out
once (both OUTSIDE the timed region), and between them a MATH-thread-only loop applies the scoped op
`reps` times inside one DeviceZoneScopedN; the test reads TRISC_1 (math). The measured cost is flat
per vector op, so the whole story is "how many vector ops does the scope run".

The two coarse knobs and the two axis-optimal tricks (vector counts in [] for a 32x32 tile):
    VectorMode  RC   -> all 4 faces                                      [32]  whole tile (baseline)
    VectorMode  R    -> faces 0,1 (top 16 rows)                          [16]  a top-half result
    VectorMode  C    -> faces 0,2 (left 16 cols)                         [16]  a left-half result
    R + ITERATIONS=2 -> top row-group of both top faces (rows 0-3)       [4]   a ROW-0 result
    C + even-parity  -> even columns of the left faces (cols 0,2,..,14)  [8]   a COL-0 result
    VectorMode  None -> face 0                                           [8]   a [0,0] / 16x16 result
    None + ITER=1    -> first vector of face 0                           [1]   a [0,0] scalar

The asymmetry between the two axis-optimal tricks is the point:
  * ROW-0 waste is on the ROW axis, which is the OUTER walk dimension — `ITERATIONS` truncates it, so
    `VectorMode::R` + `ITERATIONS=2` keeps only the first row-group (rows 0-3) of both top faces = 4
    vectors. Pure knob turn, no raw code. row 0 needs one row-group (both parities), both top faces.
  * COL-0 waste is column PARITY, which is the INNER walk dimension (vectors alternate even/odd) —
    `ITERATIONS` can't isolate it. You skip the odd-parity vectors with a custom sfpi loop that
    strides the DEST address by 2 (`dst_reg += 2`), keeping only the even vectors (which hold column
    0) of faces 0 and 2 = 8 vectors. col 0 needs every row-group (all 32 rows) but one parity.
That is why the row trick lands at 4 vectors but the column trick at 8 — and why one is easy (a knob)
and the other needs raw sfpi (the "sometimes easier, sometimes harder").

Variants (the ladder; `func` = rsqrt|recip is a parameter):
  none        — empty reps-loop, NO SFPU. Math-thread loop overhead (~0), proof the zone times SFPU.
  rc          — whole tile        (VectorMode::RC).            [32]  BASELINE.
  r           — top half          (VectorMode::R).             [16]  valid: rows 0-15.
  c           — left half         (VectorMode::C).             [16]  valid: cols 0-15.
  r_iter2     — top row-group      (R + ITERATIONS=2).         [4]   valid: row 0. iterations only.
  c_skip      — left col even-parity (C + stride-2 skip).      [8]   valid: col 0. raw sfpi.
  face        — face 0            (VectorMode::None).          [8]   valid: [0:16, 0:16].
  face_iter1  — first vector      (None + ITERATIONS=1).       [1]   valid: [0, 0].

Correctness is checked at reps=1 per variant on its valid region; the rest of the tile keeps the raw
copied input (never read). Perf is the median MATH-thread ns per call; measured, never asserted.
"""

import ttnn

TILE = 32

# CB assignment (semantic names).
CB_IN = 0  # one input tile, sharded L1 (resident)
CB_OUT = 16  # one output tile, sharded L1, fp32

FUNCS = ("rsqrt", "recip")  # the SFPU unary op (a parameter)
_FUNC_ID = {"rsqrt": 0, "recip": 1}

VARIANTS = ("none", "rc", "r", "c", "r_iter2", "c_skip", "face", "face_iter1")
BASELINE = "rc"  # the honest default: apply the SFPU to the whole tile
ABLATION = "none"  # empty reps-loop; measures math-thread loop overhead (~0)

# The DeviceZoneScopedN name wrapping the math-only SFPU loop (read from the profiler CSV).
ZONE_NAME = "STS_SFPU"

# scope name -> (method, do_sfpu, VectorMode int value, sfpu ITERATIONS).
# method: 0 = the stock scoped call (VectorMode + ITERATIONS via the wrapper); 1 = the c_skip
# custom even-parity stride body (vmode/iters ignored — it hardcodes VectorMode::C + 4 even vectors).
# VectorMode: None=0, R=1, C=2, RC=4.
_SCOPE = {
    "none": (0, 0, 4, 8),  # vmode/iters unused when do_sfpu=0
    "rc": (0, 1, 4, 8),
    "r": (0, 1, 1, 8),
    "c": (0, 1, 2, 8),
    "r_iter2": (0, 1, 1, 2),  # VectorMode::R + ITERATIONS=2 (row-0 result)
    "c_skip": (1, 1, 2, 8),  # custom even-parity stride under VectorMode::C (col-0 result)
    "face": (0, 1, 0, 8),
    "face_iter1": (0, 1, 0, 1),
}

# 32-lane vector ops each scope runs (the cost is flat per vector op).
_VECTORS = {"none": 0, "rc": 32, "r": 16, "c": 16, "r_iter2": 4, "c_skip": 8, "face": 8, "face_iter1": 1}

# one-line description of HOW each scope selects its vectors (for the report).
LABEL = {
    "none": "empty loop (no SFPU)",
    "rc": "VectorMode::RC (4 faces)",
    "r": "VectorMode::R (2 faces, top)",
    "c": "VectorMode::C (2 faces, left)",
    "r_iter2": "VectorMode::R + ITERATIONS=2",
    "c_skip": "VectorMode::C, even-parity stride (dst_reg+=2)",
    "face": "VectorMode::None (face 0)",
    "face_iter1": "VectorMode::None + ITERATIONS=1",
}

# The region of the output tile each scope leaves correct (rest keeps the raw copied input).
# (r0, r1, c0, c1) half-open; used by the test at reps=1 to slice both output and golden.
VALID_REGION = {
    "none": (0, TILE, 0, TILE),  # identity copy: whole tile equals the input
    "rc": (0, TILE, 0, TILE),
    "r": (0, 16, 0, TILE),
    "c": (0, TILE, 0, 16),
    "r_iter2": (0, 1, 0, TILE),  # a row-0 result (per-column reduce)
    "c_skip": (0, TILE, 0, 1),  # a col-0 result (per-row reduce)
    "face": (0, 16, 0, 16),
    "face_iter1": (0, 1, 0, 1),
}


def vectors(variant):
    """Number of 32-lane SFPU vector ops this scope runs (cost is ~flat per vector op)."""
    return _VECTORS[variant]


# =============================================================================
# Compute kernel — seed DEST once (pre-zone), loop the scoped SFPU `reps` times on the MATH thread
# inside one DeviceZoneScopedN, pack once (post-zone). Zone's MATH (TRISC_1) duration / reps is the
# per-call cost; unpack/pack inside the zone are ~0.
# CT args: [func_id, method, do_sfpu, vmode_val, sfpu_iters]   RT args: [reps]
# =============================================================================
_KERNEL = r"""
#include <cstdint>
#include "api/compute/common.h"
#include "api/compute/tile_move_copy.h"
#include "api/compute/pack.h"
#include "api/compute/eltwise_unary/rsqrt.h"
#include "api/compute/eltwise_unary/recip.h"
#include "api/dataflow/circular_buffer.h"
#include "tools/profiler/kernel_profiler.hpp"
#ifdef TRISC_MATH
#include "ckernel_sfpu_sqrt.h"
#endif

using ckernel::VectorMode;

// --- Coarse scoping: mirror the stock rsqrt/recip calls but thread BOTH knobs through. rsqrt_tile
// hardcodes VectorMode::RC + ITERATIONS=8; recip_tile exposes the mode but not ITERATIONS. ----------
template <int IT>
ALWI void rsqrt_scoped(uint32_t idst, VectorMode vm) {
    MATH(SFPU_UNARY_CALL(DST_SYNC_MODE, DST_ACCUM_MODE, calculate_rsqrt,
        (APPROX, IT, DST_ACCUM_MODE, false /*FAST_APPROX*/, false /*legacy_compat*/), idst, vm));
}
template <int IT>
ALWI void recip_scoped(uint32_t idst, VectorMode vm) {
    MATH(SFPU_UNARY_CALL(DST_SYNC_MODE, DST_ACCUM_MODE, calculate_reciprocal,
        (APPROX, DST_ACCUM_MODE, IT, true /*legacy_compat*/), idst, vm));
}

// --- Axis-optimal COLUMN-0 trick (c_skip): the SFPU walks a face as [rg0-even, rg0-odd, rg1-even,
// ...]; column 0 lives only in the EVEN-parity vectors, so we visit offsets 0,2,4,6 (dst_reg += 2)
// and skip the odd vectors. One body live at a time (avoids an SFPU register spill); net dst_reg
// advance is +8 == the stock ITERATIONS=8, so VectorMode::C's face-0 -> face-2 stepping composes
// unchanged, giving column 0 for all 32 rows in 8 vector ops instead of 16. -----------------------
#ifdef TRISC_MATH
sfpi_inline void cskip_rsqrt_body() {
    for (int rg = 0; rg < 4; rg++) {
        sfpi::vFloat t = ckernel::sfpu::_calculate_sqrt_body_<APPROX, true /*RECIPROCAL*/, false>(sfpi::dst_reg[0]);
        if constexpr (!DST_ACCUM_MODE) { t = sfpi::convert<sfpi::vFloat16b>(t, sfpi::RoundMode::Nearest); }
        sfpi::dst_reg[0] = t;
        sfpi::dst_reg += 2;  // skip the odd-parity vector (columns 1,3,..,15 — never column 0)
    }
}
sfpi_inline void cskip_recip_body() {
    for (int rg = 0; rg < 4; rg++) {
        sfpi::vFloat t = ckernel::sfpu::sfpu_reciprocal<APPROX>(sfpi::dst_reg[0]);
        if constexpr (!DST_ACCUM_MODE) { t = sfpi::convert<sfpi::vFloat16b>(t, sfpi::RoundMode::Nearest); }
        sfpi::dst_reg[0] = t;
        sfpi::dst_reg += 2;
    }
}
#endif
template <uint32_t func>
ALWI void cskip(uint32_t idst) {
    if constexpr (func == 0) {
        MATH((_llk_math_eltwise_unary_sfpu_params_(cskip_rsqrt_body, idst, VectorMode::C)));
    } else {
        MATH((_llk_math_eltwise_unary_sfpu_params_(cskip_recip_body, idst, VectorMode::C)));
    }
}

template <uint32_t func, uint32_t method>
ALWI void sfpu_init_for() {
    if constexpr (func == 0) {
        rsqrt_tile_init();  // sqrt/rsqrt body constants; shared by the scoped and c_skip rsqrt paths
    } else if constexpr (method == 1) {
        MATH((ckernel::sfpu::sfpu_reciprocal_init<APPROX>()));  // c_skip uses the Newton reciprocal body
    } else {
        recip_tile_init();
    }
}

template <uint32_t func, uint32_t sfpu_iters>
ALWI void sfpu_scoped(uint32_t idst, VectorMode vm) {
    if constexpr (func == 0) {
        rsqrt_scoped<sfpu_iters>(idst, vm);
    } else {
        recip_scoped<sfpu_iters>(idst, vm);
    }
}

void kernel_main() {
    constexpr uint32_t cb_in = 0, cb_out = 16;
    constexpr uint32_t func       = get_compile_time_arg_val(0);   // 0 rsqrt, 1 recip
    constexpr uint32_t method     = get_compile_time_arg_val(1);   // 0 scoped-wrapper, 1 c_skip
    constexpr uint32_t do_sfpu    = get_compile_time_arg_val(2);   // 0 = empty-loop overhead baseline
    constexpr uint32_t vmode_val  = get_compile_time_arg_val(3);   // VectorMode int (0/1/2/4)
    constexpr uint32_t sfpu_iters = get_compile_time_arg_val(4);   // 8 / 2 / 1
    const uint32_t reps = get_arg_val<uint32_t>(0);                // math-loop trip count (runtime -> no unroll)

    const VectorMode vm = static_cast<VectorMode>(vmode_val);

    compute_kernel_hw_startup(cb_in, cb_in, cb_out);
    copy_tile_init(cb_in);
    if constexpr (do_sfpu) { sfpu_init_for<func, method>(); }

    cb_reserve_back(cb_in, 1);
    cb_push_back(cb_in, 1);  // sharded input already resident — mark available once
    cb_wait_front(cb_in, 1);

    tile_regs_acquire();
    copy_tile(cb_in, 0, 0);  // seed DEST[0] once — OUTSIDE the timed zone
    {
        DeviceZoneScopedN("STS_SFPU");  // records per-TRISC; the test reads TRISC_1 (math)
        for (uint32_t r = 0; r < reps; ++r) {
            if constexpr (do_sfpu) {
                if constexpr (method == 1) {
                    cskip<func>(0);
                } else if constexpr (sfpu_iters == 8) {
                    sfpu_scoped<func, 8>(0, vm);
                } else if constexpr (sfpu_iters == 2) {
                    sfpu_scoped<func, 2>(0, vm);
                } else {
                    sfpu_scoped<func, 1>(0, vm);
                }
            }
        }
    }
    tile_regs_commit();
    tile_regs_wait();
    cb_reserve_back(cb_out, 1);
    pack_tile(0, cb_out, 0);  // pack once (post-zone) so the host can verify the scoped region
    cb_push_back(cb_out, 1);
    tile_regs_release();
    cb_pop_front(cb_in, 1);
}
"""


# =============================================================================
# Host-side sharded-L1 layout + program descriptor
# =============================================================================
def _single_core():
    return ttnn.CoreRangeSet([ttnn.CoreRange(ttnn.CoreCoord(0, 0), ttnn.CoreCoord(0, 0))])


def create_sharded_memory_config():
    """One 32x32 tile, height-sharded onto a single core."""
    return ttnn.create_sharded_memory_config(
        shape=(TILE, TILE),
        core_grid=_single_core(),
        strategy=ttnn.ShardStrategy.HEIGHT,
        orientation=ttnn.ShardOrientation.ROW_MAJOR,
        use_height_and_width_as_shard_shape=True,
    )


def create_program_descriptor(input_tensor, output_tensor, *, variant, func="rsqrt", reps=1):
    if variant not in VARIANTS:
        raise ValueError(f"variant must be one of {VARIANTS}, got {variant!r}")
    if func not in _FUNC_ID:
        raise ValueError(f"func must be one of {FUNCS}, got {func!r}")
    if reps < 1:
        raise ValueError("reps must be positive")
    if input_tensor.dtype != ttnn.bfloat16 or input_tensor.layout != ttnn.TILE_LAYOUT:
        raise ValueError("input must be bfloat16 TILE_LAYOUT")
    if output_tensor.dtype != ttnn.float32 or output_tensor.layout != ttnn.TILE_LAYOUT:
        raise ValueError("output must be float32 TILE_LAYOUT")

    method, do_sfpu, vmode_val, sfpu_iters = _SCOPE[variant]
    compute = ttnn.KernelDescriptor(
        kernel_source=_KERNEL,
        source_type=ttnn.KernelDescriptor.SourceType.SOURCE_CODE,
        core_ranges=_single_core(),
        compile_time_args=[_FUNC_ID[func], method, do_sfpu, vmode_val, sfpu_iters],
        runtime_args=[(ttnn.CoreCoord(0, 0), [reps])],
        config=ttnn.ComputeConfigDescriptor(),
    )
    cbs = [
        ttnn.cb_descriptor_from_sharded_tensor(CB_IN, input_tensor),
        ttnn.cb_descriptor_from_sharded_tensor(CB_OUT, output_tensor),
    ]
    return ttnn.ProgramDescriptor(kernels=[compute], semaphores=[], cbs=cbs)


def run_op(input_tensor, *, variant, func="rsqrt", reps=1):
    output = ttnn.allocate_tensor_on_device(
        ttnn.Shape([TILE, TILE]),
        ttnn.float32,
        ttnn.TILE_LAYOUT,
        input_tensor.device(),
        create_sharded_memory_config(),
    )
    descriptor = create_program_descriptor(input_tensor, output, variant=variant, func=func, reps=reps)
    return ttnn.generic_op([input_tensor, output], descriptor)
