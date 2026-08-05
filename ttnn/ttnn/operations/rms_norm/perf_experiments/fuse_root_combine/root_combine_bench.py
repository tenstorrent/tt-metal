# SPDX-FileCopyrightText: © 2026 Tenstorrent Inc.
# SPDX-License-Identifier: Apache-2.0

"""ISOLATED micro-benchmark of rms_norm's GROUP ROOT combine chain.

Reconstructs ONLY the root core's post-gather work, single core, everything
resident in L1 (no NoC, no DRAM, no reduce, no pass B):

    GROUP_SIZE fp32 partial tiles per tile-row  ->  group sum
                                               ->  * (1/W), + eps, rsqrt
                                               ->  ONE fp32 tile per tile-row

Layout of the partials matches the op's gather CB:
  * row-major  (page = r * GROUP_SIZE + g)  -- what the tree's D16 fold reads
  * slot-major (page = g * rows + r)        -- what HEAD's copy+add chain reads

Precision contract (FIXED, never a lever): bf16 activations feeding the partials,
fp32 partial/accumulator/out CBs, math_fidelity=HiFi2, fp32_dest_acc_en=False,
math_approx_mode=False.  Every variant runs under the identical config.

Variants (the MENU) -- accumulate mechanism x finalize placement:

  id  name                    accumulate                     finalize
  0   rmw_sep                 L1 read-modify-write (FPU add) separate (transform_in_place + handoff copy)
  1   l1acc_sep               packer L1 accumulation         separate (transform_in_place + handoff copy)
  2   l1acc_fusedout          packer L1 accumulation         fused into the copy-out (handoff copy deleted)
  3   destacc_sep             sticky DEST (pairwise FPU)     separate (transform_in_place + handoff copy)
  4   destacc_fused           sticky DEST (pairwise FPU)     fused in the SAME DEST window, one pack
  5   destchunk_fusedout      DEST in chunks of CHUNK,        fused into the copy-out
                              fp32 L1 carry between chunks
  6   floor                   nothing (CB handshake only)    -- the launch/handshake floor

Variant 0 is HEAD's approach, variant 1 is the working tree's current approach
(kernels/rms_norm_compute.cpp "Perf 1 (D16)"), so BOTH honest baselines are in
the menu.  Three further options vary ONLY the finalize's SFPU scope (identical
math on column 0 -- the only column the stat's consumer reads):
  destacc_fused_cskip   even-parity col-0 rsqrt stride  (examples/sfpu_tile_scope)
  destacc_fused_col     + mul/add-scalar scoped to VectorMode::C
  destacc_fused_sfpu1   the whole finalize as ONE 8-vector col-0 sfpi pass

MEASURED (blackhole p150b, 1350 MHz, single core, one fresh-cache profiled run per
variant, median of 3 launches, spread < 1%; ns per BLOCK_ROWS round):

  GS  rows | rmw_sep  l1acc_sep  destacc_fused  destacc_fused_sfpu1 | vs l1acc
   8   10  |  13249     11935         7409             3656         | 1.61x / 3.26x   <- PRIMARY
  32    1  |   6467      2760         1293              909         | 2.13x / 3.04x   <- SECONDARY
  28    1  |   5710      2568         1242              864         | 2.07x / 2.97x
   9    1  |   2465      1611         1015              629         | 1.59x / 2.56x
   4   32  |  31243     30755        21618             9600         | 1.42x / 3.20x
  (full sweep in test_fuse_root_combine.py::test_menu -- 12 geometries, all WIN)

Precision: every option meets the contract.  `destacc_fused` is not worse than the
current path anywhere measured (stat rel-RMS 2.19e-3 vs 3.37e-3 at the primary
geometry); the scoped finalizes are BIT-identical to the stock one.
"""

import struct


import ttnn

TILE = 32
FP32_TILE_BYTES = 4096

CB_P = 0  # cb_partials_gathered  (fp32, rows*GROUP_SIZE pages, sharded tensor)
CB_ACC = 1  # cb_row_stat          (fp32, 2*rows pages, plain L1 CB)
CB_OUT = 2  # cb_stat_handoff      (fp32, rows pages, sharded tensor)

VARIANTS = (
    "rmw_sep",
    "l1acc_sep",
    "l1acc_fusedout",
    "destacc_sep",
    "destacc_fused",
    "destchunk_fusedout",
    "destacc_fused_cskip",
    "destacc_fused_col",
    "destacc_fused_sfpu1",
    "floor",
)
# variant name -> (kernel body id, rsqrt scope).  rsqrt scope: 0 = VectorMode::RC,
# 1 = VectorMode::C (the op's shipped L6b), 2 = C + even-parity col-0 stride.
# `destacc_fused_cskip` is the SAME fused body as `destacc_fused`; it exists only to
# show whether the ORTHOGONAL rsqrt-scoping lever composes with the fusion.
_BODY = {
    "rmw_sep": (0, 1),
    "l1acc_sep": (1, 1),
    "l1acc_fusedout": (2, 1),
    "destacc_sep": (3, 1),
    "destacc_fused": (4, 1),
    "destchunk_fusedout": (5, 1),
    "destacc_fused_cskip": (4, 2),
    "destacc_fused_col": (4, 3),
    "destacc_fused_sfpu1": (4, 4),
    "floor": (6, 1),
}
# Which gather layout each variant's accumulation walk expects.
ROW_MAJOR_VARIANTS = frozenset(VARIANTS) - {"rmw_sep"}

DEST_CHUNK = 4  # partials per DEST window for `destchunk_fusedout`

_KERNEL = r"""
// ISOLATED bench kernel: the rms_norm group root's combine chain, one core.
// Every variant consumes rows*GROUP_SIZE fp32 pages of cb_p and produces `rows`
// fp32 pages in cb_out.  RAW LLK is used on purpose in the DEST variants (see
// the experiment README); the helper baselines are byte-for-byte the op's calls.
#include <cstdint>

#include "api/compute/compute_kernel_api.h"
#include "api/compute/compute_kernel_hw_startup.h"
#include "api/compute/eltwise_binary.h"
#include "api/compute/tile_move_copy.h"
#include "api/compute/pack.h"
#include "api/compute/eltwise_unary/binop_with_scalar.h"
#include "api/compute/eltwise_unary/rsqrt.h"
#include "ttnn/cpp/ttnn/kernel_lib/perf_instrumentation.hpp"
#include "ttnn/cpp/ttnn/kernel_lib/dest_helpers.hpp"
#include "ttnn/cpp/ttnn/kernel_lib/eltwise_convenience.hpp"
#include "ttnn/cpp/ttnn/kernel_lib/streaming_reduce_helpers.hpp"
#ifdef TRISC_MATH
#include "ckernel_sfpu_sqrt.h"
#endif

namespace ckl = compute_kernel_lib;

// File scope so the sfpi bodies below can fold them in as literals.
constexpr uint32_t FINALIZE_INV_W_BITS = get_compile_time_arg_val(2);
constexpr uint32_t FINALIZE_EPS_BITS = get_compile_time_arg_val(3);

// Verbatim from kernels/rms_norm_compute.cpp (Lamp L6b): rsqrt scoped to the
// tile's COLUMN faces, because a REDUCE_ROW partial only carries column 0.
#ifdef TRISC_MATH
template <bool legacy_compat = false, bool FAST_APPROX = false>
ALWI void rsqrt_tile_col(uint32_t idst) {
    SFPU_UNARY_CALL(
        DST_SYNC_MODE,
        DST_ACCUM_MODE,
        calculate_rsqrt,
        (APPROX, 8 /* ITERATIONS */, DST_ACCUM_MODE, FAST_APPROX, legacy_compat),
        idst,
        VectorMode::C);
}

// The even-parity COLUMN-0 stride (examples/sfpu_tile_scope `c_skip`), ported
// unchanged: only column 0 of the stat is ever read, and column 0 lives in the
// EVEN-parity vectors of faces 0/2, so half of VectorMode::C's vectors are dead.
sfpi_inline void cskip_rsqrt_body() {
    for (int rg = 0; rg < 4; rg++) {
        sfpi::vFloat t = ckernel::sfpu::_calculate_sqrt_body_<APPROX, true /*RECIPROCAL*/, false>(sfpi::dst_reg[0]);
        if constexpr (!DST_ACCUM_MODE) {
            t = sfpi::convert<sfpi::vFloat16b>(t, sfpi::RoundMode::Nearest);
        }
        sfpi::dst_reg[0] = t;
        sfpi::dst_reg += 2;
    }
}
#endif
ALWI void rsqrt_tile_col0(uint32_t idst) {
    MATH((_llk_math_eltwise_unary_sfpu_params_(cskip_rsqrt_body, idst, ckernel::VectorMode::C)));
}

// The WHOLE finalize -- *(1/W), +eps, rsqrt -- as ONE column-0 SFPU pass: 8
// vector ops instead of the 32+32+16 the three stock calls walk.  The scalars are
// compile-time constants so they fold into the body.
#ifdef TRISC_MATH
sfpi_inline void cskip_finalize_body() {
    constexpr float inv_w = __builtin_bit_cast(float, (uint32_t)FINALIZE_INV_W_BITS);
    constexpr float epsv = __builtin_bit_cast(float, (uint32_t)FINALIZE_EPS_BITS);
    for (int rg = 0; rg < 4; rg++) {
        sfpi::vFloat s = sfpi::dst_reg[0] * inv_w + epsv;
        sfpi::vFloat t = ckernel::sfpu::_calculate_sqrt_body_<APPROX, true /*RECIPROCAL*/, false>(s);
        if constexpr (!DST_ACCUM_MODE) {
            t = sfpi::convert<sfpi::vFloat16b>(t, sfpi::RoundMode::Nearest);
        }
        sfpi::dst_reg[0] = t;
        sfpi::dst_reg += 2;
    }
}
#endif
ALWI void finalize_tile_col0(uint32_t idst) {
    MATH((_llk_math_eltwise_unary_sfpu_params_(cskip_finalize_body, idst, ckernel::VectorMode::C)));
}

// The two stock scalar ops, scoped to the tile's COLUMN faces (same raw-LLK
// substitution the op already makes for rsqrt: the API bakes in VectorMode::RC).
ALWI void mul_unary_tile_col(uint32_t idst, uint32_t param1) {
    MATH(SFPU_UNARY_CALL(
        DST_SYNC_MODE,
        DST_ACCUM_MODE,
        calculate_binop_with_scalar,
        (APPROX, MUL_UNARY, 8 /* ITERATIONS */),
        idst,
        ckernel::VectorMode::C,
        param1));
}
ALWI void add_unary_tile_col(uint32_t idst, uint32_t param1) {
    MATH(SFPU_UNARY_CALL(
        DST_SYNC_MODE,
        DST_ACCUM_MODE,
        calculate_binop_with_scalar,
        (APPROX, ADD_UNARY, 8 /* ITERATIONS */),
        idst,
        ckernel::VectorMode::C,
        param1));
}

namespace {
constexpr uint32_t cb_p = 0;
constexpr uint32_t cb_acc = 1;
constexpr uint32_t cb_out = 2;
}  // namespace

void kernel_main() {
    constexpr uint32_t GROUP_SIZE = get_compile_time_arg_val(0);
    constexpr uint32_t VARIANT = get_compile_time_arg_val(1);
    constexpr uint32_t INV_W_BITS = get_compile_time_arg_val(2);
    constexpr uint32_t EPS_BITS = get_compile_time_arg_val(3);
    constexpr uint32_t RSQRT_COL = get_compile_time_arg_val(4);
    constexpr uint32_t CHUNK = get_compile_time_arg_val(5);

    const uint32_t rows = get_arg_val<uint32_t>(0);
    const uint32_t iters = get_arg_val<uint32_t>(1);

    compute_kernel_hw_startup(cb_p, cb_p, cb_out);

    // ---- the op's finalize, one definition (used by every variant) ---------
    auto finalize = [](uint32_t dst) {
        if constexpr (RSQRT_COL == 4) {
            rsqrt_tile_init();  // the sqrt body's constants
            finalize_tile_col0(dst);
            return;
        }
        binop_with_scalar_tile_init();
        if constexpr (RSQRT_COL == 3) {
            mul_unary_tile_col(dst, INV_W_BITS);
            add_unary_tile_col(dst, EPS_BITS);
        } else {
            mul_unary_tile(dst, INV_W_BITS);
            add_unary_tile(dst, EPS_BITS);
        }
        rsqrt_tile_init();
        if constexpr (RSQRT_COL >= 2) {
            rsqrt_tile_col0(dst);
        } else if constexpr (RSQRT_COL == 1) {
            MATH((rsqrt_tile_col(dst)));
        } else {
            rsqrt_tile(dst);
        }
    };
    // Batched form: the same three ops, inits HOISTED across the DEST batch.
    auto finalize_batch = [](uint32_t n) {
        if constexpr (RSQRT_COL == 4) {
            rsqrt_tile_init();
            for (uint32_t j = 0; j < n; ++j) {
                finalize_tile_col0(j);
            }
            return;
        }
        binop_with_scalar_tile_init();
        for (uint32_t j = 0; j < n; ++j) {
            if constexpr (RSQRT_COL == 3) {
                mul_unary_tile_col(j, INV_W_BITS);
                add_unary_tile_col(j, EPS_BITS);
            } else {
                mul_unary_tile(j, INV_W_BITS);
                add_unary_tile(j, EPS_BITS);
            }
        }
        rsqrt_tile_init();
        for (uint32_t j = 0; j < n; ++j) {
            if constexpr (RSQRT_COL >= 2) {
                rsqrt_tile_col0(j);
            } else if constexpr (RSQRT_COL == 1) {
                MATH((rsqrt_tile_col(j)));
            } else {
                rsqrt_tile(j);
            }
        }
    };

    // The op's D16 pack-accumulating fold output spec, verbatim.
    constexpr auto ROOT_FOLD_OUT = ckl::output(
        cb_acc,
        ckl::ReservePolicy::OneUpfront,
        ckl::PushPolicy::OneAtEnd,
        ckl::DataFormatReconfig::Enabled,
        ckl::PackRelu::Disabled,
        ckl::L1Accumulation::SeedFirst);

    // ---- sticky-DEST accumulation of partials [gb, ge) for `n` tile-rows ---
    // Pairwise FPU adds with acc_to_dest: DEST[j] += P[g] + P[g+1].  An odd
    // count seeds DEST[j] with a copy, then pairs.  DEST slot j <-> tile-row
    // r0 + j, so the whole batch shares ONE dst-sync window.
    auto accum_into_dest = [](uint32_t r0, uint32_t n, uint32_t gb, uint32_t ge) {
        uint32_t g = gb;
        if (((ge - gb) & 1u) != 0u) {
            copy_tile_to_dst_init_short(cb_p);
            for (uint32_t j = 0; j < n; ++j) {
                copy_tile(cb_p, (r0 + j) * GROUP_SIZE + g, j);
            }
            g += 1;
            if (g < ge) {
                add_tiles_init(cb_p, cb_p, true);
            }
        } else {
            add_tiles_init(cb_p, cb_p, false);
            for (uint32_t j = 0; j < n; ++j) {
                const uint32_t b = (r0 + j) * GROUP_SIZE + g;
                add_tiles(cb_p, cb_p, b, b + 1, j);
            }
            g += 2;
            if (g < ge) {
                add_tiles_init(cb_p, cb_p, true);
            }
        }
        for (; g < ge; g += 2) {
            for (uint32_t j = 0; j < n; ++j) {
                const uint32_t b = (r0 + j) * GROUP_SIZE + g;
                add_tiles(cb_p, cb_p, b, b + 1, j);
            }
        }
    };

    // Fused finalize-and-ship: unpack the L1 accumulator, finalize in DEST, pack
    // straight to cb_out (this is what deletes the separate stat_handoff copy).
    // Output is reserved/pushed ONE PAGE AT A TIME, exactly like the op's
    // `ckl::copy` handoff: the writer's stat multicast must be able to start on
    // row 0 while the root is still finalizing row 1.  A single push at the end
    // would be faster in this bench and slower in the op.
    auto finalize_out_from_acc = [&](uint32_t rows_) {
        const uint32_t lim = ckl::DEST_AUTO_LIMIT;
        for (uint32_t r0 = 0; r0 < rows_; r0 += lim) {
            const uint32_t n = (rows_ - r0 < lim) ? (rows_ - r0) : lim;
            cb_wait_front(cb_acc, n);
            tile_regs_acquire();
            reconfig_data_format_srca(cb_acc);
            pack_reconfig_data_format(cb_out);
            copy_tile_to_dst_init_short(cb_acc);
            for (uint32_t j = 0; j < n; ++j) {
                copy_tile(cb_acc, j, j);
            }
            finalize_batch(n);
            tile_regs_commit();
            tile_regs_wait();
            for (uint32_t j = 0; j < n; ++j) {
                cb_reserve_back(cb_out, 1);
                pack_tile<true>(j, cb_out, 0);
                cb_push_back(cb_out, 1);
            }
            tile_regs_release();
            cb_pop_front(cb_acc, n);
        }
    };

    for (uint32_t it = 0; it < iters; ++it) {
        // Re-arm the resident partials (harness overhead, identical for every
        // variant): the variant bodies POP cb_p exactly as the op's do.
        cb_reserve_back(cb_p, rows * GROUP_SIZE);
        cb_push_back(cb_p, rows * GROUP_SIZE);

        if constexpr (VARIANT == 0) {
            // ---- rmw_sep: HEAD's chain (slot-major gather layout) ----------
            {
                MaybeDeviceZoneScope("root_sum");
                ckl::copy<ckl::input(cb_p), ckl::output(cb_acc)>(ckl::EltwiseShape::tiles(rows));
                for (uint32_t g = 1; g < GROUP_SIZE; ++g) {
                    ckl::add<ckl::input(cb_acc), ckl::input(cb_p), ckl::output(cb_acc)>(
                        ckl::EltwiseShape::tiles(rows));
                }
            }
            {
                MaybeDeviceZoneScope("root_finalize");
                for (uint32_t i = 0; i < rows; ++i) {
                    ckl::transform_in_place(cb_acc, finalize);
                }
            }
            MaybeDeviceZoneScope("stat_handoff");
            ckl::copy<ckl::input(cb_acc), ckl::output(cb_out)>(ckl::EltwiseShape::tiles(rows));
        } else if constexpr (VARIANT == 1) {
            // ---- l1acc_sep: the working tree's D16 fold --------------------
            {
                MaybeDeviceZoneScope("root_sum");
                for (uint32_t r = 0; r < rows; ++r) {
                    ckl::eltwise_chain(
                        ckl::EltwiseShape::tiles(GROUP_SIZE),
                        ckl::CopyTile<ckl::input(cb_p)>{},
                        ckl::PackTile<ROOT_FOLD_OUT>{});
                }
            }
            {
                MaybeDeviceZoneScope("root_finalize");
                for (uint32_t i = 0; i < rows; ++i) {
                    ckl::transform_in_place(cb_acc, finalize);
                }
            }
            MaybeDeviceZoneScope("stat_handoff");
            ckl::copy<ckl::input(cb_acc), ckl::output(cb_out)>(ckl::EltwiseShape::tiles(rows));
        } else if constexpr (VARIANT == 2) {
            // ---- l1acc_fusedout: D16 fold + fused finalize-and-ship --------
            {
                MaybeDeviceZoneScope("root_sum");
                for (uint32_t r = 0; r < rows; ++r) {
                    ckl::eltwise_chain(
                        ckl::EltwiseShape::tiles(GROUP_SIZE),
                        ckl::CopyTile<ckl::input(cb_p)>{},
                        ckl::PackTile<ROOT_FOLD_OUT>{});
                }
            }
            MaybeDeviceZoneScope("root_finalize_out");
            finalize_out_from_acc(rows);
        } else if constexpr (VARIANT == 3) {
            // ---- destacc_sep: sticky DEST sum, finalize kept SEPARATE ------
            {
                MaybeDeviceZoneScope("root_sum");
                const uint32_t lim = ckl::DEST_AUTO_LIMIT;
                cb_wait_front(cb_p, rows * GROUP_SIZE);
                for (uint32_t r0 = 0; r0 < rows; r0 += lim) {
                    const uint32_t n = (rows - r0 < lim) ? (rows - r0) : lim;
                    cb_reserve_back(cb_acc, n);
                    tile_regs_acquire();
                    reconfig_data_format(cb_p, cb_p);
                    pack_reconfig_data_format(cb_acc);
                    accum_into_dest(r0, n, 0, GROUP_SIZE);
                    tile_regs_commit();
                    tile_regs_wait();
                    for (uint32_t j = 0; j < n; ++j) {
                        pack_tile<true>(j, cb_acc, j);
                    }
                    tile_regs_release();
                    cb_push_back(cb_acc, n);
                }
                cb_pop_front(cb_p, rows * GROUP_SIZE);
            }
            {
                MaybeDeviceZoneScope("root_finalize");
                for (uint32_t i = 0; i < rows; ++i) {
                    ckl::transform_in_place(cb_acc, finalize);
                }
            }
            MaybeDeviceZoneScope("stat_handoff");
            ckl::copy<ckl::input(cb_acc), ckl::output(cb_out)>(ckl::EltwiseShape::tiles(rows));
        } else if constexpr (VARIANT == 4) {
            // ---- destacc_fused: THE IDEA -----------------------------------
            // One dst-sync window per DEST batch of tile-rows: accumulate all
            // GROUP_SIZE partials in a sticky DEST tile, apply *(1/W), +eps and
            // rsqrt on that same tile, pack ONCE straight into cb_out.
            MaybeDeviceZoneScope("root_combine_fused");
            const uint32_t lim = ckl::DEST_AUTO_LIMIT;
            cb_wait_front(cb_p, rows * GROUP_SIZE);
            for (uint32_t r0 = 0; r0 < rows; r0 += lim) {
                const uint32_t n = (rows - r0 < lim) ? (rows - r0) : lim;
                tile_regs_acquire();
                reconfig_data_format(cb_p, cb_p);
                pack_reconfig_data_format(cb_out);
                accum_into_dest(r0, n, 0, GROUP_SIZE);
                finalize_batch(n);
                tile_regs_commit();
                tile_regs_wait();
                // One page reserved/pushed per tile-row -- the writer's multicast
                // can start on row r0+j immediately (same granularity as the op's
                // `ckl::copy` handoff).
                for (uint32_t j = 0; j < n; ++j) {
                    cb_reserve_back(cb_out, 1);
                    pack_tile<true>(j, cb_out, 0);
                    cb_push_back(cb_out, 1);
                }
                tile_regs_release();
            }
            cb_pop_front(cb_p, rows * GROUP_SIZE);
        } else if constexpr (VARIANT == 5) {
            // ---- destchunk_fusedout: BOUNDED DEST depth --------------------
            // DEST accumulates only CHUNK partials at a time and the carry
            // between chunks goes through the fp32 L1 accumulator via packer
            // L1-accumulation -- the same "bound the DEST-resident depth"
            // discipline the op's DEST_ACC_SQUARE_MAX_WT applies to pass A.
            {
                MaybeDeviceZoneScope("root_sum");
                const uint32_t lim = ckl::DEST_AUTO_LIMIT;
                cb_wait_front(cb_p, rows * GROUP_SIZE);
                cb_reserve_back(cb_acc, rows);
                for (uint32_t gb = 0; gb < GROUP_SIZE; gb += CHUNK) {
                    const uint32_t ge = (GROUP_SIZE - gb < CHUNK) ? GROUP_SIZE : (gb + CHUNK);
                    pack_reconfig_l1_acc(gb == 0 ? 0 : 1);
                    for (uint32_t r0 = 0; r0 < rows; r0 += lim) {
                        const uint32_t n = (rows - r0 < lim) ? (rows - r0) : lim;
                        tile_regs_acquire();
                        reconfig_data_format(cb_p, cb_p);
                        pack_reconfig_data_format(cb_acc);
                        accum_into_dest(r0, n, gb, ge);
                        tile_regs_commit();
                        tile_regs_wait();
                        for (uint32_t j = 0; j < n; ++j) {
                            pack_tile<true>(j, cb_acc, r0 + j);
                        }
                        tile_regs_release();
                    }
                }
                pack_reconfig_l1_acc(0);
                cb_push_back(cb_acc, rows);
                cb_pop_front(cb_p, rows * GROUP_SIZE);
            }
            MaybeDeviceZoneScope("root_finalize_out");
            finalize_out_from_acc(rows);
        } else {
            // ---- floor: CB handshake only (the launch/handshake floor) -----
            MaybeDeviceZoneScope("floor");
            cb_wait_front(cb_p, rows * GROUP_SIZE);
            cb_pop_front(cb_p, rows * GROUP_SIZE);
            cb_reserve_back(cb_out, rows);
            cb_push_back(cb_out, rows);
        }

        cb_wait_front(cb_out, rows);
        cb_pop_front(cb_out, rows);
    }
}
"""


def _f32_bits(v: float) -> int:
    return struct.unpack("I", struct.pack("f", float(v)))[0]


def _single_core():
    return ttnn.CoreRangeSet([ttnn.CoreRange(ttnn.CoreCoord(0, 0), ttnn.CoreCoord(0, 0))])


def sharded_fp32_config(h_tiles, w_tiles):
    return ttnn.create_sharded_memory_config(
        shape=(h_tiles * TILE, w_tiles * TILE),
        core_grid=_single_core(),
        strategy=ttnn.ShardStrategy.HEIGHT,
        orientation=ttnn.ShardOrientation.ROW_MAJOR,
        use_height_and_width_as_shard_shape=True,
    )


# ---------------------------------------------------------------------------
# Host-side model of the stage's input: the group's partial sum(x^2) tiles.
# ---------------------------------------------------------------------------


def build_case(rows, group_size, per_slot_w, eps, seed=0):
    """Return (partials [rows*32, GS], W, stat_ref [rows*32] float64).

    x is bf16 (the precision contract); each group member's partial is the fp32
    sum of squares over its own `per_slot_w` columns, exactly what the op's
    REDUCE_ROW produces before the combine.
    """
    import torch

    W = group_size * per_slot_w
    gen = torch.Generator().manual_seed(seed)
    x = torch.randn(rows * TILE, W, generator=gen, dtype=torch.float32)
    xb = x.to(torch.bfloat16).to(torch.float32)
    sq = xb * xb
    partials = sq.view(rows * TILE, group_size, per_slot_w).sum(-1)  # [rows*32, GS]
    total = partials.double().sum(-1)
    stat_ref = torch.rsqrt(total / float(W) + float(eps))
    return partials, W, stat_ref, xb


def pack_partials_tensor(partials, rows, group_size, row_major):
    """Lay the partials out as a [rows*32, GS*32] fp32 tile grid.

    Page (a, b) of the tile grid is tile index a*GS + b.  `row_major` puts row r's
    GROUP_SIZE partials CONTIGUOUS (page = r*GS + g, the op's D16 gather layout);
    otherwise the layout is slot-major (page = g*rows + r, HEAD's layout).
    """
    import torch

    col = torch.arange(TILE, dtype=torch.float32)
    # Non-column-0 filler: faces 1/3 (cols 16..31) are ZERO exactly as the op's
    # boot-zero leaves them; cols 1..15 carry distinct garbage so that a variant
    # reading the wrong column fails correctness instead of passing by accident.
    filler = torch.where((col >= 1) & (col < 16), col + 3.0, torch.zeros(TILE))
    t = filler.repeat(group_size).unsqueeze(0).repeat(rows * TILE, 1).contiguous()
    if row_major:
        t[:, 0::TILE] = partials
    else:
        for a in range(rows):
            for b in range(group_size):
                p = a * group_size + b
                slot, rt = divmod(p, rows)
                t[a * TILE : (a + 1) * TILE, b * TILE] = partials[rt * TILE : (rt + 1) * TILE, slot]
    return t


def make_tensors(device, partials, rows, group_size, row_major):
    import torch

    host = pack_partials_tensor(partials, rows, group_size, row_major)
    tt_p = ttnn.from_torch(
        host,
        dtype=ttnn.float32,
        layout=ttnn.TILE_LAYOUT,
        device=device,
        memory_config=sharded_fp32_config(rows, group_size),
    )
    tt_out = ttnn.from_torch(
        torch.zeros(rows * TILE, TILE, dtype=torch.float32),
        dtype=ttnn.float32,
        layout=ttnn.TILE_LAYOUT,
        device=device,
        memory_config=sharded_fp32_config(rows, 1),
    )
    return tt_p, tt_out


def create_program_descriptor(
    tt_p, tt_out, *, variant, rows, group_size, W, eps, iters, rsqrt_col=None, fp32_dest_acc_en=False
):
    body, default_rsqrt = _BODY[variant]
    rsqrt_col = default_rsqrt if rsqrt_col is None else rsqrt_col
    rt = ttnn.RuntimeArgs()
    rt[0][0] = [rows, iters]
    kernel = ttnn.KernelDescriptor(
        kernel_source=_KERNEL,
        source_type=ttnn.KernelDescriptor.SourceType.SOURCE_CODE,
        core_ranges=_single_core(),
        compile_time_args=[
            group_size,
            body,
            _f32_bits(1.0 / float(W)),
            _f32_bits(eps),
            rsqrt_col,
            DEST_CHUNK,
        ],
        runtime_args=rt,
        # PRECISION CONTRACT -- identical for every variant, never a lever.
        # `fp32_dest_acc_en` is only ever flipped by the DOMAIN test that asks
        # "does the pattern still hold if the USER asks for fp32 DEST?", never to
        # buy speed.
        config=ttnn.ComputeConfigDescriptor(
            math_fidelity=ttnn.MathFidelity.HiFi2,
            fp32_dest_acc_en=fp32_dest_acc_en,
            math_approx_mode=False,
            dst_full_sync_en=False,
        ),
    )
    cbs = [
        ttnn.cb_descriptor_from_sharded_tensor(CB_P, tt_p),
        ttnn.CBDescriptor(
            total_size=2 * rows * FP32_TILE_BYTES,
            core_ranges=_single_core(),
            format_descriptors=[
                ttnn.CBFormatDescriptor(buffer_index=CB_ACC, data_format=ttnn.float32, page_size=FP32_TILE_BYTES)
            ],
        ),
        ttnn.cb_descriptor_from_sharded_tensor(CB_OUT, tt_out),
    ]
    return ttnn.ProgramDescriptor(kernels=[kernel], semaphores=[], cbs=cbs)


def run_variant(
    device,
    *,
    variant,
    rows,
    group_size,
    per_slot_w=128,
    eps=1e-12,
    iters=1,
    seed=0,
    rsqrt_col=None,
    fp32_dest_acc_en=False,
):
    """Run one variant; returns (device stat column [rows*32], stat_ref, xb)."""
    import torch

    partials, W, stat_ref, xb = build_case(rows, group_size, per_slot_w, eps, seed)
    tt_p, tt_out = make_tensors(device, partials, rows, group_size, variant in ROW_MAJOR_VARIANTS)
    desc = create_program_descriptor(
        tt_p,
        tt_out,
        variant=variant,
        rows=rows,
        group_size=group_size,
        W=W,
        eps=eps,
        iters=iters,
        rsqrt_col=rsqrt_col,
        fp32_dest_acc_en=fp32_dest_acc_en,
    )
    out = ttnn.generic_op([tt_p, tt_out], desc)
    got = ttnn.to_torch(out).to(torch.float32)[:, 0]
    return got, stat_ref, xb
