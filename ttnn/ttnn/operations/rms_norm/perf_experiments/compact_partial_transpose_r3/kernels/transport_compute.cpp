// SPDX-FileCopyrightText: © 2026 Tenstorrent Inc.
// SPDX-License-Identifier: Apache-2.0
//
// ISOLATED PERF BENCH (perf_experiments/compact_partial_transpose_r3) -- NOT the op's kernel.
//
// BENCH B, compute half.  See transport_writer.cpp for the bench's purpose.
//
// FLAT (VARIANT 0) is the op's CURRENT root chain, carried VERBATIM from
// kernels/rms_norm_compute.cpp's `is_root != 0` branch (D22): one `cb_wait_front` over the whole
// block's GATHER_SLOTS * rows landing window, the two MANDATORY reconfigs, then per tile-row one
// DEST window of GATHER_HALF pairwise `add_tiles(..., acc_to_dest = true)`, the raw-sfpi
// `stat_finalize_payload` at D17's <STRIDE=2, ITERS=4> VectorMode::C scope, and ONE `pack_tile`.
// Members compute nothing at all (D18 has pass A's reduce pack its partial straight into the
// handoff), so their compute kernel returns immediately.
//
// COMPACT (VARIANT 1) moves work ONTO the members and takes much more off the root:
//   every core   PACK    `rows` column-shaped partials -> `rows` COLUMNS of ONE tile, by `rows`
//                        `matmul_tiles` against a one-hot bank, ACCUMULATED in one DEST tile
//                        (matmul_tiles is DST += A*B) so the whole compact tile costs ONE pack.
//   the root     FOLD    ONE DEST window of GATHER_HALF pairwise add_tiles over the GATHER_SLOTS
//                        COMPACT tiles + ONE finalize + ONE pack, independent of BLOCK_ROWS.
//   every core   UNPACK  the multicast compact stat -> `rows` column-shaped tiles, `rows`
//                        matmuls against the same bank read with matmul's srcB `transpose` flag
//                        (E_r^T == F_r), DEST-batched.
//
// RAW-LLK / RAW-API JUSTIFICATION.
//   (1) The FOLD is raw LLK in the tree already (D22), for a reason recorded there: every
//       eltwise_chain element's apply runs on EVERY inner iteration, so a finalize placed after
//       an accumulating BinaryFpu would rsqrt a PARTIAL sum GATHER_HALF times.  Re-spelling the
//       BASELINE with helpers would make it slower than the op and inflate the candidate, so it
//       is carried verbatim and the candidate is the same code with ONE window.
//   (2) The PACK / UNPACK are a COLUMN PERMUTATION, and the FPU's only horizontal-mixing
//       primitive is the matmul:
//           pack   : C = partial_r x E_r ,  E_r[0][r] = 1  ->  C[i][r] = partial_r[i][0]
//           unpack : C = compact  x E_r^T,  E_r^T[r][0] = 1 ->  C[i][0] = compact[i][r]
//       No kernel_lib helper expresses it -- the eltwise / bcast / reduce families all preserve
//       or collapse the column axis, and transpose_wh transposes the WHOLE tile, which is not
//       this.  MEASURED authorisation (bench A, blackhole p150b 1350 MHz, ns per combine round at
//       the focus geometry GROUP_SIZE = 8 / BLOCK_ROWS = 8): the root chain goes 3024 -> 770 ns
//       (3.93x), and 11487 -> 1215 ns (9.45x) at BLOCK_ROWS = 32.
//   SAFETY NOTE any later refactor must preserve: the matmul sums 32 products, so EVERY column of
//   BOTH operands must be FINITE -- an inf/NaN in an unused column becomes inf*0 = NaN and
//   poisons column 0.  `pack_tile` writes a WHOLE tile from a fully-defined DEST, and the compact
//   gather ships WHOLE tiles precisely so that no landing column is ever un-written L1.
//
// THE FINALIZE SCOPE IS THE ONE THING THAT MUST CHANGE, and it is a CORRECTNESS change, not a
// perf one.  D17's shipped <STRIDE=2, ITERS=4> VectorMode::C body reaches only columns
// 0,2,..,14, which is right for a stat that lives in column 0 but SILENTLY WRONG on a compact
// tile from BLOCK_ROWS = 2 -- the odd rows' stats are never scaled and never rsqrt-ed.  Measured,
// twice (r2 and bench A): pcc 0.99730 with rel-RMS 1036 against a 0.04 bound, i.e. a bug that
// pcc alone would have waved through.  The compact finalize must be <1,8> VectorMode::C up to
// BLOCK_ROWS 16 and <1,8> VectorMode::RC above it (faces 1/3 hold columns 16..31).  Widening C to
// RC costs a flat +452 ns/block, measured.  D17's narrow scope stays valid for the non-combine
// (local) finalize, which really does only own column 0.

#include <cstdint>

#include "api/compute/compute_kernel_api.h"
#include "api/compute/compute_kernel_hw_startup.h"
#include "api/compute/matmul.h"
#include "api/compute/eltwise_unary/rsqrt.h"
#include "ttnn/cpp/ttnn/kernel_lib/perf_instrumentation.hpp"
#include "ttnn/cpp/ttnn/kernel_lib/eltwise_convenience.hpp"

namespace ckl = compute_kernel_lib;

#ifdef TRISC_MATH
#include "ckernel_sfpu_sqrt.h"
#include "ckernel_sfpu_binop_with_unary.h"

template <int STRIDE, int ITERS>
sfpi_inline void rms_stat_scale_body(uint32_t inv_w_bits, uint32_t eps_bits) {
    const sfpi::vFloat iw = ckernel::sfpu::Converter::as_float(inv_w_bits);
    const sfpi::vFloat ep = ckernel::sfpu::Converter::as_float(eps_bits);
    for (int i = 0; i < ITERS; ++i) {
        sfpi::dst_reg[0] = sfpi::dst_reg[0] * iw + ep;
        sfpi::dst_reg += STRIDE;
    }
}

template <int STRIDE, int ITERS>
sfpi_inline void rms_stat_rsqrt_body() {
    for (int i = 0; i < ITERS; ++i) {
        sfpi::vFloat t =
            ckernel::sfpu::_calculate_sqrt_body_<APPROX, true /*RECIPROCAL*/, false /*FAST_APPROX*/>(sfpi::dst_reg[0]);
        if constexpr (!DST_ACCUM_MODE) {
            t = sfpi::convert<sfpi::vFloat16b>(t, sfpi::RoundMode::Nearest);
        }
        sfpi::dst_reg[0] = t;
        sfpi::dst_reg += STRIDE;
    }
}

ALWI void fin_skip(uint32_t idst, uint32_t iw, uint32_t eps) {
    _llk_math_eltwise_unary_sfpu_params_(rms_stat_scale_body<2, 4>, idst, VectorMode::C, iw, eps);
    _llk_math_eltwise_unary_sfpu_params_(rms_stat_rsqrt_body<2, 4>, idst, VectorMode::C);
}
ALWI void fin_cfull(uint32_t idst, uint32_t iw, uint32_t eps) {
    _llk_math_eltwise_unary_sfpu_params_(rms_stat_scale_body<1, 8>, idst, VectorMode::C, iw, eps);
    _llk_math_eltwise_unary_sfpu_params_(rms_stat_rsqrt_body<1, 8>, idst, VectorMode::C);
}
ALWI void fin_rc(uint32_t idst, uint32_t iw, uint32_t eps) {
    _llk_math_eltwise_unary_sfpu_params_(rms_stat_scale_body<1, 8>, idst, VectorMode::RC, iw, eps);
    _llk_math_eltwise_unary_sfpu_params_(rms_stat_rsqrt_body<1, 8>, idst, VectorMode::RC);
}
#endif  // TRISC_MATH

// NB: `EPS` is a MACRO in llk_math_common_api.h, hence the RMS_ prefixes (the op does the same).
template <uint32_t RMS_IW, uint32_t RMS_EPS, uint32_t RMS_FIN>
ALWI void fin_payload(uint32_t dst) {
    if constexpr (RMS_FIN == 0) {
        MATH((fin_skip(dst, RMS_IW, RMS_EPS)));
    } else if constexpr (RMS_FIN == 1) {
        MATH((fin_cfull(dst, RMS_IW, RMS_EPS)));
    } else {
        MATH((fin_rc(dst, RMS_IW, RMS_EPS)));
    }
}

namespace {
constexpr uint32_t cb_x = 0;
constexpr uint32_t cb_bank = 1;
constexpr uint32_t cb_sum_handoff = 2;
constexpr uint32_t cb_partials_gathered = 3;
constexpr uint32_t cb_stat_handoff = 4;
constexpr uint32_t cb_mcast_in = 5;
constexpr uint32_t cb_row_final = 6;
}  // namespace

void kernel_main() {
    constexpr uint32_t VARIANT = get_compile_time_arg_val(0);
    constexpr uint32_t GROUP_SIZE = get_compile_time_arg_val(1);
    constexpr uint32_t BLOCK_ROWS = get_compile_time_arg_val(2);
    constexpr uint32_t INV_W_BITS = get_compile_time_arg_val(3);
    constexpr uint32_t EPS_BITS = get_compile_time_arg_val(4);
    constexpr uint32_t FIN = get_compile_time_arg_val(5);
    constexpr uint32_t DEST_BATCH = get_compile_time_arg_val(6);

    constexpr uint32_t GATHER_SLOTS = GROUP_SIZE + GROUP_SIZE % 2;
    constexpr uint32_t GATHER_HALF = GATHER_SLOTS / 2;
    constexpr bool COMPACT = (VARIANT == 1);
    // Per-ROUND gather window; a RAGGED last block (rows < BLOCK_ROWS) is shorter on the FLAT
    // path.  Derived identically here and in the writer.

    const uint32_t num_rows = get_arg_val<uint32_t>(0);
    const uint32_t is_root = get_arg_val<uint32_t>(1);

    // FLAT: only the root computes.  COMPACT: every active core does (pack + unpack).
    const bool participates = (num_rows != 0) && (COMPACT || is_root != 0);
    if (!participates) {
        return;
    }

    if constexpr (COMPACT) {
        compute_kernel_hw_startup<ckernel::SrcOrder::Reverse>(cb_x, cb_bank, cb_sum_handoff);
        // The resident partial shard and the one-hot bank are both published ONCE and never
        // popped -- the pack indexes them at a tile offset.
        cb_reserve_back(cb_x, num_rows);
        cb_push_back(cb_x, num_rows);
        cb_wait_front(cb_x, num_rows);
        cb_reserve_back(cb_bank, BLOCK_ROWS);
        cb_push_back(cb_bank, BLOCK_ROWS);
        cb_wait_front(cb_bank, BLOCK_ROWS);
    } else {
        compute_kernel_hw_startup(cb_partials_gathered, cb_partials_gathered, cb_stat_handoff);
    }

    const uint32_t num_blocks = (num_rows + BLOCK_ROWS - 1) / BLOCK_ROWS;

    for (uint32_t blk = 0; blk < num_blocks; ++blk) {
        const uint32_t r0 = blk * BLOCK_ROWS;
        const uint32_t rows = (num_rows - r0 < BLOCK_ROWS) ? (num_rows - r0) : BLOCK_ROWS;

        if constexpr (COMPACT) {
            // ---- every core: PACK this block's partials into ONE compact tile ---------------
            MaybeDeviceZoneScope("compute_member_pack");
            cb_reserve_back(cb_sum_handoff, 1);
            tile_regs_acquire();
            // No reconfig_data_format here: matmul_init owns the operand formats, and bench A's
            // proven pack/un-pack sequence does exactly this (adding one under SrcOrder::Reverse
            // is an untested operand-order risk for zero measured gain -- every CB here is fp32).
            matmul_init(cb_x, cb_bank, 0);
            for (uint32_t r = 0; r < rows; ++r) {
                matmul_tiles(cb_x, cb_bank, r0 + r, r, 0);
            }
            tile_regs_commit();
            pack_reconfig_data_format(cb_sum_handoff);
            tile_regs_wait();
            pack_tile(0, cb_sum_handoff);
            tile_regs_release();
            cb_push_back(cb_sum_handoff, 1);
        }

        if (is_root != 0) {
            // ---- the root: the D22 fused fold.  `rows` windows FLAT, ONE window COMPACT ------
            MaybeDeviceZoneScope("compute_root_fused");
            const uint32_t windows = COMPACT ? 1u : rows;
            const uint32_t window = COMPACT ? GATHER_SLOTS : (GATHER_SLOTS * rows);
            cb_wait_front(cb_partials_gathered, window);
            // NOT optional: pass A leaves the unpacker on the bf16 input and the packer on
            // cb_sum_handoff while the gather is fp32.  Omitting these gives a uniform ~1000x
            // scale error that HOLDS pcc at 0.9997 and shows only in rel-RMS.
            reconfig_data_format(cb_partials_gathered, cb_partials_gathered);
            pack_reconfig_data_format(cb_stat_handoff);
            add_tiles_init(cb_partials_gathered, cb_partials_gathered, /*acc_to_dest=*/true);
            rsqrt_tile_init();
            for (uint32_t w = 0; w < windows; ++w) {
                const uint32_t base = w * GATHER_SLOTS;
                tile_regs_acquire();
                for (uint32_t p = 0; p < GATHER_HALF; ++p) {
                    add_tiles(cb_partials_gathered, cb_partials_gathered, base + p, base + GATHER_HALF + p, 0);
                }
                fin_payload<INV_W_BITS, EPS_BITS, FIN>(0);
                tile_regs_commit();
                // Reserve/push PER WINDOW so the writer's multicast can start on the first
                // finalized row (the op measured that overlap at zero cost).
                cb_reserve_back(cb_stat_handoff, 1);
                tile_regs_wait();
                pack_tile(0, cb_stat_handoff);
                tile_regs_release();
                cb_push_back(cb_stat_handoff, 1);
            }
            cb_pop_front(cb_partials_gathered, window);
        }

        if constexpr (COMPACT) {
            // ---- every core: UN-PACK the compact stat into `rows` column-shaped tiles --------
            // srcB `transpose` reads E_r as E_r^T, so ONE bank serves both directions.
            MaybeDeviceZoneScope("compute_recv_unpack");
            cb_wait_front(cb_mcast_in, 1);
            cb_reserve_back(cb_row_final, rows);
            matmul_init(cb_mcast_in, cb_bank, /*transpose=*/1);
            for (uint32_t b = 0; b < rows; b += DEST_BATCH) {
                const uint32_t n = (rows - b < DEST_BATCH) ? (rows - b) : DEST_BATCH;
                tile_regs_acquire();
                for (uint32_t d = 0; d < n; ++d) {
                    matmul_tiles(cb_mcast_in, cb_bank, 0, b + d, d);
                }
                tile_regs_commit();
                pack_reconfig_data_format(cb_row_final);
                tile_regs_wait();
                for (uint32_t d = 0; d < n; ++d) {
                    pack_tile(d, cb_row_final, b + d);
                }
                tile_regs_release();
            }
            cb_push_back(cb_row_final, rows);
            cb_pop_front(cb_mcast_in, 1);
        }
    }
}
