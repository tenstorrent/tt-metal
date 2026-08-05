// SPDX-FileCopyrightText: © 2026 Tenstorrent Inc.
// SPDX-License-Identifier: Apache-2.0
//
// ISOLATED PERF BENCH (perf_experiments/gather_dual_noc) -- NOT the op.
//
// The compute half of the cross-core combine, IDENTICAL in every variant.  This bake-off
// changes only WHICH RISC issues the combine's NoC traffic and synchronization, so the
// root chain here is the op's CURRENT one, carried verbatim, and never varies:
//   * D16  the fold is ONE streaming chain call per row -- CopyTile each contiguous partial
//          into DEST and PACK-ACCUMULATE it into the row's accumulator tile
//          (L1Accumulation::SeedFirst).  The running sum lives in the fp32 CB, not in a
//          DEST register that is 16-bit at fp32_dest_acc_en == False.
//   * D17  the finalize is the raw-sfpi COLUMN-SCOPED chain (`StatFinalize`), which folds
//          *(1/W) and +eps into ONE pass over DEST and walks only the even lanes of faces
//          0/2.
//   * D19  the finalize READS the accumulator and WRITES cb_stat_handoff in ONE chain.
//
// Only the ROOT computes; every other core of the group is a pure sender/receiver.

#include <cstdint>

#include "api/compute/compute_kernel_api.h"
#include "api/compute/compute_kernel_hw_startup.h"
#include "api/compute/eltwise_binary.h"
#include "api/compute/eltwise_unary/rsqrt.h"
#include "ttnn/cpp/ttnn/kernel_lib/perf_instrumentation.hpp"
#include "ttnn/cpp/ttnn/kernel_lib/eltwise_convenience.hpp"

namespace ckl = compute_kernel_lib;

// ---------------------------------------------------------------------------------------
// RAW-LLK, CARRIED VERBATIM FROM THE OP (kernels/rms_norm_compute.cpp, Perf 1 / D17).
// Not a new bypass: the baseline this bench measures against IS this code, so re-spelling
// the finalize with helper calls would make the baseline slower than the op and inflate
// every candidate's speedup.  The op's own justification, in one line: mul_unary_tile /
// add_unary_tile / rsqrt_tile all hard-code VectorMode::RC and expose no VectorMode seam,
// and column parity is the SFPU's INNER walk axis so ITERATIONS cannot reach it; the stat
// is a REDUCE_ROW column vector whose only consumer reads column 0.
// ---------------------------------------------------------------------------------------
#ifdef TRISC_MATH
#include "ckernel_sfpu_sqrt.h"              // ckernel::sfpu::_calculate_sqrt_body_
#include "ckernel_sfpu_binop_with_unary.h"  // ckernel::sfpu::Converter::as_float

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

ALWI void stat_scale_col_skip(uint32_t idst, uint32_t inv_w_bits, uint32_t eps_bits) {
    _llk_math_eltwise_unary_sfpu_params_(rms_stat_scale_body<2, 4>, idst, VectorMode::C, inv_w_bits, eps_bits);
}
ALWI void rsqrt_tile_col_skip(uint32_t idst) {
    _llk_math_eltwise_unary_sfpu_params_(rms_stat_rsqrt_body<2, 4>, idst, VectorMode::C);
}
#endif  // TRISC_MATH

template <uint32_t RMS_INV_W, uint32_t RMS_EPS>
ALWI void stat_finalize_payload(uint32_t dst) {
    MATH((stat_scale_col_skip(dst, RMS_INV_W, RMS_EPS)));
    MATH((rsqrt_tile_col_skip(dst)));
}

template <uint32_t RMS_INV_W, uint32_t RMS_EPS>
struct StatFinalize : compute_kernel_lib::UnaryOp<StatFinalize<RMS_INV_W, RMS_EPS>, compute_kernel_lib::Dst::D0> {
    static ALWI void init() { rsqrt_tile_init(); }
    static ALWI void exec_impl(uint32_t slot_offset) { stat_finalize_payload<RMS_INV_W, RMS_EPS>(slot_offset); }
};

namespace {
constexpr uint32_t cb_partials_gathered = 11;
constexpr uint32_t cb_row_stat = 14;
constexpr uint32_t cb_stat_handoff = 15;
}  // namespace

void kernel_main() {
    constexpr uint32_t GROUP_SIZE = get_compile_time_arg_val(0);
    constexpr uint32_t BLOCK_ROWS = get_compile_time_arg_val(1);
    constexpr uint32_t INV_W_BITS = get_compile_time_arg_val(2);
    constexpr uint32_t EPS_BITS = get_compile_time_arg_val(3);
    // ---- ABLATION (/perf-measure's cumulative peel), PERF MEASUREMENT ONLY ------------
    // The isolated combine's critical path is the ROOT'S SERIAL FOLD -- GROUP_SIZE tile
    // copies per row per round, 256 of them on the focus geometry (the sibling bake-off
    // perf_experiments/hierarchical_gather_r2 measured 46519 -> 18751 ns just by splitting
    // that fold over 8 gatherers).  A TRANSPORT/SYNC lever therefore shows up as a couple of
    // percent of a number it does not own.  ABLATE_FOLD strips the fold + finalize PAYLOAD
    // and keeps every CB handshake and trip count exactly as it was, which exposes the
    // combine's transport and synchronization -- the same peel that produced the op's
    // measured 16097 ns residual.  Output is garbage by construction, so an ablated run is
    // NEVER correctness-gated; every variant is gated in the un-ablated mode.
    constexpr uint32_t ABLATE_FOLD = get_compile_time_arg_val(4);
    // FOLD_STYLE: which root chain the BASELINE is.  1 = the op's CURRENT fused pairwise
    // DEST fold (Perf 2 / D22), carried verbatim -- this is the honest baseline.  0 = the
    // pre-Perf-2 per-row streaming pack-accumulate chain (D16/D19) that
    // perf_experiments/hierarchical_gather_r2 carried; kept only so the two bake-offs can be
    // read against each other.  It matters a LOT for a TRANSPORT idea: the D16 chain costs
    // ~23.5 us of the focus geometry's 41.9 us here and hides far more of the combine's NoC
    // traffic than the op actually does today.
    constexpr uint32_t FOLD_STYLE = get_compile_time_arg_val(5);
    // GATHER_SLOTS == GROUP_SIZE rounded UP TO EVEN: the fused fold walks a row's partials
    // PAIRWISE (halves p and p + GATHER_SLOTS/2), so an odd group needs one boot-zeroed pad
    // slot to pair against.  Every geometry this bench builds has an even group, so
    // GATHER_SLOTS == GROUP_SIZE and the ring layout is byte-identical to the writer's.
    constexpr uint32_t GATHER_HALF = GROUP_SIZE / 2;
    static_assert(GROUP_SIZE % 2 == 0, "this bench builds even groups only");

    const uint32_t num_rows = get_arg_val<uint32_t>(0);
    const uint32_t is_root = get_arg_val<uint32_t>(1);

    if (num_rows == 0 || is_root == 0) {
        return;  // inactive, or a plain member -- the op returns here too
    }

    // Every CB in this bench is fp32, so srcA == srcB at boot.
    compute_kernel_hw_startup(cb_partials_gathered, cb_partials_gathered, cb_stat_handoff);

    // (OneUpfront, OneAtEnd) is the policy pair L1 accumulation requires: the whole call
    // pins ONE output tile.  Hence one call per row.
    constexpr auto FOLD_TO_ROW_STAT = ckl::output(
        cb_row_stat,
        ckl::ReservePolicy::OneUpfront,
        ckl::PushPolicy::OneAtEnd,
        ckl::DataFormatReconfig::Enabled,
        ckl::PackRelu::Disabled,
        ckl::L1Accumulation::SeedFirst);

    const uint32_t num_blocks = (num_rows + BLOCK_ROWS - 1) / BLOCK_ROWS;
    for (uint32_t blk = 0; blk < num_blocks; ++blk) {
        const uint32_t r0 = blk * BLOCK_ROWS;
        const uint32_t rows = (num_rows - r0 < BLOCK_ROWS) ? (num_rows - r0) : BLOCK_ROWS;
        if constexpr (ABLATE_FOLD != 0) {
            // Same CB traffic, no math: consume the whole gather ring and publish `rows`
            // stat pages, so the reader/writer see a byte-identical handshake schedule.
            MaybeDeviceZoneScope("compute_root_sum");
            cb_wait_front(cb_partials_gathered, GROUP_SIZE * BLOCK_ROWS);
            cb_pop_front(cb_partials_gathered, GROUP_SIZE * BLOCK_ROWS);
            cb_reserve_back(cb_stat_handoff, rows);
            cb_push_back(cb_stat_handoff, rows);
            continue;
        }
        if constexpr (FOLD_STYLE != 0) {
            // ---- THE OP'S CURRENT FUSED ROOT CHAIN (Perf 2 / D22), carried verbatim -----
            // The whole block's gather window is waited/popped ONCE: the pairwise walk
            // addresses two tiles of the same CB at a stride, which a per-tile wait cannot
            // express.  Legal as it stands -- the writer publishes the block atomically and
            // the CB is sized to that same window.  Reserve/push are PER TILE-ROW so the
            // stat multicast can start on the first finalized row.
            MaybeDeviceZoneScope("compute_root_fused");
            cb_wait_front(cb_partials_gathered, GROUP_SIZE * BLOCK_ROWS);
            reconfig_data_format(cb_partials_gathered, cb_partials_gathered);
            pack_reconfig_data_format(cb_stat_handoff);
            add_tiles_init(cb_partials_gathered, cb_partials_gathered, /*acc_to_dest=*/true);
            rsqrt_tile_init();
            for (uint32_t r = 0; r < rows; ++r) {
                const uint32_t base = r * GROUP_SIZE;
                tile_regs_acquire();
                for (uint32_t p = 0; p < GATHER_HALF; ++p) {
                    add_tiles(cb_partials_gathered, cb_partials_gathered, base + p, base + GATHER_HALF + p, 0);
                }
                stat_finalize_payload<INV_W_BITS, EPS_BITS>(0);
                tile_regs_commit();
                cb_reserve_back(cb_stat_handoff, 1);
                tile_regs_wait();
                pack_tile(0, cb_stat_handoff);
                tile_regs_release();
                cb_push_back(cb_stat_handoff, 1);
            }
            cb_pop_front(cb_partials_gathered, GROUP_SIZE * BLOCK_ROWS);
            continue;
        }
        {
            MaybeDeviceZoneScope("compute_root_sum");
            for (uint32_t r = 0; r < rows; ++r) {
                ckl::eltwise_chain(
                    ckl::EltwiseShape::tiles(GROUP_SIZE),
                    ckl::CopyTile<ckl::input(cb_partials_gathered)>{},
                    ckl::PackTile<FOLD_TO_ROW_STAT>{});
            }
            if (rows < BLOCK_ROWS) {
                cb_pop_front(cb_partials_gathered, GROUP_SIZE * (BLOCK_ROWS - rows));
            }
        }
        {
            MaybeDeviceZoneScope("compute_root_finalize");
            ckl::eltwise_chain(
                ckl::EltwiseShape::tiles(rows),
                ckl::CopyTile<ckl::input(cb_row_stat)>{},
                StatFinalize<INV_W_BITS, EPS_BITS>{},
                ckl::PackTile<ckl::output(cb_stat_handoff)>{});
        }
    }
}
