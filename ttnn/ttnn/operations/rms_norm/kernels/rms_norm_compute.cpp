// SPDX-FileCopyrightText: © 2026 Tenstorrent Inc.
// SPDX-License-Identifier: Apache-2.0

// Compute kernel for rms_norm — two-pass streaming RMSNorm per row-chunk.
//
// Per row-chunk (32 rows × Wt tiles):
//   Phase 0 (HAS_GAMMA): tilize 1 gamma stick → Wt gamma tiles (asymmetric).
//   Phase 1a (INPUT_IS_RM): tilize 32 input sticks → Wt input tiles.
//   Phase 1b: external cb_wait_front(cb_input_tiles, Wt) so stage A can index Wt tiles.
//   Phase 2 (Stage A): square each input tile → cb_x_sq. NoWaitNoPop on input
//                      (input tiles stay queued for Stage D).
//   Phase 3 (Stage B): reduce SUM/REDUCE_ROW with scaler 1/W → cb_mean_sq (1 tile).
//   Phase 4 (Stage C): transform_in_place(cb_mean_sq, +eps + rsqrt).
//   Phase 5 (Stage D): x · rsqrt with BroadcastDim::Col →
//                      cb_output_tiles (no gamma) OR cb_x_norm (with gamma).
//                      Pops Wt from cb_input_tiles; cb_mean_sq held (WaitNoPop).
//   Phase 6 (Stage E, HAS_GAMMA only): x_norm · gamma with BroadcastDim::Row → cb_output_tiles.
//   Phase 7 (OUTPUT_IS_RM): untilize cb_output_tiles → cb_output_rm.
//   Phase 8: pop cb_mean_sq (1) to release rsqrt scaler for next chunk.
//
// End of kernel: pop cb_scaler (1 or 2 tiles).
//
// All work goes through kernel_lib helpers; no raw tile_regs / copy_tile / pack_tile.

#include <cstdint>

#include "api/compute/compute_kernel_api.h"
#include "api/compute/eltwise_unary/binop_with_scalar.h"
#include "api/compute/eltwise_unary/rsqrt.h"
#include "ttnn/cpp/ttnn/kernel_lib/eltwise_chain.hpp"
#include "ttnn/cpp/ttnn/kernel_lib/eltwise_misc.hpp"
#include "ttnn/cpp/ttnn/kernel_lib/reduce_helpers_compute.hpp"
#include "ttnn/cpp/ttnn/kernel_lib/streaming_reduce_helpers.hpp"
#include "ttnn/cpp/ttnn/kernel_lib/tilize_helpers.hpp"
#include "ttnn/cpp/ttnn/kernel_lib/untilize_helpers.hpp"

namespace ckl = compute_kernel_lib;
using ckl::BinaryDataFormatReconfig;
using ckl::BinaryFpu;
using ckl::BinaryFpuOp;
using ckl::BroadcastDim;
using ckl::CbIndexMode;
using ckl::CopyTile;
using ckl::CopyTilePolicy;
using ckl::CopyTileReconfig;
using ckl::Dst;
using ckl::PackTile;
using ckl::PackTileIndexMode;
using ckl::PackTilePolicy;
using ckl::PackTileReconfig;
using ckl::Square;

namespace {
// CB ids (must match program descriptor).
constexpr uint32_t cb_input_raw_rm = 0;
constexpr uint32_t cb_input_tiles = 1;
constexpr uint32_t cb_gamma_rm = 2;
constexpr uint32_t cb_gamma_tiled = 3;
constexpr uint32_t cb_scaler = 4;
constexpr uint32_t cb_output_tiles = 16;
constexpr uint32_t cb_output_rm = 17;
constexpr uint32_t cb_x_sq = 24;
constexpr uint32_t cb_mean_sq = 25;
constexpr uint32_t cb_x_norm = 26;
}  // namespace

void kernel_main() {
    // ---- Compile-time args ----
    constexpr uint32_t eps_bits = get_compile_time_arg_val(0);
    constexpr bool HAS_PARTIAL_W = get_compile_time_arg_val(1) != 0;
    constexpr bool INPUT_IS_RM = get_compile_time_arg_val(2) != 0;
    constexpr bool OUTPUT_IS_RM = get_compile_time_arg_val(3) != 0;
    constexpr bool HAS_GAMMA = get_compile_time_arg_val(4) != 0;
    constexpr uint32_t Wt = get_compile_time_arg_val(5);
    constexpr uint32_t num_chunks = get_compile_time_arg_val(6);

    // ---- Boot: hw startup. Picks the first chain's (CbA, CbB, CbOut) triple
    //      = (cb_input_tiles, cb_input_tiles, cb_x_sq) per the chain helper's
    //      D5 contract. tilize/untilize/reduce/transform_in_place each handle
    //      their own per-element init internally.
    compute_kernel_hw_startup(cb_input_tiles, cb_input_tiles, cb_x_sq);

    constexpr auto reduce_block_shape = ckl::ReduceInputBlockShape::row(Wt);
    constexpr auto partial_scaler =
        HAS_PARTIAL_W ? ckl::ReducePartialScaler::last_tile_at(1) : ckl::ReducePartialScaler::none();

    for (uint32_t chunk = 0; chunk < num_chunks; ++chunk) {
        // Phase 0: tilize gamma (1 stick → Wt tiles). Asymmetric mode:
        // cb_gamma_rm page = padded_row_bytes; compute waits 1 row, produces Wt tiles.
        if constexpr (HAS_GAMMA) {
            compute_kernel_lib::tilize<
                Wt,
                cb_gamma_rm,
                cb_gamma_tiled,
                compute_kernel_lib::tilize_config::InitUninitMode::InitAndUninit,
                compute_kernel_lib::tilize_config::WaitMode::WaitBlock,
                compute_kernel_lib::tilize_config::ReconfigureRegisterDatatypeMode::UnpackAndPackReconfigure>(
                /*num_blocks=*/1, /*total_input_pages=*/1);
        }

        // Phase 1a: tilize input sticks → Wt tiles (RM input only). Symmetric mode:
        // cb_input_raw_rm has tile-sized pages from read_sticks_for_tilize<TILE>.
        if constexpr (INPUT_IS_RM) {
            compute_kernel_lib::tilize<
                Wt,
                cb_input_raw_rm,
                cb_input_tiles,
                compute_kernel_lib::tilize_config::InitUninitMode::InitAndUninit,
                compute_kernel_lib::tilize_config::WaitMode::WaitBlock,
                compute_kernel_lib::tilize_config::ReconfigureRegisterDatatypeMode::UnpackAndPackReconfigure>(
                /*num_blocks=*/1);
        }

        // Phase 1b: wait for Wt input tiles so stage A can index BlockIter [0, Wt).
        cb_wait_front(cb_input_tiles, Wt);

        // Phase 2 — Stage A: x → x^2. NoWaitNoPop on input (caller pre-waited; tiles
        // stay queued for stage D's WaitAndPop). BlockIter walks tile 0..Wt-1.
        //
        // Format reconfig (CopyTileReconfig::Input + PackTileReconfig::Output) is
        // critical when gamma_dtype != input_dtype on the TILE input path: after
        // Phase 0's gamma tilize the unpack/pack registers are configured for
        // gamma_dtype, and without these reconfigs Stage A would read fp32
        // cb_input_tiles via the bf16 srcA path (or vice versa) and produce
        // wildly wrong x² values. For the RM input path the subsequent Phase 1a
        // tilize re-establishes the input dtype's format, but expressing the
        // reconfig at Stage A unifies both paths and is compile-time-elided when
        // the prior pack/srcA CB already matched.
        ckl::eltwise_chain(
            Wt,
            CopyTile<
                cb_input_tiles,
                Dst::D0,
                CopyTilePolicy::NoWaitNoPop,
                CbIndexMode::BlockIter,
                CopyTileReconfig::Input>{},
            Square<Dst::D0>{},
            PackTile<
                cb_x_sq,
                Dst::D0,
                PackTilePolicy::PerTileReserveAndPush,
                PackTileIndexMode::FirstTile,
                PackTileReconfig::Output>{});

        // Phase 3 — Stage B: mean(x^2) via SUM/REDUCE_ROW with scaler=1/W.
        // BulkWaitBulkPop: helper waits Wt tiles upfront, processes with indexed access,
        // pops Wt at end. partial_scaler zeros padded-W contributions when HAS_PARTIAL_W.
        ckl::reduce<ckernel::PoolType::SUM, ckernel::ReduceDim::REDUCE_ROW, ckl::ReduceInputPolicy::BulkWaitBulkPop>(
            cb_x_sq,
            cb_scaler,
            cb_mean_sq,
            reduce_block_shape,
            ckl::ReduceInputMemoryLayout::contiguous(),
            ckl::NoAccumulation{},
            ckl::NoOp{},
            partial_scaler);

        // Phase 4 — Stage C: cb_mean_sq[0] ← rsqrt(cb_mean_sq[0] + eps).
        // transform_in_place pops 1 tile from cb_mean_sq, runs lambda on DST[0],
        // packs back into cb_mean_sq. SRCA/packer reconfig bundled in.
        ckl::transform_in_place(cb_mean_sq, [&](uint32_t dst) {
            ckernel::binop_with_scalar_tile_init();
            ckernel::add_unary_tile(dst, eps_bits);
            ckernel::rsqrt_tile_init();
            ckernel::rsqrt_tile</*fast_and_approx=*/false>(dst);
        });

        // Phase 5 — Stage D: x · rsqrt with BroadcastDim::Col.
        // cb_mean_sq is REDUCE_ROW output (col 0 valid) → Col broadcast.
        // A side: WaitAndPop FirstTile (streams x, pops 1 per iter).
        // B side: WaitNoPop FirstTile (held for all Wt iters, never popped).
        if constexpr (HAS_GAMMA) {
            ckl::eltwise_chain(
                Wt,
                BinaryFpu<
                    cb_input_tiles,
                    cb_mean_sq,
                    cb_x_norm,
                    BinaryFpuOp::Mul,
                    BroadcastDim::Col,
                    BinaryDataFormatReconfig::InputAndOutput,
                    CopyTilePolicy::WaitAndPop,
                    CopyTilePolicy::WaitNoPop,
                    CbIndexMode::FirstTile,
                    Dst::D0,
                    CbIndexMode::FirstTile>{},
                PackTile<cb_x_norm, Dst::D0, PackTilePolicy::PerTileReserveAndPush>{});

            // Phase 6 — Stage E: x_norm · gamma with BroadcastDim::Row.
            // gamma row 0 valid → Row broadcast. Both sides WaitAndPop FirstTile —
            // each iter pops one from x_norm and one from gamma_tiled.
            ckl::eltwise_chain(
                Wt,
                BinaryFpu<
                    cb_x_norm,
                    cb_gamma_tiled,
                    cb_output_tiles,
                    BinaryFpuOp::Mul,
                    BroadcastDim::Row,
                    BinaryDataFormatReconfig::InputAndOutput,
                    CopyTilePolicy::WaitAndPop,
                    CopyTilePolicy::WaitAndPop,
                    CbIndexMode::FirstTile,
                    Dst::D0,
                    CbIndexMode::FirstTile>{},
                PackTile<cb_output_tiles, Dst::D0, PackTilePolicy::PerTileReserveAndPush>{});
        } else {
            ckl::eltwise_chain(
                Wt,
                BinaryFpu<
                    cb_input_tiles,
                    cb_mean_sq,
                    cb_output_tiles,
                    BinaryFpuOp::Mul,
                    BroadcastDim::Col,
                    BinaryDataFormatReconfig::InputAndOutput,
                    CopyTilePolicy::WaitAndPop,
                    CopyTilePolicy::WaitNoPop,
                    CbIndexMode::FirstTile,
                    Dst::D0,
                    CbIndexMode::FirstTile>{},
                PackTile<cb_output_tiles, Dst::D0, PackTilePolicy::PerTileReserveAndPush>{});
        }

        // Phase 7: untilize cb_output_tiles → cb_output_rm (Wt tiles → 32 sticks worth).
        if constexpr (OUTPUT_IS_RM) {
            compute_kernel_lib::untilize<
                Wt,
                cb_output_tiles,
                cb_output_rm,
                compute_kernel_lib::untilize_config::InitUninitMode::InitAndUninit,
                compute_kernel_lib::untilize_config::WaitMode::WaitBlock,
                compute_kernel_lib::untilize_config::ReconfigureRegisterDatatypeMode::UnpackAndPackReconfigure>(
                /*num_blocks=*/1);
        }

        // Phase 8: release rsqrt scaler for next chunk.
        cb_pop_front(cb_mean_sq, 1);
    }

    // End of kernel: release scaler tile(s).
    cb_pop_front(cb_scaler, HAS_PARTIAL_W ? 2 : 1);
}
