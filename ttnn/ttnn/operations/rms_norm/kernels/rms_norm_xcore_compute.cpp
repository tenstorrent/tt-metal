// SPDX-FileCopyrightText: © 2026 Tenstorrent Inc.
// SPDX-License-Identifier: Apache-2.0

// Cross-core W-split compute kernel for rms_norm (op_design.md §1 lamp 2, §5).
//
// The reduced dim W is split across a group of K cores (WIDTH/BLOCK shard, or a
// logical W-split of a wide interleaved input). Each core holds a W-slice of the
// same tile-rows resident in L1 (zero-copy sharded cb_x_in). The RMS statistic
// spans all of W, so it is a CROSS-CORE reduction:
//
//   pass 1 (local) : per tile-row, Σ_slice x²·(1/W)             -> cb_stat_local (partial)
//   cross-core     : writer gathers the K partials to the group MASTER
//   master combine : Σ_k partial_k  -> mean ; (+eps, rsqrt)     -> cb_stat_handoff (1/RMS)
//   cross-core     : writer broadcasts 1/RMS back to the group  -> cb_stat_global
//   pass 2 (local) : per tile-row, x·rstd·gamma                 -> cb_out (sharded)
//
// One gather+broadcast round per tile-row (fully synchronous, monotone counter
// semaphores in the writer) so cb_gather stays K tiles — no CB grows with the
// tile-row count. The compute<->writer handoff is via cb_stat_local (compute ->
// writer), cb_gather (writer -> master compute), cb_stat_handoff (master compute
// -> writer), cb_stat_global (writer -> compute). cb_stat_handoff is a SEPARATE
// CB from cb_stat_global (never reuse one CB for both, §7 two-consumer trap).
//
// Reused helpers: square / reduce (accumulate) / eltwise_chain mul (indexed
// resident access, same lower-level form as the R3 resident path). The one
// raw-LLK block is the master's K-partial fold + (+eps, rsqrt): no kernel-lib
// helper reduces across N distinct CB tiles element-wise with an SFPU postop —
// this is the sanctioned tensix_all_reduce fold pattern
// (examples/tensix_all_reduce _REDUCE_KERNEL), with the RMS finalize appended
// in the same DST window (identical to the streaming path's transform_in_place
// lambda body: add_unary_tile(eps) then rsqrt_tile).

#include <cstdint>

#include "api/compute/compute_kernel_api.h"
#include "api/compute/compute_kernel_hw_startup.h"
#include "api/compute/tile_move_copy.h"
#include "api/compute/eltwise_binary.h"
#include "api/compute/eltwise_unary/rsqrt.h"
#include "api/compute/eltwise_unary/binop_with_scalar.h"
#include "api/compute/reconfig_data_format.h"
#include "ttnn/cpp/ttnn/kernel_lib/eltwise_convenience.hpp"
#include "ttnn/cpp/ttnn/kernel_lib/reduce_helpers_compute.hpp"
#include "ttnn/cpp/ttnn/kernel_lib/streaming_reduce_helpers.hpp"

namespace ckl = compute_kernel_lib;

namespace {
constexpr uint32_t cb_x_in = 1;          // sharded input W-slice (zero-copy, resident)
constexpr uint32_t cb_scaler = 2;        // 1/W reduce scaler (+partial), bf16
constexpr uint32_t cb_gamma = 3;         // gamma W-slice tiles (held)
constexpr uint32_t cb_gather = 5;        // master: K partials gathered (fp32)
constexpr uint32_t cb_stat_handoff = 6;  // master: 1/RMS -> writer broadcast (fp32)
constexpr uint32_t cb_stat_global = 7;   // 1/RMS received (fp32)
constexpr uint32_t cb_out = 16;          // sharded output W-slice (zero-copy)
constexpr uint32_t cb_xsq = 24;          // x^2 (pass-1 intermediate)
constexpr uint32_t cb_stat_local = 25;   // per-tile-row local partial Σx²·(1/W) (fp32)
constexpr uint32_t cb_norm = 26;         // x·rstd (pass-2 intermediate)
}  // namespace

void kernel_main() {
    constexpr uint32_t PER_W_T = get_compile_time_arg_val(0);   // W-tiles this core holds
    constexpr uint32_t HT_LOCAL = get_compile_time_arg_val(1);  // tile-rows this core holds
    constexpr uint32_t K = get_compile_time_arg_val(2);         // group size (cores over W)
    constexpr bool HAS_GAMMA = get_compile_time_arg_val(3) != 0;
    constexpr bool HAS_PARTIAL_W = get_compile_time_arg_val(4) != 0;
    constexpr uint32_t eps_bits = get_compile_time_arg_val(5);

    const uint32_t vwt = get_arg_val<uint32_t>(0);  // valid W-tiles (<= PER_W_T)
    const uint32_t is_partial_holder = get_arg_val<uint32_t>(1);
    const uint32_t is_master = get_arg_val<uint32_t>(2);

    compute_kernel_hw_startup(cb_x_in, cb_scaler, cb_out);

    // Per-tile reduce (1 tile-row, 1 W-tile per chunk; accumulate over vwt tiles).
    constexpr auto reduce_shape = ckl::ReduceInputBlockShape::of(1, 1, 1);
    constexpr auto one_tile = ckl::EltwiseShape::of(1, 1);
    constexpr auto partial_scaler_sel = ckl::ReducePartialScaler::last_tile_at(1);

    // Arm the resident sharded input W-slice once (whole shard held; indexed access).
    const uint32_t shard_tiles = HT_LOCAL * PER_W_T;
    cb_reserve_back(cb_x_in, shard_tiles);
    cb_push_back(cb_x_in, shard_tiles);
    cb_wait_front(cb_x_in, shard_tiles);

    if constexpr (HAS_GAMMA) {
        cb_wait_front(cb_gamma, vwt);  // gamma W-slice held (read once by the reader)
    }

    for (uint32_t t = 0; t < HT_LOCAL; ++t) {
        // ---------- Pass 1: local partial Σ_slice x²·(1/W) ----------
        for (uint32_t w = 0; w < vwt; ++w) {
            const uint32_t xin_off = t * PER_W_T + w;
            ckl::eltwise_chain(
                one_tile,
                ckl::BinaryFpu<
                    cb_x_in,
                    cb_x_in,
                    ckl::BinaryFpuOp::Mul,
                    ckl::BroadcastDim::None,
                    ckl::InputLifecycle::CallerManaged,
                    ckl::InputLifecycle::CallerManaged,
                    ckl::BinaryDataFormatReconfig::Input,
                    ckl::Dst::D0,
                    ckl::OperandKind::Block,
                    ckl::OperandKind::Block,
                    ckl::TileOffset::Set,
                    ckl::TileOffset::Set>{xin_off, xin_off},
                ckl::PackTile<cb_xsq, ckl::OutputLifecycle::Streaming, ckl::PackTileReconfig::Output>{});
            const ckl::ReducePartialScaler ps = (HAS_PARTIAL_W && is_partial_holder && (w + 1 == vwt))
                                                    ? partial_scaler_sel
                                                    : ckl::ReducePartialScaler::none();
            ckl::reduce<ckernel::PoolType::SUM, ckernel::ReduceDim::REDUCE_ROW, cb_xsq, cb_scaler, cb_stat_local>(
                reduce_shape,
                ckl::ReduceInputMemoryLayout::contiguous(),
                ckl::Accumulate::at(cb_stat_local, w),
                ckl::NoOp{},
                ps);
        }
        // reduce pushed cb_stat_local (1 tile). Writer gathers it to the master.

        // ---------- Master: fold K partials -> mean; (+eps, rsqrt) -> 1/RMS ----------
        if (is_master) {
            cb_wait_front(cb_gather, K);  // writer gathered K partials for this tile-row
            cb_reserve_back(cb_stat_handoff, 1);
            // Raw-LLK fold needs the data-format reconfig the helpers do implicitly:
            // pass 1 left the unpacker configured for cb_xsq/cb_scaler; the fold reads
            // cb_gather (fp32) and packs cb_stat_handoff (fp32). Without this the
            // unpacker-A src-format LLK_ASSERT trips (hang).
            reconfig_data_format(cb_gather, cb_gather);
            pack_reconfig_data_format(cb_stat_handoff);
            tile_regs_acquire();
            // dst0 = Σ_{k=0..K-1} cb_gather[k]  (col0 holds the per-row partial;
            // other columns are ignored downstream via BroadcastDim::Col).
            uint32_t first_pair = 0;
            if (K & 1u) {
                copy_tile_to_dst_init_short(cb_gather);
                copy_tile(cb_gather, 0, 0);
                first_pair = 1;
            }
            if (K > 1) {
                add_tiles_init(cb_gather, cb_gather, /*acc_to_dest=*/true);
                for (uint32_t k = first_pair; k < K; k += 2) {
                    add_tiles(cb_gather, cb_gather, k, k + 1, 0);
                }
            }
            // RMS finalize on the summed mean (same body as the streaming
            // transform_in_place lambda): rstd = rsqrt(mean + eps).
            binop_with_scalar_tile_init();
            add_unary_tile(0, eps_bits);
            rsqrt_tile_init();
            rsqrt_tile(0);
            tile_regs_commit();
            tile_regs_wait();
            pack_tile(0, cb_stat_handoff);
            tile_regs_release();
            cb_push_back(cb_stat_handoff, 1);
            cb_pop_front(cb_gather, K);
        }

        // ---------- Pass 2: x·rstd·gamma over the W-slice (all cores) ----------
        cb_wait_front(cb_stat_global, 1);  // 1/RMS for this tile-row (broadcast landed)
        for (uint32_t w = 0; w < PER_W_T; ++w) {
            const uint32_t xin_off = t * PER_W_T + w;
            // x·rstd (rstd is REDUCE_ROW -> column-shaped -> BroadcastDim::Col).
            ckl::eltwise_chain(
                one_tile,
                ckl::BinaryFpu<
                    cb_x_in,
                    cb_stat_global,
                    ckl::BinaryFpuOp::Mul,
                    ckl::BroadcastDim::Col,
                    ckl::InputLifecycle::CallerManaged,
                    ckl::InputLifecycle::HeldBulk,
                    ckl::BinaryDataFormatReconfig::Input,
                    ckl::Dst::D0,
                    ckl::OperandKind::Block,
                    ckl::OperandKind::Col,
                    ckl::TileOffset::Set,
                    ckl::TileOffset::Unset>{xin_off, 0},
                ckl::PackTile<cb_norm, ckl::OutputLifecycle::Streaming, ckl::PackTileReconfig::Output>{});

            if (HAS_GAMMA && (w < vwt)) {
                // norm·gamma (gamma is [1,W] -> row-shaped -> BroadcastDim::Row).
                ckl::eltwise_chain(
                    one_tile,
                    ckl::BinaryFpu<
                        cb_norm,
                        cb_gamma,
                        ckl::BinaryFpuOp::Mul,
                        ckl::BroadcastDim::Row,
                        ckl::InputLifecycle::Streaming,
                        ckl::InputLifecycle::CallerManaged,
                        ckl::BinaryDataFormatReconfig::Input,
                        ckl::Dst::D0,
                        ckl::OperandKind::Scalar,
                        ckl::OperandKind::Row,
                        ckl::TileOffset::Unset,
                        ckl::TileOffset::Set>{0, w},
                    ckl::PackTile<cb_out, ckl::OutputLifecycle::Streaming, ckl::PackTileReconfig::Output>{});
            } else {
                // no gamma, or a trailing padding tile (output discarded on read-back).
                ckl::copy<cb_norm, cb_out>(one_tile);
            }
        }
        cb_pop_front(cb_stat_global, 1);
    }

    cb_pop_front(cb_x_in, shard_tiles);
    if constexpr (HAS_GAMMA) {
        cb_pop_front(cb_gamma, vwt);
    }
    cb_pop_front(cb_scaler, HAS_PARTIAL_W ? 2 : 1);
}
