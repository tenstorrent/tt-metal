// SPDX-FileCopyrightText: © 2026 Tenstorrent Inc.
// SPDX-License-Identifier: Apache-2.0

// Reader for rms_norm — WIDTH_SHARDED / BLOCK_SHARDED cross-core reduction (R5).
//
// This is the dataflow face of the design's dependent-axis SCHEME-CHANGE: the
// hidden W is split across a reduction GROUP of cores, so the RMS denominator
// spans core boundaries. Each core owns a contiguous W-slice of its tile-rows
// (a resident L1 shard). Per tile-row the flow is:
//
//   PASS 1  : stream the LOCAL W-slice -> cb_input_tiles  (compute squares +
//             reduces it into cb_partial = local Σx² / W_global, i.e. this
//             core's partial contribution to the GLOBAL mean(x²)).
//   COMBINE : reduce-root gather + broadcast-back (this file's cross-core core):
//               * non-root: unicast cb_partial -> root's cb_gather[my_index],
//                 bump the root's `progress` semaphore, then act as a
//                 ReceiverPipe for the finalized 1/rms broadcast -> cb_sumsq.
//               * root: copy its own partial into cb_gather[my_index], wait for
//                 all (GROUP_SIZE-1) peers, push cb_gather (compute folds the
//                 partials -> global mean(x²) -> transform_in_place rsqrt ->
//                 cb_rms_src), then SenderPipe-broadcast cb_rms_src -> cb_sumsq
//                 on the whole group rectangle (loopback fills its own).
//   PASS 2  : stream the LOCAL W-slice + local gamma slice again; compute
//             normalizes x·(1/rms)·gamma -> cb_output_tiles; writer drains its
//             own output shard.
//
// The gather is many->one (raw unicast + a `progress` counter; no mcast form
// exists for many->one). The broadcast is one->many via the mcast_pipe helper
// (SenderPipe/ReceiverPipe + Mcast2D host wire) — its Flag handshake is
// per-round self-contained (sender wait(ack)+set(0), receiver wait+clear), so
// multi-tile-row cores just loop. `progress` is likewise reset per round.
//
// GROUPS ARE RECTANGULAR (mcast constraint): BLOCK groups are horizontal core
// lines; a WIDTH group is the shard-grid bounding box. Ragged / RM / any
// non-rectangular geometry is routed to the interleaved streaming path by the
// program descriptor and never reaches this kernel.
//
// TILE input only (WIDTH/BLOCK + ROW_MAJOR input is EXCLUDED). ttnn's implicit
// tile zero-padding covers a non-tile-aligned global W: the padding columns of
// the last global W-tile are exactly 0, so they add 0 to Σx² and the ×1/W_global
// mean is over the true element count — no partial scaler needed here.

#include <stdint.h>

#include "api/dataflow/dataflow_api.h"
#include "api/dataflow/noc.h"
#include "api/dataflow/noc_semaphore.h"
#include "api/dataflow/endpoints.h"
#include "api/tensor/noc_traits.h"
#include "hostdevcommon/common_values.hpp"
#include "ttnn/cpp/ttnn/kernel_lib/reduce_helpers_dataflow.hpp"
#include "ttnn/cpp/ttnn/kernel_lib/mcast_pipe.hpp"

using namespace dataflow_kernel_lib;

namespace {
constexpr uint32_t cb_input_tiles = 1;
constexpr uint32_t cb_scaler = 2;
constexpr uint32_t cb_gamma_rm = 3;
constexpr uint32_t cb_gamma_tiles = 4;
constexpr uint32_t cb_sumsq = 25;
constexpr uint32_t cb_partial = 27;
constexpr uint32_t cb_gather = 28;
constexpr uint32_t cb_rms_src = 29;
constexpr uint32_t TILE_W = 32;

// Zero an L1 page word-by-word (RM gamma padding tail).
FORCE_INLINE void zero_l1_page(uint32_t l1, uint32_t nbytes) {
    volatile tt_l1_ptr uint32_t* p = reinterpret_cast<volatile tt_l1_ptr uint32_t*>(l1);
    for (uint32_t i = 0; i < nbytes / 4; ++i) {
        p[i] = 0;
    }
}
}  // namespace

void kernel_main() {
    constexpr uint32_t HAS_GAMMA = get_compile_time_arg_val(0);
    constexpr uint32_t scaler_bits = get_compile_time_arg_val(1);  // 1/W_global bits
    constexpr uint32_t origin_W = get_compile_time_arg_val(2);
    constexpr uint32_t Wt = get_compile_time_arg_val(3);  // GLOBAL tiled W stride
    constexpr uint32_t W_BLOCK_TILES = get_compile_time_arg_val(4);
    constexpr uint32_t num_w_blocks = get_compile_time_arg_val(5);  // LOCAL W-block count
    constexpr uint32_t gamma_elt = get_compile_time_arg_val(6);
    constexpr uint32_t GAMMA_IS_ROW_MAJOR = get_compile_time_arg_val(7);
    constexpr uint32_t GROUP_SIZE = get_compile_time_arg_val(8);
    constexpr uint32_t progress_sem_id = get_compile_time_arg_val(9);
    // mcast wire (host Mcast2D) at CT 10; TensorAccessors chained after it.
    constexpr auto mc = McastArgs</*CT=*/10, /*RT=*/9>();
    constexpr uint32_t ta_ct_base = mc.next_compile_time_args_offset();
    constexpr auto input_args = TensorAccessorArgs<ta_ct_base>();
    [[maybe_unused]] constexpr auto gamma_args = TensorAccessorArgs<input_args.next_compile_time_args_offset()>();

    uint32_t input_addr = get_arg_val<uint32_t>(0);
    [[maybe_unused]] uint32_t gamma_addr = get_arg_val<uint32_t>(1);
    uint32_t start_tile_row = get_arg_val<uint32_t>(2);
    uint32_t num_tile_rows = get_arg_val<uint32_t>(3);
    uint32_t w_tile_start = get_arg_val<uint32_t>(4);  // this core's W-tile offset in the global row
    uint32_t my_index = get_arg_val<uint32_t>(5);      // this core's slot in the reduction group
    uint32_t is_root = get_arg_val<uint32_t>(6);
    uint32_t root_x = get_arg_val<uint32_t>(7);  // root virtual coords (gather target)
    uint32_t root_y = get_arg_val<uint32_t>(8);

    // Reduce scaler (SUM AccumulateViaAdd does not consume it, but the compute
    // reduce<> template references cb_scaler; prepare it once so the CB is valid).
    float scaler_f = __builtin_bit_cast(float, scaler_bits);
    dataflow_kernel_lib::prepare_reduce_scaler<cb_scaler, ckernel::PoolType::SUM, ckernel::ReduceDim::REDUCE_ROW>(
        scaler_f, TILE_W);

    constexpr uint32_t wblock_cols = W_BLOCK_TILES * TILE_W;
    const uint32_t tile_bytes = get_tile_size(cb_input_tiles);
    const uint32_t sumsq_bytes = get_tile_size(cb_sumsq);
    const auto in_acc = TensorAccessor(input_args, input_addr, tile_bytes);

    Noc noc;
    Semaphore<> progress(progress_sem_id);

    for (uint32_t t = 0; t < num_tile_rows; ++t) {
        uint32_t tr = start_tile_row + t;

        // ---------- PASS 1: stream local W-slice -> cb_input_tiles ----------
        for (uint32_t b = 0; b < num_w_blocks; ++b) {
            cb_reserve_back(cb_input_tiles, W_BLOCK_TILES);
            uint32_t wp = get_write_ptr(cb_input_tiles);
            uint32_t base_tile = tr * Wt + w_tile_start + b * W_BLOCK_TILES;
            for (uint32_t wt = 0; wt < W_BLOCK_TILES; ++wt) {
                noc_async_read_page(base_tile + wt, in_acc, wp + wt * tile_bytes);
            }
            noc_async_read_barrier();
            cb_push_back(cb_input_tiles, W_BLOCK_TILES);
        }

        // ---------- COMBINE: reduce-root gather + broadcast-back ----------
        cb_wait_front(cb_partial, 1);
        uint32_t partial_rp = get_read_ptr(cb_partial);
        if (is_root) {
            cb_reserve_back(cb_gather, GROUP_SIZE);
            uint32_t gather_wp = get_write_ptr(cb_gather);
            // copy own partial into its own gather slot (local L1 read from self)
            noc_async_read(
                get_noc_addr(my_x[noc_index], my_y[noc_index], partial_rp),
                gather_wp + my_index * sumsq_bytes,
                sumsq_bytes);
            noc_async_read_barrier();
            cb_pop_front(cb_partial, 1);
            // wait for all peers' partials to land, then reset for the next round
            if constexpr (GROUP_SIZE > 1) {
                progress.wait(GROUP_SIZE - 1);
                progress.set(0);
            }
            cb_push_back(cb_gather, GROUP_SIZE);  // compute folds -> cb_rms_src

            // broadcast the finalized 1/rms to the whole group (loopback fills own cb_sumsq)
            cb_wait_front(cb_rms_src, 1);
            uint32_t rms_rp = get_read_ptr(cb_rms_src);
            cb_reserve_back(cb_sumsq, 1);
            auto sender = mc.sender(noc);
            sender.send(rms_rp, get_write_ptr(cb_sumsq), sumsq_bytes);
            cb_pop_front(cb_rms_src, 1);
            cb_push_back(cb_sumsq, 1);
        } else {
            // unicast own partial to the root's gather slot; signal after it lands
            uint32_t gather_base = get_write_ptr(cb_gather);  // same L1 offset as the root's cb_gather
            noc_async_write(
                partial_rp, get_noc_addr(root_x, root_y, gather_base + my_index * sumsq_bytes), sumsq_bytes);
            noc_async_write_barrier();
            cb_pop_front(cb_partial, 1);
            progress.up(noc, root_x, root_y, 1);
            // receive the broadcast 1/rms into cb_sumsq
            cb_reserve_back(cb_sumsq, 1);
            auto receiver = mc.receiver(noc);
            receiver.receive();
            cb_push_back(cb_sumsq, 1);
        }

        // ---------- PASS 2: re-stream local W-slice + local gamma slice ----------
        for (uint32_t b = 0; b < num_w_blocks; ++b) {
            cb_reserve_back(cb_input_tiles, W_BLOCK_TILES);
            uint32_t wp = get_write_ptr(cb_input_tiles);
            uint32_t base_tile = tr * Wt + w_tile_start + b * W_BLOCK_TILES;
            for (uint32_t wt = 0; wt < W_BLOCK_TILES; ++wt) {
                noc_async_read_page(base_tile + wt, in_acc, wp + wt * tile_bytes);
            }
            noc_async_read_barrier();
            cb_push_back(cb_input_tiles, W_BLOCK_TILES);

            if constexpr (HAS_GAMMA) {
                uint32_t g_base_tile = w_tile_start + b * W_BLOCK_TILES;  // gamma is [1,1,1,W_global]
                if constexpr (GAMMA_IS_ROW_MAJOR) {
                    // RM gamma: one W-block-wide stick slice into cb_gamma_rm (compute tilizes it).
                    const auto g_acc = TensorAccessor(gamma_args, gamma_addr);
                    cb_reserve_back(cb_gamma_rm, 1);
                    uint32_t gl1 = get_write_ptr(cb_gamma_rm);
                    uint32_t cols = origin_W - (w_tile_start + b * W_BLOCK_TILES) * TILE_W;
                    if (cols > wblock_cols) {
                        cols = wblock_cols;
                    }
                    uint32_t real_bytes = cols * gamma_elt;
                    uint32_t padded_bytes = wblock_cols * gamma_elt;
                    if (real_bytes < padded_bytes) {
                        zero_l1_page(gl1, padded_bytes);
                    }
                    noc_async_read(
                        g_acc.get_noc_addr(0, (w_tile_start + b * W_BLOCK_TILES) * TILE_W * gamma_elt),
                        gl1,
                        real_bytes);
                    noc_async_read_barrier();
                    cb_push_back(cb_gamma_rm, 1);
                } else {
                    // TILE gamma: raw tile copy at the global gamma tile offset.
                    const auto g_acc = TensorAccessor(gamma_args, gamma_addr, get_tile_size(cb_gamma_tiles));
                    for (uint32_t wt = 0; wt < W_BLOCK_TILES; ++wt) {
                        cb_reserve_back(cb_gamma_tiles, 1);
                        noc_async_read_page(g_base_tile + wt, g_acc, get_write_ptr(cb_gamma_tiles));
                        noc_async_read_barrier();
                        cb_push_back(cb_gamma_tiles, 1);
                    }
                }
            }
        }
    }
    // cb_scaler is prepared once above (producer=reader) and consumed once by the
    // compute kernel at row-loop end (consumer=compute) — the reader never pops it.
}
