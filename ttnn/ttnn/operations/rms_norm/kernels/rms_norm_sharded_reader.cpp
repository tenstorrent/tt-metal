// SPDX-FileCopyrightText: © 2026 Tenstorrent Inc.
// SPDX-License-Identifier: Apache-2.0

// Reader for rms_norm — WIDTH_SHARDED / BLOCK_SHARDED cross-core reduction
// (R5 TILE; R5a adds the ROW_MAJOR leg).
//
// This is the dataflow face of the design's dependent-axis SCHEME-CHANGE: the
// hidden W is split across a reduction GROUP of cores, so the RMS denominator
// spans core boundaries. Each core owns a contiguous W-slice of its tile-rows
// (a resident L1 shard). Per tile-row the flow is:
//
//   PASS 1  : stream the LOCAL W-slice -> cb_input_tiles (TILE) or cb_input_rm
//             (ROW_MAJOR; compute tilizes). Compute squares + reduces it into
//             cb_partial = local Σx² / W_global (this core's partial
//             contribution to the GLOBAL mean(x²)).
//   COMBINE : reduce-root gather + broadcast-back (this file's cross-core core):
//               * non-root: unicast cb_partial -> root's cb_gather[my_index],
//                 bump the root's `progress` semaphore, then act as a
//                 ReceiverPipe for the finalized 1/rms broadcast -> cb_sumsq.
//               * root: copy its own partial into cb_gather[my_index], wait for
//                 all (GROUP_SIZE-1) peers, push cb_gather (compute folds the
//                 partials -> global mean(x²) -> transform_in_place rsqrt ->
//                 cb_rms_src), then SenderPipe-broadcast cb_rms_src -> cb_sumsq
//                 on the whole group rectangle (loopback fills its own).
//   PASS 2  : re-stream the LOCAL W-slice + local gamma slice; compute
//             normalizes x·(1/rms)·gamma -> cb_output_tiles / cb_output_rm;
//             writer drains its own output shard.
//
// R5a ROW_MAJOR leg (this refinement): RM width-sharding splits each logical
// row's W across cores at sub-tile (stick) granularity, so a row is not
// contiguous in any one core's L1 and the TILE global-page addressing can't be
// used. Instead each core reads its OWN resident shard directly from local L1
// (buffer_address is the same L1 offset on every shard-grid core; the shard is
// `shard_h` sticks of `shard_w` elements, stride shard_w*elt) into cb_input_rm,
// zero-padding the sub-tile W-tail (and any H-tail) so compute always tilizes a
// clean full tile. The Σx² partial over the zero-padded tile equals the partial
// over the core's real columns (zeros add nothing) and ×1/W_global makes the
// group-sum the true global mean(x²) — no partial scaler needed. The combine +
// broadcast transport is REUSED UNCHANGED (all RM cross-core groups are
// rectangular: BLOCK grids always are, and RM+WIDTH+w_non — the only ragged
// geometry — is op-side EXCLUDED). Gamma (RM only on this leg) is read as a
// column-slice stick at the core's global W-offset (w_col_start), matching the
// input's local column layout.
//
// The gather is many->one (raw unicast + a `progress` counter). The broadcast
// is one->many via the mcast_pipe helper (SenderPipe/ReceiverPipe + Mcast2D).

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
constexpr uint32_t cb_input_rm = 0;
constexpr uint32_t cb_input_tiles = 1;
constexpr uint32_t cb_scaler = 2;
constexpr uint32_t cb_gamma_rm = 3;
constexpr uint32_t cb_gamma_tiles = 4;
constexpr uint32_t cb_sumsq = 25;
constexpr uint32_t cb_partial = 27;
constexpr uint32_t cb_gather = 28;
constexpr uint32_t cb_rms_src = 29;
constexpr uint32_t TILE_H = 32;
constexpr uint32_t TILE_W = 32;
// Arch DRAM read source alignment (Wormhole/Blackhole = 32B). The RM-gamma
// column-slice read at a sub-tile w_col_start can be sub-aligned, so it is read
// from the aligned-down base and shifted (see the RM gamma leg below).
constexpr uint32_t GAMMA_DRAM_ALIGN = 32;

// Zero an L1 page word-by-word (RM stick padding tail).
FORCE_INLINE void zero_l1_page(uint32_t l1, uint32_t nbytes) {
    volatile tt_l1_ptr uint32_t* p = reinterpret_cast<volatile tt_l1_ptr uint32_t*>(l1);
    for (uint32_t i = 0; i < nbytes / 4; ++i) {
        p[i] = 0;
    }
}
}  // namespace

void kernel_main() {
    constexpr uint32_t HAS_GAMMA = get_compile_time_arg_val(0);
    constexpr uint32_t IS_ROW_MAJOR = get_compile_time_arg_val(1);
    constexpr uint32_t scaler_bits = get_compile_time_arg_val(2);  // 1/W_global bits
    constexpr uint32_t origin_W = get_compile_time_arg_val(3);
    constexpr uint32_t Wt = get_compile_time_arg_val(4);  // GLOBAL tiled W stride (TILE leg)
    constexpr uint32_t W_BLOCK_TILES = get_compile_time_arg_val(5);
    constexpr uint32_t num_w_blocks = get_compile_time_arg_val(6);  // LOCAL W-block count
    constexpr uint32_t gamma_elt = get_compile_time_arg_val(7);
    constexpr uint32_t GAMMA_IS_ROW_MAJOR = get_compile_time_arg_val(8);
    constexpr uint32_t GROUP_SIZE = get_compile_time_arg_val(9);
    constexpr uint32_t progress_sem_id = get_compile_time_arg_val(10);
    constexpr uint32_t shard_h = get_compile_time_arg_val(11);    // Hs (RM leg): shard rows
    constexpr uint32_t shard_w = get_compile_time_arg_val(12);    // Ws (RM leg): shard cols
    constexpr uint32_t input_elt = get_compile_time_arg_val(13);  // input element bytes (RM leg)
    // mcast wire (host Mcast2D) at CT 14; TensorAccessors chained after it.
    constexpr auto mc = McastArgs</*CT=*/14, /*RT=*/10>();
    constexpr uint32_t ta_ct_base = mc.next_compile_time_args_offset();
    constexpr auto input_args = TensorAccessorArgs<ta_ct_base>();
    [[maybe_unused]] constexpr auto gamma_args = TensorAccessorArgs<input_args.next_compile_time_args_offset()>();

    uint32_t input_addr = get_arg_val<uint32_t>(0);
    [[maybe_unused]] uint32_t gamma_addr = get_arg_val<uint32_t>(1);
    uint32_t start_tile_row = get_arg_val<uint32_t>(2);
    uint32_t num_tile_rows = get_arg_val<uint32_t>(3);
    uint32_t w_tile_start = get_arg_val<uint32_t>(4);  // this core's W-tile offset (TILE leg)
    uint32_t my_index = get_arg_val<uint32_t>(5);      // this core's slot in the reduction group
    uint32_t is_root = get_arg_val<uint32_t>(6);
    uint32_t root_x = get_arg_val<uint32_t>(7);  // root virtual coords (gather target)
    uint32_t root_y = get_arg_val<uint32_t>(8);
    [[maybe_unused]] uint32_t w_col_start = get_arg_val<uint32_t>(9);  // this core's global W-col start (RM gamma)

    // Reduce scaler (SUM AccumulateViaAdd does not consume it, but the compute
    // reduce<> template references cb_scaler; prepare it once so the CB is valid).
    float scaler_f = __builtin_bit_cast(float, scaler_bits);
    dataflow_kernel_lib::prepare_reduce_scaler<cb_scaler, ckernel::PoolType::SUM, ckernel::ReduceDim::REDUCE_ROW>(
        scaler_f, TILE_W);

    constexpr uint32_t wblock_cols = W_BLOCK_TILES * TILE_W;
    constexpr uint32_t in_padded_bytes = wblock_cols * input_elt;     // RM stick page (input)
    constexpr uint32_t shard_row_bytes = shard_w * input_elt;         // RM local shard stride
    constexpr uint32_t gamma_padded_bytes = wblock_cols * gamma_elt;  // RM stick page (gamma)
    const uint32_t tile_bytes = get_tile_size(cb_input_tiles);
    const uint32_t sumsq_bytes = get_tile_size(cb_sumsq);
    const auto in_acc = TensorAccessor(input_args, input_addr, tile_bytes);

    Noc noc;
    Semaphore<> progress(progress_sem_id);

    // Read one W-block of the LOCAL RM shard into cb_input_rm (32 stick-pages).
    // Reads the core's OWN resident shard from local L1 (self NoC read): stick
    // for local row `lr` is at input_addr + lr*shard_row_bytes + block col offset.
    // Rows >= num_real_rows (H-tail) and the sub-tile W-tail are zero-padded so
    // compute tilizes a clean full tile.
    auto read_local_input_wblock = [&](uint32_t local_row_base, uint32_t num_real_rows, uint32_t b) {
        uint32_t cols = shard_w - b * wblock_cols;  // real cols of this block within the shard
        if (cols > wblock_cols) {
            cols = wblock_cols;
        }
        const uint32_t real_bytes = cols * input_elt;
        const uint32_t col_off = b * wblock_cols * input_elt;
        const bool w_tail = real_bytes < in_padded_bytes;
        for (uint32_t r = 0; r < TILE_H; ++r) {
            cb_reserve_back(cb_input_rm, 1);
            uint32_t l1 = get_write_ptr(cb_input_rm);
            const bool is_real = r < num_real_rows;
            if (!is_real || w_tail) {
                zero_l1_page(l1, in_padded_bytes);
            }
            if (is_real) {
                uint32_t src = input_addr + (local_row_base + r) * shard_row_bytes + col_off;
                noc_async_read(get_noc_addr(my_x[noc_index], my_y[noc_index], src), l1, real_bytes);
                noc_async_read_barrier();
            }
            cb_push_back(cb_input_rm, 1);
        }
    };

    // Read one W-block of the LOCAL W-slice as TILES into cb_input_tiles (TILE leg).
    auto read_tile_input_wblock = [&](uint32_t tr, uint32_t b) {
        cb_reserve_back(cb_input_tiles, W_BLOCK_TILES);
        uint32_t wp = get_write_ptr(cb_input_tiles);
        uint32_t base_tile = tr * Wt + w_tile_start + b * W_BLOCK_TILES;
        for (uint32_t wt = 0; wt < W_BLOCK_TILES; ++wt) {
            noc_async_read_page(base_tile + wt, in_acc, wp + wt * tile_bytes);
        }
        noc_async_read_barrier();
        cb_push_back(cb_input_tiles, W_BLOCK_TILES);
    };

    for (uint32_t t = 0; t < num_tile_rows; ++t) {
        uint32_t tr = start_tile_row + t;
        // RM leg: local shard row base for this tile-row + real (non-padding) rows.
        uint32_t local_row_base = t * TILE_H;
        uint32_t num_real_rows = shard_h > local_row_base ? (shard_h - local_row_base) : 0;
        if (num_real_rows > TILE_H) {
            num_real_rows = TILE_H;
        }

        // ---------- PASS 1: stream local W-slice ----------
        for (uint32_t b = 0; b < num_w_blocks; ++b) {
            if constexpr (IS_ROW_MAJOR) {
                read_local_input_wblock(local_row_base, num_real_rows, b);
            } else {
                read_tile_input_wblock(tr, b);
            }
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
            if constexpr (IS_ROW_MAJOR) {
                read_local_input_wblock(local_row_base, num_real_rows, b);
            } else {
                read_tile_input_wblock(tr, b);
            }

            if constexpr (HAS_GAMMA) {
                if constexpr (GAMMA_IS_ROW_MAJOR) {
                    // RM gamma: one W-block-wide stick slice at the core's global W-col
                    // offset (w_col_start) so it matches the input's local column layout.
                    const auto g_acc = TensorAccessor(gamma_args, gamma_addr);
                    cb_reserve_back(cb_gamma_rm, 1);
                    uint32_t gl1 = get_write_ptr(cb_gamma_rm);
                    // Global column start of this block, and real cols available.
                    uint32_t g_col0 = (IS_ROW_MAJOR ? w_col_start : w_tile_start * TILE_W) + b * wblock_cols;
                    uint32_t cols = origin_W > g_col0 ? (origin_W - g_col0) : 0;
                    if (cols > wblock_cols) {
                        cols = wblock_cols;
                    }
                    uint32_t real_bytes = cols * gamma_elt;
                    zero_l1_page(gl1, gamma_padded_bytes);
                    if (real_bytes > 0) {
                        if constexpr (IS_ROW_MAJOR) {
                            // R5a: a sub-tile w_col_start makes the gamma DRAM source
                            // byte offset sub-32B-aligned, but DRAM reads require a
                            // 32B-aligned source (odd sub-tile cores read garbage
                            // otherwise). Read from the 32B-aligned base into the page
                            // (sized with GAMMA_DRAM_ALIGN slack) and shift the window
                            // left to local col 0 with an L1 byte copy (any alignment).
                            uint32_t byte_off = g_col0 * gamma_elt;
                            uint32_t aligned_off = byte_off & ~(GAMMA_DRAM_ALIGN - 1);
                            uint32_t shift = byte_off - aligned_off;
                            uint32_t page_bytes = origin_W * gamma_elt;
                            uint32_t span = shift + real_bytes;
                            uint32_t aligned_span = (span + GAMMA_DRAM_ALIGN - 1) & ~(GAMMA_DRAM_ALIGN - 1);
                            if (aligned_off + aligned_span > page_bytes) {
                                aligned_span = page_bytes - aligned_off;
                            }
                            noc_async_read(g_acc.get_noc_addr(0, aligned_off), gl1, aligned_span);
                            noc_async_read_barrier();
                            if (shift > 0) {
                                volatile tt_l1_ptr uint8_t* p = reinterpret_cast<volatile tt_l1_ptr uint8_t*>(gl1);
                                for (uint32_t i = 0; i < real_bytes; ++i) {
                                    p[i] = p[i + shift];  // forward copy (dst < src): safe
                                }
                                for (uint32_t i = real_bytes; i < gamma_padded_bytes; ++i) {
                                    p[i] = 0;  // re-zero the W-tail after the shift
                                }
                            }
                        } else {
                            noc_async_read(g_acc.get_noc_addr(0, g_col0 * gamma_elt), gl1, real_bytes);
                            noc_async_read_barrier();
                        }
                    }
                    cb_push_back(cb_gamma_rm, 1);
                } else {
                    // TILE gamma: raw tile copy at the global gamma tile offset (TILE leg only —
                    // RM cross-core + TILE gamma is op-side EXCLUDED, sub-tile W-col offset).
                    const auto g_acc = TensorAccessor(gamma_args, gamma_addr, get_tile_size(cb_gamma_tiles));
                    uint32_t g_base_tile = w_tile_start + b * W_BLOCK_TILES;
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
