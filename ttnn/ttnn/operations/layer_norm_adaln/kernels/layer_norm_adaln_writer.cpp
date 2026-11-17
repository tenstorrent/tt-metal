// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
// SPDX-License-Identifier: Apache-2.0

// layer_norm_adaln — writer (BRISC; NoC1, or NoC0 on the NOC_SWAP_EAST_COLS easternmost grid columns).
//
//   affine_column_mcast (affine only, kernel start; layer_norm_adaln_affine.hpp; on the NoC-swapped east columns the
//                    reader runs it instead, see there): w / b row-0 tiles of this column's slice -> cb_weight_rows /
//                    cb_bias_rows. Every core of grid column x owns slice x % S, so the column's top core (injector)
//                    reads row 0 of its bw w and bw b tiles from DRAM once (one 576 B read per tile, one barrier)
//                    and multicasts them down the column; the other cores wait on a data-ready flag. Runs while the
//                    reader loads x and compute runs the stats phases (the writer is idle until the exchange), and
//                    cuts the per-call w / b DRAM traffic ~grid_y-fold (it competed with the x loads).
//   exchange_block : all-gather of per-slice (p, d, M2) payloads across the S cores of a group (S > 1),
//                    one-shot, streamed: slots are RELATIVE — position q of a member's
//                      cb_gathered_stats block holds member (my_slot + q) mod S's payload (q = 0: its own, a local
//                      copy, pushed first). Member s unicasts its stats_per_row*block_rows acc tiles to s-1, s-2, ..
//                      (its k-th destination gets them at position k) — only each tile's acc_tile_bytes / 8 prefix,
//                      where compute compacted the stats (compact_stats_payload) — all S-1 payloads issued back to
//                      back, ONE writes-flushed, then the S-1 per-sender flags (semaphore xchg_flag_sem_base + s :=
//                      block_idx + 1; stats_tail, 7th tournament: the per-peer flush + flag cost ~150 ns x (S-1)); the
//                      receiver pushes position q (stat_tiles) as soon as its sender's flag is seen, in position order,
//                      so compute folds each payload while the rest are still in flight (the exchange was NoC-volume
//                      bound at S-1 full fp32 tiles per member; the prefix is 512 B fp32 / 256 B bf16 per row).
//                      Multi-block: cb_gathered_stats holds 2 blocks, so a peer one block ahead writes the other half
//                      (it can only be ahead after receiving this core's payload b, which compute publishes only after
//                      popping gathered b-1); flags are per-sender and monotone (a peer cannot be 2 blocks ahead), so
//                      they are never reset.
//                    The compact prefix keeps every feasible blocking under the one-shot byte bound, so the former
//                    rotating ttnn.Mcast schedule (large payloads) was deleted in Perf 1 of the 6th tournament.
//                    S == 1: the payload is copied locally.
//   store_block    : cb_output_tiles -> DRAM tile by tile as compute publishes its DEST windows (writes stream
//                    alongside normalize), one writes-flushed per tile row, one write barrier at kernel end. Each core
//                    visits its row's tiles from start column `rot` (RT arg 4, core id % DRAM banks) and wraps, so the
//                    cores do not all start on bank 0 and walk the banks in lockstep (Perf 2 dram_bw); the reader
//                    and compute's normalize use the same order.
//
// Lockstep invariant: every member of a group processes identical (num_blocks, block_rows) sequences, so the
// reserved write pointer of cb_gathered_stats is identical on all members and is the multicast landing address.

#include <cstdint>

#include "api/dataflow/dataflow_api.h"
#include "api/dataflow/noc.h"
#include "ttnn/cpp/ttnn/kernel_lib/perf_instrumentation.hpp"
#include "layer_norm_adaln_affine.hpp"

namespace {

constexpr uint32_t cb_local_stats = 5;
constexpr uint32_t cb_gathered_stats = 6;
constexpr uint32_t cb_output_tiles = 14;

// One-shot all-gather, streamed (see the head comment). Raw dataflow (bypasses ttnn.Mcast / McastArgs): the helper's
// rotating sender is one sender per round with a per-round consumer-ready handshake; it has no all-to-all schedule
// where every member sends in the same round with per-sender completion flags (nor per-slot consumer pushes), and its
// concurrent multicasts are slower than S-1 unicasts here.
// Pushes cb_gathered_stats one position (stat_tiles) at a time; the caller reserved the whole block.
template <uint32_t num_slices, uint32_t flag_sem_base>
FORCE_INLINE void exchange_streamed(
    uint32_t my_slot,
    uint32_t peer_y,
    const uint32_t* peer_x,
    uint32_t seq,
    uint32_t src_l1,
    uint32_t dst_base,
    uint32_t tile_bytes,
    uint32_t stat_tiles) {
    // compact_stats_payload: each payload tile's stats are its first tile_bytes / 8 bytes (compute compacts them to
    // face 0 rows 0..7), so only that prefix of every tile is moved (8x fewer NoC bytes).
    const uint32_t prefix_bytes = tile_bytes / 8;
    const uint32_t payload_bytes = stat_tiles * tile_bytes;  // slot stride
    // stats_tail (bflush): own payload -> position 0 is a local NoC read issued first, barriered only after all S-1
    // peers' payload writes are issued; then ONE writes-flushed covers every payload and the S-1 flags follow back to
    // back. Ordering argument is exactly the former per-peer one (payload departed before its flag is issued; same
    // static unicast VC + dimension-ordered route), just applied once for all peers instead of once per peer (the
    // compact 512 B / 256 B prefix makes the payload issue cheap, so per-peer flush + flag no longer pays for itself).
    for (uint32_t t = 0; t < stat_tiles; ++t) {
        noc_async_read(get_noc_addr(src_l1 + t * tile_bytes), dst_base + t * tile_bytes, prefix_bytes);
    }
    const uint32_t my_flag = get_semaphore(flag_sem_base + my_slot);
    *reinterpret_cast<volatile tt_l1_ptr uint32_t*>(my_flag) = seq;
    uint32_t next = 1;                                           // next position to push
    uint32_t src = my_slot + 1 == num_slices ? 0 : my_slot + 1;  // its sender
    auto push_landed = [&]() {
        while (next < num_slices &&
               *reinterpret_cast<volatile tt_l1_ptr uint32_t*>(get_semaphore(flag_sem_base + src)) >= seq) {
            cb_push_back(cb_gathered_stats, stat_tiles);
            ++next;
            src = src + 1 == num_slices ? 0 : src + 1;
        }
    };
    {
        MaybeDeviceZoneScope("x_issue");
        uint32_t j = my_slot;
        for (uint32_t k = 1; k < num_slices; ++k) {
            j = j == 0 ? num_slices - 1 : j - 1;
            for (uint32_t t = 0; t < stat_tiles; ++t) {
                noc_async_write(
                    src_l1 + t * tile_bytes,
                    get_noc_addr(peer_x[j], peer_y, dst_base + k * payload_bytes + t * tile_bytes),
                    prefix_bytes);
            }
        }
        noc_async_read_barrier();
        cb_push_back(cb_gathered_stats, stat_tiles);
    }
    {
        MaybeDeviceZoneScope("x_flags");
        noc_async_writes_flushed();
        uint32_t j = my_slot;
        for (uint32_t k = 1; k < num_slices; ++k) {
            j = j == 0 ? num_slices - 1 : j - 1;
            noc_semaphore_set_remote(my_flag, get_noc_addr(peer_x[j], peer_y, my_flag));
        }
    }
    MaybeDeviceZoneScope("x_wait");
    push_landed();
    while (next < num_slices) {
        noc_semaphore_wait_min(reinterpret_cast<volatile tt_l1_ptr uint32_t*>(get_semaphore(flag_sem_base + src)), seq);
        push_landed();
    }
    // Sources (payload, own flag word) are read out before cb_local_stats is released / the flag is rewritten.
    noc_async_writes_flushed();
}

}  // namespace

void kernel_main() {
    constexpr uint32_t block_height = get_compile_time_arg_val(0);
    constexpr uint32_t block_width = get_compile_time_arg_val(1);
    constexpr uint32_t num_w_slices = get_compile_time_arg_val(2);
    constexpr uint32_t tensor_w_tiles = get_compile_time_arg_val(3);
    constexpr uint32_t acc_tile_bytes = get_compile_time_arg_val(4);
    constexpr uint32_t stats_per_row = get_compile_time_arg_val(5);
    constexpr bool has_affine = get_compile_time_arg_val(6) != 0;
    constexpr uint32_t affine_fetch_bytes = get_compile_time_arg_val(7);
    constexpr uint32_t affine_sem_id = get_compile_time_arg_val(8);
    constexpr auto output_args = TensorAccessorArgs<9>();
    // w / b accessors are declared unconditionally; without affine the output's args stand in (unused).
    [[maybe_unused]] constexpr auto weight_args = TensorAccessorArgs<output_args.next_compile_time_args_offset()>();
    [[maybe_unused]] constexpr auto bias_args = TensorAccessorArgs<weight_args.next_compile_time_args_offset()>();
    constexpr bool oneshot = get_named_compile_time_arg_val("xchg_oneshot") != 0;
    constexpr uint32_t flag_sem_base = get_named_compile_time_arg_val("xchg_flag_sem_base");
    // RT layout: [0..4] base, then (one-shot) my_slot, peer_y, peer_x[S], then the affine args.
    constexpr uint32_t xchg_rt = 5;
    constexpr uint32_t affine_rt = xchg_rt + (oneshot ? 2 + num_w_slices : 0);

    const uint32_t output_addr = get_arg_val<uint32_t>(0);
    const uint32_t row_start = get_arg_val<uint32_t>(1);
    const uint32_t group_rows = get_arg_val<uint32_t>(2);
    const uint32_t w_start = get_arg_val<uint32_t>(3);
    const uint32_t rot = get_arg_val<uint32_t>(4);  // per-core start column (DRAM bank stagger), < block_width

    const uint32_t out_tile_bytes = get_tile_size(cb_output_tiles);
    const auto output_accessor = TensorAccessor(output_args, output_addr, out_tile_bytes);

    // affine_in_reader: this kernel group's reader is the NoC1 RISC (NOC_SWAP_EAST_COLS) and owns the affine fetch.
    if constexpr (has_affine && get_named_compile_time_arg_val("affine_in_reader") == 0) {
        MaybeDeviceZoneScope("w_affine");
        affine_column_mcast<block_width, affine_fetch_bytes, affine_sem_id>(weight_args, bias_args, w_start, affine_rt);
    }

    [[maybe_unused]] uint32_t my_slot = 0;
    [[maybe_unused]] uint32_t peer_y = 0;
    [[maybe_unused]] uint32_t peer_x[num_w_slices];
    if constexpr (oneshot) {
        my_slot = get_arg_val<uint32_t>(xchg_rt + 0);
        peer_y = get_arg_val<uint32_t>(xchg_rt + 1);
        for (uint32_t j = 0; j < num_w_slices; ++j) {
            peer_x[j] = get_arg_val<uint32_t>(xchg_rt + 2 + j);
        }
    }

    uint32_t rows_done = 0;
    for (uint32_t block_idx = 0; rows_done < group_rows; ++block_idx) {
        const uint32_t block_rows = (group_rows - rows_done) < block_height ? (group_rows - rows_done) : block_height;
        const uint32_t stat_tiles = stats_per_row * block_rows;

        // exchange_block
        {
            MaybeDeviceZoneScope("w_exchange");
            cb_reserve_back(cb_gathered_stats, num_w_slices * stat_tiles);
            {
                MaybeDeviceZoneScope("w_wait_local_stats");  // starved on compute's stats phases
                cb_wait_front(cb_local_stats, stat_tiles);
            }
            {
                MaybeDeviceZoneScope("w_exchange_rounds");  // one-shot streamed all-gather (rendezvous)
                if constexpr (oneshot) {
                    exchange_streamed<num_w_slices, flag_sem_base>(  // pushes cb_gathered_stats slot by slot
                        my_slot,
                        peer_y,
                        peer_x,
                        block_idx + 1,
                        get_read_ptr(cb_local_stats),
                        get_write_ptr(cb_gathered_stats),
                        acc_tile_bytes,
                        stat_tiles);
                } else {
                    // S == 1: local copy of the own payload into slot 0.
                    noc_async_read(
                        get_noc_addr(get_read_ptr(cb_local_stats)),
                        get_write_ptr(cb_gathered_stats),
                        stat_tiles * acc_tile_bytes);
                    noc_async_read_barrier();
                }
            }
            cb_pop_front(cb_local_stats, stat_tiles);
            if constexpr (!oneshot) {
                cb_push_back(cb_gathered_stats, num_w_slices * stat_tiles);  // S == 1: whole block
            }
        }

        // store_block: compute pushes each DEST window as soon as it is packed, in the same rotated column order
        // (logical tile c = column (rot + c) mod block_width); every tile is written as soon as it lands, so a row's
        // DRAM writes overlap the rest of its normalize. Cumulative wait; the row is popped once after its flush.
        for (uint32_t r = 0; r < block_rows; ++r) {
            MaybeDeviceZoneScope("w_store_row");
            const uint32_t page0 = (row_start + rows_done + r) * tensor_w_tiles + w_start;
            uint32_t l1 = get_read_ptr(cb_output_tiles);
            uint32_t col = rot;
            {
                MaybeDeviceZoneScope("w_store_wait");  // starved until compute's first DEST window of the row lands
                cb_wait_front(cb_output_tiles, 1);
            }
            {
                MaybeDeviceZoneScope("w_store_issue");  // includes the per-tile waits on compute's normalize
                for (uint32_t c = 0; c < block_width; ++c) {
                    cb_wait_front(cb_output_tiles, c + 1);
                    noc_async_write_page(page0 + col, output_accessor, l1);
                    l1 += out_tile_bytes;
                    col = (col + 1 == block_width) ? 0 : col + 1;
                }
            }
            {
                // Only the L1 source must be consumed before the pages are released; DRAM completion is
                // awaited once at kernel end, so row r+1's writes overlap row r's in-flight ones.
                MaybeDeviceZoneScope("w_store_flush");
                noc_async_writes_flushed();
            }
            cb_pop_front(cb_output_tiles, block_width);
        }
        rows_done += block_rows;
    }
    {
        MaybeDeviceZoneScope("w_final_barrier");
        noc_async_write_barrier();
    }
}
