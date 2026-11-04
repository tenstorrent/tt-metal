// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
// SPDX-License-Identifier: Apache-2.0

// nomic_qkv_rotary_heads — writer. RISCV_1 (NCRISC) on NOC1: in triage a hung weight stream / drain shows on NCRISC.
//
// load_weight_block (every core): the row's in1 stream. In M block 0 the sender of round b (the row's right core by
//   default, or the w_mcast sender of round w_round0 + b) reads K block b of the lane's weight pages (k_block_tiles K
//   rows x lane_n_tiles / weight_page_tiles pages) from DRAM straight into ring slot b and multicasts it to the row,
//   chunk by chunk (w_mcast_chunk_k_rows K rows) as each chunk's reads land; the block's last chunk carries the ready
//   signal. A sender keeps weight_read_ahead chunks of its own reads in flight, each on its own transaction id.
//   cb_weights holds the whole K x lane (resident ring): later M blocks re-push the same slots with no reads.
// scatter_heads_block (every core): output in out_drain_tiles steps of a row, each tile straight into its q / k / v
//   page ((b * heads + head) * seq_tiles + s_t) * head_tiles + half. Rows past S of a sequence's last tile are zeroed
//   in the CB page first (selection, not multiplication). Block m - 1 drains during block m's K loop, after the
//   resident re-push of K blocks 0 .. num_k_blocks - 2, so compute never waits for weights.

#include <cstdint>

#include "api/dataflow/dataflow_api.h"
#include "api/dataflow/circular_buffer.h"
#include "api/dataflow/noc.h"
#include "api/dataflow/endpoints.h"
#include "api/core_local_mem.h"
#include "api/tensor/noc_traits.h"
#include "ttnn/cpp/ttnn/kernel_lib/mcast/kernel/mcast_args.hpp"
#include "ttnn/cpp/ttnn/kernel_lib/local_copy_helpers_dataflow.hpp"

using namespace dataflow_kernel_lib;

void kernel_main() {
    constexpr uint32_t cb_weights = get_named_compile_time_arg_val("cb_weights");
    constexpr uint32_t cb_heads_out = get_named_compile_time_arg_val("cb_heads_out");
    constexpr uint32_t block_m_tiles = get_named_compile_time_arg_val("block_m_tiles");
    constexpr uint32_t k_block_tiles = get_named_compile_time_arg_val("k_block_tiles");
    constexpr uint32_t num_k_blocks = get_named_compile_time_arg_val("num_k_blocks");
    constexpr uint32_t lane_n_tiles = get_named_compile_time_arg_val("lane_n_tiles");
    // Output pages per drain step (a divisor of lane_n_tiles): the writer waits, writes and pops this many at a time.
    constexpr uint32_t out_drain_tiles = get_named_compile_time_arg_val("out_drain_tiles");
    static_assert(lane_n_tiles % out_drain_tiles == 0, "a drain step is part of one output row");
    constexpr uint32_t weight_page_tiles = get_named_compile_time_arg_val("weight_page_tiles");
    constexpr uint32_t weight_pages_per_k_row = get_named_compile_time_arg_val("weight_pages_per_k_row");
    constexpr uint32_t head_tiles = get_named_compile_time_arg_val("head_tiles");
    constexpr uint32_t tensor_n_tiles = get_named_compile_time_arg_val("tensor_n_tiles");
    constexpr uint32_t heads = get_named_compile_time_arg_val("heads");
    constexpr uint32_t seq_tiles = get_named_compile_time_arg_val("seq_tiles");
    // Real rows of a sequence's last position tile; 0 when S is tile-aligned (nothing to zero).
    constexpr uint32_t valid_last_rows = get_named_compile_time_arg_val("valid_last_rows");
    // K rows per in1 multicast chunk: each chunk's DRAM reads retire on their own transaction id and the chunk is
    // multicast as soon as they land; the K block's last chunk carries the ready signal.
    constexpr uint32_t w_chunk_k_rows = get_named_compile_time_arg_val("w_mcast_chunk_k_rows");
    static_assert(k_block_tiles % w_chunk_k_rows == 0, "a chunk is whole K rows of one K block");
    constexpr uint32_t chunks_per_k_block = k_block_tiles / w_chunk_k_rows;
    constexpr uint32_t num_w_chunks = num_k_blocks * chunks_per_k_block;
    // Chunks of this sender's DRAM reads kept in flight: chunk c + chunks_ahead is issued when chunk c retires.
    constexpr uint32_t chunks_ahead = get_named_compile_time_arg_val("weight_read_ahead");
    static_assert(chunks_ahead >= 1 && chunks_ahead < NOC_MAX_TRANSACTION_ID, "one transaction id per chunk in flight");
    constexpr uint32_t weight_page_bytes = get_named_compile_time_arg_val("weight_page_bytes");
    constexpr auto w_args = TensorAccessorArgs<0>();
    constexpr auto q_args = TensorAccessorArgs<w_args.next_compile_time_args_offset()>();
    constexpr auto k_args = TensorAccessorArgs<q_args.next_compile_time_args_offset()>();
    constexpr auto v_args = TensorAccessorArgs<k_args.next_compile_time_args_offset()>();
    constexpr auto w_mcast = McastArgs<
        get_named_compile_time_arg_val("w_mcast_ct_offset"),
        get_named_compile_time_arg_val("w_mcast_rt_offset")>();

    constexpr uint32_t w_block_tiles = k_block_tiles * lane_n_tiles;
    constexpr uint32_t pages_per_k_row = lane_n_tiles / weight_page_tiles;
    static_assert(pages_per_k_row * weight_page_tiles == lane_n_tiles, "a lane is whole weight pages");

    uint32_t arg = 0;
    const uint32_t w_addr = get_arg_val<uint32_t>(arg++);
    const uint32_t q_addr = get_arg_val<uint32_t>(arg++);
    const uint32_t k_addr = get_arg_val<uint32_t>(arg++);
    const uint32_t v_addr = get_arg_val<uint32_t>(arg++);
    const uint32_t first_row = get_arg_val<uint32_t>(arg++);
    const uint32_t end_row = get_arg_val<uint32_t>(arg++);
    const uint32_t num_m_blocks = get_arg_val<uint32_t>(arg++);
    const uint32_t lane_n0 = get_arg_val<uint32_t>(arg++);  // first global N tile of this core's lane
    const uint32_t w_round0 = get_arg_val<uint32_t>(arg++);  // w_mcast round of K block 0 (rotating-sender stagger)

    Noc noc;
    CircularBuffer w_cb(cb_weights);
    CircularBuffer out_cb(cb_heads_out);
    constexpr uint32_t w_tile_bytes = get_tile_size(cb_weights);
    constexpr uint32_t out_tile_bytes = get_tile_size(cb_heads_out);
    static_assert(weight_page_bytes == weight_page_tiles * w_tile_bytes, "a weight page is weight_page_tiles tiles");
    constexpr uint32_t w_block_bytes = w_block_tiles * w_tile_bytes;
    constexpr uint32_t w_chunk_bytes = w_block_bytes / chunks_per_k_block;
    // M block 0 streams the empty ring from its start: K block b lands in slot b.
    const uint32_t w_ring_base = w_cb.get_write_ptr();

    const auto weights = TensorAccessor(w_args, w_addr, weight_page_bytes);
    const auto q_out = TensorAccessor(q_args, q_addr, out_tile_bytes);
    const auto k_out = TensorAccessor(k_args, k_addr, out_tile_bytes);
    const auto v_out = TensorAccessor(v_args, v_addr, out_tile_bytes);

    // K rows [chunk * w_chunk_k_rows, (chunk + 1) * w_chunk_k_rows) of the lane, one weight page run per K row.
    auto read_weights = [&](uint32_t chunk, uint32_t dst) {
        if constexpr (get_named_compile_time_arg_val("ablate_wread") == 1) {
            return;
        }
        const uint32_t first_page_in_row = lane_n0 / weight_page_tiles;
        for (uint32_t k = chunk * w_chunk_k_rows; k < (chunk + 1) * w_chunk_k_rows; ++k) {
            uint32_t page = k * weight_pages_per_k_row + first_page_in_row;
            for (uint32_t p = 0; p < pages_per_k_row; ++p, ++page, dst += weight_page_bytes) {
                noc.async_read(weights, CoreLocalMem<uint32_t>(dst), weight_page_bytes, {.page_id = page}, {});
            }
        }
    };

    // Zero rows [valid_last_rows, 32) of a bf16 tile: face rows are 32 B; faces 0/1 hold rows 0-15, faces 2/3 16-31.
    auto zero_padding_rows = [&](uint32_t tile) {
        constexpr uint32_t face_bytes = out_tile_bytes / 4;
        constexpr uint32_t face_row_bytes = face_bytes / 16;
        if constexpr (valid_last_rows < 16) {
            noc.async_write_zeros(
                CoreLocalMem<uint32_t>(tile + valid_last_rows * face_row_bytes),
                face_bytes - valid_last_rows * face_row_bytes);
            noc.async_write_zeros(
                CoreLocalMem<uint32_t>(tile + face_bytes + valid_last_rows * face_row_bytes),
                face_bytes - valid_last_rows * face_row_bytes);
            noc.async_write_zeros(CoreLocalMem<uint32_t>(tile + 2 * face_bytes), 2 * face_bytes);
        } else {
            constexpr uint32_t skip = (valid_last_rows - 16) * face_row_bytes;
            noc.async_write_zeros(CoreLocalMem<uint32_t>(tile + 2 * face_bytes + skip), face_bytes - skip);
            noc.async_write_zeros(CoreLocalMem<uint32_t>(tile + 3 * face_bytes + skip), face_bytes - skip);
        }
    };

    // Tiles [n0, n0 + out_drain_tiles) of output row `row`, read from the CB front.
    auto write_tiles = [&](uint32_t row, uint32_t n0, uint32_t read_ptr) {
        const uint32_t b = row / seq_tiles;
        const uint32_t s_t = row % seq_tiles;
        if constexpr (valid_last_rows != 0) {
            if (s_t == seq_tiles - 1) {
                for (uint32_t n = 0; n < out_drain_tiles; ++n) {
                    zero_padding_rows(read_ptr + n * out_tile_bytes);
                }
                noc.write_zeros_l1_barrier();
            }
        }
        for (uint32_t n = 0; n < out_drain_tiles; ++n) {
            const uint32_t global_n = lane_n0 + n0 + n;
            const uint32_t tensor = global_n / tensor_n_tiles;
            const uint32_t within = global_n % tensor_n_tiles;
            const uint32_t page =
                ((b * heads + within / head_tiles) * seq_tiles + s_t) * head_tiles + within % head_tiles;
            const CoreLocalMem<uint32_t> src(read_ptr + n * out_tile_bytes);
            if (tensor == 0) {
                noc.async_write(src, q_out, out_tile_bytes, {}, {.page_id = page});
            } else if (tensor == 1) {
                noc.async_write(src, k_out, out_tile_bytes, {}, {.page_id = page});
            } else {
                noc.async_write(src, v_out, out_tile_bytes, {}, {.page_id = page});
            }
        }
    };

    // One M block's output, out_drain_tiles at a time in push order (tile-row-major), as compute pushes it; flushed
    // before each pop (compute refills popped pages). Draining finer than a row lets the last row's first heads leave
    // while compute is still finishing its last heads, which shortens the op's tail.
    auto drain_block = [&](uint32_t row0) {
        for (uint32_t r = 0; r < block_m_tiles; ++r) {
            const uint32_t row = row0 + r;
            for (uint32_t n0 = 0; n0 < lane_n_tiles; n0 += out_drain_tiles) {
                out_cb.wait_front(out_drain_tiles);
                if (get_named_compile_time_arg_val("ablate_writes") == 0 && row < end_row) {
                    write_tiles(row, n0, out_cb.get_read_ptr());
                }
                noc.async_writes_flushed();
                out_cb.pop_front(out_drain_tiles);
            }
        }
    };

    auto sender = w_mcast.optional_sender(noc);
    auto receiver = w_mcast.optional_receiver(noc);
    // This core reads and multicasts K block b of M block 0 (fixed sender: every block; rotating: its phase's).
    auto sends = [&](uint32_t k_block) { return sender.has_value() && w_mcast.should_send(w_round0 + k_block); };
    auto trid = [](uint32_t chunk) { return chunk % chunks_ahead + 1; };  // in flight: chunks_ahead consecutive
    auto chunk_addr = [&](uint32_t chunk) { return w_ring_base + chunk * w_chunk_bytes; };
    auto issue_chunk = [&](uint32_t chunk) {
        if (chunk < num_w_chunks && sends(chunk / chunks_per_k_block)) {
            set_read_trid(noc, trid(chunk));
            read_weights(chunk, chunk_addr(chunk));
        }
    };

    uint32_t previous_row0 = first_row;
    for (uint32_t m_block = 0; m_block < num_m_blocks; ++m_block) {
        const uint32_t row0 = first_row + m_block * block_m_tiles;
        if (m_block == 0) {
            for (uint32_t chunk = 0; chunk < chunks_ahead; ++chunk) {
                issue_chunk(chunk);
            }
        }
        for (uint32_t k_block = 0; k_block < num_k_blocks; ++k_block) {
            if (m_block > 0 && k_block == num_k_blocks - 1) {
                drain_block(previous_row0);
            }
            w_cb.reserve_back(w_block_tiles);
            if (m_block == 0) {
                const uint32_t chunk0 = k_block * chunks_per_k_block;
                if (sends(k_block)) {
                    // Chunks 0 .. n-2: multicast as they land (the ring is empty in M block 0: no readiness wait
                    // needed); the last chunk's send() waits for the row and publishes the whole K block.
                    for (uint32_t chunk = chunk0; chunk < chunk0 + chunks_per_k_block - 1; ++chunk) {
                        async_read_barrier_with_trid(noc, trid(chunk));
                        issue_chunk(chunk + chunks_ahead);
                        sender->send_payload(chunk_addr(chunk), chunk_addr(chunk), w_chunk_bytes);
                    }
                    const uint32_t last = chunk0 + chunks_per_k_block - 1;
                    async_read_barrier_with_trid(noc, trid(last));
                    w_cb.push_back(w_block_tiles);
                    issue_chunk(last + chunks_ahead);
                    // Measurement-only ablation: multicast one tile instead of the chunk (wrong results).
                    constexpr bool ablate_wmcast = get_named_compile_time_arg_val("ablate_wmcast") == 1;
                    sender->send(chunk_addr(last), chunk_addr(last), ablate_wmcast ? w_tile_bytes : w_chunk_bytes);
                } else {
                    receiver->receive(w_round0 + k_block);
                    w_cb.push_back(w_block_tiles);
                    for (uint32_t chunk = chunk0; chunk < chunk0 + chunks_per_k_block; ++chunk) {
                        issue_chunk(chunk + chunks_ahead);
                    }
                }
            } else {
                // Resident: the slot already holds this K block (the ring is exactly one K x lane cycle).
                w_cb.push_back(w_block_tiles);
            }
        }
        if (m_block == 0) {
            set_read_trid(noc, 0);
        }
        previous_row0 = row0;
    }
    drain_block(previous_row0);
    noc.async_write_barrier();
}
