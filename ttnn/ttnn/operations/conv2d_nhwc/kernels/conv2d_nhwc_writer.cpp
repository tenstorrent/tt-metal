// SPDX-FileCopyrightText: © 2026 Tenstorrent Inc.
// SPDX-License-Identifier: Apache-2.0

// conv2d_nhwc writer (BRISC) — regimes R0 and R4/R5.
//
// Block operations owned here:
//   load_bias_slice      : core_n_tiles bias tiles, once per core (compute keeps them fronted).
//   weight K-block fill  : per (m-block, n-block), num_k_blocks K-blocks of bk x bn weight tiles
//                          (row-major [k][n] in cb_weight_tiles), double-buffered. Column tiles
//                          >= N_t (the R4/R5 slice padding, technique 5) are zero-filled.
//                          R4 with weight_mcast: only the grid-row-0 core of each column reads
//                          DRAM and multicasts the K-block down its column (one read per column).
//   act_exchange (R4/R5) : per K-block j, the owner of slice j % num_slices multicasts its
//                          tilized cb_act_send block into cb_act_tiles of every core in its group
//                          (grid row / all WIDTH cores, itself included); the others receive it.
//                          One rotating-sender multicast + handshake per K-block (McastArgs).
//   untilize_store_block : R0: per tile-row, 32 RM rows of bn*32 output channels written to DRAM
//                          at byte offset n_start*32*elem_bytes of each pixel's output stick;
//                          rows >= M or past the core's m range are dropped.
//                          R4/R5: compute untilizes into the output shard in place; only a WIDTH
//                          shard's partial last tile-row is copied here (its valid rows).

#include <cstdint>
#include "api/dataflow/dataflow_api.h"
#include "api/dataflow/noc.h"
#include "ttnn/cpp/ttnn/kernel_lib/mcast/kernel/mcast_args.hpp"
#include "ttnn/cpp/ttnn/kernel_lib/local_copy_helpers_dataflow.hpp"

constexpr uint32_t cb_weight_tiles = get_compile_time_arg_val(0);
constexpr uint32_t cb_bias_tiles = get_compile_time_arg_val(1);
constexpr uint32_t cb_out_rm = get_compile_time_arg_val(2);
constexpr uint32_t block_m_tiles = get_compile_time_arg_val(3);
constexpr uint32_t block_k_tiles = get_compile_time_arg_val(4);
constexpr uint32_t block_n_tiles = get_compile_time_arg_val(5);
constexpr uint32_t num_k_blocks = get_compile_time_arg_val(6);
constexpr uint32_t num_n_blocks = get_compile_time_arg_val(7);
constexpr uint32_t N_t = get_compile_time_arg_val(8);
constexpr uint32_t core_n_tiles = get_compile_time_arg_val(9);
constexpr uint32_t has_bias = get_compile_time_arg_val(10);
constexpr uint32_t M = get_compile_time_arg_val(11);
constexpr uint32_t elem_bytes = get_compile_time_arg_val(12);
constexpr bool k_exchange = get_compile_time_arg_val(13) != 0;
constexpr uint32_t cb_act_send = get_compile_time_arg_val(14);
constexpr uint32_t cb_act_tiles = get_compile_time_arg_val(15);
constexpr uint32_t kblocks_per_slice = get_compile_time_arg_val(16);
constexpr bool has_out_tail = get_compile_time_arg_val(17) != 0;  // WIDTH shard, M % 32 != 0
constexpr bool act_mcast = get_compile_time_arg_val(18) != 0;     // > 1 channel slice
constexpr bool weight_mcast = get_compile_time_arg_val(19) != 0;  // BLOCK: weights down each column
// cb_weight_tiles depth in K-blocks; the weight stream runs weight_depth - 2 blocks ahead.
constexpr uint32_t weight_depth = get_compile_time_arg_val(20);
constexpr uint32_t weight_lead = weight_depth - 2;
// Weight K-rows per K-block: bk (dense) or the kH*kW taps (R11 depthwise: tile (tap, ct) of the
// prepared [taps*32, C_out] weight, row 0 = the tap's per-channel weights).
constexpr uint32_t weight_k_rows = get_compile_time_arg_val(21);
// R3 (HEIGHT): the whole weight slice (num_k_blocks K-blocks) is resident in cb_weight_tiles for
// the kernel's life — read (and multicast) once, before the first m-block, as ONE transfer instead
// of a serial read -> multicast round per K-block per m-block. Compute reads it in place.
constexpr bool weights_resident = get_compile_time_arg_val(22) != 0;
// Resident weights: K-rows (bn tiles each) per read/multicast chunk.
constexpr uint32_t weight_chunk_rows = get_compile_time_arg_val(23);
constexpr uint32_t WEIGHT_CHUNK_LEAD = 2;  // chunks whose reads are in flight ahead of the multicast
constexpr uint32_t MAX_WEIGHT_TRIDS = 15;  // read transaction ids 1..15
constexpr auto weight_args = TensorAccessorArgs<24>();
[[maybe_unused]] constexpr auto bias_args = TensorAccessorArgs<weight_args.next_compile_time_args_offset()>();
constexpr auto output_args = TensorAccessorArgs<bias_args.next_compile_time_args_offset()>();
// Split gather (R3): the reader's gather compile-time args follow (this instance's own window / rm
// CBs); the writer gathers the odd m-blocks once its weights and bias are loaded.
#define GATHER_CT_BASE (output_args.next_compile_time_args_offset())
#include "conv2d_nhwc_gather.hpp"
constexpr uint32_t WRITER_RT_ARGS = 10;  // the writer's own runtime args; the reader's follow (split gather)
constexpr auto act_channel = dataflow_kernel_lib::McastArgs<
    get_named_compile_time_arg_val("act_mcast_ct_offset"),
    get_named_compile_time_arg_val("act_mcast_rt_offset")>();
constexpr auto weight_channel = dataflow_kernel_lib::McastArgs<
    get_named_compile_time_arg_val("w_mcast_ct_offset"),
    get_named_compile_time_arg_val("w_mcast_rt_offset")>();

constexpr uint32_t TILE_DIM = 32;
constexpr uint32_t out_row_bytes = block_n_tiles * TILE_DIM * elem_bytes;
constexpr uint32_t act_block_tiles = block_m_tiles * block_k_tiles;
// R4/R5 K order interleaves the slices: K-block j is own K-block j / num_slices of slice
// j % num_slices; its weight rows start at slice * slice_k_tiles + (j / num_slices) * bk
// (prepared K order stays [slice][kh][kw x slice_width]). R0 / one slice: rows j * bk.
constexpr uint32_t num_slices = act_mcast ? num_k_blocks / kblocks_per_slice : 1;
constexpr uint32_t slice_k_tiles = kblocks_per_slice * block_k_tiles;
FORCE_INLINE uint32_t kblock_owner(uint32_t k_block) { return k_block % num_slices; }
FORCE_INLINE uint32_t kblock_first_row(uint32_t k_block) {
    return act_mcast ? kblock_owner(k_block) * slice_k_tiles + (k_block / num_slices) * block_k_tiles
                     : k_block * block_k_tiles;
}

FORCE_INLINE void zero_fill(uint64_t zeros_noc, uint32_t dst, uint32_t bytes) {
    while (bytes > 0) {
        const uint32_t chunk = bytes < MEM_ZEROS_SIZE ? bytes : MEM_ZEROS_SIZE;
        noc_async_read(zeros_noc, dst, chunk);
        dst += chunk;
        bytes -= chunk;
    }
}

void kernel_main() {
    const uint32_t weight_addr = get_arg_val<uint32_t>(0);
    const uint32_t bias_addr = get_arg_val<uint32_t>(1);
    const uint32_t output_addr = get_arg_val<uint32_t>(2);
    const uint32_t m_tile_start = get_arg_val<uint32_t>(3);
    const uint32_t m_tiles_this_core = get_arg_val<uint32_t>(4);
    const uint32_t num_m_blocks = get_arg_val<uint32_t>(5);
    const uint32_t n_tile_start = get_arg_val<uint32_t>(6);
    const uint32_t my_slice = get_arg_val<uint32_t>(7);                         // R4/R5: this core's channel slice
    const bool is_sink = get_arg_val<uint32_t>(8) != 0;                         // receive-only rectangle filler (WIDTH)
    const bool reads_weights = !weight_mcast || get_arg_val<uint32_t>(9) != 0;  // column's weight reader

    const auto weights = TensorAccessor(weight_args, weight_addr);
    const auto output = TensorAccessor(output_args, output_addr);
    const uint32_t weight_tile_bytes = get_tile_size(cb_weight_tiles);
    const uint64_t zeros_noc = get_noc_addr(MEM_ZEROS_BASE);
    Noc noc;
    auto act_sender = act_channel.optional_sender(noc);
    auto act_receiver = act_channel.optional_receiver(noc);
    [[maybe_unused]] const uint32_t act_block_bytes = act_block_tiles * get_tile_size(cb_act_tiles);
    auto weight_sender = weight_channel.optional_sender(noc);
    auto weight_receiver = weight_channel.optional_receiver(noc);
    const uint32_t weight_block_bytes = weight_k_rows * block_n_tiles * weight_tile_bytes;

    // ---- sink: keep the group's landing-slot cadence and acks, nothing else ----
    if constexpr (act_mcast) {
        if (is_sink) {
            for (uint32_t m_block = 0; m_block < num_m_blocks; ++m_block) {
                for (uint32_t k_block = 0; k_block < num_k_blocks; ++k_block) {
                    cb_reserve_back(cb_act_tiles, act_block_tiles);
                    act_receiver->receive(kblock_owner(k_block));
                    cb_push_back(cb_act_tiles, act_block_tiles);
                    cb_wait_front(cb_act_tiles, act_block_tiles);
                    cb_pop_front(cb_act_tiles, act_block_tiles);
                }
            }
            return;
        }
    }

    // ---- load_bias_slice ---- (after the resident weights: the bias is first needed at the end
    // of the first m-block, the weights at its first matmul)
    auto load_bias = [&]() {
        if constexpr (has_bias) {
            const auto bias = TensorAccessor(bias_args, bias_addr);
            const uint32_t bias_tile_bytes = get_tile_size(cb_bias_tiles);
            cb_reserve_back(cb_bias_tiles, core_n_tiles);
            uint32_t dst = get_write_ptr(cb_bias_tiles);
            for (uint32_t j = 0; j < core_n_tiles; ++j) {
                if (n_tile_start + j < N_t) {
                    noc_async_read(bias.get_noc_addr(n_tile_start + j), dst, bias_tile_bytes);
                } else {
                    zero_fill(zeros_noc, dst, bias_tile_bytes);
                }
                dst += bias_tile_bytes;
            }
            noc_async_read_barrier();
            cb_push_back(cb_bias_tiles, core_n_tiles);
        }
    };
    if constexpr (!weights_resident) {
        load_bias();
    }

    constexpr uint32_t weight_block_tiles = weight_k_rows * block_n_tiles;
    // read_weight_block: K-block k_block's bk x bn weight tiles (columns from n0) into dst, no
    // barrier. Column tiles >= N_t (slice padding) are zero-filled.
    auto read_weight_block = [&](uint32_t k_block, uint32_t n0, uint32_t dst) {
        uint32_t row_page = kblock_first_row(k_block) * N_t + n0;
        for (uint32_t kk = 0; kk < weight_k_rows; ++kk) {
            for (uint32_t j = 0; j < block_n_tiles; ++j) {
                if (n0 + j < N_t) {
                    noc_async_read(weights.get_noc_addr(row_page + j), dst, weight_tile_bytes);
                } else {
                    zero_fill(zeros_noc, dst, weight_tile_bytes);
                }
                dst += weight_tile_bytes;
            }
            row_page += N_t;
        }
    };

    // ---- load_resident_weights (R3): every K-block once, before any m-block ----
    // The sender reads the slice in chunks of weight_chunk_rows K-rows, each tagged with its own read
    // transaction id, and multicasts chunk c as soon as its reads land while chunks c+1.. are still
    // in flight (the read issue is RISC-bound: it overlaps the multicast). Receivers push per chunk.
    if constexpr (weights_resident) {
        constexpr uint32_t total_rows = num_k_blocks * weight_k_rows;
        constexpr uint32_t num_chunks = (total_rows + weight_chunk_rows - 1) / weight_chunk_rows;
        static_assert(!weights_resident || num_chunks <= MAX_WEIGHT_TRIDS, "one read transaction id per chunk");
        const uint32_t row_bytes = block_n_tiles * weight_tile_bytes;
        cb_reserve_back(cb_weight_tiles, total_rows * block_n_tiles);
        const uint32_t slot = get_write_ptr(cb_weight_tiles);  // lockstep across the group
        auto chunk_rows = [&](uint32_t c) {
            const uint32_t r0 = c * weight_chunk_rows;
            return total_rows - r0 < weight_chunk_rows ? total_rows - r0 : weight_chunk_rows;
        };
        // One HEIGHT core owns every output channel and its K order is the prepared one, so chunk c
        // is the contiguous page range [r0 * N_t, (r0 + rows) * N_t) of the DRAM-interleaved weight:
        // page p lives in bank p % NUM_DRAM_BANKS at a per-bank stride of one aligned page. Issue
        // bank by bank with stateful reads: one address computation per bank, not per tile.
        static_assert(!weights_resident || (block_n_tiles == N_t && !act_mcast), "resident = whole rows, one slice");
        auto issue_chunk = [&](uint32_t c) {
            dataflow_kernel_lib::set_read_trid(noc, c + 1);
            const uint32_t p0 = c * weight_chunk_rows * N_t;
            const uint32_t np = chunk_rows(c) * N_t;
            const uint32_t bank_stride = weights.get_aligned_page_size();
            const uint32_t dst0 = slot + p0 * weight_tile_bytes;
            for (uint32_t b = 0; b < NUM_DRAM_BANKS && b < np; ++b) {
                const uint64_t a = weights.get_noc_addr(p0 + b);
                noc_async_read_set_state(a);
                uint32_t src = static_cast<uint32_t>(a);
                uint32_t dst = dst0 + b * weight_tile_bytes;
                for (uint32_t q = b; q < np; q += NUM_DRAM_BANKS) {
                    noc_async_read_with_state(src, dst, weight_tile_bytes);
                    src += bank_stride;
                    dst += NUM_DRAM_BANKS * weight_tile_bytes;
                }
            }
        };
        if (reads_weights) {
            for (uint32_t c = 0; c < num_chunks && c < WEIGHT_CHUNK_LEAD; ++c) {
                issue_chunk(c);
            }
            for (uint32_t c = 0; c < num_chunks; ++c) {
                dataflow_kernel_lib::async_read_barrier_with_trid(noc, c + 1);
                if constexpr (weight_mcast) {
                    // The resident region is never rewritten: no source guard needed (the kernel-end
                    // write barrier drains the multicast).
                    const uint32_t a = slot + c * weight_chunk_rows * row_bytes;
                    weight_sender->template send<dataflow_kernel_lib::SourceL1Guard::CallerManaged>(
                        a, a, chunk_rows(c) * row_bytes);  // self-excluded
                }
                cb_push_back(cb_weight_tiles, chunk_rows(c) * block_n_tiles);
                if (c + WEIGHT_CHUNK_LEAD < num_chunks) {
                    issue_chunk(c + WEIGHT_CHUNK_LEAD);
                }
            }
            dataflow_kernel_lib::set_read_trid(noc, 0);
        } else {
            for (uint32_t c = 0; c < num_chunks; ++c) {
                weight_receiver->receive();
                cb_push_back(cb_weight_tiles, chunk_rows(c) * block_n_tiles);
            }
        }
        if (is_sink) {
            return;  // an input core holding no output shard: it only receives the multicast
        }
        load_bias();
    }

    // ---- split gather (R3): the odd m-blocks' fetch + im2col gather, into this RISC's own CBs ----
    if constexpr (conv_gather::num_gather_roles > 1) {
        conv_gather::gather_m_blocks(WRITER_RT_ARGS, 1);
    }

    const uint32_t m_tile_end = m_tile_start + m_tiles_this_core;
    for (uint32_t m_block = 0; m_block < num_m_blocks; ++m_block) {
        const uint32_t t0 = m_tile_start + m_block * block_m_tiles;
        const uint32_t t1 = (t0 + block_m_tiles < m_tile_end) ? t0 + block_m_tiles : m_tile_end;
        for (uint32_t n_block = 0; n_block < num_n_blocks; ++n_block) {
            const uint32_t n0 = n_tile_start + n_block * block_n_tiles;

            // ---- weight K-blocks for this (m, n) block ----
            // issue_weights: reserve K-block k_block's slot and (column reader) issue its DRAM
            // reads without waiting; land_weights: barrier, column multicast / receive, push.
            // R4/R5 runs the act exchange of a K-block between the two, so the weight DRAM
            // latency overlaps the row handshake instead of serializing ahead of it.
            auto issue_weights = [&](uint32_t k_block) {
                cb_reserve_back(cb_weight_tiles, weight_block_tiles);
                const uint32_t weight_slot = get_write_ptr(cb_weight_tiles);  // lockstep across the column
                if (reads_weights) {
                    read_weight_block(k_block, n0, weight_slot);
                }
                return weight_slot;
            };
            auto land_weights = [&](uint32_t weight_slot) {
                if (reads_weights) {
                    noc_async_read_barrier();
                    if constexpr (weight_mcast) {
                        weight_sender->send(weight_slot, weight_slot, weight_block_bytes);  // self-excluded
                    }
                } else {
                    weight_receiver->receive();
                }
                cb_push_back(cb_weight_tiles, weight_block_tiles);
            };

            // Weight block w is issued before act block w - weight_lead and landed after it; its
            // slot frees when compute is done with w - weight_depth (= the act block 2 back, the
            // same gate as the double-buffered landing reserve).
            // Resident weights: nothing to stream (w_next starts past the last K-block).
            uint32_t w_next = weights_resident ? num_k_blocks : 0;
            for (; w_next < weight_lead && w_next < num_k_blocks; ++w_next) {
                land_weights(issue_weights(w_next));
            }
            for (uint32_t k_block = 0; k_block < num_k_blocks; ++k_block) {
                const bool w_pending = w_next < num_k_blocks;
                const uint32_t weight_slot = w_pending ? issue_weights(w_next) : 0;
                // ---- act_exchange: K-block k_block's in0, from its slice owner ----
                if constexpr (act_mcast) {
                    const uint32_t owner = kblock_owner(k_block);
                    cb_reserve_back(cb_act_tiles, act_block_tiles);
                    const uint32_t landing = get_write_ptr(cb_act_tiles);  // identical on every group core
                    if (owner == my_slice) {
                        cb_wait_front(cb_act_send, act_block_tiles);
                        act_sender->send(get_read_ptr(cb_act_send), landing, act_block_bytes);
                        cb_pop_front(cb_act_send, act_block_tiles);
                    } else {
                        act_receiver->receive(owner);  // round = owner -> sender phase
                    }
                    cb_push_back(cb_act_tiles, act_block_tiles);
                }
                if (w_pending) {
                    land_weights(weight_slot);
                    ++w_next;
                }
            }

            if constexpr (k_exchange) {
                // Output is untilized into the shard in place; copy only a staged partial tail.
                if (has_out_tail && m_block + 1 == num_m_blocks) {
                    cb_wait_front(cb_out_rm, block_n_tiles);
                    const uint32_t last_tile = m_tile_end - 1;
                    const uint32_t valid_rows = M - last_tile * TILE_DIM;
                    const uint32_t dst = output_addr + (last_tile - m_tile_start) * TILE_DIM * out_row_bytes;
                    noc_async_write(get_read_ptr(cb_out_rm), get_noc_addr(dst), valid_rows * out_row_bytes);
                    noc_async_write_barrier();
                    cb_pop_front(cb_out_rm, block_n_tiles);
                }
                continue;
            }

            // ---- untilize_store_block: drain bm tile-rows ----
            const uint32_t col_offset_bytes = n0 * TILE_DIM * elem_bytes;
            for (uint32_t t = 0; t < block_m_tiles; ++t) {
                cb_wait_front(cb_out_rm, block_n_tiles);
                const uint32_t tile = t0 + t;
                if (tile < t1) {
                    uint32_t src = get_read_ptr(cb_out_rm);
                    uint32_t pix = tile * TILE_DIM;
                    for (uint32_t r = 0; r < TILE_DIM && pix < M; ++r, ++pix) {
                        noc_async_write(src, output.get_noc_addr(pix, col_offset_bytes), out_row_bytes);
                        src += out_row_bytes;
                    }
                    // Only the source slots must be drained before they are recycled; the
                    // DRAM-side completion is awaited once at kernel end.
                    noc_async_writes_flushed();
                }
                cb_pop_front(cb_out_rm, block_n_tiles);
            }
        }
    }
    noc_async_write_barrier();
}
