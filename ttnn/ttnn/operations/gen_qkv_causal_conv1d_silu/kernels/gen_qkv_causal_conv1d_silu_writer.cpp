// SPDX-FileCopyrightText: © 2026 Tenstorrent Inc.
// SPDX-License-Identifier: Apache-2.0

// gen_qkv_causal_conv1d_silu writer (BRISC / NoC1) — store_block.
//
// This core owns channel tiles [ch_group_tile_start, ch_group_tile_start + block_ch_tiles) of every
// tile-row in its token range. Global channel tile cg routes to q if cg < Q_t, to k if
// cg < Q_t + K_t, else to v (page r * X_t + (cg - base_X)). A channel group is at most three
// contiguous segments (one per output), resolved once; per tile-row: wait block_ch_tiles, issue all
// writes, flush (source L1 read out), pop. Final write barrier before return (BB-03).
//
// Raw NoC writes rather than write_sticks_after_untilize: outputs are TILE pages routed to three
// tensors; that helper writes RM sticks to one tensor.
//
// Split reader (ROW_MAJOR regime, split_rows > 0; Refinement 2): this RISC also reads body rows
// [32 - split_rows, 32) of every input tile-row on its own NoC (NoC1), whose DRAM endpoints are
// separate ports from the reader's (NoC0) — decode is DRAM-request-rate-bound per port. The reader
// posts each tile-row's L1 destination in cb_split_requests; this kernel issues the reads, waits for
// them, and posts a token in cb_split_done (the reader pushes the rows after it). Before waiting on
// output tile-row m it serves every row through the end of the NEXT block: the rows of m's block are
// required (compute tilizes a whole block before its first output), the next block's are the
// prefetch that keeps reads overlapped with compute. Deadlock-free: the next block's first request
// needs compute to have released block(m) - 1, whose outputs this kernel has already drained.

#include <cstdint>

#include "api/dataflow/dataflow_api.h"

namespace {

constexpr uint32_t cb_output_tiles = 3;
constexpr uint32_t cb_split_requests = 6;  // reader -> this kernel: one L1 row address per tile-row
constexpr uint32_t cb_split_done = 7;      // this kernel -> reader: one token per tile-row

constexpr uint32_t TILE_DIM = 32;

struct Segment {
    uint32_t first_col;  // first tile column inside the destination tensor
    uint32_t count;      // tiles of this core's group that go to the destination
    uint32_t cb_offset;  // tile offset of the segment inside a tile-row of cb_output_tiles
};

// Intersection of [group_lo, group_hi) with the destination's channel-tile range [lo, hi).
FORCE_INLINE Segment make_segment(uint32_t group_lo, uint32_t group_hi, uint32_t lo, uint32_t hi) {
    const uint32_t a = group_lo > lo ? group_lo : lo;
    const uint32_t b = group_hi < hi ? group_hi : hi;
    if (a >= b) {
        return Segment{0, 0, 0};
    }
    return Segment{a - lo, b - a, a - group_lo};
}

template <uint32_t tile_bytes, typename Accessor>
FORCE_INLINE void write_segment(
    const Accessor& acc, const Segment& seg, uint32_t dst_width_tiles, uint32_t tile_row, uint32_t l1_row_base) {
    const uint32_t first_page = tile_row * dst_width_tiles + seg.first_col;
    const uint32_t src = l1_row_base + seg.cb_offset * tile_bytes;
    for (uint32_t i = 0; i < seg.count; ++i) {
        noc_async_write(src + i * tile_bytes, acc.get_noc_addr(first_page + i), tile_bytes);
    }
}

}  // namespace

void kernel_main() {
    constexpr uint32_t block_ch_tiles = get_compile_time_arg_val(0);
    constexpr uint32_t q_tiles = get_compile_time_arg_val(1);
    constexpr uint32_t k_tiles = get_compile_time_arg_val(2);
    constexpr uint32_t v_tiles = get_compile_time_arg_val(3);
    constexpr uint32_t tile_bytes = get_compile_time_arg_val(4);
    constexpr uint32_t block_rows = get_compile_time_arg_val(5);
    constexpr uint32_t split_rows = get_compile_time_arg_val(6);   // 0: the reader reads every row
    constexpr uint32_t stick_bytes = get_compile_time_arg_val(7);  // one token row of this core's channels
    constexpr auto q_args = TensorAccessorArgs<8>();
    constexpr auto k_args = TensorAccessorArgs<q_args.next_compile_time_args_offset()>();
    constexpr auto v_args = TensorAccessorArgs<k_args.next_compile_time_args_offset()>();
    [[maybe_unused]] constexpr auto input_args = TensorAccessorArgs<v_args.next_compile_time_args_offset()>();
    static_assert(split_rows <= TILE_DIM, "split_rows is a row count inside one tile-row");

    uint32_t arg = 0;
    const uint32_t q_addr = get_arg_val<uint32_t>(arg++);
    const uint32_t k_addr = get_arg_val<uint32_t>(arg++);
    const uint32_t v_addr = get_arg_val<uint32_t>(arg++);
    [[maybe_unused]] const uint32_t input_addr = get_arg_val<uint32_t>(arg++);
    const uint32_t ch_group_tile_start = get_arg_val<uint32_t>(arg++);
    const uint32_t core_seq_start = get_arg_val<uint32_t>(arg++);
    const uint32_t core_seq_tiles = get_arg_val<uint32_t>(arg++);
    [[maybe_unused]] const uint32_t input_col_offset_tiles = get_arg_val<uint32_t>(arg++);

    const auto q_acc = TensorAccessor(q_args, q_addr);
    const auto k_acc = TensorAccessor(k_args, k_addr);
    const auto v_acc = TensorAccessor(v_args, v_addr);

    const uint32_t group_hi = ch_group_tile_start + block_ch_tiles;
    const Segment q_seg = make_segment(ch_group_tile_start, group_hi, 0, q_tiles);
    const Segment k_seg = make_segment(ch_group_tile_start, group_hi, q_tiles, q_tiles + k_tiles);
    const Segment v_seg = make_segment(ch_group_tile_start, group_hi, q_tiles + k_tiles, q_tiles + k_tiles + v_tiles);

    // ---- split reader: serve tile-rows [0, rows_served) of this core's range ----
    [[maybe_unused]] uint32_t rows_served = 0;
    [[maybe_unused]] auto serve_split_rows_through = [&](uint32_t end) {
        if constexpr (split_rows > 0) {
            const auto input_acc = TensorAccessor(input_args, input_addr);
            const uint32_t input_byte_offset =
                (input_col_offset_tiles + ch_group_tile_start) * (stick_bytes / block_ch_tiles);
            for (; rows_served < end; ++rows_served) {
                cb_wait_front(cb_split_requests, 1);
                invalidate_l1_cache();  // the address was stored by the reader RISC
                const uint32_t dst = *reinterpret_cast<volatile tt_l1_ptr uint32_t*>(get_read_ptr(cb_split_requests));
                cb_pop_front(cb_split_requests, 1);
                const uint32_t token0 = (core_seq_start + rows_served) * TILE_DIM;
                for (uint32_t i = TILE_DIM - split_rows; i < TILE_DIM; ++i) {
                    noc_async_read(
                        input_acc.get_noc_addr(token0 + i, input_byte_offset), dst + i * stick_bytes, stick_bytes);
                }
                noc_async_read_barrier();
                cb_reserve_back(cb_split_done, 1);
                cb_push_back(cb_split_done, 1);
            }
        }
    };

    for (uint32_t r = core_seq_start; r < core_seq_start + core_seq_tiles; ++r) {
        if constexpr (split_rows > 0) {
            const uint32_t m = r - core_seq_start;
            const uint32_t next_block_end = (m / block_rows + 2) * block_rows;
            serve_split_rows_through(next_block_end < core_seq_tiles ? next_block_end : core_seq_tiles);
        }
        cb_wait_front(cb_output_tiles, block_ch_tiles);
        const uint32_t l1_row_base = get_read_ptr(cb_output_tiles);
        write_segment<tile_bytes>(q_acc, q_seg, q_tiles, r, l1_row_base);
        write_segment<tile_bytes>(k_acc, k_seg, k_tiles, r, l1_row_base);
        write_segment<tile_bytes>(v_acc, v_seg, v_tiles, r, l1_row_base);
        noc_async_writes_flushed();
        cb_pop_front(cb_output_tiles, block_ch_tiles);
    }
    noc_async_write_barrier();
}
