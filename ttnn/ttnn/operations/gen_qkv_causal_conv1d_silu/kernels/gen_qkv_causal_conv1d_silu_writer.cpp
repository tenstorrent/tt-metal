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

#include <cstdint>

#include "api/dataflow/dataflow_api.h"

namespace {

constexpr uint32_t cb_output_tiles = 3;

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
    constexpr auto q_args = TensorAccessorArgs<5>();
    constexpr auto k_args = TensorAccessorArgs<q_args.next_compile_time_args_offset()>();
    constexpr auto v_args = TensorAccessorArgs<k_args.next_compile_time_args_offset()>();

    uint32_t arg = 0;
    const uint32_t q_addr = get_arg_val<uint32_t>(arg++);
    const uint32_t k_addr = get_arg_val<uint32_t>(arg++);
    const uint32_t v_addr = get_arg_val<uint32_t>(arg++);
    const uint32_t ch_group_tile_start = get_arg_val<uint32_t>(arg++);
    const uint32_t core_seq_start = get_arg_val<uint32_t>(arg++);
    const uint32_t core_seq_tiles = get_arg_val<uint32_t>(arg++);

    const auto q_acc = TensorAccessor(q_args, q_addr);
    const auto k_acc = TensorAccessor(k_args, k_addr);
    const auto v_acc = TensorAccessor(v_args, v_addr);

    const uint32_t group_hi = ch_group_tile_start + block_ch_tiles;
    const Segment q_seg = make_segment(ch_group_tile_start, group_hi, 0, q_tiles);
    const Segment k_seg = make_segment(ch_group_tile_start, group_hi, q_tiles, q_tiles + k_tiles);
    const Segment v_seg = make_segment(ch_group_tile_start, group_hi, q_tiles + k_tiles, q_tiles + k_tiles + v_tiles);

    for (uint32_t r = core_seq_start; r < core_seq_start + core_seq_tiles; ++r) {
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
