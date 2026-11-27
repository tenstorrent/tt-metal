// SPDX-FileCopyrightText: © 2026 Tenstorrent Inc.
// SPDX-License-Identifier: Apache-2.0

// g6_fill writer (BRISC / NoC1), regime R1: tile row-0 segment scatter.
//
// The only kernel of the op. It writes one face row of e to the row-0 segment of face 0 and
// face 1 of every tile this core owns. Rows 1-31 and faces 2-3 are padding of the rank-1
// tile plane [1, N] and are not written. The source row is either built in cb_fill_row or,
// with fill_row_in_common_args, read in place from the common runtime args, where the host
// put one copy of e per segment word.
//
// Raw dataflow API, no kernel_lib helper: no helper writes one constant L1 run to a set of
// (page, in-page offset) destinations of an interleaved tensor (op_design.md, API Mapping).
// The row is built with RISC word stores because no helper writes an arbitrary 32-bit
// constant; it is 16 words, written once per core.

#include "api/dataflow/dataflow_api.h"

namespace {

// e x (segment_bytes / 4) at the CB base. cb_fill_row is a private source buffer of this
// thread, so it is never reserved, pushed or popped.
FORCE_INLINE uint32_t build_fill_row(uint32_t cb_fill_row, uint32_t segment_words, uint32_t fill_bits) {
    const uint32_t row_addr = get_write_ptr(cb_fill_row);
    volatile tt_l1_ptr uint32_t* row = reinterpret_cast<volatile tt_l1_ptr uint32_t*>(row_addr);
    for (uint32_t w = 0; w < segment_words; ++w) {
        row[w] = fill_bits;
    }
    return row_addr;
}

// Non-posted one-packet writes so the closing barrier waits for every ack.
template <uint32_t segment_bytes, uint32_t segments_per_tile, uint32_t segment_stride_bytes, typename Accessor>
FORCE_INLINE void scatter_row0_block(const Accessor& dst, uint32_t row_addr, uint32_t first_tile, uint32_t num_tiles) {
    for (uint32_t tile = first_tile; tile < first_tile + num_tiles; ++tile) {
        for (uint32_t segment = 0; segment < segments_per_tile; ++segment) {
            noc_async_write_one_packet(row_addr, dst.get_noc_addr(tile, segment * segment_stride_bytes), segment_bytes);
        }
    }
}

}  // namespace

void kernel_main() {
    constexpr uint32_t cb_fill_row = get_compile_time_arg_val(0);
    constexpr uint32_t segment_bytes = get_compile_time_arg_val(1);
    constexpr uint32_t segments_per_tile = get_compile_time_arg_val(2);
    constexpr uint32_t segment_stride_bytes = get_compile_time_arg_val(3);
    constexpr uint32_t page_bytes = get_compile_time_arg_val(4);
    constexpr uint32_t block_w_tiles = get_compile_time_arg_val(5);
    constexpr bool fill_row_in_common_args = get_compile_time_arg_val(6) != 0;
    constexpr uint32_t dst_addr_arg = get_compile_time_arg_val(7);
    constexpr auto dst_args = TensorAccessorArgs<8>();

    constexpr uint32_t segment_words = segment_bytes / sizeof(uint32_t);

    static_assert(segment_bytes % sizeof(uint32_t) == 0, "segment must hold whole 32-bit elements");
    static_assert(segment_bytes <= NOC_MAX_BURST_SIZE, "a one-packet write cannot exceed one burst");
    static_assert(block_w_tiles > 0, "block_w_tiles must be positive");
    static_assert(
        !fill_row_in_common_args || dst_addr_arg >= segment_words, "the in-place fill run must precede dst_addr");

    // e is a common runtime arg, never a CT arg: the cached program must take a new value on
    // every call. Layout (host-owned): copies of e from index 0, then dst_addr at dst_addr_arg.
    uint32_t row_addr;
    if constexpr (fill_row_in_common_args) {
        // Blackhole DRAM writes need source and destination congruent mod 16; every segment
        // starts 16-B aligned in its page, and the common-arg base is L1-aligned.
        row_addr = get_common_arg_addr(0);
        ASSERT((row_addr & 0xF) == 0);
    } else {
        DeviceZoneScopedN("G6FILL-BUILD-ROW");
        row_addr = build_fill_row(cb_fill_row, segment_words, get_common_arg_val<uint32_t>(0));
    }

    {
        DeviceZoneScopedN("G6FILL-ISSUE");
        // The argument reads sit inside the zone so no work of the core falls outside one (BB-21).
        const uint32_t dst_addr = get_common_arg_val<uint32_t>(dst_addr_arg);
        const uint32_t start_tile = get_arg_val<uint32_t>(0);
        const uint32_t core_w_tiles = get_arg_val<uint32_t>(1);
        const auto dst = TensorAccessor(dst_args, dst_addr, page_bytes);
        for (uint32_t done = 0; done < core_w_tiles; done += block_w_tiles) {
            const uint32_t remaining = core_w_tiles - done;
            const uint32_t this_block = remaining < block_w_tiles ? remaining : block_w_tiles;
            scatter_row0_block<segment_bytes, segments_per_tile, segment_stride_bytes>(
                dst, row_addr, start_tile + done, this_block);
        }
    }

    {
        DeviceZoneScopedN("G6FILL-DRAIN");
        noc_async_write_barrier();
    }
}
