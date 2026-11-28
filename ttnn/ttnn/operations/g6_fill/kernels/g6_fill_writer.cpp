// SPDX-FileCopyrightText: © 2026 Tenstorrent Inc.
// SPDX-License-Identifier: Apache-2.0

// g6_fill writer (BRISC / NoC1): scatter one constant run of e over the units this core owns.
//
// The only kernel of the op. Both regimes are this one loop; only the CT geometry differs.
// - R1 (TILE): a unit is a tile page. Its two segments are row 0 of face 0 and face 1, at
//   offsets {0, face_bytes}. Rows 1-31 and faces 2-3 are padding of the rank-1 tile plane
//   [1, N] and are not written.
// - R2 (ROW_MAJOR): a unit is the chunk of page 0 at offset k * rm_chunk_bytes. Writes are
//   clamped to the 4N logical bytes, so only the last chunk is short.
// The source run is either built in cb_fill_row or, with fill_row_in_common_args (R1 only),
// read in place from the common runtime args, where the host put one copy of e per word.
//
// Raw dataflow API, no kernel_lib helper: no helper writes one constant L1 run to a set of
// (page, in-page offset) destinations of an interleaved tensor (op_design.md, API Mapping).
// The run is built with RISC word stores because no helper writes an arbitrary 32-bit
// constant; it is segment_bytes / 4 words (R1: 16; R2: at most RM_CHUNK_CAP_BYTES / 4),
// written once per core.

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

// Segment s of unit u: segment_bytes at byte u * unit_byte_stride + s * segment_stride_bytes of
// page u * unit_page_stride, clamped to the page's first extent_bytes. Non-posted one-packet
// writes so the closing barrier waits for every ack.
template <
    uint32_t segment_bytes,
    uint32_t segments_per_unit,
    uint32_t segment_stride_bytes,
    uint32_t unit_page_stride,
    uint32_t unit_byte_stride,
    uint32_t extent_bytes,
    typename Accessor>
FORCE_INLINE void scatter_block(const Accessor& dst, uint32_t row_addr, uint32_t first_unit, uint32_t num_units) {
    for (uint32_t unit = first_unit; unit < first_unit + num_units; ++unit) {
        const uint32_t page = unit * unit_page_stride;
        const uint32_t unit_offset = unit * unit_byte_stride;
        for (uint32_t segment = 0; segment < segments_per_unit; ++segment) {
            const uint32_t offset = unit_offset + segment * segment_stride_bytes;
            const uint32_t left = extent_bytes - offset;
            noc_async_write_one_packet(
                row_addr, dst.get_noc_addr(page, offset), left < segment_bytes ? left : segment_bytes);
        }
    }
}

}  // namespace

void kernel_main() {
    constexpr uint32_t cb_fill_row = get_compile_time_arg_val(0);
    constexpr uint32_t segment_bytes = get_compile_time_arg_val(1);
    constexpr uint32_t segments_per_unit = get_compile_time_arg_val(2);
    constexpr uint32_t segment_stride_bytes = get_compile_time_arg_val(3);
    constexpr uint32_t unit_page_stride = get_compile_time_arg_val(4);
    constexpr uint32_t unit_byte_stride = get_compile_time_arg_val(5);
    constexpr uint32_t extent_bytes = get_compile_time_arg_val(6);
    constexpr uint32_t page_bytes = get_compile_time_arg_val(7);
    constexpr uint32_t block_units = get_compile_time_arg_val(8);
    constexpr bool fill_row_in_common_args = get_compile_time_arg_val(9) != 0;
    constexpr uint32_t dst_addr_arg = get_compile_time_arg_val(10);
    constexpr auto dst_args = TensorAccessorArgs<11>();

    constexpr uint32_t segment_words = segment_bytes / sizeof(uint32_t);

    static_assert(segment_bytes % sizeof(uint32_t) == 0, "segment must hold whole 32-bit elements");
    static_assert(segment_bytes <= NOC_MAX_BURST_SIZE, "a one-packet write cannot exceed one burst");
    // Blackhole DRAM writes need source and destination congruent mod 16: the run starts at an
    // L1-aligned base, so every write must start 16-B aligned in its page.
    static_assert(unit_byte_stride % 16 == 0 && segment_stride_bytes % 16 == 0, "write starts must be 16-B aligned");
    static_assert(block_units > 0, "block_units must be positive");
    static_assert(
        !fill_row_in_common_args || dst_addr_arg >= segment_words, "the in-place fill run must precede dst_addr");

    // e is a common runtime arg, never a CT arg: the cached program must take a new value on
    // every call. Layout (host-owned): copies of e from index 0, then dst_addr at dst_addr_arg.
    uint32_t row_addr;
    if constexpr (fill_row_in_common_args) {
        row_addr = get_common_arg_addr(0);
    } else {
        DeviceZoneScopedN("G6FILL-BUILD-ROW");
        row_addr = build_fill_row(cb_fill_row, segment_words, get_common_arg_val<uint32_t>(0));
    }
    ASSERT((row_addr & 0xF) == 0);

    {
        DeviceZoneScopedN("G6FILL-ISSUE");
        // The argument reads sit inside the zone so no work of the core falls outside one (BB-21).
        const uint32_t dst_addr = get_common_arg_val<uint32_t>(dst_addr_arg);
        const uint32_t start_unit = get_arg_val<uint32_t>(0);
        const uint32_t core_units = get_arg_val<uint32_t>(1);
        const auto dst = TensorAccessor(dst_args, dst_addr, page_bytes);
        for (uint32_t done = 0; done < core_units; done += block_units) {
            const uint32_t remaining = core_units - done;
            const uint32_t this_block = remaining < block_units ? remaining : block_units;
            scatter_block<
                segment_bytes,
                segments_per_unit,
                segment_stride_bytes,
                unit_page_stride,
                unit_byte_stride,
                extent_bytes>(dst, row_addr, start_unit + done, this_block);
        }
    }

    {
        DeviceZoneScopedN("G6FILL-DRAIN");
        noc_async_write_barrier();
    }
}
