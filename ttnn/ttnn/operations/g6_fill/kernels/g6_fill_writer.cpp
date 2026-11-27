// SPDX-FileCopyrightText: © 2026 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

// g6_fill writer (BRISC / NoC1): the op's only kernel. Regime R1 of op_design.md.
//
// A rank-1 TILE tensor [N] is the tile plane [1, N]: tile t holds positions 32t..32t+31 in its
// row 0, split into two 64-B row-0 segments (face 0 row 0 at byte 0, face 1 row 0 at byte
// face_bytes). Every other byte of the tile is padding and is not written.
//
// Raw dataflow API instead of kernel_lib: no helper writes an arbitrary 32-bit constant run to
// (page, in-page offset) destinations of an interleaved tensor (gap recorded in op_design.md,
// "Helpers considered and rejected"). cb_fill_row has no producer/consumer cycle: this thread
// builds it once and every write reads it, so there is no reserve/push/wait/pop.

#include <cstdint>

#include "api/dataflow/dataflow_api.h"
#include "api/debug/kernel_profiler.hpp"
#include "ttnn/cpp/ttnn/kernel_lib/l1_helpers.hpp"

namespace {

// Common runtime args; order fixed by g6_fill_program_descriptor.py (COMMON_ARG_*).
constexpr uint32_t COMMON_ARG_DST_ADDR = 0;
constexpr uint32_t COMMON_ARG_FILL_BITS = 1;

// Unique runtime args; order fixed by g6_fill_program_descriptor.py (UNIQUE_ARG_*).
constexpr uint32_t UNIQUE_ARG_START_TILE = 0;
constexpr uint32_t UNIQUE_ARG_CORE_W_TILES = 1;
constexpr uint32_t UNIQUE_ARG_BLOCK_W_TILES = 2;

template <uint32_t segments_per_tile, uint32_t face_bytes, uint32_t segment_bytes, typename Accessor>
FORCE_INLINE void scatter_row0_block(
    const Accessor& dst, uint32_t fill_row_addr, uint32_t first_tile, uint32_t end_tile) {
    for (uint32_t tile = first_tile; tile < end_tile; ++tile) {
        for (uint32_t segment = 0; segment < segments_per_tile; ++segment) {
            // Non-posted: each write is acked, so the final barrier proves it landed (BB-03).
            noc_async_write_one_packet(fill_row_addr, dst.get_noc_addr(tile, segment * face_bytes), segment_bytes);
        }
    }
}

}  // namespace

void kernel_main() {
    constexpr uint32_t cb_fill_row = get_compile_time_arg_val(0);
    constexpr uint32_t segment_elems = get_compile_time_arg_val(1);
    constexpr uint32_t segments_per_tile = get_compile_time_arg_val(2);
    constexpr uint32_t face_bytes = get_compile_time_arg_val(3);
    constexpr auto dst_args = TensorAccessorArgs<4>();

    // Elements are 32-bit patterns (float32 / int32), asserted on the host.
    constexpr uint32_t segment_bytes = segment_elems * sizeof(uint32_t);
    static_assert(segment_bytes <= NOC_MAX_BURST_SIZE, "a one-packet write must fit one burst");
    static_assert(face_bytes % 16 == 0, "BH DRAM writes need L1 offset == NoC offset (mod 16)");

    const uint32_t dst_addr = get_common_arg_val<uint32_t>(COMMON_ARG_DST_ADDR);
    const uint32_t fill_bits = get_common_arg_val<uint32_t>(COMMON_ARG_FILL_BITS);
    const uint32_t start_tile = get_arg_val<uint32_t>(UNIQUE_ARG_START_TILE);
    const uint32_t core_w_tiles = get_arg_val<uint32_t>(UNIQUE_ARG_CORE_W_TILES);
    const uint32_t block_w_tiles = get_arg_val<uint32_t>(UNIQUE_ARG_BLOCK_W_TILES);

    const uint32_t fill_row_addr = get_write_ptr(cb_fill_row);

    {
        DeviceZoneScopedN("G6FILL-BUILD-ROW");
        // Built before the first write is issued: the NoC reads this run straight from L1, and a
        // stale run would carry the previous call's value at the same CB address.
        volatile tt_l1_ptr uint32_t* fill_row = dataflow_kernel_lib::addr_to_l1_ptr(fill_row_addr);
        for (uint32_t i = 0; i < segment_elems; ++i) {
            fill_row[i] = fill_bits;
        }
    }

    {
        DeviceZoneScopedN("G6FILL-ISSUE");
        const auto dst = TensorAccessor(dst_args, dst_addr);
        const uint32_t end_tile = start_tile + core_w_tiles;
        for (uint32_t block_start = start_tile; block_start < end_tile; block_start += block_w_tiles) {
            const uint32_t block_end = block_start + block_w_tiles < end_tile ? block_start + block_w_tiles : end_tile;
            scatter_row0_block<segments_per_tile, face_bytes, segment_bytes>(
                dst, fill_row_addr, block_start, block_end);
        }
    }

    {
        DeviceZoneScopedN("G6FILL-DRAIN");
        noc_async_write_barrier();
    }
}
