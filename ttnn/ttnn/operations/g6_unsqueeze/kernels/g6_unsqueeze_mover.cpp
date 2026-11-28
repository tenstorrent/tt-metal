// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

// g6_unsqueeze mover: one instance per data-movement RISC of a mover core.
//
// Moves row 0 of input face h of every owned page into column 0 of output face
// faces_per_face_row*h of the same page id. Each (page, half) is staged in one
// face slot: a face_row_bytes DRAM read lands in row 0, RISC-V word moves turn
// that row into column 0 in place, and the whole slot is written as one face.
//
// No kernel_lib helper expresses this: copy_tile_column (tile_column_dataflow.hpp)
// copies column to column on a DFB tile, and NoC self-copies cannot move single
// 4-byte values whose source and destination offsets differ modulo the 16-B L1
// NoC alignment. Values move only as uint32_t loads/stores and NoC copies, so
// every bit pattern (NaN payloads, -0.0, subnormals) survives.
//
// The face slots are single-thread scratch: this RISC both fills and drains
// them, so the CB carries no reserve/push/wait/pop.

#include <cstdint>

#include "api/dataflow/dataflow_api.h"
#include "api/debug/kernel_profiler.hpp"

void kernel_main() {
    constexpr uint32_t cb_face_slots = get_compile_time_arg_val(0);
    constexpr uint32_t first_half = get_compile_time_arg_val(1);
    constexpr uint32_t block_tile_halves = get_compile_time_arg_val(2);
    constexpr uint32_t block_w_tiles = get_compile_time_arg_val(3);
    constexpr uint32_t staging_depth = get_compile_time_arg_val(4);
    constexpr uint32_t face_h = get_compile_time_arg_val(5);
    constexpr uint32_t face_w = get_compile_time_arg_val(6);
    constexpr uint32_t face_bytes = get_compile_time_arg_val(7);
    constexpr uint32_t face_row_bytes = get_compile_time_arg_val(8);
    constexpr uint32_t out_half_stride_bytes = get_compile_time_arg_val(9);
    constexpr uint32_t in_page_bytes = get_compile_time_arg_val(10);
    constexpr uint32_t out_page_bytes = get_compile_time_arg_val(11);
    constexpr auto in_args = TensorAccessorArgs<12>();
    constexpr auto out_args = TensorAccessorArgs<in_args.next_compile_time_args_offset()>();

    // Destinations j*face_w (j >= 1) never alias sources 1..face_w-1 only for a square face.
    static_assert(face_h == face_w, "in-place row-to-column move needs square faces");
    static_assert(face_row_bytes == face_w * sizeof(uint32_t), "each word move carries one 4-byte element");
    static_assert(face_bytes == face_h * face_row_bytes, "a slot is one face");
    static_assert(block_tile_halves >= 1 && block_w_tiles >= 1 && staging_depth >= 1, "block knobs start at 1");
    static_assert(face_bytes <= NOC_MAX_BURST_SIZE, "each face write is one NoC packet");

    constexpr uint32_t slots_per_block = block_w_tiles * block_tile_halves;
    constexpr uint32_t group_bytes = slots_per_block * face_bytes;

    uint32_t in_addr, out_addr, first_tile, core_w_tiles, slots_base;
    {
        DeviceZoneScopedN("G6U-SETUP");
        in_addr = get_common_arg_val<uint32_t>(0);
        out_addr = get_common_arg_val<uint32_t>(1);
        first_tile = get_arg_val<uint32_t>(0);
        core_w_tiles = get_arg_val<uint32_t>(1);
        slots_base = get_write_ptr(cb_face_slots);
    }

    const auto in_accessor = TensorAccessor(in_args, in_addr, in_page_bytes);
    const auto out_accessor = TensorAccessor(out_args, out_addr, out_page_bytes);
    const uint32_t num_blocks = (core_w_tiles + block_w_tiles - 1) / block_w_tiles;

    auto block_first_tile = [&](uint32_t block_idx) { return first_tile + block_idx * block_w_tiles; };
    auto block_extent = [&](uint32_t block_idx) {
        const uint32_t remaining = core_w_tiles - block_idx * block_w_tiles;
        return remaining < block_w_tiles ? remaining : block_w_tiles;
    };
    auto group_base = [&](uint32_t block_idx) { return slots_base + (block_idx % staging_depth) * group_bytes; };
    auto gather_rows_block = [&](uint32_t block_idx) {
        DeviceZoneScopedN("G6U-GATHER-ISSUE");
        const uint32_t tile0 = block_first_tile(block_idx);
        const uint32_t num_slots = block_extent(block_idx) * block_tile_halves;
        uint32_t slot = group_base(block_idx);
        for (uint32_t s = 0; s < num_slots; ++s) {
            const uint32_t half = first_half + s % block_tile_halves;
            noc_async_read(
                in_accessor.get_noc_addr(tile0 + s / block_tile_halves, half * face_bytes), slot, face_row_bytes);
            slot += face_bytes;
        }
    };

    auto wait_rows = [&]() {
        DeviceZoneScopedN("G6U-READ-BARRIER");
        // Also invalidates the RISC-V L1 cache, so the word moves below see the landed rows.
        noc_async_read_barrier();
    };

    auto wait_slot_writes_left = [&]() {
        DeviceZoneScopedN("G6U-WRITE-FLUSH");
        noc_async_writes_flushed();
    };

    // Programs the write command buffer with a face's destination and size; the send that follows the
    // move then writes only the two address-low registers. Returns the destination's local address.
    auto arm_face_write = [&](uint32_t tile_id, uint32_t half) {
        const uint64_t dst = out_accessor.get_noc_addr(tile_id, half * out_half_stride_bytes);
        noc_async_write_one_packet_set_state(dst, face_bytes);
        return static_cast<uint32_t>(dst);
    };

    // All loads are issued before any store so they pipeline instead of each waiting on its store.
    auto move_row_to_column = [](uint32_t slot) {
        volatile tt_l1_ptr uint32_t* face = reinterpret_cast<volatile tt_l1_ptr uint32_t*>(slot);
        uint32_t row[face_w];
#pragma GCC unroll 16
        for (uint32_t j = 1; j < face_w; ++j) {
            row[j] = face[j];
        }
#pragma GCC unroll 16
        for (uint32_t j = 1; j < face_w; ++j) {
            face[j * face_w] = row[j];
        }
    };

    // Expects the write of the block's first slot armed; arms each next slot while the current face leaves.
    auto scatter_store_block = [&](uint32_t block_idx, uint32_t armed_dst) {
        DeviceZoneScopedN("G6U-SCATTER-STORE");
        const uint32_t tile0 = block_first_tile(block_idx);
        const uint32_t num_slots = block_extent(block_idx) * block_tile_halves;
        uint32_t slot = group_base(block_idx);
        for (uint32_t s = 0; s < num_slots; ++s) {
            move_row_to_column(slot);
            noc_async_write_one_packet_with_state(slot, armed_dst);
            slot += face_bytes;
            if (s + 1 < num_slots) {
                armed_dst =
                    arm_face_write(tile0 + (s + 1) / block_tile_halves, first_half + (s + 1) % block_tile_halves);
            }
        }
    };

    gather_rows_block(0);
    for (uint32_t block_idx = 0; block_idx < num_blocks; ++block_idx) {
        const bool has_next = block_idx + 1 < num_blocks;
        // Armed in the shadow of the read round trip; nothing below touches the write command buffer before the send.
        const uint32_t armed_dst = arm_face_write(block_first_tile(block_idx), first_half);
        wait_rows();
        if constexpr (staging_depth > 1) {
            // Next block lands in another slot group while this one is moved and stored.
            if (has_next) {
                if (block_idx + 1 >= staging_depth) {
                    wait_slot_writes_left();
                }
                gather_rows_block(block_idx + 1);
            }
        }
        scatter_store_block(block_idx, armed_dst);
        if constexpr (staging_depth == 1) {
            if (has_next) {
                wait_slot_writes_left();
                gather_rows_block(block_idx + 1);
            }
        }
    }

    {
        DeviceZoneScopedN("G6U-WRITE-BARRIER");
        noc_async_write_barrier();
    }
}
