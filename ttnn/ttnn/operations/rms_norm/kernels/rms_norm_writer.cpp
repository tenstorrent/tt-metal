// SPDX-FileCopyrightText: © 2026 Tenstorrent Inc.
// SPDX-License-Identifier: Apache-2.0

// rms_norm writer (RISCV_0 / NOC_1).
//
// Block operations owned here (op_design.md §Block schedule / §Dataflow Strategy):
//   load_gamma     — once, before the first output: this core's gamma slice into cb_gamma_tiles, row 0 of each
//                    tile valid (consumed only through BroadcastDim::Row).
//                      RM gamma:   one DRAM read of the slice into cb_gamma_rm_stick, then per tile two local
//                                  16-element face-row copies (face 0 row 0, face 1 row 0); one barrier.
//                      TILE gamma: two DRAM reads per tile (row 0 of faces 0 and 1); one barrier.
//                      bfloat8_b TILE gamma: one whole-page read per tile (the shared exponents precede the
//                                  mantissas, so face-row byte offsets do not apply); one barrier.
//   combine_block  — R2 only, once per block: partial -> root's cb_rowsum_gather (unicast + semaphore inc);
//                    root waits all partials, hands them to its compute, then multicasts the finalized
//                    inverse-RMS (cb_inv_rms_root) into every member's cb_inv_rms (loopback includes the root).
//   store_block    — per chunk: valid tiles (TILE) / valid stick bytes (RM) to DRAM, one flush per chunk.
//
// Sharded placements (Refinement 2): the output shard is written zero-copy — TILE: cb_out_tiles IS the output
// shard, so store_block only retires the chunk (wait / pop, no NoC); RM: the untilized sticks are copied into the
// local output shard (local NoC writes at the shard stick stride). The combine is unchanged: its W-groups are the
// shard grid's rows of cores (host-side Mcast grouping). A sharded RM slice may start mid-tile: the RM gamma slice
// is read from the DRAM-alignment unit below its first element, and face rows the NoC cannot copy (not 16 B
// aligned) are copied by the RISC (2 x 16 elements per tile).
//
// Helper deviations (design rejections R-b, R-c, R-d): the RM stick store is a raw loop rather than
// write_sticks_after_untilize (nominal stride/pop on the ragged chunk); the gamma face-row scatter uses plain
// self-aimed noc_async_read (the source address changes on every copy, so the stateful set_read_state /
// read_with_state pair would have to be re-armed per read); the many-to-one gather is raw unicast (the Mcast
// helper is one-to-many — it is used for the return leg).

#include <cstdint>

#include "api/dataflow/dataflow_api.h"
#include "ttnn/cpp/ttnn/kernel_lib/mcast/kernel/mcast_args.hpp"

void kernel_main() {
    constexpr uint32_t cb_gamma_rm_stick = get_compile_time_arg_val(0);
    constexpr uint32_t cb_gamma_tiles = get_compile_time_arg_val(1);
    constexpr uint32_t cb_rowsum_partial = get_compile_time_arg_val(2);
    constexpr uint32_t cb_rowsum_gather = get_compile_time_arg_val(3);
    constexpr uint32_t cb_inv_rms_root = get_compile_time_arg_val(4);
    constexpr uint32_t cb_inv_rms = get_compile_time_arg_val(5);
    constexpr uint32_t cb_out_tiles = get_compile_time_arg_val(6);
    constexpr uint32_t cb_out_rm_sticks = get_compile_time_arg_val(7);
    constexpr bool is_rm = get_compile_time_arg_val(8) != 0;
    constexpr bool has_gamma = get_compile_time_arg_val(9) != 0;
    constexpr bool gamma_is_rm = get_compile_time_arg_val(10) != 0;
    constexpr bool is_w_split = get_compile_time_arg_val(11) != 0;
    constexpr uint32_t block_rows = get_compile_time_arg_val(12);
    constexpr uint32_t chunk_w_tiles = get_compile_time_arg_val(13);
    constexpr uint32_t chunks_per_row = get_compile_time_arg_val(14);
    constexpr uint32_t w_group_size = get_compile_time_arg_val(15);
    constexpr uint32_t out_elem_bytes = get_compile_time_arg_val(16);
    constexpr uint32_t gamma_elem_bytes = get_compile_time_arg_val(17);
    constexpr uint32_t tensor_w_tiles = get_compile_time_arg_val(18);
    [[maybe_unused]] constexpr uint32_t tensor_w_elems = get_compile_time_arg_val(19);
    constexpr bool out_sharded = get_compile_time_arg_val(20) != 0;
    constexpr uint32_t sem_gather_id = get_compile_time_arg_val(21);
    constexpr uint32_t gamma_face_row_read_bytes = get_compile_time_arg_val(22);
    constexpr bool gamma_is_block_float = get_compile_time_arg_val(23) != 0;
    constexpr uint32_t shard_stick_bytes = get_compile_time_arg_val(24);  // RM output shard: local stick stride
    constexpr uint32_t gamma_read_align = get_compile_time_arg_val(25);   // DRAM alignment (bytes)
    constexpr auto out_args = TensorAccessorArgs<26>();
    constexpr auto gamma_args = TensorAccessorArgs<out_args.next_compile_time_args_offset()>();
    constexpr auto inv_mcast = dataflow_kernel_lib::McastArgs<
        get_named_compile_time_arg_val("inv_mcast_ct_offset"),
        get_named_compile_time_arg_val("inv_mcast_rt_offset")>();

    constexpr uint32_t TILE_W = 32;
    constexpr uint32_t TILE_H = 32;
    constexpr uint32_t FACE_W = 16;
    constexpr uint32_t row_stride_tiles = chunks_per_row * chunk_w_tiles;
    constexpr uint32_t chunk_tiles = block_rows * chunk_w_tiles;
    constexpr uint32_t out_tile_col_bytes = TILE_W * out_elem_bytes;
    constexpr uint32_t out_stick_stride_bytes = chunk_w_tiles * out_tile_col_bytes;

    const uint32_t out_addr = get_arg_val<uint32_t>(0);
    const uint32_t gamma_addr = get_arg_val<uint32_t>(1);
    const uint32_t tile_row_start = get_arg_val<uint32_t>(2);
    const uint32_t core_rows_tiles = get_arg_val<uint32_t>(3);
    const uint32_t slice_start_tile = get_arg_val<uint32_t>(4);
    const uint32_t slice_w_tiles = get_arg_val<uint32_t>(5);
    const uint32_t member_idx = get_arg_val<uint32_t>(6);
    const uint32_t root_x = get_arg_val<uint32_t>(7);
    const uint32_t root_y = get_arg_val<uint32_t>(8);
    const uint32_t row_sticks_end = get_arg_val<uint32_t>(9);      // RM: sticks of the row space below this are valid
    const uint32_t slice_valid_elems = get_arg_val<uint32_t>(10);  // valid elements of this core's W slice
    const uint32_t out_col0_bytes = get_arg_val<uint32_t>(11);     // RM: slice start within a destination stick
    const uint32_t gamma_col0_elem = get_arg_val<uint32_t>(12);    // global element index of the slice start

    const auto out_accessor = TensorAccessor(out_args, out_addr);

    // ---------------- load_gamma ----------------
    if constexpr (has_gamma) {
        const auto gamma_accessor = TensorAccessor(gamma_args, gamma_addr);
        constexpr uint32_t face_row_bytes = FACE_W * gamma_elem_bytes;
        constexpr uint32_t face1_offset = FACE_W * FACE_W * gamma_elem_bytes;  // row 0 of face 1
        cb_reserve_back(cb_gamma_tiles, row_stride_tiles);
        const uint32_t gamma_base = get_write_ptr(cb_gamma_tiles);
        const uint32_t gamma_tile_bytes = get_tile_size(cb_gamma_tiles);
        if constexpr (gamma_is_rm) {
            const uint32_t stage = get_write_ptr(cb_gamma_rm_stick);
            const uint32_t src_offset = gamma_col0_elem * gamma_elem_bytes;
            const uint32_t lead = src_offset & (gamma_read_align - 1);  // 0 unless a sharded RM slice starts mid-unit
            noc_async_read(
                gamma_accessor.get_noc_addr(0, src_offset - lead), stage, lead + slice_valid_elems * gamma_elem_bytes);
            noc_async_read_barrier();
            for (uint32_t j = 0; j < slice_w_tiles; ++j) {
                const uint32_t src = stage + lead + j * TILE_W * gamma_elem_bytes;
                const uint32_t dst = gamma_base + j * gamma_tile_bytes;
                if ((src & 15) == 0) {
                    noc_async_read(get_noc_addr(src), dst, face_row_bytes);
                    noc_async_read(get_noc_addr(src + face_row_bytes), dst + face1_offset, face_row_bytes);
                } else {
                    // 2-byte granular source (fp32 RM shard of a bf16 gamma): 2 x 16 elements, RISC copy.
                    auto* s16 = reinterpret_cast<volatile tt_l1_ptr uint16_t*>(src);
                    auto* d0 = reinterpret_cast<volatile tt_l1_ptr uint16_t*>(dst);
                    auto* d1 = reinterpret_cast<volatile tt_l1_ptr uint16_t*>(dst + face1_offset);
                    for (uint32_t h = 0; h < face_row_bytes / 2; ++h) {
                        d0[h] = s16[h];
                        d1[h] = s16[face_row_bytes / 2 + h];
                    }
                }
            }
        } else if constexpr (gamma_is_block_float) {
            for (uint32_t j = 0; j < slice_w_tiles; ++j) {
                noc_async_read_page(slice_start_tile + j, gamma_accessor, gamma_base + j * gamma_tile_bytes);
            }
        } else {
            for (uint32_t j = 0; j < slice_w_tiles; ++j) {
                const uint32_t page = slice_start_tile + j;
                const uint32_t dst = gamma_base + j * gamma_tile_bytes;
                noc_async_read(gamma_accessor.get_noc_addr(page, 0), dst, gamma_face_row_read_bytes);
                noc_async_read(
                    gamma_accessor.get_noc_addr(page, face1_offset), dst + face1_offset, gamma_face_row_read_bytes);
            }
        }
        noc_async_read_barrier();
        cb_push_back(cb_gamma_tiles, row_stride_tiles);
    }

    Noc noc;
    const uint32_t num_blocks = (core_rows_tiles + block_rows - 1) / block_rows;
    const uint32_t sem_gather_addr = get_semaphore(sem_gather_id);

    // RM per-chunk valid element count of this core's slice (same as the reader's).
    auto chunk_valid_elems = [&](uint32_t ci) -> uint32_t {
        const uint32_t chunk_elem0 = ci * chunk_w_tiles * TILE_W;
        if (chunk_elem0 >= slice_valid_elems) {
            return 0;
        }
        const uint32_t elems_left = slice_valid_elems - chunk_elem0;
        return chunk_w_tiles * TILE_W < elems_left ? chunk_w_tiles * TILE_W : elems_left;
    };

    for (uint32_t block_idx = 0; block_idx < num_blocks; ++block_idx) {
        const uint32_t block_row0 = block_idx * block_rows;
        const uint32_t remaining_rows = core_rows_tiles - block_row0;
        const uint32_t valid_rows = remaining_rows < block_rows ? remaining_rows : block_rows;

        // ---------------- combine_block (R2) ----------------
        if constexpr (is_w_split) {
            // Reserve the multicast destination before the partial leaves: the root multicasts block b only
            // after every member's block-b partial arrived, so no member's live inverse-RMS is overwritten.
            cb_reserve_back(cb_inv_rms, block_rows);
            cb_wait_front(cb_rowsum_partial, block_rows);
            const uint32_t partial_src = get_read_ptr(cb_rowsum_partial);
            const uint32_t partial_bytes = get_tile_size(cb_rowsum_partial);
            // gather capacity == one block's push, so its write pointer is the CB base on every core (M1/M6).
            const uint32_t gather_base = get_write_ptr(cb_rowsum_gather);
            for (uint32_t r = 0; r < block_rows; ++r) {
                noc_async_write(
                    partial_src + r * partial_bytes,
                    get_noc_addr(root_x, root_y, gather_base + (r * w_group_size + member_idx) * partial_bytes),
                    partial_bytes);
            }
            noc_async_write_barrier();
            cb_pop_front(cb_rowsum_partial, block_rows);

            if (member_idx == 0) {
                cb_reserve_back(cb_rowsum_gather, w_group_size * block_rows);
                noc_semaphore_wait_min(
                    reinterpret_cast<volatile tt_l1_ptr uint32_t*>(sem_gather_addr),
                    (w_group_size - 1) * (block_idx + 1));
                cb_push_back(cb_rowsum_gather, w_group_size * block_rows);

                cb_wait_front(cb_inv_rms_root, block_rows);
                auto sender = inv_mcast.optional_sender(noc);
                sender->send(get_read_ptr(cb_inv_rms_root), get_write_ptr(cb_inv_rms), block_rows * partial_bytes);
                cb_pop_front(cb_inv_rms_root, block_rows);
            } else {
                noc_semaphore_inc(get_noc_addr(root_x, root_y, sem_gather_addr), 1);
                auto receiver = inv_mcast.optional_receiver(noc);
                receiver->receive();
            }
            cb_push_back(cb_inv_rms, block_rows);
        }

        // ---------------- store_block ----------------
        for (uint32_t ci = 0; ci < chunks_per_row; ++ci) {
            if constexpr (!is_rm) {
                const uint32_t chunk_col0 = ci * chunk_w_tiles;
                const uint32_t chunk_valid =
                    chunk_col0 >= slice_w_tiles
                        ? 0
                        : (slice_w_tiles - chunk_col0 < chunk_w_tiles ? slice_w_tiles - chunk_col0 : chunk_w_tiles);
                cb_wait_front(cb_out_tiles, chunk_tiles);
                if constexpr (out_sharded) {  // compute packed straight into the output shard
                    cb_pop_front(cb_out_tiles, chunk_tiles);
                    continue;
                }
                const uint32_t l1_base = get_read_ptr(cb_out_tiles);
                const uint32_t tile_bytes = get_tile_size(cb_out_tiles);
                for (uint32_t r = 0; r < valid_rows; ++r) {
                    const uint32_t page_row = (tile_row_start + block_row0 + r) * tensor_w_tiles;
                    uint32_t l1_addr = l1_base + r * chunk_w_tiles * tile_bytes;
                    for (uint32_t col = 0; col < chunk_valid; ++col) {
                        noc_async_write_page(page_row + slice_start_tile + chunk_col0 + col, out_accessor, l1_addr);
                        l1_addr += tile_bytes;
                    }
                }
                noc_async_writes_flushed();
                cb_pop_front(cb_out_tiles, chunk_tiles);
            } else {
                const uint32_t valid_bytes = chunk_valid_elems(ci) * out_elem_bytes;
                const uint32_t byte_offset = out_col0_bytes + ci * chunk_w_tiles * out_tile_col_bytes;
                cb_wait_front(cb_out_rm_sticks, chunk_tiles);
                const uint32_t l1_base = get_read_ptr(cb_out_rm_sticks);
                const uint32_t slab_bytes = chunk_w_tiles * get_tile_size(cb_out_rm_sticks);
                if (valid_bytes > 0) {
                    for (uint32_t r = 0; r < valid_rows; ++r) {
                        const uint32_t stick0 = (tile_row_start + block_row0 + r) * TILE_H;
                        if (stick0 >= row_sticks_end) {
                            break;
                        }
                        const uint32_t sticks_valid =
                            row_sticks_end - stick0 < TILE_H ? row_sticks_end - stick0 : TILE_H;
                        const uint32_t slab = l1_base + r * slab_bytes;
                        for (uint32_t i = 0; i < sticks_valid; ++i) {
                            uint64_t dst;
                            if constexpr (out_sharded) {
                                dst = get_noc_addr(out_addr + (stick0 + i) * shard_stick_bytes + byte_offset);
                            } else {
                                dst = out_accessor.get_noc_addr(stick0 + i, byte_offset);
                            }
                            noc_async_write(slab + i * out_stick_stride_bytes, dst, valid_bytes);
                        }
                    }
                }
                noc_async_writes_flushed();
                cb_pop_front(cb_out_rm_sticks, chunk_tiles);
            }
        }
    }
    noc_async_write_barrier();
    noc_async_atomic_barrier();
}
