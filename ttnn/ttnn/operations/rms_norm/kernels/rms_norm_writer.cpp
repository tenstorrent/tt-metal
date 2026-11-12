// SPDX-FileCopyrightText: © 2026 Tenstorrent Inc.
// SPDX-License-Identifier: Apache-2.0

// rms_norm writer (regime R1). store_chunk per row-chunk, mirroring the reader's batching:
//   TILE: wait c tiles, <= c tile writes (pad columns >= core_w_tiles skipped), one barrier, pop c.
//   RM:   write_sticks_after_untilize for full-width chunks (valid sticks / valid bytes only);
//         the tail chunk of a K>1 row keeps the nominal c*32-element L1 stride (hand-written,
//         see tilize_helpers_dataflow.inl:208-209).
// HEIGHT_SHARDED (placement):
//   RESIDENT     — cb_out / cb_out_rm IS the output shard: compute packs in place; the writer only
//                  drains the credit (no NoC traffic).
//   LOCAL_STICKS — RM: the valid sticks of each untilized row-chunk are copied into this core's output
//                  shard (self-aimed L1 write; the shard's stick count need not be a multiple of 32).
//
// R3 W-split (w_split): before a block's output, the writer runs the cross-core combine of that block:
//   gather   — every core of the row-group writes its cb_partial tiles (one per row, Σx² over its W slice)
//              into the root's cb_gather at slot row*G + slot, then bumps the root's gather semaphore.
//              The root (slot 0, also a worker) waits for the G-1 remote arrivals (cumulative count, no
//              reset) and publishes the block's G*rows tiles to its compute kernel.
//   rstd     — the root's compute reduces the gathered partials (·1/W, +eps, rsqrt) into cb_rstd_root; the
//              root's writer multicasts them into cb_rstd of every core of the group (loopback included, so
//              the root's own cb_rstd is filled by the same send); receivers just receive. cb_rstd's
//              producer is the writer on every core.
// Addresses: cb_gather / cb_rstd are allocated on every core and every core walks them identically (whole
// CB per block), so the local write pointer IS the remote one.

#include <cstdint>

#include "api/dataflow/dataflow_api.h"
#include "api/dataflow/noc.h"
#include "ttnn/cpp/ttnn/kernel_lib/mcast/kernel/mcast_args.hpp"
#include "ttnn/cpp/ttnn/kernel_lib/tilize_helpers_dataflow.hpp"

void kernel_main() {
    constexpr uint32_t cb_rstd = get_compile_time_arg_val(4);
    constexpr uint32_t cb_out = get_compile_time_arg_val(7);
    constexpr uint32_t cb_out_rm = get_compile_time_arg_val(8);

    constexpr uint32_t block_rows = get_compile_time_arg_val(9);
    constexpr uint32_t w_chunk_tiles = get_compile_time_arg_val(10);
    constexpr uint32_t num_w_chunks = get_compile_time_arg_val(11);
    constexpr uint32_t core_w_tiles = get_compile_time_arg_val(12);
    constexpr uint32_t tensor_w_tiles = get_compile_time_arg_val(13);
    constexpr bool is_rm = get_compile_time_arg_val(14) != 0;
    [[maybe_unused]] constexpr uint32_t W = get_compile_time_arg_val(15);
    [[maybe_unused]] constexpr uint32_t num_valid_sticks = get_compile_time_arg_val(16);
    constexpr uint32_t out_elem = get_compile_time_arg_val(17);
    constexpr uint32_t placement = get_compile_time_arg_val(18);
    constexpr uint32_t stick_page_bytes = get_compile_time_arg_val(19);
    constexpr bool w_split = get_compile_time_arg_val(20) != 0;
    constexpr uint32_t group_size = get_compile_time_arg_val(21);
    constexpr uint32_t cb_partial = get_compile_time_arg_val(22);
    constexpr uint32_t cb_gather = get_compile_time_arg_val(23);
    constexpr uint32_t cb_rstd_root = get_compile_time_arg_val(24);
    constexpr uint32_t sem_gather_id = get_compile_time_arg_val(25);
    constexpr bool out_resident = placement == 1;
    constexpr bool out_local_sticks = placement == 2;
    constexpr auto out_args = TensorAccessorArgs<26>();
    constexpr auto rstd_mcast = dataflow_kernel_lib::McastArgs<
        get_named_compile_time_arg_val("rstd_mcast_ct_offset"),
        get_named_compile_time_arg_val("rstd_mcast_rt_offset")>();

    const uint32_t out_addr = get_arg_val<uint32_t>(0);
    const uint32_t start_tile_row = get_arg_val<uint32_t>(1);
    const uint32_t num_tile_rows = get_arg_val<uint32_t>(2);
    const uint32_t w_tile_start = get_arg_val<uint32_t>(3);
    const uint32_t stick_base = get_arg_val<uint32_t>(4);
    const uint32_t stick_end = get_arg_val<uint32_t>(5);
    const uint32_t w_elem_start = get_arg_val<uint32_t>(6);
    const uint32_t w_elem_end = get_arg_val<uint32_t>(7);
    [[maybe_unused]] const bool is_root = get_arg_val<uint32_t>(8) != 0;
    [[maybe_unused]] const uint32_t slot = get_arg_val<uint32_t>(9);
    [[maybe_unused]] const uint32_t root_x = get_arg_val<uint32_t>(10);
    [[maybe_unused]] const uint32_t root_y = get_arg_val<uint32_t>(11);

    constexpr uint32_t chunk_elems = w_chunk_tiles * 32;
    constexpr uint32_t row_stride_tiles = num_w_chunks * w_chunk_tiles;
    const auto out_acc = TensorAccessor(out_args, out_addr);
    constexpr uint32_t cb_resident = is_rm ? cb_out_rm : cb_out;

    if constexpr (out_resident && !w_split) {
        cb_wait_front(cb_resident, num_tile_rows * row_stride_tiles);
        cb_pop_front(cb_resident, num_tile_rows * row_stride_tiles);
        return;
    }

    // ---- R3 combine state ----
    [[maybe_unused]] uint32_t gather_base = 0;
    [[maybe_unused]] uint64_t root_sem_noc = 0;
    [[maybe_unused]] volatile tt_l1_ptr uint32_t* sem_ptr = nullptr;
    if constexpr (w_split) {
        gather_base = get_write_ptr(cb_gather);  // == the root's: every block fills the whole CB from its base
        const uint32_t sem_addr = get_semaphore(sem_gather_id);
        root_sem_noc = get_noc_addr(root_x, root_y, sem_addr);
        sem_ptr = reinterpret_cast<volatile tt_l1_ptr uint32_t*>(sem_addr);
    }
    Noc noc;

    const uint32_t num_blocks = (num_tile_rows + block_rows - 1) / block_rows;
    for (uint32_t block = 0; block < num_blocks; ++block) {
        const uint32_t first_row = block * block_rows;
        const uint32_t rows_this_block =
            (num_tile_rows - first_row < block_rows) ? (num_tile_rows - first_row) : block_rows;

        if constexpr (w_split) {
            // ---- gather: this core's partials -> root's cb_gather, tile (row i, slot) at i*G + slot ----
            constexpr uint32_t partial_bytes = get_tile_size(cb_partial);
            if (is_root) {
                cb_reserve_back(cb_gather, group_size * rows_this_block);  // root compute released block-1
            }
            cb_wait_front(cb_partial, rows_this_block);
            const uint32_t src = get_read_ptr(cb_partial);
            for (uint32_t i = 0; i < rows_this_block; ++i) {
                noc_async_write(
                    src + i * partial_bytes,
                    get_noc_addr(root_x, root_y, gather_base + (i * group_size + slot) * partial_bytes),
                    partial_bytes);
            }
            noc_async_write_barrier();
            cb_pop_front(cb_partial, rows_this_block);
            if (is_root) {
                noc_semaphore_wait_min(sem_ptr, (group_size - 1) * (block + 1));
                cb_push_back(cb_gather, group_size * rows_this_block);
            } else {
                noc_semaphore_inc(root_sem_noc, 1);
                noc_async_atomic_barrier();
            }

            // ---- rstd: root multicasts the combined rstd into every group core's cb_rstd ----
            constexpr uint32_t rstd_bytes = get_tile_size(cb_rstd);
            cb_reserve_back(cb_rstd, rows_this_block);
            const uint32_t rstd_dst = get_write_ptr(cb_rstd);  // same on every core of the group
            // Role-polymorphic pipes: a kernel group may hold only receivers (a ragged last slice column).
            if (is_root) {
                cb_wait_front(cb_rstd_root, rows_this_block);
                auto sender = rstd_mcast.optional_sender(noc);
                ASSERT(sender.has_value());
                sender->send(get_read_ptr(cb_rstd_root), rstd_dst, rows_this_block * rstd_bytes);
                cb_pop_front(cb_rstd_root, rows_this_block);
            } else {
                auto receiver = rstd_mcast.optional_receiver(noc);
                ASSERT(receiver.has_value());
                receiver->receive();
            }
            cb_push_back(cb_rstd, rows_this_block);
        }

        if constexpr (out_resident) {
            continue;  // output is produced in place in the resident shard (drained below)
        }
        // Compute emits chunk-major (all rows of chunk k, then chunk k+1), so the writer walks the same order.
        for (uint32_t k = 0; k < num_w_chunks; ++k) {
            for (uint32_t i = 0; i < rows_this_block; ++i) {
                const uint32_t tile_row = start_tile_row + first_row + i;
                if constexpr (!is_rm) {
                    constexpr uint32_t out_tile_bytes = get_tile_size(cb_out);
                    cb_wait_front(cb_out, w_chunk_tiles);
                    uint32_t l1 = get_read_ptr(cb_out);
                    for (uint32_t j = 0; j < w_chunk_tiles; ++j) {
                        const uint32_t col = k * w_chunk_tiles + j;
                        if (col < core_w_tiles) {
                            noc_async_write_page(tile_row * tensor_w_tiles + w_tile_start + col, out_acc, l1);
                        }
                        l1 += out_tile_bytes;
                    }
                    noc_async_write_barrier();
                    cb_pop_front(cb_out, w_chunk_tiles);
                } else {
                    const uint32_t first_stick = stick_base + (first_row + i) * 32;
                    const uint32_t sticks = (stick_end - first_stick < 32) ? (stick_end - first_stick) : 32;
                    const uint32_t elem_off = w_elem_start + k * chunk_elems;
                    const uint32_t valid_elems =
                        (elem_off >= w_elem_end)
                            ? 0
                            : ((w_elem_end - elem_off < chunk_elems) ? (w_elem_end - elem_off) : chunk_elems);
                    const uint32_t valid_bytes = valid_elems * out_elem;
                    if (!out_local_sticks && (valid_elems + 31) / 32 == w_chunk_tiles) {
                        dataflow_kernel_lib::write_sticks_after_untilize<cb_out_rm>(
                            out_acc, sticks, valid_bytes, first_stick, elem_off * out_elem);
                    } else {
                        cb_wait_front(cb_out_rm, w_chunk_tiles);
                        uint32_t l1 = get_read_ptr(cb_out_rm);
                        for (uint32_t s = 0; s < sticks; ++s) {
                            uint64_t dst;
                            if constexpr (out_local_sticks) {
                                dst = get_noc_addr(
                                    out_addr + (first_stick - stick_base + s) * stick_page_bytes +
                                    (elem_off - w_elem_start) * out_elem);
                            } else {
                                dst = out_acc.get_noc_addr(first_stick + s, elem_off * out_elem);
                            }
                            if (valid_bytes != 0) {
                                noc_async_write(l1, dst, valid_bytes);
                            }
                            l1 += chunk_elems * out_elem;
                        }
                        noc_async_write_barrier();
                        cb_pop_front(cb_out_rm, w_chunk_tiles);
                    }
                }
            }
        }
    }

    if constexpr (out_resident) {
        cb_wait_front(cb_resident, num_tile_rows * row_stride_tiles);
        cb_pop_front(cb_resident, num_tile_rows * row_stride_tiles);
    }
}
