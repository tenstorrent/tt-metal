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
// R3 W-split (w_split): before a block's output, the writer runs the cross-core combine of that block. The
// combine is a row-wise reduce-scatter: row i of the block is combined by its owner, slot o = i / q
// (q = combine_rows = ceil(block_rows / G)), which holds local row l = i % q.
//   gather   — every core of the row-group writes its cb_partial tiles of owner o's rows (Σx² over its W slice,
//              contiguous in cb_partial) into o's cb_gather at tiles [slot*q, slot*q + n_o) — slot-major, so
//              it is ONE contiguous write per owner — then bumps each owner's gather semaphore once. An owner
//              waits for the G-1 remote arrivals (cumulative count over the blocks it owns rows in) and
//              publishes the full G*q quantum to its compute kernel (which reduces its n_own rows).
//   relay    — each owner's compute reduces its rows (·1/W, +eps, rsqrt) into cb_rstd_root; with more than one
//              owner, every owner writes those rows into the root's cb_rstd_gather at tile i and bumps the
//              root's relay semaphore.
//   rstd     — the root multicasts the block's rstd rows (cb_rstd_gather, or its own cb_rstd_root when it is the
//              only owner) into cb_rstd of every core of the group (loopback included); receivers just receive.
//              cb_rstd's producer is the writer on every core.
// Addresses: cb_gather / cb_rstd / cb_rstd_gather are allocated on every core and walked identically (whole
// quantum per block), so the local write pointer IS the remote one. Partials / rstd rows move as whole contiguous
// tile runs (one NoC write per owner) rather than per-tile column-0 faces: with many rows per block the per-write
// issue cost, not the bytes, bounds the gather.

#include <cstdint>

#include "api/dataflow/dataflow_api.h"
#include "api/dataflow/noc.h"
#include "api/debug/kernel_profiler.hpp"
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
    constexpr uint32_t combine_rows = get_compile_time_arg_val(26);  // q: rows per combine owner
    constexpr uint32_t cb_rstd_gather = get_compile_time_arg_val(27);
    constexpr uint32_t sem_rstd_gather_id = get_compile_time_arg_val(28);
    constexpr bool out_resident = placement == 1;
    constexpr bool out_local_sticks = placement == 2;
    constexpr auto out_args = TensorAccessorArgs<29>();
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
    constexpr uint32_t group_noc_arg = 12;  // R3: G words, NoC (x | y << 16) of the row-group's cores by slot

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
    [[maybe_unused]] uint32_t rstd_gather_base = 0;
    [[maybe_unused]] uint32_t gather_sem_addr = 0;
    [[maybe_unused]] uint32_t rstd_sem_addr = 0;
    [[maybe_unused]] uint32_t owned_blocks = 0;     // blocks this core combined rows in (gather semaphore target)
    [[maybe_unused]] uint32_t relays_expected = 0;  // root: cumulative remote owners relaying rstd rows
    if constexpr (w_split) {
        gather_base = get_write_ptr(cb_gather);  // == every owner's: each owned block pushes the whole CB
        if constexpr (combine_rows < block_rows) {
            rstd_gather_base = get_write_ptr(cb_rstd_gather);  // raw scratch on the root, never pushed
        }
        gather_sem_addr = get_semaphore(sem_gather_id);
        rstd_sem_addr = get_semaphore(sem_rstd_gather_id);
    }
    auto slot_noc = [&](uint32_t s, uint32_t l1_addr) -> uint64_t {
        const uint32_t xy = get_arg_val<uint32_t>(group_noc_arg + s);
        return get_noc_addr(xy & 0xFFFF, xy >> 16, l1_addr);
    };
    Noc noc;

    const uint32_t num_blocks = (num_tile_rows + block_rows - 1) / block_rows;
    for (uint32_t block = 0; block < num_blocks; ++block) {
        const uint32_t first_row = block * block_rows;
        const uint32_t rows_this_block =
            (num_tile_rows - first_row < block_rows) ? (num_tile_rows - first_row) : block_rows;

        if constexpr (w_split) {
            constexpr uint32_t partial_bytes = get_tile_size(cb_partial);
            constexpr uint32_t face_bytes = partial_bytes / 4;
            // Owners of this block: slots 0 .. num_owners-1; this core owns n_own rows from row slot*q.
            const uint32_t num_owners = (rows_this_block + combine_rows - 1) / combine_rows;
            const uint32_t own_lo = slot * combine_rows;
            const uint32_t n_own =
                own_lo >= rows_this_block
                    ? 0
                    : ((rows_this_block - own_lo < combine_rows) ? (rows_this_block - own_lo) : combine_rows);
            if (n_own > 0) {
                cb_reserve_back(cb_gather, group_size * combine_rows);  // this core's compute released block-1
            }
            // ---- gather: owner o's rows of this core's partials -> o's cb_gather tiles [slot*q, slot*q + n_o) ----
            {
                DeviceZoneScopedN("W_SEND");
                // Pipelined with the partial reduce (one output tile per push): owner o's rows go out as soon as they
                // are packed, so the NoC transfer of owners 0..o-1 overlaps the reduce of the later rows.
                for (uint32_t o = 0; o < num_owners; ++o) {
                    const uint32_t lo = o * combine_rows;
                    const uint32_t n_o = (rows_this_block - lo < combine_rows) ? (rows_this_block - lo) : combine_rows;
                    cb_wait_front(cb_partial, lo + n_o);
                    const uint32_t src = get_read_ptr(cb_partial) + lo * partial_bytes;
                    const uint32_t dst = gather_base + slot * combine_rows * partial_bytes;
                    // A partial is column-0-valid, and column 0 lives only in faces 0 and 2: only those two faces
                    // cross the NoC (the row all-to-all on one NoC ring is bandwidth-bound — bytes x hops).
                    for (uint32_t t = 0; t < n_o; ++t) {
                        const uint32_t s_t = src + t * partial_bytes;
                        const uint64_t d_t = slot_noc(o, dst + t * partial_bytes);
                        noc_async_write(s_t, d_t, face_bytes);
                        noc_async_write(s_t + 2 * face_bytes, d_t + 2 * face_bytes, face_bytes);
                    }
                    if (o != slot) {
                        noc_async_write_barrier();  // the owner's partials have landed before its count moves
                        noc_semaphore_inc(slot_noc(o, gather_sem_addr), 1);
                    }
                }
                noc_async_write_barrier();
                noc_async_atomic_barrier();
                cb_pop_front(cb_partial, rows_this_block);
            }
            if (n_own > 0) {
                ++owned_blocks;
                DeviceZoneScopedN("W_SEMWAIT");
                noc_semaphore_wait_min(
                    reinterpret_cast<volatile tt_l1_ptr uint32_t*>(gather_sem_addr), (group_size - 1) * owned_blocks);
                cb_push_back(cb_gather, group_size * combine_rows);
            }

            // ---- relay: owners' rstd rows -> root's cb_rstd_gather (multi-owner blocks only) ----
            constexpr uint32_t rstd_bytes = get_tile_size(cb_rstd);
            uint32_t mcast_src = 0;
            if (n_own > 0) {
                cb_wait_front(cb_rstd_root, combine_rows);  // n_own rows + (q - n_own) unused pages
                mcast_src = get_read_ptr(cb_rstd_root);
            }
            if constexpr (combine_rows < block_rows) {
                if (num_owners > 1) {
                    if (n_own > 0) {
                        noc_async_write(
                            mcast_src, slot_noc(0, rstd_gather_base + own_lo * rstd_bytes), n_own * rstd_bytes);
                        noc_async_write_barrier();
                        cb_pop_front(cb_rstd_root, combine_rows);
                        if (!is_root) {
                            noc_semaphore_inc(slot_noc(0, rstd_sem_addr), 1);
                            noc_async_atomic_barrier();
                        }
                    }
                    if (is_root) {
                        relays_expected += num_owners - 1;
                        noc_semaphore_wait_min(
                            reinterpret_cast<volatile tt_l1_ptr uint32_t*>(rstd_sem_addr), relays_expected);
                        mcast_src = rstd_gather_base;
                    }
                }
            }

            // ---- rstd: root multicasts the block's rstd rows into every group core's cb_rstd ----
            cb_reserve_back(cb_rstd, rows_this_block);
            const uint32_t rstd_dst = get_write_ptr(cb_rstd);  // same on every core of the group
            // Role-polymorphic pipes: a kernel group may hold only receivers (a ragged last slice column).
            if (is_root) {
                auto sender = rstd_mcast.optional_sender(noc);
                ASSERT(sender.has_value());
                sender->send(mcast_src, rstd_dst, rows_this_block * rstd_bytes);
                if (num_owners == 1 || combine_rows == block_rows) {
                    cb_pop_front(cb_rstd_root, combine_rows);  // single owner: multicast straight from it
                }
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
