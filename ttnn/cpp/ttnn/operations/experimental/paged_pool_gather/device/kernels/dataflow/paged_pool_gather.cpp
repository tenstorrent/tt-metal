// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0
//
// paged_pool_gather data-movement kernel. Per work item (user b, compact block blk): for each of the
// slots_per_block pool slots of the block, look the pool id up in the user's pool-id row, translate the
// pool's logical block through the user's page-table row, and copy its P token rows out of every tile of
// the source block into the same rows of an L1 tile block (a P-row run inside each 16-row tile face is
// P*32 contiguous bytes: two runs per tile, one per face column). Invalid pools (sentinel / out of range)
// and the padding slots are zero-filled. The finished block is written as Wt whole tiles. Two L1 blocks
// alternate so the writes of one item drain while the next item's reads are in flight.

#include "api/dataflow/dataflow_api.h"
#include "api/dataflow/noc.h"
#include "api/dataflow/circular_buffer.h"
#include "api/core_local_mem.h"
#include "api/tensor/noc_traits.h"

void kernel_main() {
    constexpr uint32_t Wt = get_compile_time_arg_val(0);
    constexpr uint32_t P = get_compile_time_arg_val(1);
    constexpr uint32_t N = get_compile_time_arg_val(2);
    constexpr uint32_t out_blocks = get_compile_time_arg_val(3);
    constexpr uint32_t blocks_per_seq = get_compile_time_arg_val(4);
    constexpr uint32_t pt_row_bytes = get_compile_time_arg_val(5);
    constexpr uint32_t ids_row_bytes = get_compile_time_arg_val(6);
    constexpr uint32_t slots_per_block = get_compile_time_arg_val(7);
    constexpr auto cache_a = TensorAccessorArgs<8>();
    constexpr auto pt_a = TensorAccessorArgs<cache_a.next_compile_time_args_offset()>();
    constexpr auto ids_a = TensorAccessorArgs<pt_a.next_compile_time_args_offset()>();
    constexpr auto out_a = TensorAccessorArgs<ids_a.next_compile_time_args_offset()>();

    constexpr uint32_t tile_bytes = 2048;  // bf16 32x32
    constexpr uint32_t face_bytes = 512;   // 16x16 bf16
    constexpr uint32_t row_bytes = 32;     // 16 bf16 of one face row
    constexpr uint32_t run_bytes = P * row_bytes;
    constexpr uint32_t block_bytes = Wt * tile_bytes;
    constexpr uint32_t SENTINEL = 0xFFFFFFFFu;
    constexpr uint32_t max_pool = (blocks_per_seq * 32u) / P;  // pool ids >= this are out of the user's range

    const uint32_t item0 = get_arg_val<uint32_t>(0);
    const uint32_t n_items = get_arg_val<uint32_t>(1);
    const uint32_t stride = get_arg_val<uint32_t>(2);
    const uint32_t cache_addr = get_arg_val<uint32_t>(3);
    const uint32_t pt_addr = get_arg_val<uint32_t>(4);
    const uint32_t ids_addr = get_arg_val<uint32_t>(5);
    const uint32_t out_addr = get_arg_val<uint32_t>(6);

    const auto cache_acc = TensorAccessor(cache_a, cache_addr, tile_bytes);
    const auto pt_acc = TensorAccessor(pt_a, pt_addr, pt_row_bytes);
    const auto ids_acc = TensorAccessor(ids_a, ids_addr, ids_row_bytes);
    const auto out_acc = TensorAccessor(out_a, out_addr, tile_bytes);

    // L1 scratch (64 B aligned): two tile blocks, the pool-id row, the page-table fragments
    auto align64 = [](uint32_t a) { return (a + 63u) & ~63u; };
    const uint32_t blk_base = align64(CircularBuffer(0).get_write_ptr());
    const uint32_t ids_base = align64(CircularBuffer(1).get_write_ptr());
    const uint32_t ptf_base = align64(CircularBuffer(2).get_write_ptr());

    Noc noc;
    for (uint32_t it = 0; it < n_items; it++) {
        const uint32_t item = item0 + it * stride;
        const uint32_t b = item / out_blocks;
        const uint32_t blk = item % out_blocks;
        const uint32_t buf = blk_base + (it & 1u) * block_bytes;
        // the writes issued from this buffer two items ago must have drained before it is refilled
        noc.async_write_barrier();

        // the user's pool-id row (one page) and, per slot, the 64 B page-table fragment of its block
        noc.async_read(ids_acc, CoreLocalMem<uint32_t>(ids_base), ids_row_bytes, {.page_id = b}, {});
        noc.async_read_barrier();
        const CoreLocalMem<volatile uint32_t> ids(ids_base);
        uint32_t pool_of[slots_per_block];
        uint32_t lblock_of[slots_per_block];
        for (uint32_t j = 0; j < slots_per_block; j++) {
            const uint32_t slot = blk * slots_per_block + j;
            uint32_t p = (slot < N) ? ids[slot] : SENTINEL;
            if (p >= max_pool) {
                p = SENTINEL;  // sentinel or outside the user's pages: zero rows
            }
            pool_of[j] = p;
            if (p != SENTINEL) {
                const uint32_t lblock = (p * P) / 32u;
                lblock_of[j] = lblock;
                const uint32_t off = (lblock * 4u) & ~63u;
                noc.async_read(
                    pt_acc, CoreLocalMem<uint32_t>(ptf_base + j * 64u), 64u, {.page_id = b, .offset_bytes = off}, {});
            }
        }
        noc.async_read_barrier();

        // the pools' rows into the tile block (or zeros)
        for (uint32_t j = 0; j < slots_per_block; j++) {
            const uint32_t p = pool_of[j];
            const uint32_t drow = j * P;  // destination row 0 of the slot inside the block
            const uint32_t doff = (drow / 16u) * 2u * face_bytes + (drow % 16u) * row_bytes;
            if (p == SENTINEL) {
                for (uint32_t t = 0; t < Wt; t++) {
                    for (uint32_t half = 0; half < 2; half++) {
                        const CoreLocalMem<volatile uint32_t> z(buf + t * tile_bytes + doff + half * face_bytes);
                        for (uint32_t w = 0; w < run_bytes / 4u; w++) {
                            z[w] = 0;
                        }
                    }
                }
                continue;
            }
            const CoreLocalMem<volatile uint32_t> frag(ptf_base + j * 64u);
            const uint32_t pblock = frag[(lblock_of[j] * 4u & 63u) / 4u];
            const uint32_t srow = (p * P) % 32u;
            const uint32_t soff = (srow / 16u) * 2u * face_bytes + (srow % 16u) * row_bytes;
            for (uint32_t t = 0; t < Wt; t++) {
                const uint32_t page = pblock * Wt + t;
                for (uint32_t half = 0; half < 2; half++) {
                    noc.async_read(
                        cache_acc,
                        CoreLocalMem<uint32_t>(buf + t * tile_bytes + doff + half * face_bytes),
                        run_bytes,
                        {.page_id = page, .offset_bytes = soff + half * face_bytes},
                        {});
                }
            }
        }
        noc.async_read_barrier();

        // the finished block: Wt whole tiles of user b's compact page blk
        const uint32_t opage0 = (b * out_blocks + blk) * Wt;
        for (uint32_t t = 0; t < Wt; t++) {
            noc.async_write(
                CoreLocalMem<uint32_t>(buf + t * tile_bytes), out_acc, tile_bytes, {}, {.page_id = opage0 + t});
        }
    }
    noc.async_write_barrier();
}
