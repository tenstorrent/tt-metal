// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include <cstdint>
#include <algorithm>
#include "api/compute/common.h"
#include "api/compute/tile_move_copy.h"
#include "api/compute/eltwise_binary.h"
#include "api/dataflow/circular_buffer.h"

// Direct (one-shot) reduce-scatter reduce kernel: one N-way sum per chunk, where the N blocks of
// cb_reduce are this device's own slice (block 0) plus the N-1 contributions the other devices unicast
// into staging. Block b tile t sits at cb index b*tile_granularity + t.
//
// The N blocks are folded pairwise into DST (num_devices/2 math ops per output tile). The first fold is
// a non-accumulating add (or a copy when N is odd) so nothing depends on DST being zero at acquire.
//
// The parity picks which half of cb_reduce holds this invocation's data, and is needed only on the
// aliased path, where cb_reduce is this core's staging shard and remote senders write into it with no
// flow control (half_stride_tiles is 0 otherwise, making the offset vanish).

namespace detail {

// copy-pasted from api/dataflow/dataflow_api.h
FORCE_INLINE
void noc_semaphore_set(volatile tt_l1_ptr uint32_t* sem_addr, uint32_t val) {
    // set semaphore value to val
    (*sem_addr) = val;
}

}  // namespace detail
void kernel_main() {
    constexpr uint32_t tile_granularity = get_compile_time_arg_val(0);
    constexpr uint32_t num_devices = get_compile_time_arg_val(1);
    constexpr uint32_t cb_reduce_id = get_compile_time_arg_val(2);
    constexpr uint32_t cb_out_id = get_compile_time_arg_val(3);
    constexpr uint32_t half_stride_tiles = get_compile_time_arg_val(4);

    constexpr uint32_t block_stride = tile_granularity;
    constexpr uint32_t group_pages = num_devices * tile_granularity;

    // Up first: its args are all compile-time, so no runtime-state access need precede it.
    compute_kernel_hw_startup(cb_reduce_id, cb_reduce_id, cb_out_id);

    uint32_t arg_idx = 0;
    // This core's partition of every slice: chunk_count chunks / tile_count tiles.
    const uint32_t chunk_count = get_arg_val<uint32_t>(arg_idx++);
    const uint32_t tile_count = get_arg_val<uint32_t>(arg_idx++);
    const uint32_t gen_addr = get_arg_val<uint32_t>(arg_idx++);  // this core's private invocation counter

    auto* gen_ptr = reinterpret_cast<volatile tt_l1_ptr uint32_t*>(gen_addr);
    invalidate_l1_cache();
    const uint32_t invocation = *gen_ptr;
    // Constant tile-index offset into the parity half. The CB read pointer still walks half 0 one group
    // per chunk (the reader pushes there), so this offset alone reaches the right half -- indices stay
    // inside the CB because chunk_count <= chunks_per_slice.
    const uint32_t half_off = (invocation & 1u) * half_stride_tiles;

    CircularBuffer cb_reduce(cb_reduce_id);
    CircularBuffer cb_out(cb_out_id);

    if constexpr (num_devices % 2 == 0) {
        add_tiles_init(cb_reduce_id, cb_reduce_id, true);
    }

    uint32_t tiles_done = 0;
    for (uint32_t k = 0; k < chunk_count; ++k) {
        // Only n = tiles_in_chunk tiles are valid in the last (partial) chunk; the CB slot is always a
        // full group so the reader's coalesced reads stay aligned.
        const uint32_t n = std::min(tile_granularity, tile_count - tiles_done);

        cb_reduce.wait_front(group_pages);
        tile_regs_acquire();

        uint32_t block = 0;
        if constexpr (num_devices % 2 != 0) {
            // Odd block count: seed DST with block0, leaving an even number to fold pairwise.
            copy_tile_init(cb_reduce_id);
            for (uint32_t t = 0; t < n; ++t) {
                copy_tile(cb_reduce_id, half_off + t, t);
            }
            add_tiles_init(cb_reduce_id, cb_reduce_id, true);
            block = 1;
        }
        for (; block + 1 < num_devices; block += 2) {
            const uint32_t a = half_off + block * block_stride;
            const uint32_t b = half_off + (block + 1) * block_stride;
            for (uint32_t t = 0; t < n; ++t) {
                add_tiles(cb_reduce_id, cb_reduce_id, a + t, b + t, t);
            }
        }
        tile_regs_commit();
        cb_reduce.pop_front(group_pages);

        cb_out.reserve_back(tile_granularity);
        tile_regs_wait();
        for (uint32_t t = 0; t < n; ++t) {
            pack_tile(t, cb_out_id, t);
        }
        tile_regs_release();
        cb_out.push_back(tile_granularity);

        tiles_done += n;
    }

    // Exactly one TRISC advances the shared counter, and only after all of this launch's work (pack is
    // last in the unpack -> math -> pack chain). See the per-launch-state note at the top.
    PACK((*gen_ptr = invocation + 1));
}
