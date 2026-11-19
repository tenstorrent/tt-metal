// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <cstdint>

#include "api/dataflow/dataflow_api.h"
#include "ttnn/cpp/ttnn/operations/experimental/kda/device/kernels/persistent_counter.hpp"

// The output gate's operands, shared in groups: the cores whose blocks need the same K blocks of an operand (the rows
// of x a share of every head's chunks needs, or a head's columns of W). A group's leader reads each K block from DRAM
// and multicasts it into the other members' buffers, at the same address on every member. The groups are a table in
// the common runtime args from `table`: per group an offset to its record, then the records: the leader's NoC
// x | y << 16, the receivers, the rectangles, then per rectangle its lowest and highest corners' NoC x | y << 16 and
// its cores. Counts go to persistent counters (persistent_counter.hpp): a receiver counts its free
// slot on the leader's `ready`, the leader counts each block landed on the receivers' `valid`.
namespace kda_gate {

struct Group {
    uint32_t leader_xy;
    uint32_t receivers;
    uint32_t rectangles;
    uint32_t first_rectangle;  // common arg index
};

FORCE_INLINE Group group(uint32_t table, uint32_t index) {
    const uint32_t record = table + get_common_arg_val<uint32_t>(table + index);
    return {
        get_common_arg_val<uint32_t>(record),
        get_common_arg_val<uint32_t>(record + 1),
        get_common_arg_val<uint32_t>(record + 2),
        record + 3};
}

FORCE_INLINE uint64_t noc_address(uint32_t xy, uint32_t address) {
    return get_noc_addr(xy & 0xFFFF, xy >> 16, address);
}

// One operand's K blocks of `block_bytes` through `cb` (`block_tiles` pages each). The leader stages each with
// read(block, slot) and sends it on; the receivers take it in their own slot.
struct Share {
    Group group;
    uint32_t cb;
    uint32_t block_tiles;
    uint32_t block_bytes;
    uint32_t ready;
    uint32_t valid;
    bool leader = group.leader_xy == (uint32_t{my_x[noc_index]} | uint32_t{my_y[noc_index]} << 16);

    template <typename Read>
    FORCE_INLINE void block(uint32_t b, const Read& read) const {
        cb_reserve_back(cb, block_tiles);
        const uint32_t slot = get_write_ptr(cb);
        if (leader) {
            read(b, slot);
            noc_async_read_barrier();
            kda_counter::wait(ready, group.receivers * (b + 1));
            for_each_rectangle([&](uint32_t start, uint32_t end, uint32_t cores) {
                if (cores == 1) {
                    noc_async_write(slot, noc_address(start, slot), block_bytes);
                } else {
                    noc_async_write_multicast(slot, multicast_address(start, end, slot), block_bytes, cores);
                }
            });
            // Every member holds the block before any learns of it.
            noc_async_write_barrier();
            for_each_rectangle([&](uint32_t start, uint32_t end, uint32_t cores) {
                if (cores == 1) {
                    noc_semaphore_inc(noc_address(start, valid), 1);
                } else {
                    noc_semaphore_inc_multicast(multicast_address(start, end, valid), 1, cores);
                }
            });
        } else {
            noc_semaphore_inc(noc_address(group.leader_xy, ready), 1);
            kda_counter::wait(valid, b + 1);
        }
        cb_push_back(cb, block_tiles);
    }

    // After the last of `blocks` blocks: retire the counts and take this call's off the counters.
    FORCE_INLINE void finish(uint32_t blocks) const {
        noc_async_atomic_barrier();
        if (leader) {
            kda_counter::consume(ready, group.receivers * blocks);
        } else {
            kda_counter::consume(valid, blocks);
        }
    }

    template <typename Read>
    FORCE_INLINE void all(uint32_t blocks, const Read& read) const {
        for (uint32_t b = 0; b < blocks; ++b) {
            block(b, read);
        }
        finish(blocks);
    }

private:
    // NoC 0 multicasts from the rectangle's lowest corner, NoC 1 from its highest.
    static FORCE_INLINE uint64_t multicast_address(uint32_t low, uint32_t high, uint32_t address) {
        const uint32_t start = noc_index == 0 ? low : high;
        const uint32_t end = noc_index == 0 ? high : low;
        return get_noc_multicast_addr(start & 0xFFFF, start >> 16, end & 0xFFFF, end >> 16, address);
    }

    template <typename F>
    FORCE_INLINE void for_each_rectangle(const F& f) const {
        for (uint32_t r = 0; r < group.rectangles; ++r) {
            const uint32_t arg = group.first_rectangle + 3 * r;
            f(get_common_arg_val<uint32_t>(arg),
              get_common_arg_val<uint32_t>(arg + 1),
              get_common_arg_val<uint32_t>(arg + 2));
        }
    }
};

}  // namespace kda_gate
