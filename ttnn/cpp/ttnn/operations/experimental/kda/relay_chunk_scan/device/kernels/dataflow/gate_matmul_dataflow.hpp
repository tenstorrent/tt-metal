// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <cstdint>

#include "api/dataflow/dataflow_api.h"
#include "ttnn/cpp/ttnn/operations/experimental/kda/device/kernels/persistent_counter.hpp"

// The output gate's operands, shared in sets: a set's leader reads each K block of the operand from DRAM into its own
// buffer and sends it into its receivers', at the same address on each, a multicast per rectangle of them. The sets
// are a table in the common runtime args from `table`: per group of cores an offset to its record, then the records:
// the sets, then per set its leader's NoC x | y << 16, its receivers and their rectangles, then per rectangle its
// lowest and highest cores' NoC x | y << 16 and its cores. A core of the group leads a set or is in a rectangle of one.
//
// Counts go to two persistent counters (persistent_counter.hpp) from common arg `counters`: `ready` on the leader,
// where its receivers count the slots they free, and `valid` on the receivers, where the leader counts the blocks.
// A receiver announces every free slot, so blocks are in flight in every slot.
namespace kda_gate {

FORCE_INLINE uint64_t noc_address(uint32_t xy, uint32_t address) {
    return get_noc_addr(xy & 0xFFFF, xy >> 16, address);
}

// One operand's K blocks of `block_bytes` through `cb` (`block_tiles` pages each, the CB a whole number of blocks).
// step() never blocks, so one RISC can move two operands; all() moves one.
class Share {
public:
    Share(uint32_t table, uint32_t group, uint32_t cb, uint32_t block_tiles, uint32_t block_bytes, uint32_t counters) :
        cb_(cb),
        block_tiles_(block_tiles),
        block_bytes_(block_bytes),
        slots_(get_local_cb_interface(cb).fifo_size / block_bytes),
        ready_(get_common_arg_val<uint32_t>(counters)),
        valid_(get_common_arg_val<uint32_t>(counters + 1)) {
        const uint32_t me = uint32_t{my_x[noc_index]} | uint32_t{my_y[noc_index]} << 16;
        const uint32_t record = table + get_common_arg_val<uint32_t>(table + group);
        uint32_t set = record + 1;
        for (uint32_t s = 0; s < get_common_arg_val<uint32_t>(record); ++s) {
            const uint32_t rectangles = get_common_arg_val<uint32_t>(set + 2);
            if (get_common_arg_val<uint32_t>(set) == me) {
                leader_ = true;
                set_ = set;
            }
            for (uint32_t r = 0; r < rectangles; ++r) {
                const uint32_t low = get_common_arg_val<uint32_t>(set + 3 + 3 * r);
                const uint32_t high = get_common_arg_val<uint32_t>(set + 4 + 3 * r);
                if ((me & 0xFFFF) >= (low & 0xFFFF) && (me & 0xFFFF) <= (high & 0xFFFF) && (me >> 16) >= (low >> 16) &&
                    (me >> 16) <= (high >> 16)) {
                    set_ = set;
                }
            }
            set += 3 + 3 * rectangles;
        }
    }

    // Take every free slot (the leader reads its block, a receiver announces the slot) and push each block to compute
    // as it lands; the leader sends its blocks on as its receivers free their slots, and takes a slot again only once
    // its block has been sent. True once `blocks` blocks have moved.
    template <typename Read>
    FORCE_INLINE bool step(uint32_t blocks, const Read& read) {
        while (reserved_ < blocks && reserved_ - (leader_ ? sent_ : pushed_) < slots_ &&
               cb_pages_reservable_at_back(cb_, (reserved_ - pushed_ + 1) * block_tiles_)) {
            cb_reserve_back(cb_, (reserved_ - pushed_ + 1) * block_tiles_);
            if (leader_) {
                read(reserved_, slot(reserved_));
            } else {
                noc_semaphore_inc(noc_address(get_common_arg_val<uint32_t>(set_), ready_), 1);
            }
            ++reserved_;
        }
        if (!leader_) {
            while (pushed_ < reserved_ && counted(valid_, pushed_ + 1)) {
                cb_push_back(cb_, block_tiles_);
                ++pushed_;
            }
            return pushed_ == blocks;
        }
        if (pushed_ < reserved_) {
            noc_async_read_barrier();
            cb_push_back(cb_, (reserved_ - pushed_) * block_tiles_);
            pushed_ = reserved_;
        }
        while (sent_ < pushed_ && counted(ready_, receivers() * (sent_ + 1))) {
            send(slot(sent_));
            ++sent_;
        }
        return sent_ == blocks;
    }

    // After the last of `blocks` blocks: retire the counts and take this call's off the counters.
    FORCE_INLINE void finish(uint32_t blocks) const {
        noc_async_atomic_barrier();
        if (leader_) {
            kda_counter::consume(ready_, receivers() * blocks);
        } else {
            kda_counter::consume(valid_, blocks);
        }
    }

    template <typename Read>
    FORCE_INLINE void all(uint32_t blocks, const Read& read) {
        while (!step(blocks, read)) {
        }
        finish(blocks);
    }

private:
    FORCE_INLINE uint32_t receivers() const { return get_common_arg_val<uint32_t>(set_ + 1); }

    // Block b's slot: the CB holds whole blocks from its base, a block per push.
    FORCE_INLINE uint32_t slot(uint32_t b) const {
        const auto& fifo = get_local_cb_interface(cb_);
        return fifo.fifo_limit - fifo.fifo_size + b % slots_ * block_bytes_;
    }

    static FORCE_INLINE bool counted(uint32_t counter, uint32_t count) {
        invalidate_l1_cache();
        return *reinterpret_cast<volatile tt_l1_ptr uint32_t*>(counter) >= count;
    }

    // NoC 0 multicasts from the rectangle's lowest corner, NoC 1 from its highest.
    static FORCE_INLINE uint64_t multicast_address(uint32_t low, uint32_t high, uint32_t address) {
        const uint32_t start = noc_index == 0 ? low : high;
        const uint32_t end = noc_index == 0 ? high : low;
        return get_noc_multicast_addr(start & 0xFFFF, start >> 16, end & 0xFFFF, end >> 16, address);
    }

    // Each of the set's rectangles, as (lowest, highest, cores).
    template <typename F>
    FORCE_INLINE void for_each_rectangle(const F& f) const {
        for (uint32_t r = 0; r < get_common_arg_val<uint32_t>(set_ + 2); ++r) {
            f(get_common_arg_val<uint32_t>(set_ + 3 + 3 * r),
              get_common_arg_val<uint32_t>(set_ + 4 + 3 * r),
              get_common_arg_val<uint32_t>(set_ + 5 + 3 * r));
        }
    }

    // A block, then its count: on the same NoC and VC, so in order once the data is flushed.
    FORCE_INLINE void send(uint32_t data) const {
        for_each_rectangle([&](uint32_t low, uint32_t high, uint32_t cores) {
            if (cores == 1) {
                noc_async_write(data, noc_address(low, data), block_bytes_);
            } else {
                noc_async_write_multicast(data, multicast_address(low, high, data), block_bytes_, cores);
            }
        });
        noc_async_writes_flushed();
        for_each_rectangle([&](uint32_t low, uint32_t high, uint32_t cores) {
            if (cores == 1) {
                noc_semaphore_inc(noc_address(low, valid_), 1);
            } else {
                noc_semaphore_inc_multicast(multicast_address(low, high, valid_), 1, cores);
            }
        });
    }

    uint32_t cb_;
    uint32_t block_tiles_;
    uint32_t block_bytes_;
    uint32_t slots_;
    uint32_t ready_;
    uint32_t valid_;
    bool leader_ = false;
    uint32_t set_ = 0;
    uint32_t reserved_ = 0;
    uint32_t sent_ = 0;
    uint32_t pushed_ = 0;
};

}  // namespace kda_gate
