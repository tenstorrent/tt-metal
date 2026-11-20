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
// The leader counts the blocks it sends on the receivers, in the persistent counter (persistent_counter.hpp) at common
// arg `valid`. Each receiver tells the leader how many of its slots it has taken, in its own word of the leader's
// `progress` words, one per NoC node of the set's rectangles (a count per receiver: one shared count would let a
// receiver that is ahead stand in for one behind, whose slot would then be overwritten). A slot is free until first
// taken, so a receiver writes only once it takes a slot a second time, after the leader's first block: by then the
// leader has cleared its words.
namespace kda_gate {

// The progress words in the gate cores' progress CB: x's from its start, W's from here.
constexpr uint32_t w_progress = 128;

FORCE_INLINE uint64_t noc_address(uint32_t xy, uint32_t address) {
    return get_noc_addr(xy & 0xFFFF, xy >> 16, address);
}

// One operand's K blocks of `block_bytes` through `cb` (`block_tiles` pages each, the CB a whole number of blocks).
// step() never blocks, so one RISC can move two operands; all() moves one.
class Share {
public:
    Share(
        uint32_t table,
        uint32_t group,
        uint32_t cb,
        uint32_t block_tiles,
        uint32_t block_bytes,
        uint32_t valid,
        uint32_t progress) :
        cb_(cb),
        block_tiles_(block_tiles),
        block_bytes_(block_bytes),
        slots_(get_local_cb_interface(cb).fifo_size / block_bytes),
        valid_(get_common_arg_val<uint32_t>(valid)),
        progress_(progress) {
        const uint32_t me = uint32_t{my_x[noc_index]} | uint32_t{my_y[noc_index]} << 16;
        const uint32_t record = table + get_common_arg_val<uint32_t>(table + group);
        uint32_t set = record + 1;
        for (uint32_t s = 0; s < get_common_arg_val<uint32_t>(record); ++s) {
            const uint32_t rectangles = get_common_arg_val<uint32_t>(set + 2);
            // A node's word: the nodes of the set's earlier rectangles, then its place in its own.
            uint32_t nodes = 0;
            for (uint32_t r = 0; r < rectangles; ++r) {
                const uint32_t low = get_common_arg_val<uint32_t>(set + 3 + 3 * r);
                const uint32_t high = get_common_arg_val<uint32_t>(set + 4 + 3 * r);
                const uint32_t width = (high & 0xFFFF) - (low & 0xFFFF) + 1;
                if ((me & 0xFFFF) >= (low & 0xFFFF) && (me & 0xFFFF) <= (high & 0xFFFF) && (me >> 16) >= (low >> 16) &&
                    (me >> 16) <= (high >> 16)) {
                    in_ = set;
                    index_ = nodes + (me & 0xFFFF) - (low & 0xFFFF) + ((me >> 16) - (low >> 16)) * width;
                }
                nodes += width * ((high >> 16) - (low >> 16) + 1);
            }
            if (get_common_arg_val<uint32_t>(set) == me) {
                out_ = set;
                nodes_ = nodes;
                for (uint32_t i = 0; i < nodes; ++i) {
                    reinterpret_cast<volatile tt_l1_ptr uint32_t*>(progress_)[i] = 0;
                }
            }
            set += 3 + 3 * rectangles;
        }
    }

    // Take every free slot (a leader reads its block, a receiver tells its leader), and push each block to compute: a
    // receiver's as it lands, a leader's once it has sent it, as its receivers free their slots. True once `blocks`
    // blocks have moved.
    template <typename Read>
    FORCE_INLINE bool step(uint32_t blocks, const Read& read) {
        if (in_) {
            take(blocks, read);
            while (pushed_ < reserved_ && counted(valid_, pushed_ + 1)) {
                cb_push_back(cb_, block_tiles_);
                ++pushed_;
            }
        } else {
            // Send what has landed, then read into the slots that frees: the sends never wait on later reads.
            land();
            // A block goes to compute once it has been sent: compute then never reuses a slot's storage (the CB's
            // aliases) while a block is still being sent from it.
            while (pushed_ < loaded_ && taken(pushed_ + 1)) {
                send(slot(pushed_));
                cb_push_back(cb_, block_tiles_);
                ++pushed_;
            }
            take(blocks, read);
            land();
            if (pushed_ == blocks) {
                noc_async_read_set_trid(0);
            }
        }
        return pushed_ == blocks;
    }

    // Take every free slot: a leader reads its block into it (under the slot's read transaction id), a receiver tells
    // its leader.
    template <typename Read>
    FORCE_INLINE void take(uint32_t blocks, const Read& read) {
        while (reserved_ < blocks && reserved_ - pushed_ < slots_ &&
               cb_pages_reservable_at_back(cb_, (reserved_ - pushed_ + 1) * block_tiles_)) {
            cb_reserve_back(cb_, (reserved_ - pushed_ + 1) * block_tiles_);
            if (in_) {
                if (reserved_ >= slots_) {
                    noc_inline_dw_write(
                        noc_address(get_common_arg_val<uint32_t>(in_), progress_ + index_ * sizeof(uint32_t)),
                        reserved_ + 1);
                }
            } else {
                noc_async_read_set_trid(1 + reserved_ % slots_);
                read(reserved_, slot(reserved_));
            }
            ++reserved_;
        }
    }

    // A leader's blocks whose reads have landed.
    FORCE_INLINE void land() {
        while (loaded_ < reserved_ && ncrisc_noc_read_with_transaction_id_flushed(noc_index, 1 + loaded_ % slots_)) {
            ++loaded_;
        }
    }

    // After the last of `blocks` blocks: retire the counts and take this call's off the counter.
    FORCE_INLINE void finish(uint32_t blocks) const {
        noc_async_atomic_barrier();
        noc_async_write_barrier();
        if (in_) {
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
    FORCE_INLINE uint32_t receivers() const { return get_common_arg_val<uint32_t>(out_ + 1); }

    // Block b's slot: the CB holds whole blocks from its base, a block per push.
    FORCE_INLINE uint32_t slot(uint32_t b) const {
        const auto& fifo = get_local_cb_interface(cb_);
        return fifo.fifo_limit - fifo.fifo_size + b % slots_ * block_bytes_;
    }

    // Whether every receiver has taken `slots` slots (each holds its first ones from the start): the words of the
    // nodes that are not receivers stay clear.
    FORCE_INLINE bool taken(uint32_t slots) const {
        if (slots <= slots_) {
            return true;
        }
        invalidate_l1_cache();
        const auto* progress = reinterpret_cast<volatile tt_l1_ptr uint32_t*>(progress_);
        uint32_t ready = 0;
        for (uint32_t i = 0; i < nodes_; ++i) {
            ready += progress[i] >= slots;
        }
        return ready == receivers();
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
        for (uint32_t r = 0; r < get_common_arg_val<uint32_t>(out_ + 2); ++r) {
            f(get_common_arg_val<uint32_t>(out_ + 3 + 3 * r),
              get_common_arg_val<uint32_t>(out_ + 4 + 3 * r),
              get_common_arg_val<uint32_t>(out_ + 5 + 3 * r));
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
    uint32_t valid_;
    uint32_t progress_;
    // The common arg indices of the set this core leads and of the one it receives in (0: none).
    uint32_t out_ = 0;
    uint32_t in_ = 0;
    // This receiver's word, and the leader's count of words.
    uint32_t index_ = 0;
    uint32_t nodes_ = 0;
    uint32_t reserved_ = 0;
    uint32_t loaded_ = 0;
    uint32_t pushed_ = 0;
};

// The pair's partial sums of each other's head: once this core's are formed (cb_pair, pushed by compute), tell the
// pair's core (at NoC x | y << 16 `pair_xy`) its partials may land in this core's cb_partial (the matmul is done
// with the x blocks that share its storage), send this core's into the pair's once it says the same, and hand the
// pair's to compute as they land. `ready` and `landed` are program semaphores.
FORCE_INLINE void swap_partials(
    uint32_t cb_pair, uint32_t cb_partial, uint32_t tiles, uint32_t pair_xy, uint32_t ready, uint32_t landed) {
    cb_wait_front(cb_pair, tiles);
    cb_reserve_back(cb_partial, tiles);
    noc_semaphore_inc(noc_address(pair_xy, ready), 1);
    noc_semaphore_wait(reinterpret_cast<volatile tt_l1_ptr uint32_t*>(ready), 1);
    noc_async_write(
        get_read_ptr(cb_pair), noc_address(pair_xy, get_write_ptr(cb_partial)), tiles * get_tile_size(cb_pair));
    noc_async_writes_flushed();
    noc_semaphore_inc(noc_address(pair_xy, landed), 1);
    noc_semaphore_wait(reinterpret_cast<volatile tt_l1_ptr uint32_t*>(landed), 1);
    cb_push_back(cb_partial, tiles);
    cb_pop_front(cb_pair, tiles);
    noc_async_atomic_barrier();
    noc_async_write_barrier();
}

// The output gate's operands as a kernel moves them. Where the buffers fit (`ways` 2), the heads of a batch pair up,
// h with h + heads / 2, and each core forms both heads' columns of its rows over its head's half of K (the first half
// for a pair's first head), then swaps partial sums with the pair's core (swap_partials); else (`ways` 1) each core
// forms its head's over all of K. Compile-time args from `ct`: x's, W's and the progress CBs,
// the rows and a head's columns (tiles) a core forms, the K block and K (tiles), the K tiles of the local input (0:
// none), W's tile columns, the heads per batch, the x groups (two per place: the pairs' first heads' cores, then their
// second's; W's follow, one per head), where the group table and the gate's common args start, the CBs of the partial
// sums of the pair's head and of the pair's of this core's head and their swap's semaphores, then the input's, the
// local input's and W's accessor args. Common args: the input, the local input, W, the TP rank, and x's and W's block
// counts. Last of the compile-time args before the accessors: `ways`.
template <uint32_t ct>
struct Operands {
    static constexpr uint32_t cb_x = get_compile_time_arg_val(ct);
    static constexpr uint32_t cb_w = get_compile_time_arg_val(ct + 1);
    static constexpr uint32_t cb_progress = get_compile_time_arg_val(ct + 2);
    static constexpr uint32_t rows = get_compile_time_arg_val(ct + 3);
    static constexpr uint32_t columns = get_compile_time_arg_val(ct + 4);
    static constexpr uint32_t kb = get_compile_time_arg_val(ct + 5);
    static constexpr uint32_t k = get_compile_time_arg_val(ct + 6);
    static constexpr uint32_t local_k = get_compile_time_arg_val(ct + 7);
    static constexpr uint32_t width = get_compile_time_arg_val(ct + 8);
    static constexpr uint32_t heads = get_compile_time_arg_val(ct + 9);
    static constexpr uint32_t x_groups = get_compile_time_arg_val(ct + 10);
    static constexpr uint32_t table = get_compile_time_arg_val(ct + 11);
    static constexpr uint32_t common = get_compile_time_arg_val(ct + 12);
    static constexpr uint32_t cb_pair = get_compile_time_arg_val(ct + 13);
    static constexpr uint32_t cb_partial = get_compile_time_arg_val(ct + 14);
    static constexpr uint32_t ready_id = get_compile_time_arg_val(ct + 15);
    static constexpr uint32_t partial_id = get_compile_time_arg_val(ct + 16);
    static constexpr uint32_t ways = get_compile_time_arg_val(ct + 17);
    static constexpr auto input_args = TensorAccessorArgs<ct + 18>();
    static constexpr auto local_args = TensorAccessorArgs<input_args.next_compile_time_args_offset()>();
    static constexpr auto weight_args = TensorAccessorArgs<local_args.next_compile_time_args_offset()>();
    // Each core's share of K.
    static constexpr uint32_t blocks = k / kb / ways;

    static uint32_t half(uint32_t head) { return ways == 2 && head % heads >= heads / 2; }
    // A pair's first head.
    static uint32_t first(uint32_t head) { return head - half(head) * (heads / 2); }

    // x of a correction core's chunks, or its head's W: whichever the other kernel leaves it.
    static void move_x(uint32_t group, uint32_t first_row, uint32_t head) {
        x_share(ways * group + half(head)).all(blocks, x_read(first_row, half(head)));
    }
    static void move_w(uint32_t head) { w_share(head).all(blocks, w_read(head)); }
    // The partial sums with the pair's core, at NoC x | y << 16 `pair_xy`.
    static void swap(uint32_t pair_xy) {
        swap_partials(cb_pair, cb_partial, rows * columns, pair_xy, get_semaphore(ready_id), get_semaphore(partial_id));
    }

    // x's tile rows [first_row, first_row + rows), in the x group `group`.
    static Share x_share(uint32_t group) {
        return Share(
            table, group, cb_x, rows * kb, rows * kb * get_tile_size(cb_x), common + 4, get_write_ptr(cb_progress));
    }
    static auto x_read(uint32_t first_row, uint32_t half) {
        const auto input = TensorAccessor(input_args, get_common_arg_val<uint32_t>(common));
        const auto local = TensorAccessor(local_args, get_common_arg_val<uint32_t>(common + 1));
        const uint32_t tile = get_tile_size(cb_x);
        // The K tiles of this rank's TP slice come from the local input.
        const uint32_t local_begin = get_common_arg_val<uint32_t>(common + 3) * local_k;
        return [=](uint32_t b, uint32_t slot) {
            for (uint32_t m = 0; m < rows; ++m) {
                for (uint32_t i = 0; i < kb; ++i) {
                    const uint32_t kk = (half * blocks + b) * kb + i;
                    const uint32_t destination = slot + (m * kb + i) * tile;
                    if (kk - local_begin < local_k) {
                        noc_async_read_page((first_row + m) * local_k + kk - local_begin, local, destination);
                    } else {
                        noc_async_read_page((first_row + m) * k + kk, input, destination);
                    }
                }
            }
        };
    }

    // The pair's W columns over the head's half of K, in the head's W group.
    static Share w_share(uint32_t head) {
        return Share(
            table,
            x_groups + head,
            cb_w,
            kb * ways * columns,
            kb * ways * columns * get_tile_size(cb_w),
            common + 5,
            get_write_ptr(cb_progress) + w_progress);
    }
    static auto w_read(uint32_t head) {
        const auto weight = TensorAccessor(weight_args, get_common_arg_val<uint32_t>(common + 2));
        const uint32_t tile = get_tile_size(cb_w);
        const uint32_t first_column = first(head) % heads * columns;
        const uint32_t k_begin = half(head) * blocks * kb;
        return [=](uint32_t b, uint32_t slot) {
            for (uint32_t i = 0; i < kb; ++i) {
                for (uint32_t pair = 0; pair < ways; ++pair) {
                    for (uint32_t j = 0; j < columns; ++j) {
                        noc_async_read_page(
                            (k_begin + b * kb + i) * width + first_column + pair * (heads / 2) * columns + j,
                            weight,
                            slot + ((ways * i + pair) * columns + j) * tile);
                    }
                }
            }
        };
    }
};

}  // namespace kda_gate
