// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

// router_input_feeder: feeds the router compute its inputs and builds a share of the joint routing rows
// Runs on: router cores, RISCV_1 (NCRISC), NoC 0
// Responsibilities:
//  - reads the router bias and expert index tiles from DRAM into CBs 5 / 11 (once a call)
//  - waits for the matmul cores' partial logits in cb_logits (engine-done semaphore), then the cb_ready page (live
//    row tiles, chip index, TP position) to the compute
//  - hands the other chips' logits partials (landed over the fabric) to the compute (cb_arrival_order) in a fixed
//    order
//  - builds tokens 8 .. 15 of the joint routing rows in L1 once the top-K is done
//  - under ROUTING_PLANNER: waits for its rows word, then routing_planner::plan if the host gave it an instance
//    (planner_slot currently places every instance on a BRISC)
#include "ttnn/cpp/ttnn/operations/experimental/moe_block/moe_head/device/kernels/row_block.hpp"
#include "ttnn/cpp/ttnn/operations/experimental/moe_block/common/kernels/tile_bytes.hpp"
#include "ttnn/cpp/ttnn/operations/experimental/moe_block/moe_head/device/kernels/router_common.hpp"
#include "ttnn/cpp/ttnn/operations/experimental/moe_block/moe_head/device/kernels/routing_planner.hpp"

#include <cstdint>
#include "api/dataflow/dataflow_api.h"
#include "api/debug/kernel_profiler.hpp"

// Compile-time args (head_router_program.cpp, the router's dataflow kernels)
constexpr uint32_t expert_tiles = get_compile_time_arg_val(1);  // (score tiles a token column)
constexpr uint32_t tp_size = get_compile_time_arg_val(2);       // (the chips summing a logits partial each)
constexpr uint32_t top_k = get_compile_time_arg_val(3);
constexpr auto bias_accessor_args = TensorAccessorArgs<4>();
constexpr auto expert_index_accessor_args = TensorAccessorArgs<bias_accessor_args.next_compile_time_args_offset()>();

// Circular buffers (head_router_program.cpp; router_compute.cpp's names)
constexpr uint32_t cb_arrival_order = 4;  // the arrivals' slots, in arrival order (a uint32 page each)
constexpr uint32_t cb_bias = 5;           // the score bias^T
constexpr uint32_t cb_expert_index = 11;  // the expert ids (uint16), a tile per score tile
constexpr uint32_t cb_ready = 12;         // the compute's page: the partials are in; this chip's index, TP position

// This kernel's args (head_router_program.cpp input_feeder_rt), read once by kernel_main
struct InputFeederArgs {
    uint32_t bias_address;          // [0] bias^T (DRAM)
    uint32_t expert_index_address;  // [1] the expert index tiles (DRAM)
    // ([2] .. [1 + tp_size]: my rows a TP position - not read here)
    uint32_t arrivals;               // [2 + tp_size] every other chip's partial of my owner rows (slot = its source)
    uint32_t arrival_words_address;  // [3 + tp_size] the arrivals' words (a word a slot, 16 B apart)
    // [4 + tp_size] ..: the slots in their fixed sum order [tp_size - 1], read in place
    const volatile tt_l1_ptr uint32_t* arrival_order;
    // the tail, after the order: the top-K's done words, the rows' L1
    uint32_t done_words;    // the done words (16 B apart; 64: the top-K, 112: the PACK's go, 144: the live row tiles)
    uint32_t best_page_l1;  // the cb_best page
    uint32_t rows_l1;       // the joint routing rows
    uint32_t scale_bits;    // the route scale's fp32 bits
    uint32_t scores_l1;     // my block of the logits: the scores
    // (the routing exchange, after the tail: my planner rows word, my planner instance's region (0: none here), then
    // routing_planner's args)
    uint32_t planner_rows_word;
    uint32_t planner_region_base;
    routing_planner::PlannerArgs planner;
    uint32_t engine_done_semaphore_address;  // (common [1])
    uint32_t arrival_slot(uint32_t position) const { return arrival_order[position]; }
};
InputFeederArgs read_input_feeder_args(const volatile tt_l1_ptr uint32_t* args, const RouterCommonArgs& common_args) {
    constexpr uint32_t arrivals_at = 2 + tp_size, tail_at = arrivals_at + 2 + (tp_size - 1);
    constexpr uint32_t planner_at = tail_at + 5;
    const uint32_t planner_region_base = args[planner_at + 1];
    return InputFeederArgs{
        args[0],
        args[1],
        args[arrivals_at],
        args[arrivals_at + 1],
        args + arrivals_at + 2,
        args[tail_at],
        args[tail_at + 1],
        args[tail_at + 2],
        args[tail_at + 3],
        args[tail_at + 4],
        args[planner_at],
        planner_region_base,
        // (no instance here: no planner args follow)
        planner_region_base
            ? routing_planner::read_planner_args(args + planner_at + 2, moe_block_args::common_pointer(2))
            : routing_planner::PlannerArgs{},
        common_args.engine_done_semaphore_address};
}

// One row block of this kernel's work: kernel_main below runs it once per block (row_block.hpp).
void process_row_block(const InputFeederArgs& args, uint32_t block) {
    {  // (the top-K's done words: zero before anyone can write them - they are written after the scores; with the
       // split word 128 too, the partner's best K landed - the arena is call scratch: stale, it let the merge take a
       // stale page, the picks a call behind)
        // (and 144: the BRISC's live row tiles, written after my cb_bias push - in block 0 only: in a later row block
        // the bias is already there, so the BRISC may write 144 before I get here; I clear it after its last read
        // instead)
        const uint32_t words_to_clear = block == 0 ? 10 : 9;
        for (uint32_t i = 0; i < words_to_clear; ++i) {
            *reinterpret_cast<volatile tt_l1_ptr uint32_t*>(args.done_words + 16 * i) = 0;
        }
    }
    if (block == 0) {  // (bias and expert indices: the call's, once - the compute reads them every block, never pops)
        const auto bias_accessor =
            TensorAccessor(bias_accessor_args, args.bias_address, moe_block_tiles::kFp32TileBytes);
        const auto expert_index_accessor =
            TensorAccessor(expert_index_accessor_args, args.expert_index_address, moe_block_tiles::kBf16TileBytes);
        cb_reserve_back(cb_bias, expert_tiles);
        cb_reserve_back(cb_expert_index, expert_tiles);
        for (uint32_t tile = 0; tile < expert_tiles; ++tile) {
            noc_async_read(
                bias_accessor.get_noc_addr(tile),
                get_write_ptr(cb_bias) + tile * moe_block_tiles::kFp32TileBytes,
                moe_block_tiles::kFp32TileBytes);
        }
        for (uint32_t tile = 0; tile < expert_tiles; ++tile) {
            noc_async_read(
                expert_index_accessor.get_noc_addr(tile),
                get_write_ptr(cb_expert_index) + tile * moe_block_tiles::kBf16TileBytes,
                moe_block_tiles::kBf16TileBytes);
        }
        noc_async_read_barrier();
        cb_push_back(cb_bias, expert_tiles);
        cb_push_back(cb_expert_index, expert_tiles);
    }
#ifdef ROUTER_ITERATIONS_ON_ROUTER
    constexpr uint32_t router_iterations = ROUTER_ITERATIONS_ON_ROUTER;  // (router iterations: my row tile of each)
#else
    constexpr uint32_t router_iterations = 1;
#endif
    volatile tt_l1_ptr uint32_t* arrival_words =
        reinterpret_cast<volatile tt_l1_ptr uint32_t*>(args.arrival_words_address);
    for (uint32_t iteration = 0; iteration < router_iterations; ++iteration) {
        {  // the engine computed my partial logits (every engine core its expert tiles x its rows, written into my
           // cb_logits with an increment of this counter a tile): all tp_size x expert_tiles in -> the compute
           // (cb_ready)
            DeviceZoneScopedN("SE-RENGWAIT");
            volatile tt_l1_ptr uint32_t* engine_done =
                reinterpret_cast<volatile tt_l1_ptr uint32_t*>(args.engine_done_semaphore_address);
            noc_semaphore_wait_min(engine_done, expert_tiles);  // (my own block's tiles: the others' go to the ports)
            noc_semaphore_inc(get_noc_addr(args.engine_done_semaphore_address), 0u - expert_tiles);  // (re-arm)
            noc_async_atomic_barrier();
            cb_reserve_back(cb_ready, 1);
#ifdef PADDED_CHUNK
            {  // (the BRISC's live row tiles into the ready page: the compute's three threads read the same word)
                volatile tt_l1_ptr uint32_t* live_word =
                    reinterpret_cast<volatile tt_l1_ptr uint32_t*>(args.done_words + 144);
                uint32_t live_tiles;
                while (!((live_tiles = *live_word) >> 31)) {
                    invalidate_l1_cache();
                }
                *reinterpret_cast<volatile tt_l1_ptr uint32_t*>(get_write_ptr(cb_ready)) = live_tiles;
            }
#endif
#ifdef ROUTER_MY_CHIP
            {  // (this chip's index and TP position for the compute: one program for the mesh)
                volatile tt_l1_ptr uint32_t* ready_page =
                    reinterpret_cast<volatile tt_l1_ptr uint32_t*>(get_write_ptr(cb_ready));
                ready_page[1] = ROUTER_MY_CHIP;
                ready_page[2] = ROUTER_TP_POSITION;
            }
#endif
            cb_push_back(cb_ready, 1);
            // (and word 112 for the compute's PACK thread: the ready wait holds only UNPACK, and PACK's hand-off
            // pushes (REDUCE_SCATTER_FINAL_COUNTER_CB) would hand cb_logits to the ports before the engine's partials
            // landed - stale logits: the last call's planner overlay in it, K3's picks 99.10%)
            *reinterpret_cast<volatile tt_l1_ptr uint32_t*>(args.done_words + 112) = 1;
        }
        if (args.arrivals) {  // the sources' partials as they land (a fused increment of the source slot's own word,
                              // 16 B apart): each slot to the compute (cb_arrival_order) in a fixed order - the order
                              // args, the predicted landing order - so the sum is the same every run at ~no wait (the
                              // head is always deterministic)
            DeviceZoneScopedN("SE-RLWAIT");
            for (uint32_t arrival = 0, seen_slots = 0; arrival < args.arrivals;) {
                invalidate_l1_cache();
                for (uint32_t slot = 0; slot < args.arrivals; ++slot) {
                    if (slot != args.arrival_slot(arrival)) {  // (the fixed order)
                        continue;
                    }
                    if ((seen_slots >> slot & 1) || arrival_words[4 * slot] == 0) {
                        continue;
                    }
                    // (re-arm: one increment a call - the source's next one comes after this chip's exchange)
                    arrival_words[4 * slot] = 0;
                    seen_slots |= 1u << slot;
                    ++arrival;
                    cb_reserve_back(cb_arrival_order, 1);
                    *reinterpret_cast<volatile tt_l1_ptr uint32_t*>(get_write_ptr(cb_arrival_order)) = slot;
                    cb_push_back(cb_arrival_order, 1);
                }
            }
        }
        {  // (my share of the joint rows - tokens 8 .. 15 - once the top-K is done)
            volatile tt_l1_ptr uint32_t* top_k_done =
                reinterpret_cast<volatile tt_l1_ptr uint32_t*>(args.done_words + 64);
            while (*top_k_done == 0) {
                invalidate_l1_cache();
            }
            *top_k_done = 0;
            DeviceZoneScopedN("SE-RROWS");
#ifdef PADDED_CHUNK
            // (my iterations' live row tiles: the BRISC's, done words + 144 - an all-padding tile has no rows)
            if ((*reinterpret_cast<volatile tt_l1_ptr uint32_t*>(args.done_words + 144) >> iteration) & 1)
#endif
            {
                set_l1_data_cache<true>();
                invalidate_l1_cache();
                routing_rows::build<top_k, expert_tiles>(
                    args.best_page_l1, args.scores_l1, args.rows_l1, args.scale_bits, 8, 16);
                set_l1_data_cache<false>();
            }
            *reinterpret_cast<volatile tt_l1_ptr uint32_t*>(args.done_words + 16) = 1;
        }
    }  // (router iterations)
#ifdef PADDED_CHUNK
    if constexpr (row_block::BLOCKS > 1) {
        // (the live row tiles read: zero for the next row block's BRISC write - see the top)
        *reinterpret_cast<volatile tt_l1_ptr uint32_t*>(args.done_words + 144) = 0;
    }
#endif
    {  // the second planner instance on this core (the routing exchange + planner; ROUTING_PLANNER: always here)
        noc_semaphore_wait(
            reinterpret_cast<volatile tt_l1_ptr uint32_t*>(args.planner_rows_word), ROUTING_EXCHANGE_ROUTER_ROW_TILES);
        *reinterpret_cast<volatile tt_l1_ptr uint32_t*>(args.planner_rows_word) = 0;
        if (args.planner_region_base) {  // (no instance here when the chip has fewer experts than planner RISCs)
            routing_planner::plan(args.planner, args.planner_region_base);
        }
    }
}

void kernel_main() {
    moe_block_args::load();  // (this core's args from the DRAM table; a compute kernel waits for its page)
    const InputFeederArgs args = read_input_feeder_args(
        moe_block_args::arg_pointer(0), read_router_common_args(moe_block_args::common_pointer(0)));
    for (uint32_t block = 0; block < row_block::BLOCKS; ++block) {
        row_block::begin(block);
        process_row_block(args, block);
        row_block::end(block);
    }
}
