// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

// router_output_writer: writes the router's outputs to DRAM and runs the core's routing planner instance
// Runs on: router cores, RISCV_0 (BRISC), NoC 1
// Responsibilities:
//  - builds tokens 0 .. 7 of the joint routing rows, waits for the other shares, writes the owner page to the routing
//    rows in DRAM, then +1 on every planner core's rows words; writes the compute's route words to DRAM
//  - padded chunk: reads actual_start / actual_end from DRAM, publishes the live row tiles, zero routes for padding
//  - router iterations: +1 on the ports' and router core 0's iteration words; router core 0 tells the matmul cores go
//  - under ROUTING_PLANNER: once every router core's rows are in, routing_planner::plan (reads the exchange's DRAM
//    slots; writes counts / regions / token index / table / plan to DRAM)
// Note: fabric-written DRAM is read on NoC 1, the fabric routers' NoC: the other chips' routing packets land in the
//  DRAM slots through its endpoint (a read through the other NoC's DRAM endpoint can return stale data)
#include "ttnn/cpp/ttnn/operations/experimental/moe_block/moe_head/device/kernels/row_block.hpp"
#include "ttnn/cpp/ttnn/operations/experimental/moe_block/common/kernels/tile_bytes.hpp"
#include "ttnn/cpp/ttnn/operations/experimental/moe_block/moe_head/device/kernels/router_common.hpp"
#include "ttnn/cpp/ttnn/operations/experimental/moe_block/moe_head/device/kernels/routing_planner.hpp"

#include <cstdint>
#include "api/dataflow/dataflow_api.h"
#include "api/debug/kernel_profiler.hpp"

// Compile-time args (head_router_program.cpp, the router's dataflow kernels)
constexpr uint32_t expert_tiles = get_compile_time_arg_val(1);  // (score tiles a token column)
constexpr uint32_t top_k = get_compile_time_arg_val(3);
constexpr auto rows_accessor_args = TensorAccessorArgs<4>();  // (the routing rows in DRAM)
#ifdef ROUTER_ROUTE_WORDS
constexpr auto route_words_accessor_args = TensorAccessorArgs<ROUTER_ROUTE_ACCESSOR_CT_OFFSET>();
#endif
constexpr uint32_t cb_bias = 5;  // (the score bias^T: the input feeder pushes it once the done words are zero)

// This kernel's args (head_router_program.cpp output_writer_rt), read once by kernel_main
struct OutputWriterArgs {
    uint32_t rows_address;         // [0] the routing rows (DRAM)
    uint32_t owner_page;           // [1] my owner page of them (the first row block's)
    uint32_t scale_bits;           // [2] the route scale's fp32 bits
    uint32_t rows_l1;              // [3] the joint routing rows (L1)
    uint32_t done_words;           // [4] the top-K's done words (16 B apart)
    uint32_t best_page_l1;         // [5] the cb_best page: the top K of every token column, sorted descending
    uint32_t route_words_l1;       // [6] the compute's route words (L1; also the padded metadata's scratch)
    uint32_t route_words_address;  // [7] the route words (DRAM, the first row block's page)
    uint32_t first_token;          // [8] my row tile's first token
    uint32_t scores_l1;            // [9] my block of the logits: the scores
#ifdef ROUTER_ITERATIONS_ON_ROUTER
    // router iterations (ROUTER_ITERATION_ARGS args from [10])
    bool router_core_zero;             // am I router core 0
    uint32_t iteration_word;           // my chip's iteration word ("rit")
    uint32_t router_core_zero_noc_xy;  // router core 0's NoC x | y << 16
    uint32_t active_logits_ports;      // the ports that relay logits
    uint32_t ports;                    // every port core of the chip, then their NoC x | y << 16
    const volatile tt_l1_ptr uint32_t* port_noc_xy;
    uint32_t engine_phase_words;  // the matmul cores' phase words address
    uint32_t engine_columns;      // the matmul cores' columns, then their NoC x's
    const volatile tt_l1_ptr uint32_t* engine_column_noc_x;
    uint32_t engine_rows;  // the matmul cores' rows, then their NoC y's
    const volatile tt_l1_ptr uint32_t* engine_row_noc_y;
#endif
    // the routing exchange (after the iteration args): every planner core's rows words, my planner instance's region
    // (0: none here), the planner cores' NoC (x, y) [ROUTING_EXCHANGE_ROUTER_CORES], then routing_planner's args
    uint32_t planner_rows_word;
    uint32_t planner_second_rows_word;
    uint32_t planner_region_base;
    const volatile tt_l1_ptr uint32_t* planner_core_noc;
    routing_planner::PlannerArgs planner;
#ifdef PADDED_CHUNK
    RouterPaddingArgs padding;  // (common [2] ..)
#endif
    uint32_t planner_core_noc_x(uint32_t core) const { return planner_core_noc[2 * core]; }
    uint32_t planner_core_noc_y(uint32_t core) const { return planner_core_noc[2 * core + 1]; }
};
OutputWriterArgs read_output_writer_args(const volatile tt_l1_ptr uint32_t* args) {
    OutputWriterArgs result{};
    result.rows_address = args[0];
    result.owner_page = args[1];
    result.scale_bits = args[2];
    result.rows_l1 = args[3];
    result.done_words = args[4];
    result.best_page_l1 = args[5];
    result.route_words_l1 = args[6];
    result.route_words_address = args[7];
    result.first_token = args[8];
    result.scores_l1 = args[9];
#ifdef ROUTER_ITERATIONS_ON_ROUTER
    {
        uint32_t index = 10;
        result.router_core_zero = args[index++] != 0;
        result.iteration_word = args[index++];
        result.router_core_zero_noc_xy = args[index++];
        result.active_logits_ports = args[index++];
        result.ports = args[index++];
        result.port_noc_xy = args + index;
        index += result.ports;
        result.engine_phase_words = args[index++];
        result.engine_columns = args[index++];
        result.engine_column_noc_x = args + index;
        index += result.engine_columns;
        result.engine_rows = args[index++];
        result.engine_row_noc_y = args + index;
    }
#endif
    constexpr uint32_t routing_at = 10 + ROUTER_ITERATION_ARGS;
    result.planner_rows_word = args[routing_at];
    result.planner_second_rows_word = args[routing_at + 1];
    result.planner_region_base = args[routing_at + 2];
    result.planner_core_noc = args + routing_at + 3;
    if (result.planner_region_base) {  // (no instance here: no planner args follow)
        result.planner = routing_planner::read_planner_args(
            args + routing_at + 3 + 2 * ROUTING_EXCHANGE_ROUTER_CORES, moe_block_args::common_pointer(2));
    }
#ifdef PADDED_CHUNK
    result.padding = read_router_padding_args(moe_block_args::common_pointer(2));
#endif
    return result;
}

// One row block of this kernel's work: kernel_main below runs it once per block (row_block.hpp).
void process_row_block(const OutputWriterArgs& args, uint32_t block) {
#if defined(ROUTER_ROUTE_WORDS) && defined(PADDED_CHUNK)
    // (a padded chunk's actual_start / actual_end, for the route words: into the route words' L1 (256 B) - the compute
    // writes them only after word 80, by when they are read - early, read once the rows are written, long after)
    const uint32_t metadata_l1 = padded_metadata_slot(args.route_words_l1);
    padded_metadata_read(args.padding.forward, metadata_l1);
    PaddedRows my_padded_rows{};  // (my SP row's real rows, read below)
#endif
    // (row blocks: block b's owner rows ROW_BLOCK_ROUTER_PAGE_STRIDE pages on, its route words
    // ROW_BLOCK_ROUTE_WORDS_STRIDE bytes on - row_block.hpp)
    const uint32_t owner_page = args.owner_page + block * ROW_BLOCK_ROUTER_PAGE_STRIDE;
    // the top K of every token column, sorted descending in rows 0 .. top_k - 1 of the best page;
    // tokens 0 .. 7 here, 8 .. 15 the NCRISC, 16 .. 23 the compute's MATH thread, 24 .. 31 its PACK (done words 16 /
    // 32 / 48)
    const auto rows_accessor =
        TensorAccessor(rows_accessor_args, args.rows_address, tt::constants::TILE_HEIGHT * sizeof(uint32_t) * top_k);
#ifdef ROUTER_ITERATIONS_ON_ROUTER
    // router iterations: am I router core 0, my chip's iteration word, router core 0's NoC, the logits ports' count,
    // the ports, the matmul cores' phase words, the matmul cores' NoC x's and y's (OutputWriterArgs)
    constexpr uint32_t router_iterations = ROUTER_ITERATIONS_ON_ROUTER;
#else
    constexpr uint32_t router_iterations = 1;
#endif
#ifdef PADDED_CHUNK
    // a padded forward: my SP row's real rows now, and which of my iterations' row tiles hold a real token (bit
    // iteration, bit 31 valid) - an all-padding tile's top-K and joint rows are not computed (its route words are
    // zero). Published at done words + 144 once the NCRISC zeroed the words (the arena is call scratch: its cb_bias
    // push comes after), for it and, through each iteration's ready page, the compute's three threads alike
    noc_async_read_barrier();
    invalidate_l1_cache();
    my_padded_rows = padded_rows_of(metadata_l1, args.padding.forward);
    uint32_t live_tiles = 0x80000000u;
    for (uint32_t iteration = 0; iteration < router_iterations; ++iteration) {
#ifdef ROUTER_ITERATIONS_ON_ROUTER
        const uint32_t first_token =
            args.first_token + iteration * tt::constants::TILE_HEIGHT * ROUTER_ROW_TILES_PER_ITERATION;
#else
        const uint32_t first_token = args.first_token;
#endif
        if (my_padded_rows.count(
                args.padding.first_row + block * ROW_BLOCK_PADDED_ROW_STRIDE + first_token,
                tt::constants::TILE_HEIGHT)) {
            live_tiles |= 1u << iteration;
        }
    }
    auto live_iteration = [&](uint32_t iteration) -> bool { return (live_tiles >> iteration) & 1; };
#else
    auto live_iteration = [](uint32_t) -> bool { return true; };
#endif
    auto write_route_words = [&]([[maybe_unused]] uint32_t iteration) {  // dispatch prep: my owned tokens' route
                                                                         // words (the compute's, done word 96)
#ifdef ROUTER_ROUTE_WORDS
        volatile tt_l1_ptr uint32_t* route_words_ready =
            reinterpret_cast<volatile tt_l1_ptr uint32_t*>(args.done_words + 96);
        while (*route_words_ready == 0) {
            invalidate_l1_cache();
        }
        *route_words_ready = 0;  // (re-arm)
#ifdef ROUTER_ITERATIONS_ON_ROUTER
        const uint32_t first_token =  // (my row tile of iteration `iteration`)
            args.first_token + iteration * tt::constants::TILE_HEIGHT * ROUTER_ROW_TILES_PER_ITERATION;
#else
        const uint32_t first_token = args.first_token;
#endif
#ifdef PADDED_CHUNK
        {  // (padding tokens - their rows not real - get no route: not dispatched)
            invalidate_l1_cache();
            volatile tt_l1_ptr uint32_t* token_words =
                reinterpret_cast<volatile tt_l1_ptr uint32_t*>(args.route_words_l1);
            // (token column's row in my SP row: first_row + column)
            const uint32_t first_row = args.padding.first_row + block * ROW_BLOCK_PADDED_ROW_STRIDE + first_token;
            for (uint32_t column = 0; column < 32; ++column) {
                if (!my_padded_rows.real(first_row + column)) {
                    for (uint32_t i = 0; i < ROUTER_ROUTE_WORDS_PER_TOKEN; ++i) {
                        token_words[ROUTER_ROUTE_WORDS_PER_TOKEN * column + i] = 0;
                    }
                }
            }
        }
#endif
        const auto route_words_accessor = TensorAccessor(
            route_words_accessor_args,
            args.route_words_address + block * ROW_BLOCK_ROUTE_WORDS_STRIDE,
            ROUTER_ROUTE_PAGE_BYTES);
        noc_async_write(
            args.route_words_l1,
            route_words_accessor.get_noc_addr(0) + first_token * sizeof(uint32_t) * ROUTER_ROUTE_WORDS_PER_TOKEN,
            tt::constants::TILE_HEIGHT * sizeof(uint32_t) * ROUTER_ROUTE_WORDS_PER_TOKEN);
        noc_async_write_barrier();
#endif
    };
    cb_wait_front(cb_bias, expert_tiles);
#ifdef PADDED_CHUNK
    *reinterpret_cast<volatile tt_l1_ptr uint32_t*>(args.done_words + 144) = live_tiles;
#endif
    for (uint32_t iteration = 0; iteration < router_iterations; ++iteration) {
        volatile tt_l1_ptr uint32_t* top_k_done = reinterpret_cast<volatile tt_l1_ptr uint32_t*>(args.done_words);
        while (*top_k_done == 0) {
            invalidate_l1_cache();
        }
        *top_k_done = 0;  // (re-arm)
        {
            {
                DeviceZoneScopedN("SE-RROWS");
                if (live_iteration(iteration)) {  // (an all-padding tile: no rows)
                    set_l1_data_cache<true>();    // (write-through; the packed tiles read after an invalidate)
                    invalidate_l1_cache();
                    routing_rows::build<top_k, expert_tiles>(
                        args.best_page_l1, args.scores_l1, args.rows_l1, args.scale_bits, 0, 8);
                    set_l1_data_cache<false>();
                }
            }
            for (uint32_t share = 1; share < 4; ++share) {  // (the other shares' done words: 16, 32, 48)
                volatile tt_l1_ptr uint32_t* share_done =
                    reinterpret_cast<volatile tt_l1_ptr uint32_t*>(args.done_words + 16 * share);
                while (*share_done == 0) {
                    invalidate_l1_cache();
                }
                *share_done = 0;  // (re-arm)
            }
#ifdef ROUTER_ROUTE_WORDS
            // (every share of the rows written)
            *reinterpret_cast<volatile tt_l1_ptr uint32_t*>(args.done_words + 80) = 1;
#endif
        }
#ifdef ROUTER_ITERATIONS_ON_ROUTER
        noc_async_write(
            args.rows_l1,
            rows_accessor.get_noc_addr(owner_page + iteration * ROUTER_ROW_TILES_PER_ITERATION),
            tt::constants::TILE_HEIGHT * sizeof(uint32_t) * top_k);  // (my row tile of iteration `iteration`)
#else
        noc_async_write(
            args.rows_l1,
            rows_accessor.get_noc_addr(owner_page),
            tt::constants::TILE_HEIGHT * sizeof(uint32_t) * top_k);
#endif
        noc_async_write_barrier();
        {  // (my rows of this iteration are written: every router core's two planner RISCs told - ROUTING_PLANNER:
           // always here)
            for (uint32_t core = 0; core < ROUTING_EXCHANGE_ROUTER_CORES; ++core) {
                const uint32_t noc_x = args.planner_core_noc_x(core), noc_y = args.planner_core_noc_y(core);
                noc_semaphore_inc(get_noc_addr(noc_x, noc_y, args.planner_rows_word), 1);
                noc_semaphore_inc(get_noc_addr(noc_x, noc_y, args.planner_second_rows_word), 1);
            }
        }
#ifdef ROUTER_ITERATIONS_ON_ROUTER
        if (iteration + 1 < router_iterations) {  // (not the last iteration: its route words now - the L1 is reused -
                                                  // then this core is free)
            write_route_words(iteration);
            noc_async_atomic_barrier();
            // (every port: the chips sending me logits may send iteration + 1)
            for (uint32_t port = 0; port < args.ports; ++port) {
                const uint32_t port_noc_xy = args.port_noc_xy[port];
                noc_semaphore_inc(get_noc_addr(port_noc_xy & 0xFFFF, port_noc_xy >> 16, args.iteration_word + 16), 1);
            }
            noc_semaphore_inc(
                get_noc_addr(
                    args.router_core_zero_noc_xy & 0xFFFF, args.router_core_zero_noc_xy >> 16, args.iteration_word),
                1);
            if (args.router_core_zero) {  // (router core 0: every router core and logits port done with it -> every
                                          // matmul core's go)
                volatile tt_l1_ptr uint32_t* iteration_count =
                    reinterpret_cast<volatile tt_l1_ptr uint32_t*>(args.iteration_word);
                while (*iteration_count <
                       (ROUTER_ROW_TILES_PER_ITERATION + args.active_logits_ports) * (iteration + 1)) {
                    invalidate_l1_cache();
                }
                for (uint32_t column = 0; column < args.engine_columns; ++column) {
                    for (uint32_t row = 0; row < args.engine_rows; ++row) {
                        noc_semaphore_inc(
                            get_noc_addr(
                                args.engine_column_noc_x[column],
                                args.engine_row_noc_y[row],
                                args.engine_phase_words + ROUTER_GO_WORD),
                            1);
                    }
                }
                if (iteration + 2 == router_iterations) {  // (re-arm: the last count came)
                    noc_semaphore_inc(
                        get_noc_addr(args.iteration_word),
                        0u - (ROUTER_ROW_TILES_PER_ITERATION + args.active_logits_ports) * (router_iterations - 1));
                }
            }
            noc_async_atomic_barrier();
        }
#endif
    }
    {  // the routing exchange + planner: every router core's rows written, then the routing planner (ROUTING_PLANNER:
       // always here)
        DeviceZoneScopedN("SE-RPLAN");
        noc_semaphore_wait(
            reinterpret_cast<volatile tt_l1_ptr uint32_t*>(args.planner_rows_word), ROUTING_EXCHANGE_ROUTER_ROW_TILES);
        *reinterpret_cast<volatile tt_l1_ptr uint32_t*>(args.planner_rows_word) = 0;  // (re-arm: all increments landed)
        if (args.planner_region_base) {  // (no instance here: fewer planner instances than router cores)
            routing_planner::plan(args.planner, args.planner_region_base);
        }
    }
    write_route_words(router_iterations - 1);  // (the last iteration's after the planner: off the routing chain)
}

void kernel_main() {
    moe_block_args::load();  // (this core's args from the DRAM table; a compute kernel waits for its page)
    const OutputWriterArgs args = read_output_writer_args(moe_block_args::arg_pointer(0));
    for (uint32_t block = 0; block < row_block::BLOCKS; ++block) {
        row_block::begin(block);
        process_row_block(args, block);
        row_block::end(block);
    }
}
