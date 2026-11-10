// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

// routing_planner_core: the routing planner on a planner-only core (row blocks)
// Runs on: planner-only cores, RISCV_0 (BRISC), NoC 1; or RISCV_1 (NCRISC), NoC 0 (the core's second planner instance)
// Responsibilities:
//  - waits for every router core's rows (ROUTING_EXCHANGE_ROUTER_ROW_TILES increments of its rows word by
//    router_output_writer), then re-arms the word
//  - with an instance here (its region arg non-zero): routing_planner::plan - reads the routing rows and the
//    exchange's DRAM slots, writes counts / regions / token index / table / plan to DRAM
// Note: fabric-written DRAM is read on NoC 1, the fabric routers' NoC (a read through the other NoC's DRAM endpoint
//  can return stale data); planner_slot currently gives the NCRISC (NoC 0) instance no region: it only waits

// A planner-only core's kernel (row blocks: the routing planner's instances past the router cores, BRISC and NCRISC):
// once every router core's rows are written (ROUTING_EXCHANGE_ROUTER_ROW_TILES increments of my rows word),
// routing_planner::plan.
#include "ttnn/cpp/ttnn/operations/experimental/moe_block/moe_head/device/kernels/row_block.hpp"
#include <cstdint>
#include "api/dataflow/dataflow_api.h"
#include "api/debug/kernel_profiler.hpp"
#include "ttnn/cpp/ttnn/operations/experimental/moe_block/moe_head/device/kernels/routing_planner.hpp"

// This core's args (head_router_program.cpp), read once in kernel_main
struct PlannerCoreArgs {
    uint32_t rows_word_address;            // [0] my rows word (the router cores count their rows on it)
    uint32_t region_base;                  // [1] my instance's region (0: no instance here)
    routing_planner::PlannerArgs planner;  // [2] on: routing_planner::plan's (only with an instance here)
};

// One row block of this kernel's work: kernel_main below runs it once per block (row_block.hpp).
void process_row_block(const PlannerCoreArgs& args) {
    noc_semaphore_wait(
        reinterpret_cast<volatile tt_l1_ptr uint32_t*>(args.rows_word_address), ROUTING_EXCHANGE_ROUTER_ROW_TILES);
    *reinterpret_cast<volatile tt_l1_ptr uint32_t*>(args.rows_word_address) = 0;  // (re-arm: all increments landed)
    if (args.region_base) {
        DeviceZoneScopedN("SE-RPLAN");
        routing_planner::plan(args.planner, args.region_base);
    }
}

void kernel_main() {
    moe_block_args::load();  // (this core's args from the DRAM table; a compute kernel waits for its page)
    PlannerCoreArgs args{moe_block_args::arg(0), moe_block_args::arg(1), routing_planner::PlannerArgs{}};
    if (args.region_base) {  // (no instance here: no planner args follow)
        args.planner =
            routing_planner::read_planner_args(moe_block_args::arg_pointer(2), moe_block_args::common_pointer(2));
    }
    for (uint32_t block = 0; block < row_block::BLOCKS; ++block) {
        row_block::begin(block);
        process_row_block(args);
        row_block::end(block);
    }
}
