// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

// x_gather_watcher: tells the matmul cores when each gathered x shard has fully landed on this chip
// Runs on: the watcher core (one a chip), RISCV_1 (NCRISC), NoC 1 (RISCV_1_default)
// Responsibilities:
//  - polls its L1 counters, which the upstream fabric_port_sender cores increment over the fabric (a shard a stream)
//  - once every stream carrying shard k of chunk m counted it: +1 on XREADY on every matmul core
//    (shared_expert_row_dataflow waits on it); re-arms the counters
//  - with a second Y direction (WATCHER_ROUTING_ARGS_OFFSET): that direction's routing exchange send, once the planner
//    instances are ready, over the fabric into the destination chips' DRAM slots (router_fabric_sends.hpp)
#include <cstdint>
#include "api/dataflow/dataflow_api.h"
#include "ttnn/cpp/ttnn/operations/experimental/moe_block/moe_head/device/kernels/row_block.hpp"
#ifdef SKIP_PADDING_CHUNKS
#include "ttnn/cpp/ttnn/operations/experimental/moe_block/moe_head/device/kernels/padded_rows.hpp"
#endif
#ifdef WATCHER_ROUTING_ARGS_OFFSET
#include "api/debug/kernel_profiler.hpp"
// (no public device-side fabric API: the packet headers, the fabric connection, the header pool, the routes)
#include "hostdev/fabric_edm_packet_header.hpp"
#include "tt_metal/fabric/hw/inc/edm_fabric/edm_fabric_worker_adapters.hpp"
#include "tt_metal/fabric/hw/inc/packet_header_pool.h"
#include "tt_metal/fabric/hw/inc/tt_fabric_api.h"
using namespace tt::tt_fabric;
#include "ttnn/cpp/ttnn/operations/experimental/moe_block/moe_head/device/kernels/router_fabric_sends.hpp"
#endif

// This kernel's runtime args (head_gather_program.cpp ChipProgram::gather, `watcher_args`), read once in kernel_main.
// Chunk m's shard of rank k (k = 1 .. tp_size - 1 in the engine's order) is in once each stream carrying it counted it
// (a stream counts every shard it sends: m (its entries a chunk) + its entry's index + 1): then XREADY + 1 on every
// engine core (XREADY = m (tp_size - 1) + k).
struct XGatherWatcherArgs {
    uint32_t num_chunks;                        // [0] the call's row chunks
    uint32_t x_ready_semaphore_address;         // [1] XREADY (its semaphore id: get_semaphore)
    uint32_t num_ranks;                         // [2] the shards that land here: tp_size - 1
    uint32_t num_engine_cores;                  // [3]
    uint32_t num_counters;                      // [4]
    volatile tt_l1_ptr uint32_t* counters;      // in place: num_counters x (counter address, its count a call)
    volatile tt_l1_ptr uint32_t* engine_cores;  // in place: num_engine_cores x (NoC x, NoC y)
    // in place, per rank: its streams n, then n x (the stream's counter address, its entry's index, its entries a
    // chunk)
    volatile tt_l1_ptr uint32_t* rank_streams;
#ifdef SKIP_PADDING_CHUNKS
    // (common args: ChipProgram::live_args) padded_rows.hpp's LiveChunkArgs; [10] rounded up to 64 B: the metadata
    // is read into it
    LiveChunkArgs live_chunk_args;
    uint32_t live_chunk_scratch_address;
#endif
#ifdef WATCHER_ROUTING_ARGS_OFFSET
    // (from WATCHER_ROUTING_ARGS_OFFSET: ChipProgram::y_args of the second Y direction, without its hellos) its send
    // block, then whether it has remote destinations; then (if so) the hello's 3 words (unused here) and the fabric
    // connection: its eth channel, teardown and buffer-index semaphores
    RoutingExchangeSendArgs routing_exchange;
    uint32_t has_remote_destinations;
    uint32_t fabric_eth_channel, fabric_teardown_semaphore, fabric_buffer_index_semaphore;
#endif
};

FORCE_INLINE XGatherWatcherArgs read_x_gather_watcher_args() {
    XGatherWatcherArgs args;
    uint32_t arg_index = 0;
    args.num_chunks = moe_block_args::arg(arg_index++);
    args.x_ready_semaphore_address = get_semaphore(moe_block_args::arg(arg_index++));
    args.num_ranks = moe_block_args::arg(arg_index++);
    args.num_engine_cores = moe_block_args::arg(arg_index++);
    args.num_counters = moe_block_args::arg(arg_index++);
    args.counters = moe_block_args::arg_pointer(arg_index);
    arg_index += 2 * args.num_counters;
    args.engine_cores = moe_block_args::arg_pointer(arg_index);
    arg_index += 2 * args.num_engine_cores;
    args.rank_streams = moe_block_args::arg_pointer(arg_index);
#ifdef SKIP_PADDING_CHUNKS
    args.live_chunk_args = read_live_chunk_args(moe_block_args::common_pointer(0));
    args.live_chunk_scratch_address = padded_metadata_slot(args.live_chunk_args.metadata_scratch);
#endif
#ifdef WATCHER_ROUTING_ARGS_OFFSET
    arg_index = WATCHER_ROUTING_ARGS_OFFSET;
    args.routing_exchange = read_routing_exchange_send_args(arg_index);
    args.has_remote_destinations = moe_block_args::arg(arg_index++);
    args.fabric_eth_channel = args.fabric_teardown_semaphore = args.fabric_buffer_index_semaphore = 0;
    if (args.has_remote_destinations) {
        arg_index += 3;  // (the Y hello's words: the BRISC's direction only)
        args.fabric_eth_channel = moe_block_args::arg(arg_index++);
        args.fabric_teardown_semaphore = moe_block_args::arg(arg_index++);
        args.fabric_buffer_index_semaphore = moe_block_args::arg(arg_index++);
    }
#endif
    return args;
}

// One row block of this kernel's work: kernel_main below runs it once per block (row_block.hpp).
void process_row_block(uint32_t block, const XGatherWatcherArgs& args) {
    // (row blocks: every block allocates its packet headers anew - the last block's packets were flushed by its
    // connection's close)
#ifdef WATCHER_ROUTING_ARGS_OFFSET
    PacketHeaderPool::reset();
#endif
#ifdef SKIP_PADDING_CHUNKS
    // (a padded forward: the row chunks up to the last with a real row - the counters count those; block b's first row:
    // ROW_BLOCK_PADDED_ROW_STRIDE rows on)
    const uint32_t num_live_chunks = padded_live_chunks(
        args.live_chunk_args,
        args.live_chunk_args.first_row + block * ROW_BLOCK_PADDED_ROW_STRIDE,
        args.live_chunk_scratch_address);
#else
    const uint32_t num_live_chunks = args.num_chunks;
#endif
    for (uint32_t chunk = 0; chunk < num_live_chunks; ++chunk) {
        uint32_t word = 0;  // (the rank's words in rank_streams)
        for (uint32_t rank = 0; rank < args.num_ranks; ++rank) {
            const uint32_t num_streams = args.rank_streams[word++];
            for (uint32_t stream = 0; stream < num_streams; ++stream, word += 3) {
                noc_semaphore_wait_min(
                    reinterpret_cast<volatile tt_l1_ptr uint32_t*>(args.rank_streams[word]),
                    chunk * args.rank_streams[word + 2] + args.rank_streams[word + 1] + 1);
            }
            for (uint32_t core = 0; core < args.num_engine_cores; ++core) {
                noc_semaphore_inc(
                    get_noc_addr(
                        args.engine_cores[2 * core], args.engine_cores[2 * core + 1], args.x_ready_semaphore_address),
                    1);
            }
        }
    }
    for (uint32_t i = 0; i < args.num_counters; ++i) {  // (re-arm)
        noc_semaphore_inc(
            get_noc_addr(args.counters[2 * i]), 0u - args.counters[2 * i + 1] / args.num_chunks * num_live_chunks);
    }
    noc_async_atomic_barrier();
#ifdef WATCHER_ROUTING_ARGS_OFFSET
    {  // (the routing exchange's second Y direction: after the watch)
        RoutingExchangeSend routing_exchange{args.routing_exchange};
        if (args.has_remote_destinations) {
            auto connection = WorkerToFabricEdmSender::build_from_eth_channel<ProgrammableCoreType::TENSIX>(
                args.fabric_eth_channel, args.fabric_teardown_semaphore, args.fabric_buffer_index_semaphore);
            volatile tt_l1_ptr PACKET_HEADER_TYPE* header = PacketHeaderPool::allocate_header();
            connection.open();
            routing_exchange.poll(
                connection,
                header,
                true,
                ROUTING_EXCHANGE_NUM_CHIPS,
                ROUTING_EXCHANGE_PLANNER_INSTANCES,
                ROUTING_EXCHANGE_EXPERTS_PER_CHIP,
                ROUTING_EXCHANGE_MY_CHIP,
                ROUTING_EXCHANGE_HEADER_BYTES,
                ROUTING_EXCHANGE_EXPERT_COUNT_BLOCK_BYTES,
                ROUTING_EXCHANGE_INSTANCE_BLOCK_BYTES,
                ROUTING_EXCHANGE_MAX_PACKET_PAYLOAD_BYTES);
            connection.close();
        }
        noc_async_atomic_barrier();
    }
#endif
}

void kernel_main() {
    moe_block_args::load();  // (this core's args from the DRAM table; a compute kernel waits for its page)
    const XGatherWatcherArgs args = read_x_gather_watcher_args();
    for (uint32_t block = 0; block < row_block::BLOCKS; ++block) {
        row_block::begin(block);
        process_row_block(block, args);
        row_block::end(block);
    }
}
