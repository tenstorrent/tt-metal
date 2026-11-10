// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

// shared_expert_row_dataflow: the shared expert's row-direction dataflow on a matmul core (x in, x / h along the row)
// Runs on: matmul cores (the shared expert's engine_columns x engine_rows grid), RISCV_0 (BRISC), NoC 0
// Responsibilities:
//  - reads this core's steps of the chip's own x shard from DRAM into x-own L1; then +1 on every port core's
//    "x loaded" word and the row lead's word, and lets its NCRISC's W load start
//  - row lead: reads the row's router x from the row cores' x-own L1 and multicasts it along the row (RX, x-ready word)
//  - x steps along the row into CB 0 (kernel_lib multicasts x0 / x1), a gathered shard once XREADY (x_gather_watcher)
//    says it landed; h along the row into every row core's h-all slot (multicast h, CB 3)
//  - hand-offs to the port cores (+1 on their mailbox words): the latent block packed (LATENT_CB), each reduce-scatter
//    sub-block done (REDUCE_SCATTER_FINAL_COUNTER_CB); publishes the live row chunks to its NCRISC and compute
// Note: row multicasts on NoC 0, column multicasts (shared_expert_column_dataflow) on NoC 1: same-NoC row and column
//  multicasts stall each other
#include "ttnn/cpp/ttnn/operations/experimental/moe_block/moe_head/device/kernels/row_block.hpp"
#include "ttnn/cpp/ttnn/operations/experimental/moe_block/common/kernels/tile_bytes.hpp"
#include "ttnn/cpp/ttnn/operations/experimental/moe_block/moe_head/device/kernels/shared_expert.hpp"

#include <cstdint>
#include "api/dataflow/dataflow_api.h"
#include "api/dataflow/noc.h"
#include "ttnn/cpp/ttnn/kernel_lib/mcast/kernel/mcast_args.hpp"
#include "api/debug/kernel_profiler.hpp"
#ifdef SKIP_PADDING_CHUNKS
#include "ttnn/cpp/ttnn/operations/experimental/moe_block/moe_head/device/kernels/padded_rows.hpp"
#endif
using namespace dataflow_kernel_lib;
// Compile-time args (head_engine_program.cpp row_ct, then x's accessor args)
constexpr uint32_t engine_columns = get_compile_time_arg_val(0), row_tiles_per_core = get_compile_time_arg_val(1),
                   k_step_tiles = get_compile_time_arg_val(2);
constexpr uint32_t gate_up_k_steps = get_compile_time_arg_val(3), gate_up_passes = get_compile_time_arg_val(4),
                   row_chunks = get_compile_time_arg_val(5);
// (the x families: family 0 / 1 - the even / odd K step positions - each spread over its subset of the row's columns)
constexpr uint32_t max_column_pairs = get_compile_time_arg_val(6), family_0_columns = get_compile_time_arg_val(7),
                   family_1_columns = get_compile_time_arg_val(8);
constexpr uint32_t x_ready_semaphore_id = get_compile_time_arg_val(9);
constexpr auto x_args = TensorAccessorArgs<10>();
// Circular buffers
constexpr uint32_t cb_x = 0, cb_h_local = 2, cb_h_gathered = 3;
// (steps a family a pass; per owner column of the family)
constexpr uint32_t family_steps = gate_up_k_steps / 2, family_0_steps_per_owner = family_steps / family_0_columns,
                   family_1_steps_per_owner = family_steps / family_1_columns;
// x position j (the j-th K step this chip processes: its own shard's first, then each shard as it lands): family j % 2,
// owner = the family's subset[(j / 2) % family columns] (the family's columns divide a shard's steps / 2, so the
// rotation over (shard, pass, step) rounds keeps every position on one owner), its local index (j / 2) / family columns
FORCE_INLINE uint32_t owner_slot(uint32_t position) {
    return (position & 1) ? (position / 2) / family_1_columns : (position / 2) / family_0_columns;
}

using ArgTable = const volatile tt_l1_ptr uint32_t*;
// This kernel's args (head_engine_program.cpp engine(): the core's list, its block of the DRAM table -
// head_workload.cpp head_args_to_dram; common arg 0 is ewords), read once by kernel_main
struct RowDataflowArgs {
    uint32_t phase_words_address;  // common [0]: the router-phase words (ewords)
    uint32_t column;               // [0] my column
    uint32_t row;                  // [1] my row
    uint32_t family_0_index;       // [2] my index in x family 0 (engine_columns: not in it)
    uint32_t family_1_index;       // [3] my index in x family 1 (engine_columns: not in it)
    uint32_t row_noc_y;            // [4] my row's NoC y
    uint32_t x_address;            // [5] x (row block 0's)
    uint32_t x_k_tiles;            // [6] x's K tiles (the chip's slice)
    uint32_t x_first_k_tile;       // [7] its first K tile
    uint32_t x_own_address;        // [8] x own: my steps of every chunk
    uint32_t x_in_address;         // [9] CB 0 (x steps)
    uint32_t h_local_address;      // [10] CB 2 (my h columns)
    uint32_t h_gathered_address;   // [11] CB 3 (the row's h)
    uint32_t row_first_noc_x;      // [12] the row's first core's NoC x
    uint32_t row_last_noc_x;       // [13] its last core's
    uint32_t router_x_address;     // [14] RX (the router's x buffers)
    // [15 ..] family 0's columns [family_0_columns], family 1's [family_1_columns]; per column (its owned steps, its
    // family-1 base slot); then the row's NoC x [engine_columns]
    ArgTable family_table;
    uint32_t steps_per_shard;  // then: K steps a shard, the shards, the shard order (own, then arrival) [shards]
    uint32_t shards;
    ArgTable shard_order_table;
    // then the gather's ports: their count, their "x loaded" word, (NoC x, y) a port, my x-loaded semaphore id
    uint32_t ports;
    uint32_t x_loaded_word;
    ArgTable port_table;
    uint32_t x_loaded_semaphore_id;
    // (then the router-done word and y's address: not mine) the reduce-scatter hand-offs: count, the mailbox word,
    // (NoC x, y) a hand-off
    uint32_t handoffs;
    uint32_t handoff_mailbox_word;
    ArgTable handoff_table;
#ifdef LATENT_CB
    // then the latent's destinations: count, their word, (NoC x, y) a destination
    uint32_t latent_destinations;
    uint32_t latent_word;
    ArgTable latent_table;
#endif
#ifdef SKIP_PADDING_CHUNKS
    LiveChunkArgs live_chunk_args;  // common args from SKIP_PADDING_ARGS_OFFSET (head_chip_program.hpp live_args)
#endif

    uint32_t family_0_column(uint32_t i) const { return family_table[i]; }
    uint32_t family_1_column(uint32_t i) const { return family_table[family_0_columns + i]; }
    uint32_t column_owned_steps(uint32_t column_index) const {
        return family_table[family_0_columns + family_1_columns + 2 * column_index];
    }
    uint32_t column_family_1_base(uint32_t column_index) const {
        return family_table[family_0_columns + family_1_columns + 2 * column_index + 1];
    }
    uint32_t column_noc_x(uint32_t column_index) const {
        return family_table[family_0_columns + family_1_columns + 2 * engine_columns + column_index];
    }
    uint32_t shard_order(uint32_t rank) const { return shard_order_table[rank]; }
    uint32_t port_noc_x(uint32_t port) const { return port_table[2 * port]; }
    uint32_t port_noc_y(uint32_t port) const { return port_table[2 * port + 1]; }
    uint32_t handoff_noc_x(uint32_t handoff) const { return handoff_table[2 * handoff]; }
    uint32_t handoff_noc_y(uint32_t handoff) const { return handoff_table[2 * handoff + 1]; }
#ifdef LATENT_CB
    uint32_t latent_noc_x(uint32_t destination) const { return latent_table[2 * destination]; }
    uint32_t latent_noc_y(uint32_t destination) const { return latent_table[2 * destination + 1]; }
#endif
};
RowDataflowArgs read_row_dataflow_args() {
    RowDataflowArgs args;
    args.phase_words_address = moe_block_args::common(0);
    args.column = moe_block_args::arg(0);
    args.row = moe_block_args::arg(1);
    args.family_0_index = moe_block_args::arg(2);
    args.family_1_index = moe_block_args::arg(3);
    args.row_noc_y = moe_block_args::arg(4);
    args.x_address = moe_block_args::arg(5);
    args.x_k_tiles = moe_block_args::arg(6);
    args.x_first_k_tile = moe_block_args::arg(7);
    args.x_own_address = moe_block_args::arg(8);
    args.x_in_address = moe_block_args::arg(9);
    args.h_local_address = moe_block_args::arg(10);
    args.h_gathered_address = moe_block_args::arg(11);
    args.row_first_noc_x = moe_block_args::arg(12);
    args.row_last_noc_x = moe_block_args::arg(13);
    args.router_x_address = moe_block_args::arg(14);
    constexpr uint32_t family_table_at = 15;
    args.family_table = moe_block_args::arg_pointer(family_table_at);
    const uint32_t shard_order_at =
        family_table_at + family_0_columns + family_1_columns + 2 * engine_columns + engine_columns;
    args.steps_per_shard = moe_block_args::arg(shard_order_at);
    args.shards = moe_block_args::arg(shard_order_at + 1);
    args.shard_order_table = moe_block_args::arg_pointer(shard_order_at + 2);
    const uint32_t ports_at = shard_order_at + 2 + args.shards;
    args.ports = moe_block_args::arg(ports_at);
    args.x_loaded_word = moe_block_args::arg(ports_at + 1);
    args.port_table = moe_block_args::arg_pointer(ports_at + 2);
    args.x_loaded_semaphore_id = moe_block_args::arg(ports_at + 2 + 2 * args.ports);
    const uint32_t handoffs_at = ports_at + 5 + 2 * args.ports;  // (past the router-done word and y's address)
    args.handoffs = moe_block_args::arg(handoffs_at);
    args.handoff_mailbox_word = moe_block_args::arg(handoffs_at + 1);
    args.handoff_table = moe_block_args::arg_pointer(handoffs_at + 2);
#ifdef LATENT_CB
    const uint32_t latent_at = handoffs_at + 2 + 2 * args.handoffs;
    args.latent_destinations = moe_block_args::arg(latent_at);
    args.latent_word = moe_block_args::arg(latent_at + 1);
    args.latent_table = moe_block_args::arg_pointer(latent_at + 2);
#endif
#ifdef SKIP_PADDING_CHUNKS
    args.live_chunk_args = read_live_chunk_args(moe_block_args::common_pointer(SKIP_PADDING_ARGS_OFFSET));
#endif
    return args;
}

// One row block of this kernel's work: kernel_main below runs it once per block (row_block.hpp).
// chunks_before_block: the row chunks of the earlier blocks - the h-all slots alternate over every chunk of the call,
// the blocks' one after another (the compute pops cb_h_gathered a chunk at a time)
void process_row_block(const RowDataflowArgs& args, uint32_t block, uint32_t& chunks_before_block) {
    const uint32_t x_address = args.x_address + block * ROW_BLOCK_X_STRIDE_BYTES;  // (block b's x)
    const uint32_t steps_per_shard = args.steps_per_shard, shards = args.shards;
    // (position j's K step: its shard's rank in the shard order, its step there)
    auto step_of = [&](uint32_t position) {
        return args.shard_order(position / steps_per_shard) * steps_per_shard + position % steps_per_shard;
    };
    // (my steps a chunk)
    const uint32_t owned_steps_per_chunk = (args.family_0_index < engine_columns ? family_0_steps_per_owner : 0) +
                                           (args.family_1_index < engine_columns ? family_1_steps_per_owner : 0);
    auto is_mine = [&](uint32_t position) {
        return (position & 1) ? ((position / 2) % family_1_columns == args.family_1_index)
                              : ((position / 2) % family_0_columns == args.family_0_index);
    };
    // my steps' x-own slot: [chunk][family 0 slots, then family 1][row_tiles_per_core][k_step_tiles]
    auto own_address = [&](uint32_t chunk, uint32_t position) {  // (every row chunk's x own resident)
        const uint32_t slot = (position & 1) ? (args.family_0_index < engine_columns ? family_0_steps_per_owner : 0) +
                                                   owner_slot(position)
                                             : owner_slot(position);
        return args.x_own_address + ((chunk * owned_steps_per_chunk + slot) * row_tiles_per_core) * k_step_tiles *
                                        moe_block_tiles::kBf16TileBytes;
    };
    auto send_router_x = [&]() {  // (called after my own-shard x load: the gather's ports wait on every matmul core's
        // - a row lead's x load behind its ~20 us of router x held the gather back that long: GLM-5.3 ring 640 235.8
        // -> 219.2 us)
        if (args.column == 0) {  // the router's x: the row's first core reads the row's x (row_tiles_per_core row
                                 // tiles, the chip's K slice) a K chunk
            // a round into RX and multicasts it along the row on NOC0 (the NCRISC's W_r^T goes down the columns on
            // NOC1), then sets the round on the row's RX-ready word (from my own source word, ROUTER_X_SENT_FLAG_ID:
            // the NCRISC re-arms the ready word)
            DeviceZoneScopedN("SE-RTRX");
            constexpr uint32_t router_x_round_bytes =
                ROUTER_K_CHUNK_TILES * ROUTER_CORE_ROW_TILES *
                moe_block_tiles::kBf16TileBytes;  // (ROUTER_CORE_ROW_TILES: my row_tiles_per_core rows of every chunk
                                                  // - row tile t of chunk t / row_tiles_per_core)
            volatile tt_l1_ptr uint32_t* router_x_sent_word =
                reinterpret_cast<volatile tt_l1_ptr uint32_t*>(args.phase_words_address + ROUTER_X_SENT_FLAG_ID);
            constexpr uint32_t router_k_chunks = ROUTER_K_TILES / ROUTER_K_CHUNK_TILES;
#ifdef ROUTER_ITERATIONS
            constexpr uint32_t router_iterations =
                ROUTER_ITERATIONS;  // (router iterations: ROUTER_CORE_ROW_TILES of the row's rows each)
#else
            constexpr uint32_t router_iterations = 1;
#endif
            constexpr uint32_t router_rounds =
                router_k_chunks * router_iterations;  // (the rounds of every iteration: their counts run on)
            volatile tt_l1_ptr uint32_t* x_ack_word =
                reinterpret_cast<volatile tt_l1_ptr uint32_t*>(args.phase_words_address + X_ACKNOWLEDGE_FLAG_ID);
            {  // (the row's x is the own shard the row's cores just loaded into x own: read from their L1, not the 2 MB
               // again from DRAM beside the own-x loads and W_r^T - once every row core counted its load on my
               // X_READ_LOCAL_FLAG_ID word)
                volatile tt_l1_ptr uint32_t* x_read_local_word =
                    reinterpret_cast<volatile tt_l1_ptr uint32_t*>(args.phase_words_address + X_READ_LOCAL_FLAG_ID);
                while (*x_read_local_word < engine_columns) {
                    invalidate_l1_cache();
                }
                *x_read_local_word = 0;  // (re-armed: one count a row core a call, all in)
            }
            for (uint32_t round = 0; round < router_rounds; ++round) {
                // (round k_chunk of iteration `iteration`)
                const uint32_t k_chunk = round % router_k_chunks, iteration = round / router_k_chunks;
                // (two buffers, alternating over every round of the call - the blocks' one after another: the receivers
                // push a round each, the compute pops; past two rounds of a block the row's acks first)
                const uint32_t router_x_buffer =
                    args.router_x_address + ((block * router_rounds + round) & 1) * router_x_round_bytes;
                if (router_rounds > 2 && round >= 2) {
                    while (*x_ack_word < (round - 1) * engine_columns) {
                        invalidate_l1_cache();
                    }
                }
                for (uint32_t row_tile = 0; row_tile < ROUTER_CORE_ROW_TILES; ++row_tile) {
                    for (uint32_t chunk_k_tile = 0; chunk_k_tile < ROUTER_K_CHUNK_TILES; ++chunk_k_tile) {
                        // (the row's row tile call_row_tile of the call)
                        const uint32_t call_row_tile = iteration * ROUTER_CORE_ROW_TILES + row_tile;
                        // (K tile k_tile of the chip's slice: own-shard position k_tile / k_step_tiles - its owner
                        // column and x-own slot as x_gather.hpp's XGatherDestinationTables::destination; row tile t:
                        // chunk t / row_tiles_per_core, local row tile t % row_tiles_per_core)
                        const uint32_t k_tile = k_chunk * ROUTER_K_CHUNK_TILES + chunk_k_tile,
                                       position = k_tile / k_step_tiles, family_step = position / 2;
                        const uint32_t owner_column = (position & 1)
                                                          ? args.family_1_column(family_step % family_1_columns)
                                                          : args.family_0_column(family_step % family_0_columns);
                        const uint32_t owner_steps = args.column_owned_steps(owner_column),
                                       owner_family_1_base = args.column_family_1_base(owner_column);
                        const uint32_t slot = (position & 1) ? owner_family_1_base + family_step / family_1_columns
                                                             : family_step / family_0_columns;
                        const uint32_t source_address =
                            args.x_own_address +
                            ((((call_row_tile / row_tiles_per_core) * owner_steps + slot) * row_tiles_per_core +
                              call_row_tile % row_tiles_per_core) *
                                 k_step_tiles +
                             k_tile % k_step_tiles) *
                                moe_block_tiles::kBf16TileBytes;
                        noc_async_read(
                            get_noc_addr(args.column_noc_x(owner_column), args.row_noc_y, source_address),
                            router_x_buffer +
                                (chunk_k_tile * ROUTER_CORE_ROW_TILES + row_tile) * moe_block_tiles::kBf16TileBytes,
                            moe_block_tiles::kBf16TileBytes);
                    }
                }
                noc_async_read_barrier();
                noc_async_write_multicast_loopback_src(
                    router_x_buffer,
                    get_noc_multicast_addr(
                        args.row_first_noc_x, args.row_noc_y, args.row_last_noc_x, args.row_noc_y, router_x_buffer),
                    router_x_round_bytes,
                    engine_columns);
                noc_async_writes_flushed();
                *router_x_sent_word = round + 1;
                noc_semaphore_set_multicast_loopback_src(
                    args.phase_words_address + ROUTER_X_SENT_FLAG_ID,
                    get_noc_multicast_addr(
                        args.row_first_noc_x,
                        args.row_noc_y,
                        args.row_last_noc_x,
                        args.row_noc_y,
                        args.phase_words_address + ROUTER_X_READY_FLAG_ID),
                    engine_columns);
                noc_async_writes_flushed();
            }
            if (router_rounds > 2) {  // (re-arm: every ack landed - the last were waited for)
                noc_semaphore_inc(
                    get_noc_addr(args.phase_words_address + X_ACKNOWLEDGE_FLAG_ID),
                    0u - (router_rounds - 2) * engine_columns);
                noc_async_atomic_barrier();
            }
        }
    };
    const auto x_accessor = TensorAccessor(x_args, x_address, moe_block_tiles::kBf16TileBytes);
    // (one owned step of chunk `chunk` from DRAM x [rows, x_k_tiles])
    auto load_owned_step = [&](uint32_t chunk, uint32_t position) {
        const uint32_t step = step_of(position);
        if (!(is_mine(position) && step * k_step_tiles >= args.x_first_k_tile &&
              step * k_step_tiles < args.x_first_k_tile + args.x_k_tiles)) {
            return;
        }
        for (uint32_t row_tile = 0; row_tile < row_tiles_per_core; ++row_tile) {
            for (uint32_t k_tile = 0; k_tile < k_step_tiles; ++k_tile) {
                noc_async_read(
                    x_accessor.get_noc_addr(
                        global_row_tile(chunk, args.row * row_tiles_per_core + row_tile) * args.x_k_tiles +
                        step * k_step_tiles + k_tile - args.x_first_k_tile),
                    own_address(chunk, position) + (row_tile * k_step_tiles + k_tile) * moe_block_tiles::kBf16TileBytes,
                    moe_block_tiles::kBf16TileBytes);
            }
        }
    };
    if (args.x_k_tiles) {  // (alone) my steps of every chunk from DRAM x [rows, x_k_tiles]: columns x_first_k_tile ..
        DeviceZoneScopedN("SE-XLOAD");
        for (uint32_t chunk = 0; chunk < row_chunks; ++chunk) {
            for (uint32_t position = 0; position < gate_up_k_steps; ++position) {
                load_owned_step(chunk, position);
            }
        }
        noc_async_read_barrier();
    }
    {  // (my own-shard x is loaded: count it on the gather's ports - they send it from here - and let my W load start)
        for (uint32_t port = 0; port < args.ports; ++port) {
            noc_semaphore_inc(get_noc_addr(args.port_noc_x(port), args.port_noc_y(port), args.x_loaded_word), 1);
        }
        noc_semaphore_inc(
            get_noc_addr(args.column_noc_x(0), args.row_noc_y, args.phase_words_address + X_READ_LOCAL_FLAG_ID),
            1);  // (and on my row's lead: the router's x)
        noc_async_atomic_barrier();
        *reinterpret_cast<volatile tt_l1_ptr uint32_t*>(get_semaphore(args.x_loaded_semaphore_id)) = 1;
    }
#ifdef SKIP_PADDING_CHUNKS
    // a padded forward: the row chunks up to the last with a real row only (the rest are padding: no x, no shared
    // expert, no reduce-scatter) - for my NCRISC and compute too (ewords + LIVE_CHUNKS_FLAG_ID, zeroed at my end).
    // After my x load and its count on the gather's ports (they go first), before the router's x (my NCRISC's weights
    // wait on it)
    const uint32_t live_chunks = padded_live_chunks(
        args.live_chunk_args,
        args.live_chunk_args.first_row + block * ROW_BLOCK_PADDED_ROW_STRIDE,
        padded_metadata_slot(args.live_chunk_args.metadata_scratch));
    *reinterpret_cast<volatile tt_l1_ptr uint32_t*>(args.phase_words_address + LIVE_CHUNKS_FLAG_ID) = live_chunks;
#else
    constexpr uint32_t live_chunks = row_chunks;
#endif
    send_router_x();  // (the router's x after my own x)
    constexpr auto x_family_0_multicast =
        McastArgs<get_named_compile_time_arg_val("x0_ct_offset"), get_named_compile_time_arg_val("x0_rt_offset")>();
    constexpr auto x_family_1_multicast =
        McastArgs<get_named_compile_time_arg_val("x1_ct_offset"), get_named_compile_time_arg_val("x1_rt_offset")>();
    constexpr auto h_multicast =
        McastArgs<get_named_compile_time_arg_val("h_ct_offset"), get_named_compile_time_arg_val("h_rt_offset")>();
    Noc noc;
    auto h_sender = h_multicast.optional_sender(noc);
    auto h_receiver = h_multicast.optional_receiver(noc);
    auto x_family_0_sender = x_family_0_multicast.optional_sender(noc);
    auto x_family_0_receiver = x_family_0_multicast.optional_receiver(noc);
    auto x_family_1_sender = x_family_1_multicast.optional_sender(noc);
    auto x_family_1_receiver = x_family_1_multicast.optional_receiver(noc);
    volatile tt_l1_ptr uint32_t* x_ready =
        reinterpret_cast<volatile tt_l1_ptr uint32_t*>(get_semaphore(x_ready_semaphore_id));
    for (uint32_t chunk = 0; chunk < live_chunks; ++chunk) {
        {
            DeviceZoneScopedN("SE-XROUNDS");
            // shard-major: each shard (own first, then as it lands) through every pass before the next shard
            for (uint32_t shard = 0; shard < shards; ++shard) {
                for (uint32_t pass = 0; pass < gate_up_passes; ++pass) {
                    for (uint32_t step = 0; step < steps_per_shard; ++step) {
                        // (position; its family position % 2 = step % 2: a block's rounds are even)
                        const uint32_t position = shard * steps_per_shard + step;
                        const uint32_t round =
                            ((chunk * shards + shard) * gate_up_passes + pass) * (steps_per_shard / 2) +
                            step / 2;  // (the family's)
                        cb_reserve_back(cb_x, row_tiles_per_core * k_step_tiles);
                        const uint32_t x_destination = args.x_in_address + (position & 1) * row_tiles_per_core *
                                                                               k_step_tiles *
                                                                               moe_block_tiles::kBf16TileBytes;
                        if (is_mine(position)) {  // (position: the shard of rank position / steps_per_shard must be
                                                  // in - the own shard always is)
                            if (position >= steps_per_shard) {
                                noc_semaphore_wait_min(x_ready, chunk * (shards - 1) + position / steps_per_shard);
                            }
                            constexpr uint32_t x_step_bytes =
                                row_tiles_per_core * k_step_tiles * moe_block_tiles::kBf16TileBytes;
                            if (position & 1) {
                                x_family_1_sender->send(own_address(chunk, position), x_destination, x_step_bytes);
                            } else {
                                x_family_0_sender->send(own_address(chunk, position), x_destination, x_step_bytes);
                            }
                        } else {
                            if (position & 1) {
                                x_family_1_receiver->receive(round);
                            } else {
                                x_family_0_receiver->receive(round);
                            }
                        }
                        cb_push_back(cb_x, row_tiles_per_core * k_step_tiles);
                    }
                }
            }
        }
#ifdef LATENT_CB
        {  // (the latent's all-to-all: once it is packed, +1 on each destination's port word of mine)
            DeviceZoneScopedN("SE-LHAND");
            // (chunk `chunk`'s block; the words count up over the chunks)
            cb_wait_front(LATENT_CB, (chunk + 1) * LATENT_PAGES);
            for (uint32_t destination = 0; destination < args.latent_destinations; ++destination) {
                noc_semaphore_inc(
                    get_noc_addr(args.latent_noc_x(destination), args.latent_noc_y(destination), args.latent_word), 1);
            }
        }
#endif
        {  // (its own zone: the hand-offs below are not h exchange)
            DeviceZoneScopedN("SE-HX");
            // h: every core's h block [row_tiles_per_core][max_column_pairs] into every row core's h-all slot
            // (chunk % 2) as block `column` ([engine_columns][row_tiles_per_core][max_column_pairs]): one kernel_lib
            // rotating round a column; the receivers' readiness ack is the slot's credit
            constexpr uint32_t h_gathered_tiles = engine_columns * row_tiles_per_core * max_column_pairs;
            cb_wait_front(cb_h_local, row_tiles_per_core * max_column_pairs);
            cb_reserve_back(cb_h_gathered, h_gathered_tiles);
            const uint32_t h_slot_address =
                args.h_gathered_address + (SHARED_EXPERT_H_SLOTS > 1 ? ((chunks_before_block + chunk) & 1) : 0) *
                                              h_gathered_tiles *
                                              moe_block_tiles::kBf16TileBytes;  // (h-all slots: SHARED_EXPERT_H_SLOTS)
            for (uint32_t column = 0; column < engine_columns; ++column) {
                if (column == args.column) {
                    h_sender->send(
                        args.h_local_address,
                        h_slot_address +
                            column * row_tiles_per_core * max_column_pairs * moe_block_tiles::kBf16TileBytes,
                        row_tiles_per_core * max_column_pairs * moe_block_tiles::kBf16TileBytes);
                } else {
                    h_receiver->receive(chunk * engine_columns + column);
                }
                cb_push_back(
                    cb_h_gathered,
                    row_tiles_per_core *
                        max_column_pairs);  // (column `column`'s block: the down's first pass starts on its K tiles)
            }
            cb_pop_front(cb_h_local, row_tiles_per_core * max_column_pairs);
        }
        {  // (down + reduce-scatter) each finished sub-block's hand-off: +1 on its port's mailbox word of mine (the
           // words count up over the chunks: the port's targets do)
            DeviceZoneScopedN("SE-HAND");
            for (uint32_t handoff = 0; handoff < args.handoffs; ++handoff) {
                cb_wait_front(REDUCE_SCATTER_FINAL_COUNTER_CB, handoff + 1);
                noc_semaphore_inc(
                    get_noc_addr(args.handoff_noc_x(handoff), args.handoff_noc_y(handoff), args.handoff_mailbox_word),
                    1);
            }
            if constexpr (row_chunks > 1 || row_block::BLOCKS > 1) {
                cb_pop_front(
                    REDUCE_SCATTER_FINAL_COUNTER_CB,
                    args.handoffs);  // (the next chunk's - or block's - hand-offs from the front again)
            }
        }
    }
#ifdef LATENT_CB
    if constexpr (row_block::BLOCKS > 1) {
        // (row blocks: the latent's whole region popped - the compute pads the skipped chunks - so the next block's
        // packs start at its start, where the ports read them)
        cb_wait_front(LATENT_CB, row_chunks * LATENT_PAGES);
        cb_pop_front(LATENT_CB, row_chunks * LATENT_PAGES);
    }
#endif
    chunks_before_block += live_chunks;
    noc_async_atomic_barrier();
#ifdef SKIP_PADDING_CHUNKS
    *reinterpret_cast<volatile tt_l1_ptr uint32_t*>(args.phase_words_address + LIVE_CHUNKS_FLAG_ID) =
        0;  // (re-arm: my NCRISC and compute read it)
#endif
}

void kernel_main() {
    moe_block_args::load();  // (this core's args from the DRAM table; a compute kernel waits for its page)
    const RowDataflowArgs args = read_row_dataflow_args();
    uint32_t chunks_before_block = 0;  // (a launch starts the count)
    for (uint32_t block = 0; block < row_block::BLOCKS; ++block) {
        row_block::begin(block);
        process_row_block(args, block, chunks_before_block);
        row_block::end(block);
    }
}
