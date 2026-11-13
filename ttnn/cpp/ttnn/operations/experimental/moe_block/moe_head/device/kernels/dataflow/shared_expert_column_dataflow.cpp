// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

// shared_expert_column_dataflow: the shared expert's column-direction dataflow on a matmul core (W, outputs)
// Runs on: matmul cores (the shared expert's engine_columns x engine_rows grid), RISCV_1 (NCRISC), NoC 1
// Responsibilities:
//  - router: the column lead reads W_r^T expert tiles from DRAM and multicasts them down the column; pushes the x / W
//    rounds to the compute; writes its partial logits to the router core's HOLD or the port's sub-block (+ counter);
//    then +1 on every port's "router done" word
//  - after its BRISC's x load: reads its W blocks from DRAM (windowed prefetch) and multicasts each down the column
//    into the W slots (CB 1, the W-valid / W-free semaphores)
//  - reduce-scatter: pushes the fabric-landed arrival zones to the compute, credits the ports' zone-free words, writes
//    the final tiles to y in DRAM, frees the final block once the ports read it; latent arrivals counted ->
//    LATENT_READY_CB
//  - dispatch prep: the compute's rows into the dispatch buffer in DRAM (posted writes, a fence at the end)
// Note: column multicasts on NoC 1, row multicasts (shared_expert_row_dataflow) on NoC 0: same-NoC row and column
//  multicasts stall each other
#include "ttnn/cpp/ttnn/operations/experimental/moe_block/moe_head/device/kernels/row_block.hpp"
#include "ttnn/cpp/ttnn/operations/experimental/moe_block/common/kernels/tile_bytes.hpp"
#include "ttnn/cpp/ttnn/operations/experimental/moe_block/moe_head/device/kernels/shared_expert.hpp"

#include <cstdint>
#include "api/dataflow/dataflow_api.h"
#include "api/debug/kernel_profiler.hpp"
// Compile-time args (head_engine_program.cpp column_ct, then the accessors' args)
constexpr uint32_t engine_rows = get_compile_time_arg_val(0), row_tiles_per_core = get_compile_time_arg_val(1),
                   k_step_tiles = get_compile_time_arg_val(2);
constexpr uint32_t gate_up_k_steps = get_compile_time_arg_val(3), gate_up_passes = get_compile_time_arg_val(4),
                   row_chunks = get_compile_time_arg_val(5);
constexpr uint32_t down_k_step_tiles = get_compile_time_arg_val(6), down_k_steps = get_compile_time_arg_val(7),
                   down_passes = get_compile_time_arg_val(8);
constexpr uint32_t w_slots = get_compile_time_arg_val(9), w_slot_tiles = get_compile_time_arg_val(10),
                   w_tile_bytes = get_compile_time_arg_val(11);
// (the W slots' semaphores: slot s's W-valid and W-free semaphore ids, from these on)
constexpr uint32_t w_valid_semaphore_id = get_compile_time_arg_val(12),
                   w_free_semaphore_id = get_compile_time_arg_val(13);
constexpr auto w_args = TensorAccessorArgs<14>();
constexpr auto y_args = TensorAccessorArgs<w_args.next_compile_time_args_offset()>();
constexpr auto dispatch_buffer_args =
    TensorAccessorArgs<DISPATCH_PREP_ACCESSOR_CT_OFFSET>();  // (the dispatch buffer [N S, W] row-major)
constexpr auto router_x_args =
    TensorAccessorArgs<ROUTER_ACCESSOR_CT_OFFSET>();  // (the chip's x shard [rows, H / TP], DRAM)
constexpr auto router_w_args =
    TensorAccessorArgs<router_x_args.next_compile_time_args_offset()>();  // (W_r^T [E, H / TP])
// Circular buffers
constexpr uint32_t cb_w = 1;
constexpr uint32_t w_blocks_per_chunk =
    gate_up_passes * gate_up_k_steps + down_passes * down_k_steps;  // (W blocks a chunk)
#ifdef ROUTER_ITERATIONS
constexpr uint32_t router_iterations =
    ROUTER_ITERATIONS;  // (router iterations: the router phase over ROUTER_CORE_ROW_TILES of my rows at a time)
#else
constexpr uint32_t router_iterations = 1;
#endif
constexpr uint32_t down_k_steps_divisor = down_k_steps ? down_k_steps : 1;  // (never divided by 0)
FORCE_INLINE volatile tt_l1_ptr uint32_t* semaphore_word(uint32_t semaphore_id) {
    return reinterpret_cast<volatile tt_l1_ptr uint32_t*>(get_semaphore(semaphore_id));
}
// (a row of an untilized dispatch-prep block: DISPATCH_PREP_FULL_COLUMN_TILES tiles' bf16 columns)
constexpr uint32_t prep_row_bytes = DISPATCH_PREP_FULL_COLUMN_TILES * tt::constants::TILE_WIDTH * sizeof(uint16_t);

using ArgTable = const volatile tt_l1_ptr uint32_t*;
// This kernel's args (head_engine_program.cpp engine(): the core's list, its block of the DRAM table -
// head_workload.cpp head_args_to_dram; common arg 0 is ewords), read once by kernel_main
struct ColumnDataflowArgs {
    uint32_t phase_words_address;  // common [0]: the router-phase words (ewords)
    uint32_t row;                  // [0] my row
    uint32_t column_noc_x;         // [1] my column's NoC x
    uint32_t w_own_offset;         // [2] my W-own region's offset in its DRAM bank (per-bank layout)
    uint32_t w_own_bank;           // [3] the DRAM bank holding it
    // the router logits' args, from [4]: x (not read here), W_r^T, my column's expert tiles (valid count, first), RX
    // (not read here), RW, the row's far NoC x (not read here), the row's x lead (NoC x, y), my column has experts;
    // then per row tile of mine (ROUTER_CORE_ROW_TILES x router_iterations of them) its destination: (NoC x | y << 16,
    // base, counter)
    uint32_t router_w_address;          // [5]
    uint32_t router_expert_tiles;       // [6]
    uint32_t router_first_expert_tile;  // [7]
    uint32_t router_w_buffer;           // [9] RW
    uint32_t router_x_lead_noc_x;       // [11]
    uint32_t router_x_lead_noc_y;       // [12]
    bool has_router_experts;            // [13]
    ArgTable router_destination_table;  // [14 ..]
    // then the column's pass widths: gate / up [gate_up_passes], down [down_passes]
    ArgTable gate_up_pass_width_table;
    ArgTable down_pass_width_table;
    uint32_t column_top_noc_y;     // then the column's first row's NoC y
    uint32_t column_bottom_noc_y;  // its last row's
    uint32_t w_address;            // W in DRAM
    uint32_t w_own_address;        // my W-own (L1)
    uint32_t w_in_address;         // CB 1 (the W slots)
    uint32_t w_valid_words;        // the W-valid source words (4, 16 B apart)
    uint32_t y_address;            // y (row block 0's)
    ArgTable row_noc_y_table;      // the column's NoC y [engine_rows]
    uint32_t steps_per_shard;      // K steps a shard, the shards, the shard order (own, then arrival) [shards]
    uint32_t shards;
    ArgTable shard_order_table;
    // then the gather's ports: their count, (their "x loaded" word: not read here), (NoC x, y) a port, my x-loaded
    // semaphore id, the "router done" word
    uint32_t ports;
    ArgTable port_table;
    uint32_t x_loaded_semaphore_id;
#ifdef ROUTER_DONE_CREDIT
    uint32_t router_done_word;
#endif
    // the reduce-scatter's arrivals per direction in zone-slot order (a slot's word: its counter threshold | its zone
    // pages << 16), then the final sub-block to y: its tiles, its first y tile (chunk 0's), its y row stride, its
    // width, the arrival counters' addresses
    uint32_t forward_arrival_slots;
    ArgTable forward_arrival_table;
    uint32_t backward_arrival_slots;
    ArgTable backward_arrival_table;
    uint32_t final_tiles;
    uint32_t final_first_y_tile;
    uint32_t final_y_row_stride;
    uint32_t final_width;
    uint32_t forward_arrival_counter_address;
    uint32_t backward_arrival_counter_address;
    // (row chunks: the counters count up over them - a chunk's thresholds + its index x the arrivals a chunk) then
    // the final sub-block's first y tile of chunks 1 .. [row_chunks - 1], the ports facing my forward / backward
    // upstream (NoC x | y << 16), their zone-free word, my final-block-free counter, the ports reading my final block;
    // REDUCE_SCATTER_FINAL_ROWS: then each (chunk, row tile)'s own first y tile [row_chunks][row_tiles_per_core]
    ArgTable final_chunk_table;
    uint32_t forward_upstream_port;
    uint32_t backward_upstream_port;
    uint32_t zone_free_word;
    uint32_t final_free_counter_address;
    uint32_t final_block_readers;
#ifdef LATENT_CB
    // then the latent: my rows of it, its sources (each other chip's slice of my rows), their arrival counter
    uint32_t latent_rows;
    uint32_t latent_sources;
    uint32_t latent_counter_address;
#endif
    // then the dispatch prep: its units (chunk-major, as many a chunk), the dispatch buffer, per unit (buffer row,
    // column offset, row bytes)
    uint32_t prep_units;
    uint32_t dispatch_buffer_address;
    ArgTable prep_unit_table;

    volatile tt_l1_ptr uint32_t* phase_word(uint32_t offset) const {
        return reinterpret_cast<volatile tt_l1_ptr uint32_t*>(phase_words_address + offset);
    }
    uint32_t router_destination_noc_xy(uint32_t call_row_tile) const {
        return router_destination_table[3 * call_row_tile];
    }
    uint32_t router_destination_base(uint32_t call_row_tile) const {
        return router_destination_table[3 * call_row_tile + 1];
    }
    uint32_t router_destination_counter(uint32_t call_row_tile) const {
        return router_destination_table[3 * call_row_tile + 2];
    }
    uint32_t gate_up_pass_width(uint32_t pass) const { return gate_up_pass_width_table[pass]; }
    uint32_t down_pass_width(uint32_t pass) const { return down_pass_width_table[pass]; }
    uint32_t row_noc_y(uint32_t row_index) const { return row_noc_y_table[row_index]; }
    uint32_t shard_order(uint32_t rank) const { return shard_order_table[rank]; }
    uint32_t port_noc_x(uint32_t port) const { return port_table[2 * port]; }
    uint32_t port_noc_y(uint32_t port) const { return port_table[2 * port + 1]; }
    uint32_t forward_arrival(uint32_t slot) const { return forward_arrival_table[slot]; }
    uint32_t backward_arrival(uint32_t slot) const { return backward_arrival_table[slot]; }
    uint32_t final_chunk_first_y_tile(uint32_t chunk) const { return final_chunk_table[chunk - 1]; }  // (chunk >= 1)
#ifdef REDUCE_SCATTER_FINAL_ROWS
    uint32_t final_row_first_y_tile(uint32_t chunk, uint32_t row_tile) const {
        return final_chunk_table[row_chunks + 4 + chunk * row_tiles_per_core + row_tile];
    }
#endif
    uint32_t prep_unit_row(uint32_t unit) const { return prep_unit_table[3 * unit]; }
    uint32_t prep_unit_column_offset(uint32_t unit) const { return prep_unit_table[3 * unit + 1]; }
    uint32_t prep_unit_row_bytes(uint32_t unit) const { return prep_unit_table[3 * unit + 2]; }
};
ColumnDataflowArgs read_column_dataflow_args() {
    ColumnDataflowArgs args;
    args.phase_words_address = moe_block_args::common(0);
    args.row = moe_block_args::arg(0);
    args.column_noc_x = moe_block_args::arg(1);
    args.w_own_offset = moe_block_args::arg(2);
    args.w_own_bank = moe_block_args::arg(3);
    constexpr uint32_t router_at = 4;
    args.router_w_address = moe_block_args::arg(router_at + 1);
    args.router_expert_tiles = moe_block_args::arg(router_at + 2);
    args.router_first_expert_tile = moe_block_args::arg(router_at + 3);
    args.router_w_buffer = moe_block_args::arg(router_at + 5);
    args.router_x_lead_noc_x = moe_block_args::arg(router_at + 7);
    args.router_x_lead_noc_y = moe_block_args::arg(router_at + 8);
    args.has_router_experts = moe_block_args::arg(router_at + 9);
    args.router_destination_table = moe_block_args::arg_pointer(router_at + 10);
    constexpr uint32_t pass_widths_at = router_at + 10 + ROUTER_DEPTH * ROUTER_CORE_ROW_TILES * router_iterations;
    args.gate_up_pass_width_table = moe_block_args::arg_pointer(pass_widths_at);
    args.down_pass_width_table = moe_block_args::arg_pointer(pass_widths_at + gate_up_passes);
    constexpr uint32_t column_at = pass_widths_at + gate_up_passes + down_passes;
    args.column_top_noc_y = moe_block_args::arg(column_at);
    args.column_bottom_noc_y = moe_block_args::arg(column_at + 1);
    args.w_address = moe_block_args::arg(column_at + 2);
    args.w_own_address = moe_block_args::arg(column_at + 3);
    args.w_in_address = moe_block_args::arg(column_at + 4);
    args.w_valid_words = moe_block_args::arg(column_at + 5);
    args.y_address = moe_block_args::arg(column_at + 6);
    constexpr uint32_t row_noc_y_at = column_at + 7;
    args.row_noc_y_table = moe_block_args::arg_pointer(row_noc_y_at);
    args.steps_per_shard = moe_block_args::arg(row_noc_y_at + engine_rows);
    args.shards = moe_block_args::arg(row_noc_y_at + engine_rows + 1);
    args.shard_order_table = moe_block_args::arg_pointer(row_noc_y_at + engine_rows + 2);
    const uint32_t ports_at = row_noc_y_at + engine_rows + 2 + args.shards;  // (past the shard order: the tail)
    args.ports = moe_block_args::arg(ports_at);
    args.port_table = moe_block_args::arg_pointer(ports_at + 2);
    args.x_loaded_semaphore_id = moe_block_args::arg(ports_at + 2 + 2 * args.ports);
#ifdef ROUTER_DONE_CREDIT
    args.router_done_word = moe_block_args::arg(ports_at + 3 + 2 * args.ports);
#endif
    const uint32_t forward_arrivals_at = ports_at + 4 + 2 * args.ports;  // (past the tail)
    args.forward_arrival_slots = moe_block_args::arg(forward_arrivals_at);
    args.forward_arrival_table = moe_block_args::arg_pointer(forward_arrivals_at + 1);
    const uint32_t backward_arrivals_at = forward_arrivals_at + 1 + args.forward_arrival_slots;
    args.backward_arrival_slots = moe_block_args::arg(backward_arrivals_at);
    args.backward_arrival_table = moe_block_args::arg_pointer(backward_arrivals_at + 1);
    const uint32_t final_at = backward_arrivals_at + 1 + args.backward_arrival_slots;
    args.final_tiles = moe_block_args::arg(final_at);
    args.final_first_y_tile = moe_block_args::arg(final_at + 1);
    args.final_y_row_stride = moe_block_args::arg(final_at + 2);
    args.final_width = moe_block_args::arg(final_at + 3);
    args.forward_arrival_counter_address = moe_block_args::arg(final_at + 4);
    args.backward_arrival_counter_address = moe_block_args::arg(final_at + 5);
    const uint32_t final_chunk_at = final_at + 6;
    args.final_chunk_table = moe_block_args::arg_pointer(final_chunk_at);
    args.forward_upstream_port = row_chunks > 1 ? moe_block_args::arg(final_chunk_at + row_chunks - 1) : 0;
    args.backward_upstream_port = row_chunks > 1 ? moe_block_args::arg(final_chunk_at + row_chunks) : 0;
    args.zone_free_word = row_chunks > 1 ? moe_block_args::arg(final_chunk_at + row_chunks + 1) : 0;
    args.final_free_counter_address = row_chunks > 1 ? moe_block_args::arg(final_chunk_at + row_chunks + 2) : 0;
    args.final_block_readers = row_chunks > 1 ? moe_block_args::arg(final_chunk_at + row_chunks + 3) : 0;
#ifdef REDUCE_SCATTER_FINAL_ROWS  // (a core's row_tiles_per_core rows straddle owner blocks: each (chunk, row tile)'s
                                  // own y base)
    constexpr uint32_t final_chunk_args = row_chunks + 4 + row_chunks * row_tiles_per_core;
#else
    constexpr uint32_t final_chunk_args = row_chunks > 1 ? row_chunks + 4 : 0;
#endif
#ifdef LATENT_CB
    const uint32_t latent_at = final_chunk_at + final_chunk_args;
    args.latent_rows = moe_block_args::arg(latent_at);
    args.latent_sources = moe_block_args::arg(latent_at + 1);
    args.latent_counter_address = moe_block_args::arg(latent_at + 2);
    const uint32_t prep_at = final_chunk_at + 3 + final_chunk_args;
#else
    const uint32_t prep_at = final_chunk_at + final_chunk_args;
#endif
    args.prep_units = moe_block_args::arg(prep_at);
    args.dispatch_buffer_address = moe_block_args::arg(prep_at + 1);
    args.prep_unit_table = moe_block_args::arg_pointer(prep_at + 2);
    return args;
}

// One row block of this kernel's work: kernel_main below runs it once per block (row_block.hpp).
// w_rounds_before_block: the W rounds of the earlier blocks - a round's slot (and its W-valid / W-free semaphore) is
// its index over every round of the call, the blocks' one after another (the compute pops cb_w a round at a time; the
// semaphores start at 0 every block)
void process_row_block(const ColumnDataflowArgs& args, uint32_t block, uint32_t& w_rounds_before_block) {
    auto w_slot_of = [&](uint32_t round) { return (w_rounds_before_block + round) % w_slots; };
    const uint32_t y_address = args.y_address + block * ROW_BLOCK_Y_STRIDE_BYTES;  // (block b's y)
    const uint32_t steps_per_shard = args.steps_per_shard;
    // the column's W stream: W block w of a chunk (gate / up pass w / gate_up_k_steps, step w % gate_up_k_steps; then
    // down) held by row w % engine_rows, multicast down the column into W slot (round % w_slots); every core pushes
    // what lands and credits the slot's next holder. A round's W block (natural index): a gate / up pass's steps in
    // this chip's processing order, then down's
    auto w_block_tiles = [&](uint32_t w_block) {  // (tiles)
        return w_block < gate_up_passes * gate_up_k_steps
                   ? k_step_tiles * args.gate_up_pass_width(w_block / gate_up_k_steps)
                   : down_k_step_tiles *
                         args.down_pass_width((w_block - gate_up_passes * gate_up_k_steps) / down_k_steps_divisor);
    };
    // (a chunk's round -> W block, and my W blocks' offsets in my W-own, tabled once: computed in place, the round's
    // W block (4 runtime divisions - steps_per_shard is a runtime arg) and the offset's walk over my W blocks ran on
    // every pass of the stream's loop below (the prefetch scan, the send's skip, the credit) - ~340 cycles a pass, a W
    // round ~2.7 passes; tabled ~285 (K3 head handshake-only gate / up 48.8 -> 41.1 us, traced head ~-4 us; GLM-5.3
    // -5.6 us))
    static_assert(w_blocks_per_chunk <= 1024, "W rounds a chunk");
    uint16_t round_w_block[w_blocks_per_chunk], my_w_block_offset[w_blocks_per_chunk / engine_rows + 1];
    for (uint32_t round = 0; round < w_blocks_per_chunk; ++round) {
        uint32_t w_block = round;
        if (round < gate_up_passes * gate_up_k_steps) {
            const uint32_t shard_rank = round / (gate_up_passes * steps_per_shard),
                           shard_round = round % (gate_up_passes * steps_per_shard),
                           pass = shard_round / steps_per_shard, step = shard_round % steps_per_shard;
            // (shard-major: shard, pass, step)
            w_block = pass * gate_up_k_steps + args.shard_order(shard_rank) * steps_per_shard + step;
        }
        round_w_block[round] = w_block;
    }
    for (uint32_t my_index = 0, offset = 0; args.row + my_index * engine_rows < w_blocks_per_chunk;
         ++my_index) {  // (my W blocks: row, row + engine_rows, ..)
        my_w_block_offset[my_index] = offset;
        offset += w_block_tiles(args.row + my_index * engine_rows);
    }
    auto w_block_of_round = [&](uint32_t round) -> uint32_t { return round_w_block[round % w_blocks_per_chunk]; };
    const auto y_accessor = TensorAccessor(y_args, y_address, moe_block_tiles::kBf16TileBytes);
    // my W blocks from DRAM, once: my W-own is one contiguous region in one bank (per-bank layout, w_own_offset its
    // offset), read W block by W block in large bursts, every read issued now - the k-th W block's under transaction
    // id min(k, 14) + 1; a W block is sent once its reads landed (the column streams while the weights are still
    // arriving)
    // the router's logits on the engine: the row's first core reads the row's x (row_tiles_per_core row tiles, the
    // chip's K slice) and multicasts it along the row; the column's first core reads the column's W_r^T expert tiles
    // and multicasts them down the column; then the compute's partial logits go to the router cores' HOLD (the main
    // loop below)
    for (uint32_t iteration = 0; iteration < router_iterations;
         ++iteration) {  // (router iterations: my rows iteration x ROUTER_CORE_ROW_TILES ..)
        {
            // K chunks of ROUTER_K_CHUNK_TILES tiles, at most two rounds, a buffer each (no slot reused: no acks): the
            // row's first core's BRISC multicasts the row's x along the row (NOC0), the column's first core multicasts
            // its W_r^T expert tiles down the column (here, NOC1 - same-NoC row and column multicasts stall each
            // other); a lead reads round k_chunk + 1 while round k_chunk's multicast goes; every core pushes a round
            // once its words say it landed
            DeviceZoneScopedN("SE-RTRLD");
            // (the column's W_r^T lead on row NoC x % engine_rows, not row 0: the NOC1 read responses come back along
            // the lead's row (-y, then -x), and 11 leads on one row shared its links - ~3 us a round's reads vs ~0.6
            // for the x leads, who sit on different rows)
            const uint32_t router_w_lead_noc_y = args.row_noc_y(args.column_noc_x % engine_rows);
            const bool is_router_w_lead = args.router_expert_tiles && args.row == args.column_noc_x % engine_rows;
            const auto router_w_accessor =
                TensorAccessor(router_w_args, args.router_w_address, moe_block_tiles::kBf16TileBytes);
            constexpr uint32_t router_k_chunks = ROUTER_K_TILES / ROUTER_K_CHUNK_TILES,
                               router_w_round_bytes = ROUTER_EXPERT_TILES_PER_COLUMN * ROUTER_K_CHUNK_TILES *
                                                      moe_block_tiles::kBf16TileBytes;
            constexpr uint32_t router_rounds =
                router_k_chunks *
                router_iterations;  // (the rounds of every iteration: their counts run on across them)
            // (two buffers: past two rounds a buffer is reused - every receiver acks a round once its compute popped it
            // (its next reserve went through): +1 on the row's x lead and the column's W lead, who wait for all of
            // them)
            for (uint32_t k_chunk = 0; k_chunk < router_k_chunks; ++k_chunk) {
                const uint32_t round = iteration * router_k_chunks + k_chunk;  // (the round over every iteration)
                cb_reserve_back(ROUTER_X_CB, ROUTER_K_CHUNK_TILES * ROUTER_CORE_ROW_TILES);
                cb_reserve_back(ROUTER_W_CB, ROUTER_EXPERT_TILES_PER_COLUMN * ROUTER_K_CHUNK_TILES);
                if (router_rounds > 2 && round >= 2) {
                    noc_semaphore_inc(
                        get_noc_addr(
                            args.router_x_lead_noc_x,
                            args.router_x_lead_noc_y,
                            args.phase_words_address + X_ACKNOWLEDGE_FLAG_ID),
                        1);
                    if (args.has_router_experts) {
                        noc_semaphore_inc(
                            get_noc_addr(
                                args.column_noc_x,
                                router_w_lead_noc_y,
                                args.phase_words_address + W_ACKNOWLEDGE_FLAG_ID),
                            1);
                    }
                }
                if (is_router_w_lead) {  // (the column's first core: the chunk of its expert tiles of W_r^T down the
                                         // column)
                    const uint32_t router_w_buffer_address =
                        args.router_w_buffer +
                        ((block * router_rounds + round) & 1) * router_w_round_bytes;  // (over the call's rounds)
                    if (router_rounds > 2 && round >= 2) {
                        while (*args.phase_word(W_ACKNOWLEDGE_FLAG_ID) < (round - 1) * engine_rows) {
                            invalidate_l1_cache();
                        }
                    }
                    for (uint32_t expert_tile = 0; expert_tile < args.router_expert_tiles; ++expert_tile) {
                        for (uint32_t chunk_k_tile = 0; chunk_k_tile < ROUTER_K_CHUNK_TILES; ++chunk_k_tile) {
                            noc_async_read(
                                router_w_accessor.get_noc_addr(
                                    (args.router_first_expert_tile + expert_tile) * ROUTER_K_TILES +
                                    k_chunk * ROUTER_K_CHUNK_TILES + chunk_k_tile),
                                router_w_buffer_address + (expert_tile * ROUTER_K_CHUNK_TILES + chunk_k_tile) *
                                                              moe_block_tiles::kBf16TileBytes,
                                moe_block_tiles::kBf16TileBytes);
                        }
                    }
                    noc_async_read_barrier();
                    noc_async_write_multicast_loopback_src(
                        router_w_buffer_address,
                        get_noc_multicast_addr(
                            args.column_noc_x,
                            args.column_bottom_noc_y,
                            args.column_noc_x,
                            args.column_top_noc_y,
                            router_w_buffer_address),
                        args.router_expert_tiles * ROUTER_K_CHUNK_TILES * moe_block_tiles::kBf16TileBytes,
                        engine_rows);
                    noc_async_writes_flushed();
                    *args.phase_word(ROUTER_W_READY_FLAG_ID) = round + 1;
                    noc_semaphore_set_multicast_loopback_src(
                        (args.phase_words_address + ROUTER_W_READY_FLAG_ID),
                        get_noc_multicast_addr(
                            args.column_noc_x,
                            args.column_bottom_noc_y,
                            args.column_noc_x,
                            args.column_top_noc_y,
                            (args.phase_words_address + ROUTER_W_READY_FLAG_ID)),
                        engine_rows);
                }
                while (*args.phase_word(ROUTER_X_READY_FLAG_ID) < round + 1) {
                    invalidate_l1_cache();
                }
                cb_push_back(ROUTER_X_CB, ROUTER_K_CHUNK_TILES * ROUTER_CORE_ROW_TILES);
                while (args.has_router_experts && *args.phase_word(ROUTER_W_READY_FLAG_ID) < round + 1) {
                    invalidate_l1_cache();
                }
                cb_push_back(ROUTER_W_CB, ROUTER_EXPERT_TILES_PER_COLUMN * ROUTER_K_CHUNK_TILES);
            }
            if (iteration + 1 == router_iterations) {
                noc_async_writes_flushed();  // (my sets' source words read: re-arm them)
                *args.phase_word(ROUTER_X_READY_FLAG_ID) =
                    0;  // (re-armed: no more sets come this call - the x lead sets from its own word, ROUTER_X_SENT)
                *args.phase_word(ROUTER_W_READY_FLAG_ID) = 0;
                if (router_rounds > 2 && is_router_w_lead) {  // (every ack landed: the last were waited for)
                    noc_semaphore_inc(
                        get_noc_addr(args.phase_words_address + W_ACKNOWLEDGE_FLAG_ID),
                        0u - (router_rounds - 2) * engine_rows);
                    noc_async_atomic_barrier();
                }
            }
        }
        {  // my partial logits (the compute's, a few us later) to their router cores' HOLD - before my W stream starts
            DeviceZoneScopedN("SE-RTROUT");
#ifdef ROUTER_ITERATIONS
            if (iteration > 0) {  // (router core 0's go: every router core and logits port of the chip is done with
                                  // iteration - 1)
                while (*args.phase_word(ROUTER_GO_FLAG_ID) < iteration) {
                    invalidate_l1_cache();
                }
            }
#endif
            cb_wait_front(ROUTER_OUT_CB, ROUTER_EXPERT_TILES_PER_COLUMN * ROUTER_CORE_ROW_TILES);
            // (row tile `row_tile` of mine: my own owner's block into its router core's HOLD, another owner's straight
            // into the port sub-block it leaves from - (NoC, base, count) a row tile; the count router_expert_tiles
            // tiles, after they landed)
            const uint32_t router_out_address = get_read_ptr(ROUTER_OUT_CB);
            for (uint32_t row_tile = 0; row_tile < ROUTER_CORE_ROW_TILES; ++row_tile) {
                const uint32_t call_row_tile =
                    iteration * ROUTER_CORE_ROW_TILES + row_tile;  // (my row tile call_row_tile of the call)
                const uint32_t destination_noc_xy = args.router_destination_noc_xy(call_row_tile),
                               destination_noc_x = destination_noc_xy & 0xFFFF,
                               destination_noc_y = destination_noc_xy >> 16;
                const uint32_t destination_base = args.router_destination_base(call_row_tile),
                               destination_counter = args.router_destination_counter(call_row_tile);
                for (uint32_t expert_tile = 0; expert_tile < args.router_expert_tiles; ++expert_tile) {
                    noc_async_write(
                        router_out_address +
                            (expert_tile * ROUTER_CORE_ROW_TILES + row_tile) * moe_block_tiles::kFp32TileBytes,
                        get_noc_addr(
                            destination_noc_x,
                            destination_noc_y,
                            destination_base +
                                (args.router_first_expert_tile + expert_tile) * moe_block_tiles::kFp32TileBytes),
                        moe_block_tiles::kFp32TileBytes);
                }
                noc_async_write_barrier();  // (landed before the count)
                if (args.router_expert_tiles) {
                    noc_semaphore_inc(
                        get_noc_addr(destination_noc_x, destination_noc_y, destination_counter),
                        args.router_expert_tiles);
                }
            }
            cb_pop_front(ROUTER_OUT_CB, ROUTER_EXPERT_TILES_PER_COLUMN * ROUTER_CORE_ROW_TILES);
        }
    }  // (router iterations)
#ifdef ROUTER_ITERATIONS
    *args.phase_word(ROUTER_GO_FLAG_ID) =
        0;  // (re-armed: router core 0's last go came before my last iteration's sends)
#endif
#ifdef ROUTER_DONE_CREDIT
    {  // my router phase is over - x rows / W_r^T read, the partials out: +1 on every port's "router done" word (the
       // ports credit this chip's arrival zones / tell the latent's senders "started" once every core's is: the
       // router's buffers overlay them - rtr_overlay)
        for (uint32_t port = 0; port < args.ports; ++port) {
            noc_semaphore_inc(get_noc_addr(args.port_noc_x(port), args.port_noc_y(port), args.router_done_word), 1);
        }
    }
#endif
    {  // (after my core's x load: the DRAM serves the x first - the gather is waiting on it)
        while (*semaphore_word(args.x_loaded_semaphore_id) == 0) {
            invalidate_l1_cache();
        }
    }
#ifdef SKIP_PADDING_CHUNKS
    uint32_t live_chunks;  // (a padded forward: the row chunks up to the last with a real row - my BRISC's, in ewords)
    {
        volatile tt_l1_ptr uint32_t* live_chunks_word = args.phase_word(LIVE_CHUNKS_FLAG_ID);
        while ((live_chunks = *live_chunks_word) == 0) {
            invalidate_l1_cache();
        }
    }
#else
    constexpr uint32_t live_chunks = row_chunks;
#endif
    // the weights' DRAM reads go in stream order a window ahead of the sends (w_reads_in_flight_max of my W blocks in
    // flight): spread over gate / up at the rate the stream uses them - a burst of all 35 MB at the start starved the
    // gather
    constexpr uint8_t w_read_noc = noc_index;  // (the W DRAM reads: the NCRISC's own NoC)
    constexpr uint32_t w_reads_in_flight_max = 3;
    constexpr uint32_t w_lookahead_rounds =
        6;  // (the window in rounds: K3 head traced 440 -> 397 us at 6, flat over 5-8)
    uint8_t transaction_id_of[w_blocks_per_chunk / engine_rows + 1];  // (a core's W blocks: every engine_rows-th of
                                                                      // the chunk's w_blocks_per_chunk)
    // (prefetch cursor over the first chunk's rounds; W blocks issued; issued unsent)
    uint32_t prefetch_round = 0, w_blocks_issued = 0, w_blocks_pending = 0;
#ifdef SHARED_EXPERT_W_RING_SLOTS
    // (one chunk: my W-own a ring of SHARED_EXPERT_W_RING_SLOTS W block slots - a W block read into slot (its issue
    // index % SHARED_EXPERT_W_RING_SLOTS) once the W block SHARED_EXPERT_W_RING_SLOTS before it was sent (multicast
    // flushed); the DRAM region keeps every W block)
    uint8_t ring_slot_of[w_blocks_per_chunk / engine_rows + 1];
    uint32_t w_blocks_sent = 0;
#endif
#ifdef SHARED_EXPERT_W_RING_SLOTS
    const uint32_t prefetch_rounds =
        live_chunks * w_blocks_per_chunk;  // (rounds the prefetch walks: every chunk reads its W blocks again)
#else
    constexpr uint32_t prefetch_rounds = w_blocks_per_chunk;  // (the first chunk's: resident after it)
#endif
    auto w_own_offset_of = [&](uint32_t w_block) {  // (w_block: one of mine)
        return static_cast<uint32_t>(my_w_block_offset[w_block / engine_rows]);
    };
    uint32_t rounds_landed = 0;                           // (rounds landed here - the window's base)
    auto prefetch_w_blocks = [&](uint32_t until_round) {  // (issue my W blocks of rounds < until_round, and the window)
        // (the window in the column's rounds, not my W blocks - a W block only once its round is within
        // w_lookahead_rounds of the stream; 3 of mine in flight on every core put ~30 rounds of the whole grid in the
        // DRAM queues at once, and the first-needed W block landed ~50 us after the burst was issued)
        while (prefetch_round < prefetch_rounds &&
               (prefetch_round < until_round ||
                (w_blocks_pending < w_reads_in_flight_max && prefetch_round < rounds_landed + w_lookahead_rounds))) {
            const uint32_t w_block = w_block_of_round(prefetch_round);
            if (w_block % engine_rows != args.row) {
                ++prefetch_round;
                continue;
            }
#ifdef SHARED_EXPERT_W_RING_SLOTS
            if (w_blocks_issued >= w_blocks_sent + SHARED_EXPERT_W_RING_SLOTS) {
                break;  // (its slot still sending)
            }
            const uint32_t w_own_tile_offset = (w_blocks_issued % SHARED_EXPERT_W_RING_SLOTS) * w_slot_tiles;
            ring_slot_of[w_block / engine_rows] = w_blocks_issued % SHARED_EXPERT_W_RING_SLOTS;
#endif
            ++prefetch_round;
            const uint32_t transaction_id = (w_blocks_issued++ % 14) + 1, offset = w_own_offset_of(w_block);
#ifndef SHARED_EXPERT_W_RING_SLOTS
            const uint32_t w_own_tile_offset = offset;
#endif
            transaction_id_of[w_block / engine_rows] = transaction_id;
            noc_async_read_set_trid(transaction_id, w_read_noc);
            noc_async_read(
                get_noc_addr_from_bank_id<true>(
                    args.w_own_bank, args.w_address + args.w_own_offset + offset * w_tile_bytes, w_read_noc),
                args.w_own_address + w_own_tile_offset * w_tile_bytes,
                w_block_tiles(w_block) * w_tile_bytes,
                w_read_noc);
            ++w_blocks_pending;
        }
    };
    const uint32_t w_rounds = live_chunks * w_blocks_per_chunk;  // (W rounds a call)
    // (send, its offset in my W-own, pushed, credited)
    uint32_t send_round = 0, send_offset = 0, pushed_rounds = 0, credited_rounds = w_slots;
    uint32_t slot_reuses[w_slots] = {};  // (a slot's sends past its first: the W-free count it waits for)
    while (send_round < w_rounds && w_block_of_round(send_round) % engine_rows != args.row) {
        ++send_round;  // (skip rounds I do not hold)
    }
    // (down + reduce-scatter) arrivals per direction in zone-slot order: slot i credited (its pages pushed - the
    // word's high half: P, or with compact zones my tiles of its block) once my arrival counter reached its threshold
    // (the low half; 0: nothing comes - a block I hold no tile of); the final sub-block to y
    volatile tt_l1_ptr uint32_t* forward_arrival_counter =
        reinterpret_cast<volatile tt_l1_ptr uint32_t*>(args.forward_arrival_counter_address);
    volatile tt_l1_ptr uint32_t* backward_arrival_counter =
        reinterpret_cast<volatile tt_l1_ptr uint32_t*>(args.backward_arrival_counter_address);
    const uint32_t forward_arrival_slots = args.forward_arrival_slots,
                   backward_arrival_slots = args.backward_arrival_slots, final_tiles = args.final_tiles,
                   final_width = args.final_width;
    // (forward / backward slots credited; final tiles written to y)
    uint32_t forward_slots_done = 0, backward_slots_done = 0, final_tiles_written = 0;
#ifdef REDUCE_SCATTER_FINAL_ROWS
    const uint32_t final_free_credits = live_chunks - 1;
#else
    const uint32_t final_free_credits = row_chunks > 1 ? live_chunks - 1 : 0;
#endif
    // (the thresholds' max; the zone pages my slots take)
    uint32_t forward_threshold_max = 0, backward_threshold_max = 0, forward_zone_pages = 0, backward_zone_pages = 0;
    for (uint32_t slot = 0; slot < forward_arrival_slots; ++slot) {
        const uint32_t arrival = args.forward_arrival(slot);
        forward_threshold_max = (arrival & 0xFFFF) > forward_threshold_max ? (arrival & 0xFFFF) : forward_threshold_max;
        forward_zone_pages += arrival >> 16;
    }
    for (uint32_t slot = 0; slot < backward_arrival_slots; ++slot) {
        const uint32_t arrival = args.backward_arrival(slot);
        backward_threshold_max =
            (arrival & 0xFFFF) > backward_threshold_max ? (arrival & 0xFFFF) : backward_threshold_max;
        backward_zone_pages += arrival >> 16;
    }
    // (zone-free signals sent a direction; final-block-free credits pushed)
    uint32_t forward_zone_free_signals = 0, backward_zone_free_signals = 0, final_free_pushed = 0;
    volatile tt_l1_ptr uint32_t* final_free_counter =
        reinterpret_cast<volatile tt_l1_ptr uint32_t*>(args.final_free_counter_address);
#ifdef LATENT_CB
    // (the latent: once every other chip's slice of my rows has landed (latent_counter: latent_sources arrivals a
    // chunk), the compute's prep may untilize them)
    volatile tt_l1_ptr uint32_t* latent_counter =
        reinterpret_cast<volatile tt_l1_ptr uint32_t*>(args.latent_counter_address);
    const uint32_t latent_chunks =
        args.latent_rows == 0 || args.latent_sources == 0 ? 0 : live_chunks;  // (chunks to signal)
    uint32_t latent_chunks_ready = 0;
#else
    constexpr uint32_t latent_chunks = 0, latent_chunks_ready = 0;
#endif
    const uint32_t prep_units =
        args.prep_units / row_chunks * live_chunks;  // (prep units: chunk-major, as many a chunk)
    const auto dispatch_buffer_accessor =
        TensorAccessor(dispatch_buffer_args, args.dispatch_buffer_address, DISPATCH_PREP_ROW_BYTES);
    // (the last prep unit: the fence's source / rows - mine: nothing else writes them)
    uint32_t fence_source = 0, fence_row = 0, fence_column_offset = 0;
    uint32_t prep_units_written = 0;
    while (send_round < w_rounds || pushed_rounds < w_rounds || credited_rounds < w_rounds ||
           forward_slots_done < live_chunks * forward_arrival_slots ||
           backward_slots_done < live_chunks * backward_arrival_slots ||
           final_tiles_written < live_chunks * final_tiles || final_free_pushed < final_free_credits ||
           latent_chunks_ready < latent_chunks || prep_units_written < prep_units) {
        invalidate_l1_cache();
#ifdef LATENT_CB
        if (latent_chunks_ready < latent_chunks && *latent_counter >= (latent_chunks_ready + 1) * args.latent_sources) {
            cb_reserve_back(LATENT_READY_CB, 1);  // (the compute's prep may untilize the arrivals)
            cb_push_back(LATENT_READY_CB, 1);
            ++latent_chunks_ready;
        }
#endif
        if (prep_units_written < prep_units &&
            cb_pages_available_at_front(
                DISPATCH_PREP_OUT_CB, DISPATCH_PREP_FULL_COLUMN_TILES)) {  // a prep unit's 32 rows to the buffer
            const uint32_t prep_source = get_read_ptr(DISPATCH_PREP_OUT_CB);
            const uint32_t buffer_row = args.prep_unit_row(prep_units_written) + block * ROW_BLOCK_DISPATCH_ROW_STRIDE,
                           column_offset = args.prep_unit_column_offset(prep_units_written),
                           unit_row_bytes = args.prep_unit_row_bytes(prep_units_written);
            // (posted: non-posted DRAM writes on NOC1 - their acks - under the W stream's NOC1 DRAM reads wedged a
            // chip's NOC1, ~1 in 200 calls with the prep in every chunk; landed by the fence at the end:
            // test_posted_fence)
            for (uint32_t i = 0; i < 32; ++i) {
                noc_async_write<NOC_MAX_BURST_SIZE + 1, true, true>(
                    prep_source + i * prep_row_bytes,
                    dispatch_buffer_accessor.get_noc_addr(buffer_row + i) + column_offset,
                    unit_row_bytes);
            }
            noc_async_posted_writes_flushed();  // (the out page may be refilled once they left L1)
            fence_source = prep_source, fence_row = buffer_row, fence_column_offset = column_offset;
            cb_pop_front(DISPATCH_PREP_OUT_CB, DISPATCH_PREP_FULL_COLUMN_TILES);
            ++prep_units_written;
        }
        if (forward_slots_done < live_chunks * forward_arrival_slots) {
            const uint32_t slot = forward_slots_done % forward_arrival_slots,
                           arrival_chunk = forward_slots_done / forward_arrival_slots;
            if (row_chunks > 1 && slot == 0 &&
                arrival_chunk > forward_zone_free_signals) {  // (chunk arrival_chunk's first slot: chunk
                                                              // arrival_chunk - 1's zone all read (popped) - +1 on
                // the zone-free word of the port facing my forward upstream: it credits the upstream's sends)
                if (cb_pages_reservable_at_back(REDUCE_SCATTER_ARRIVAL_FORWARD_CB, REDUCE_SCATTER_ZONE_TILES_FORWARD)) {
                    // (only a core something lands on signals: one with only "nothing comes" slots runs ahead of
                    // the credit - its signals would count for a slower one's)
                    if (forward_threshold_max) {
                        noc_semaphore_inc(
                            get_noc_addr(
                                args.forward_upstream_port & 0xFFFF,
                                args.forward_upstream_port >> 16,
                                args.zone_free_word),
                            1);
                    }
                    ++forward_zone_free_signals;
                }
            } else {
                const uint32_t arrival = args.forward_arrival(slot), threshold = arrival & 0xFFFF,
                               zone_pages = arrival >> 16;
                if (threshold == 0 || *forward_arrival_counter >= arrival_chunk * forward_threshold_max + threshold) {
                    if (zone_pages) {
                        cb_reserve_back(REDUCE_SCATTER_ARRIVAL_FORWARD_CB, zone_pages);
                        cb_push_back(REDUCE_SCATTER_ARRIVAL_FORWARD_CB, zone_pages);
                    }
                    ++forward_slots_done;
                    if ((row_chunks > 1 || row_block::BLOCKS > 1) && slot + 1 == forward_arrival_slots &&
                        REDUCE_SCATTER_ZONE_TILES_FORWARD > forward_zone_pages) {  // (the chunk padded to the whole
                        // zone: the compute pops it all, the next chunk's slots from the zone's start again)
                        cb_reserve_back(
                            REDUCE_SCATTER_ARRIVAL_FORWARD_CB, REDUCE_SCATTER_ZONE_TILES_FORWARD - forward_zone_pages);
                        cb_push_back(
                            REDUCE_SCATTER_ARRIVAL_FORWARD_CB, REDUCE_SCATTER_ZONE_TILES_FORWARD - forward_zone_pages);
                    }
                }
            }
        }
        if (backward_slots_done < live_chunks * backward_arrival_slots) {
            const uint32_t slot = backward_slots_done % backward_arrival_slots,
                           arrival_chunk = backward_slots_done / backward_arrival_slots;
            if (row_chunks > 1 && slot == 0 && arrival_chunk > backward_zone_free_signals) {
                if (cb_pages_reservable_at_back(
                        REDUCE_SCATTER_ARRIVAL_BACKWARD_CB, REDUCE_SCATTER_ZONE_TILES_BACKWARD)) {
                    if (backward_threshold_max) {
                        noc_semaphore_inc(
                            get_noc_addr(
                                args.backward_upstream_port & 0xFFFF,
                                args.backward_upstream_port >> 16,
                                args.zone_free_word),
                            1);
                    }
                    ++backward_zone_free_signals;
                }
            } else {
                const uint32_t arrival = args.backward_arrival(slot), threshold = arrival & 0xFFFF,
                               zone_pages = arrival >> 16;
                if (threshold == 0 || *backward_arrival_counter >= arrival_chunk * backward_threshold_max + threshold) {
                    if (zone_pages) {
                        cb_reserve_back(REDUCE_SCATTER_ARRIVAL_BACKWARD_CB, zone_pages);
                        cb_push_back(REDUCE_SCATTER_ARRIVAL_BACKWARD_CB, zone_pages);
                    }
                    ++backward_slots_done;
                    if ((row_chunks > 1 || row_block::BLOCKS > 1) && slot + 1 == backward_arrival_slots &&
                        REDUCE_SCATTER_ZONE_TILES_BACKWARD > backward_zone_pages) {
                        cb_reserve_back(
                            REDUCE_SCATTER_ARRIVAL_BACKWARD_CB,
                            REDUCE_SCATTER_ZONE_TILES_BACKWARD - backward_zone_pages);
                        cb_push_back(
                            REDUCE_SCATTER_ARRIVAL_BACKWARD_CB,
                            REDUCE_SCATTER_ZONE_TILES_BACKWARD - backward_zone_pages);
                    }
                }
            }
        }
        if (final_tiles_written < live_chunks * final_tiles &&
            cb_pages_available_at_front(REDUCE_SCATTER_OUTPUT_FORWARD_CB, 1)) {
            const uint32_t final_chunk = final_tiles_written / final_tiles,
                           final_tile = final_tiles_written % final_tiles;
#ifdef REDUCE_SCATTER_FINAL_ROWS
            const uint32_t y_tile =
                args.final_row_first_y_tile(final_chunk, final_tile / final_width) + final_tile % final_width;
#elif defined(SHARED_EXPERT_REDUCE_SCATTER_PASSES)
            // (the rows scatter's pass-major sub-block: pass p's final_rows x pass_width tiles after the earlier
            // passes' - widths split_even(final_width, SHARED_EXPERT_REDUCE_SCATTER_PASSES))
            const uint32_t chunk_first_y_tile =
                final_chunk == 0 ? args.final_first_y_tile : args.final_chunk_first_y_tile(final_chunk);
            const uint32_t final_rows = final_tiles / final_width,
                           base_pass_width = final_width / SHARED_EXPERT_REDUCE_SCATTER_PASSES,
                           wider_passes = final_width % SHARED_EXPERT_REDUCE_SCATTER_PASSES;
            uint32_t pass = 0, pass_tile = final_tile, pass_first_column = 0;
            while (pass_tile >= final_rows * (base_pass_width + (pass < wider_passes ? 1 : 0))) {
                pass_tile -= final_rows * (base_pass_width + (pass < wider_passes ? 1 : 0));
                pass_first_column += base_pass_width + (pass < wider_passes ? 1 : 0);
                ++pass;
            }
            const uint32_t pass_width = base_pass_width + (pass < wider_passes ? 1 : 0);
            const uint32_t y_tile = chunk_first_y_tile + (pass_tile / pass_width) * args.final_y_row_stride +
                                    pass_first_column + pass_tile % pass_width;
#else
            const uint32_t chunk_first_y_tile =
                final_chunk == 0 ? args.final_first_y_tile : args.final_chunk_first_y_tile(final_chunk);
            const uint32_t y_tile =
                chunk_first_y_tile + (final_tile / final_width) * args.final_y_row_stride + final_tile % final_width;
#endif
            cb_wait_front(REDUCE_SCATTER_OUTPUT_FORWARD_CB, 1);
            noc_async_write(
                get_read_ptr(REDUCE_SCATTER_OUTPUT_FORWARD_CB),
                y_accessor.get_noc_addr(y_tile),
                moe_block_tiles::kBf16TileBytes);
            noc_async_writes_flushed();  // (the page read out: free for the next final tiles - its ack is the kernel
                                         // end's barrier; a barrier a tile serialized ~35 DRAM round trips behind the
                                         // last final tile)
            cb_pop_front(REDUCE_SCATTER_OUTPUT_FORWARD_CB, 1);
            ++final_tiles_written;
        }
        if (final_free_pushed < final_free_credits &&
            *final_free_counter >= (final_free_pushed + 1) * args.final_block_readers) {  // (the ports read chunk
            // final_free_pushed's final block: the compute may write it again)
            cb_reserve_back(REDUCE_SCATTER_FINAL_FREE_CB, 1);
            cb_push_back(REDUCE_SCATTER_FINAL_FREE_CB, 1);
            ++final_free_pushed;
        }
        if (prefetch_round < prefetch_rounds) {
            prefetch_w_blocks(0);  // (keep the window full)
        }
        if (send_round < w_rounds &&
            (send_round < w_slots || *semaphore_word(w_free_semaphore_id + w_slot_of(send_round)) >=
                                         engine_rows * (slot_reuses[w_slot_of(send_round)] + 1))) {  // my W block:
                                                                                                     // down the column
            const uint32_t w_block = w_block_of_round(send_round), w_tiles = w_block_tiles(w_block),
                           my_index = w_block / engine_rows;  // (my_index: its index among my W blocks)
            send_offset = 0;                                  // (its offset in my W-own: my W blocks before it)
#ifdef SHARED_EXPERT_W_RING_SLOTS
            send_offset = ring_slot_of[my_index] * w_slot_tiles;
#else
            send_offset = w_own_offset_of(w_block);
#endif
            if (send_round < prefetch_rounds) {  // (first chunk, or every chunk with the ring: its reads issued and
                                                 // landed)
                prefetch_w_blocks(send_round + 1);
                noc_async_read_barrier_with_trid(transaction_id_of[my_index], w_read_noc);
                --w_blocks_pending;
            }
            const uint32_t slot_address = args.w_in_address + w_slot_of(send_round) * w_slot_tiles * w_tile_bytes,
                           w_valid_word = args.w_valid_words + (send_round & 3) * 16;
            *reinterpret_cast<volatile tt_l1_ptr uint32_t*>(w_valid_word) = send_round + 1;
            noc_async_write_multicast_loopback_src(
                args.w_own_address + send_offset * w_tile_bytes,
                get_noc_multicast_addr(
                    args.column_noc_x,
                    args.column_bottom_noc_y,
                    args.column_noc_x,
                    args.column_top_noc_y,
                    slot_address),
                w_tiles * w_tile_bytes,
                engine_rows);
            noc_async_writes_flushed();
            noc_semaphore_set_multicast_loopback_src(
                w_valid_word,
                get_noc_multicast_addr(
                    args.column_noc_x,
                    args.column_bottom_noc_y,
                    args.column_noc_x,
                    args.column_top_noc_y,
                    get_semaphore(w_valid_semaphore_id + w_slot_of(send_round))),
                engine_rows);
            noc_async_writes_flushed();
            if (send_round >= w_slots) {
                ++slot_reuses[w_slot_of(send_round)];
            }
            ++send_round;
#ifdef SHARED_EXPERT_W_RING_SLOTS
            ++w_blocks_sent;  // (flushed: its slot may take a later W block)
#endif
            while (send_round < w_rounds && w_block_of_round(send_round) % engine_rows != args.row) {
                ++send_round;
            }
        }
        if (pushed_rounds < w_rounds &&
            (*semaphore_word(w_valid_semaphore_id + w_slot_of(pushed_rounds)) >= pushed_rounds + 1)) {  // W round
                                                                                                        // landed here
            cb_reserve_back(cb_w, w_slot_tiles);
            cb_push_back(cb_w, w_slot_tiles);
            rounds_landed = ++pushed_rounds;
        }
        if (credited_rounds < w_rounds && credited_rounds < pushed_rounds + w_slots &&
            cb_pages_reservable_at_back(cb_w, w_slot_tiles * (credited_rounds - pushed_rounds + 1))) {  // its slot free
            noc_semaphore_inc(
                get_noc_addr(
                    args.column_noc_x,
                    args.row_noc_y(w_block_of_round(credited_rounds) % engine_rows),
                    get_semaphore(w_free_semaphore_id + w_slot_of(credited_rounds))),
                1);
            ++credited_rounds;
        }
    }
#ifdef LATENT_CB
    if constexpr (row_block::BLOCKS > 1) {
        for (; latent_chunks_ready < live_chunks;
             ++latent_chunks_ready) {  // (row blocks: a page a chunk in every block - the compute pops live_chunks -
                                       // fillers past mine)
            cb_reserve_back(LATENT_READY_CB, 1);
            cb_push_back(LATENT_READY_CB, 1);
        }
    }
#endif
    w_rounds_before_block += w_rounds;
    if (prep_units_written) {  // (the prep's fence: a non-posted rewrite of 4 B in every DRAM bank - the last unit's
                               // rows, one a bank - after the posted writes on the same VC: its ack means they all
                               // landed)
        for (uint32_t i = 0; i < NUM_DRAM_BANKS; ++i) {
            noc_async_write(
                fence_source + i * prep_row_bytes,
                dispatch_buffer_accessor.get_noc_addr(fence_row + i) + fence_column_offset,
                4);
        }
    }
    noc_async_write_barrier();
    {  // re-arm my arrival counters (every increment of the call landed: each was waited to its final count)
        if (forward_threshold_max) {
            noc_semaphore_inc(
                get_noc_addr(reinterpret_cast<uint32_t>(forward_arrival_counter)),
                0u - live_chunks * forward_threshold_max);
        }
        if (backward_threshold_max) {
            noc_semaphore_inc(
                get_noc_addr(reinterpret_cast<uint32_t>(backward_arrival_counter)),
                0u - live_chunks * backward_threshold_max);
        }
        if (final_free_credits && args.final_block_readers) {
            noc_semaphore_inc(
                get_noc_addr(reinterpret_cast<uint32_t>(final_free_counter)),
                0u - final_free_credits * args.final_block_readers);
        }
    }
#ifdef LATENT_CB
    if (args.latent_rows && args.latent_sources) {
        noc_semaphore_inc(
            get_noc_addr(reinterpret_cast<uint32_t>(latent_counter)),
            0u - live_chunks * args.latent_sources);  // (re-arm)
    }
#endif
    noc_async_atomic_barrier();
}

void kernel_main() {
    moe_block_args::load();  // (this core's args from the DRAM table; a compute kernel waits for its page)
    const ColumnDataflowArgs args = read_column_dataflow_args();
    uint32_t w_rounds_before_block = 0;  // (a launch starts the count)
    for (uint32_t block = 0; block < row_block::BLOCKS; ++block) {
        row_block::begin(block);
        process_row_block(args, block, w_rounds_before_block);
        row_block::end(block);
    }
}
