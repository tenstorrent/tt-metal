// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

// The shared expert's matmul cores' compute kernel (one kernel a column group): the router's logits matmul, then per
// row chunk gate / up with the activation (h), down + the reduce-scatter's adds, and the dispatch prep's untilize.
#include "ttnn/cpp/ttnn/operations/experimental/moe_block/moe_head/device/kernels/row_block.hpp"
#include "ttnn/cpp/ttnn/operations/experimental/moe_block/moe_head/device/kernels/shared_expert.hpp"
#include "ttnn/cpp/ttnn/operations/experimental/moe_block/common/kernels/tile_bytes.hpp"

#include <cstdint>
#include "api/compute/compute_kernel_api.h"
#include "api/compute/matmul.h"
#include "api/compute/pack.h"
#include "api/compute/cb_api.h"
#include "api/compute/tile_move_copy.h"
#include "api/compute/reconfig_data_format.h"
#include "api/compute/eltwise_binary.h"
#include "api/compute/pack_untilize.h"
#include "api/debug/kernel_profiler.hpp"

// The GLU activation of gate / up: common/kernels/expert_activation.hpp (shared with the routed experts) - its kind the
// named compile-time arg "expert_activation", its constants the EXPERT_ACTIVATION_* defines
#include "ttnn/cpp/ttnn/operations/experimental/moe_block/common/kernels/expert_activation.hpp"
constexpr expert_activation::Kind activation =
    static_cast<expert_activation::Kind>(get_named_compile_time_arg_val("expert_activation"));

// Compile-time args (head_engine_program.cpp, the shared expert's compute kernel)
constexpr uint32_t row_tiles_per_core = get_compile_time_arg_val(0);  // (row tiles of a chunk this core computes)
constexpr uint32_t k_step_tiles = get_compile_time_arg_val(1);        // (K tiles a gate / up step)
constexpr uint32_t gate_up_passes = get_compile_time_arg_val(2);      // (gate / up passes; latent-down passes after)
constexpr uint32_t row_chunks = get_compile_time_arg_val(3);
constexpr uint32_t max_column_pairs = get_compile_time_arg_val(4);   // (h tiles a row of a core, at most)
constexpr uint32_t down_k_step_tiles = get_compile_time_arg_val(5);  // (K tiles a down step)
constexpr uint32_t down_k_steps = get_compile_time_arg_val(6);
constexpr uint32_t down_passes = get_compile_time_arg_val(7);
constexpr uint32_t w_slot_tiles = get_compile_time_arg_val(8);  // (a W slot: the tiles a step pops)
constexpr uint32_t k_shards = get_compile_time_arg_val(9);      // (K shards: one a TP chip)
constexpr uint32_t k_steps_per_shard = get_compile_time_arg_val(10);
constexpr uint32_t engine_columns = get_compile_time_arg_val(11);
// (h of the whole row, gathered: [column][row tile][max_column_pairs])
constexpr uint32_t h_gathered_tiles = engine_columns * row_tiles_per_core * max_column_pairs;

// Circular buffers
constexpr uint32_t cb_x = 0;             // x steps (the row's x, gathered along the row)
constexpr uint32_t cb_w = 1;             // W steps (gate / up, then down; the column's W stream)
constexpr uint32_t cb_h_local = 2;       // this core's h columns of the chunk
constexpr uint32_t cb_h_gathered = 3;    // h of every column of the row (the down's input)
constexpr uint32_t cb_partial_sums = 5;  // fp32 gate / up partials of the K shards (packer L1 accumulation)

// The matmuls' fidelities, named at every call (shared_expert.hpp matmul_block_init_at / matmul_block_at): the
// router's logits at HiFi3 (the routing picks need it); the shared expert's gate / up / down at the kernel's configured
// fidelity (the compute config's math_fidelity: MoEDtypes::shared_fidelity, HiFi2 by default)
constexpr MathFidelity router_logits_fidelity = MathFidelity::HiFi3;
constexpr MathFidelity shared_expert_fidelity = configured_fidelity;

// The core's program of ops (ReduceScatterPlan::compute_args, ChipProgram::prep_compute): an op is kind | index << 8
enum class OperationKind : uint32_t {
    DownPass = 0,      // index: the down pass
    HandOff = 1,       // a reduce-scatter sub-block's start part complete (the BRISC's hand-off)
    Finish = 2,        // index: the reduce-scatter entry
    DispatchPrep = 3,  // index: the prep units' table (its arg index in the full list)
};

// A reduce-scatter entry (6 args a finish: ReduceScatterPlan::compute_args)
struct ReduceScatterEntry {
    uint32_t tiles;               // its tiles
    uint32_t mode;                // bit 0 a forward arrival, bit 1 a backward arrival, bit 2 the output - else relayed
    uint32_t forward_slot;        // the forward arrival's zone offset (tiles)
    uint32_t backward_slot;       // the backward arrival's zone offset (tiles)
    uint32_t hold_offset;         // its held partial in REDUCE_SCATTER_HOLD_CB
    uint32_t final_block_offset;  // its tiles in REDUCE_SCATTER_FINAL_BLOCK_CB (relayed)
};

// This core's args, read once (kernel_main): the column group's block (head_engine_program.cpp: the gate / up pass
// widths, the down pass widths, the h columns' first K tiles, the shared expert's tiles), then the core's program
// (prep_compute: its ops, the down passes' tables, the reduce-scatter entries, the dispatch prep's units) - the page
// the core's BRISC hands over from its block of the DRAM table (head_workload.cpp head_args_to_dram). The tables are
// read in place: views of the args in L1.
struct SharedExpertComputeArgs {
    const volatile tt_l1_ptr uint32_t* all_args;                // the full list (a prep op's table index is absolute)
    const volatile tt_l1_ptr uint32_t* gate_up_pass_widths;     // [gate_up_passes]: the column's pass widths (tiles)
    const volatile tt_l1_ptr uint32_t* h_column_first_k_tiles;  // [engine_columns + 1]: h column c's first K tile of
                                                                // the down (the last: the shared expert's tiles)
    uint32_t num_operations;
    const volatile tt_l1_ptr uint32_t* operations;       // [num_operations]: kind | index << 8
    const volatile tt_l1_ptr uint32_t* down_pass_table;  // [down_passes][1 + row_tiles_per_core]: the pass width, then
                                                         // row r's pack base (bit 31: REDUCE_SCATTER_HOLD_CB)
    const volatile tt_l1_ptr uint32_t* entries;          // [REDUCE_SCATTER_ENTRIES][6]: ReduceScatterEntry
#ifdef SKIP_PADDING_CHUNKS
    uint32_t phase_words_address;  // common [SKIP_PADDING_ARGS_OFFSET]: the router-phase words (live chunks word)
#endif

    uint32_t gate_up_pass_width(uint32_t pass) const { return gate_up_pass_widths[pass]; }
    uint32_t h_column_first_k_tile(uint32_t column) const { return h_column_first_k_tiles[column]; }
    uint32_t operation(uint32_t index) const { return operations[index]; }
    uint32_t down_pass_width(uint32_t pass) const { return down_pass_table[pass * (1 + row_tiles_per_core)]; }
    uint32_t down_pass_row_base(uint32_t pass, uint32_t row) const {
        return down_pass_table[pass * (1 + row_tiles_per_core) + 1 + row];
    }
    ReduceScatterEntry entry(uint32_t index) const {
        const volatile tt_l1_ptr uint32_t* words = entries + 6 * index;
        return ReduceScatterEntry{words[0], words[1], words[2], words[3], words[4], words[5]};
    }
    uint32_t entry_mode(uint32_t index) const { return entries[6 * index + 1]; }
    // a dispatch prep op's table at arg `table`: the units a chunk, then (L1 offset, tiles | latent << 16) a unit
    uint32_t prep_units_per_chunk(uint32_t table) const { return all_args[table]; }
    uint32_t prep_unit_offset(uint32_t table, uint32_t unit) const { return all_args[table + 1 + 2 * unit]; }
    uint32_t prep_unit_flags(uint32_t table, uint32_t unit) const { return all_args[table + 2 + 2 * unit]; }
};

// The router's logits: my expert tiles x my row tiles over the chip's K slice (every chain starts here)
void router_logits() {
    DeviceZoneScopedN("SE-RTRMM");
    // (every core the same count; past the experts: unused tiles)
    constexpr uint32_t expert_tiles = ROUTER_EXPERT_TILES_PER_COLUMN;
#ifdef ROUTER_ITERATIONS
    constexpr uint32_t router_iterations = ROUTER_ITERATIONS;  // (ROUTER_CORE_ROW_TILES of my rows each)
#else
    constexpr uint32_t router_iterations = 1;
#endif
    for (uint32_t iteration = 0; iteration < router_iterations; ++iteration) {
        // (a K chunk's expert_tiles x row tiles partials in DEST, then fp32 into ROUTER_OUT_CB with the packer's L1
        // accumulation)
        reconfig_data_format(ROUTER_X_CB, ROUTER_W_CB);
        pack_reconfig_data_format(ROUTER_OUT_CB);
        // (row groups of ROUTER_ROW_TILES_PER_GROUP rows: expert_tiles x ROUTER_ROW_TILES_PER_GROUP tiles in DEST - all
        // rows at once where they fit; W_r^T's tiles transposed, one row tile a block)
        matmul_block_init_at<router_logits_fidelity>(
            ROUTER_W_CB,
            ROUTER_X_CB,
            /*transpose=*/1,
            ROUTER_ROW_TILES_PER_GROUP,
            /*row_tiles=*/1,
            ROUTER_K_CHUNK_TILES);
        cb_reserve_back(ROUTER_OUT_CB, expert_tiles * ROUTER_CORE_ROW_TILES);
        for (uint32_t k_chunk = 0; k_chunk < ROUTER_K_TILES / ROUTER_K_CHUNK_TILES; ++k_chunk) {
            cb_wait_front(ROUTER_X_CB, ROUTER_K_CHUNK_TILES * ROUTER_CORE_ROW_TILES);
            cb_wait_front(ROUTER_W_CB, expert_tiles * ROUTER_K_CHUNK_TILES);
            pack_reconfig_l1_acc(k_chunk > 0 ? 1 : 0);
            for (uint32_t row_group = 0; row_group < ROUTER_CORE_ROW_TILES; row_group += ROUTER_ROW_TILES_PER_GROUP) {
                tile_regs_acquire();
                for (uint32_t expert = 0; expert < expert_tiles; ++expert) {
                    for (uint32_t k = 0; k < ROUTER_K_CHUNK_TILES; ++k) {
                        matmul_block_at<router_logits_fidelity>(
                            ROUTER_W_CB,
                            ROUTER_X_CB,
                            expert * ROUTER_K_CHUNK_TILES + k,
                            k * ROUTER_CORE_ROW_TILES + row_group,
                            expert * ROUTER_ROW_TILES_PER_GROUP,
                            /*transpose=*/1,
                            ROUTER_ROW_TILES_PER_GROUP,
                            /*row_tiles=*/1,
                            ROUTER_K_CHUNK_TILES);
                    }
                }
                tile_regs_commit();
                tile_regs_wait();
                for (uint32_t expert = 0; expert < expert_tiles; ++expert) {
                    for (uint32_t row = 0; row < ROUTER_ROW_TILES_PER_GROUP; ++row) {
                        pack_tile<true>(
                            expert * ROUTER_ROW_TILES_PER_GROUP + row,
                            ROUTER_OUT_CB,
                            expert * ROUTER_CORE_ROW_TILES + row_group + row);
                    }
                }
                tile_regs_release();
            }
            cb_pop_front(ROUTER_X_CB, ROUTER_K_CHUNK_TILES * ROUTER_CORE_ROW_TILES);
            cb_pop_front(ROUTER_W_CB, expert_tiles * ROUTER_K_CHUNK_TILES);
        }
        pack_reconfig_l1_acc(0);
        cb_push_back(ROUTER_OUT_CB, expert_tiles * ROUTER_CORE_ROW_TILES);
        reconfig_data_format(cb_w, cb_x);  // (back to gate / up's formats)
    }
}

// One row block of this kernel's work: kernel_main below runs it once per block (row_block.hpp).
// output_tiles_pushed: REDUCE_SCATTER_OUTPUT_FORWARD_CB tiles pushed over the whole call - its blocks end at multiples
// of SHARED_EXPERT_DEST_TILES (the NCRISC reads it page by page, pack_tile does not wrap): a block's adds continue
// where the last block's left it, as a chunk's do after the last chunk's.
void process_row_block(const SharedExpertComputeArgs& args, uint32_t& output_tiles_pushed) {
    router_logits();
    expert_activation::init<activation>();
#ifdef SKIP_PADDING_CHUNKS
    uint32_t live_chunks;  // (a padded forward: the row chunks up to the last with a real row - the BRISC's word)
    {
        volatile tt_l1_ptr uint32_t* live_chunks_word =
            reinterpret_cast<volatile tt_l1_ptr uint32_t*>(args.phase_words_address + LIVE_CHUNKS_FLAG_ID);
        while ((live_chunks = *live_chunks_word) == 0) {
            invalidate_l1_cache();
        }
    }
#else
    constexpr uint32_t live_chunks = row_chunks;
#endif
    for (uint32_t chunk = 0; chunk < live_chunks; ++chunk) {
        // gate / up: per pass all of K in DEST, the activation, h packed once
        {
            DeviceZoneScopedN("SE-GU");
            if (row_chunks > 1 && chunk > 0) {
                reconfig_data_format(cb_w, cb_x);  // (the last chunk's down / reduce-scatter left other formats)
            }
            cb_reserve_back(cb_h_local, row_tiles_per_core * max_column_pairs);
            // shard-major: shard k (own first, then as each lands) through every pass; a pass's fp32 partial is packed
            // onto its fixed cb_partial_sums slot (row tiles x pass width tiles at the pass's offset) with the packer's
            // L1 accumulation - shard 0 writes it, shards 1 .. k_shards - 2 add onto it in L1 - and only the last shard
            // brings the sum back into DEST (unpacked straight to DEST, fp32) under its own matmuls; then the
            // activation and h. Shard k_shards - 2 pushes each pass's slot as it is done (the write pointer then sits
            // at the next pass's slot); the chunk's tail is pushed and popped as padding, so every chunk starts at the
            // region's start
            if (k_shards > 1) {
                cb_reserve_back(cb_partial_sums, SHARED_EXPERT_ACCUMULATOR_PAGES);
            }
            // (cb_partial_sums pages pushed this chunk - the pack offsets are relative to the write pointer)
            uint32_t partial_pages_pushed = 0;
            for (uint32_t shard = 0; shard < k_shards; ++shard) {
                uint32_t h_column_offset = 0, latent_column_offset = 0, partial_offset = 0;
                for (uint32_t pass = 0; pass < gate_up_passes; ++pass) {
                    const uint32_t pass_width = args.gate_up_pass_width(pass);
                    const uint32_t pass_tiles = row_tiles_per_core * pass_width;
                    tile_regs_acquire();
                    if (shard + 1 == k_shards && shard > 0) {
                        cb_wait_front(cb_partial_sums, pass_tiles);
                        reconfig_data_format_srca(cb_w, cb_partial_sums);
                        copy_init(cb_partial_sums);
                        for (uint32_t tile = 0; tile < pass_tiles; ++tile) {
                            copy_tile(cb_partial_sums, tile, tile);
                        }
                        cb_pop_front(cb_partial_sums, pass_tiles);
                        reconfig_data_format_srca(cb_partial_sums, cb_w);
                    }
                    matmul_block_init_at<shared_expert_fidelity>(
                        cb_x, cb_w, /*transpose=*/0, pass_width, row_tiles_per_core, k_step_tiles);
                    for (uint32_t step = 0; step < k_steps_per_shard; ++step) {
                        cb_wait_front(cb_x, row_tiles_per_core * k_step_tiles);
                        cb_wait_front(cb_w, w_slot_tiles);
                        for (uint32_t k = 0; k < k_step_tiles; ++k) {
                            matmul_block_at<shared_expert_fidelity>(
                                cb_x,
                                cb_w,
                                k,
                                k * pass_width,
                                0,
                                /*transpose=*/false,
                                pass_width,
                                row_tiles_per_core,
                                k_step_tiles);
                        }
                        cb_pop_front(cb_x, row_tiles_per_core * k_step_tiles);
                        cb_pop_front(cb_w, w_slot_tiles);
                    }
                    tile_regs_commit();
                    if (shard + 1 < k_shards) {
                        tile_regs_wait();
                        pack_reconfig_data_format(cb_partial_sums);
                        pack_reconfig_l1_acc(shard > 0 ? 1 : 0);
                        for (uint32_t tile = 0; tile < pass_tiles; ++tile) {
                            pack_tile<true>(tile, cb_partial_sums, partial_offset + tile - partial_pages_pushed);
                        }
                        pack_reconfig_l1_acc(0);
                        tile_regs_release();
                        partial_offset += pass_tiles;
                        if (shard + 2 == k_shards) {  // (complete but for the last shard's: the last shard may read it)
                            cb_push_back(cb_partial_sums, pass_tiles);
                            partial_pages_pushed += pass_tiles;
                        }
                        continue;
                    }
                    tile_regs_wait();
#ifdef LATENT_CB
                    // (a latent-down pass: no activation, the chip's latent columns packed bf16 to LATENT_CB)
                    if (pass >= SHARED_EXPERT_GATE_UP_PASSES) {
                        pack_reconfig_data_format(LATENT_CB);
                        for (uint32_t row = 0; row < row_tiles_per_core; ++row) {
                            for (uint32_t tile = 0; tile < pass_width; ++tile) {
                                pack_tile<true>(
                                    row * pass_width + tile,
                                    LATENT_CB,
                                    row * LATENT_MAX_COLUMN_TILES + latent_column_offset + tile);
                            }
                        }
                        tile_regs_release();
                        latent_column_offset += pass_width;
                        continue;
                    }
#endif
                    pack_reconfig_data_format(cb_h_local);
                    // the activation on the PACK thread's SFPU, (gate, up) tile pairs in place
                    PACK(TT_SETC16(DEST_TARGET_REG_CFG_MATH_Offset_ADDR32, ckernel::packer::get_packer_dest_offset()));
                    for (uint32_t tile = 0; tile < pass_tiles; tile += 2) {
                        expert_activation::tile_pair<activation>(tile);
                    }
                    PACK(TTI_STALLWAIT(p_stall::STALL_PACK, p_stall::WAIT_SFPU));
                    for (uint32_t row = 0; row < row_tiles_per_core; ++row) {
                        for (uint32_t tile = 0; tile < pass_width; tile += 2) {
                            pack_tile<true>(
                                row * pass_width + tile,
                                cb_h_local,
                                row * max_column_pairs + h_column_offset + tile / 2);
                        }
                    }
                    tile_regs_release();
                    h_column_offset += pass_width / 2;
                }
            }
            cb_push_back(cb_h_local, row_tiles_per_core * max_column_pairs);
            // (the region's tail: padding, so the next chunk starts at its start)
            if (k_shards > 1 && partial_pages_pushed < SHARED_EXPERT_ACCUMULATOR_PAGES) {
                cb_push_back(cb_partial_sums, SHARED_EXPERT_ACCUMULATOR_PAGES - partial_pages_pushed);
                cb_wait_front(cb_partial_sums, SHARED_EXPERT_ACCUMULATOR_PAGES - partial_pages_pushed);
                cb_pop_front(cb_partial_sums, SHARED_EXPERT_ACCUMULATOR_PAGES - partial_pages_pushed);
            }
#ifdef LATENT_CB
            // (the latent: the BRISC hands it to the ports, the NCRISC writes mine; with chunks a chunk's latent pages,
            // the next chunk's packed past them)
            cb_reserve_back(LATENT_CB, LATENT_PAGES);
            cb_push_back(LATENT_CB, LATENT_PAGES);
#endif
        }
        {
            DeviceZoneScopedN("SE-DOWNRS");
            // (down + reduce-scatter) the host's program of ops: a down pass - all of K in DEST over h-gathered and the
            // streamed W_down; row r's tiles packed at its base: into REDUCE_SCATTER_HOLD_CB (fp32: an entry that adds
            // an arrival here) or REDUCE_SCATTER_FINAL_BLOCK_CB (bf16: a start entry, exchanged as is); a hand-off -
            // entry e's start sub-block is complete (a REDUCE_SCATTER_FINAL_COUNTER_CB page: the BRISC's hand-off); a
            // finish - the held partial + the arrival(s) in fp32 DEST, packed bf16 into the final block (relay: handed
            // off) or REDUCE_SCATTER_OUTPUT_FORWARD_CB (fin: y). Partials accumulate fp32 in L1; only the exchanged
            // sub-blocks are bf16.
            // (h-gathered columns waited for: the first pass follows the exchange's rounds)
            uint32_t down_passes_done = 0, h_columns_waited = 0;
            if (row_chunks > 1 && chunk > 0) {  // (the ports read the last chunk's final block: mine to write again -
                                                // the NCRISC's credit)
                cb_wait_front(REDUCE_SCATTER_FINAL_FREE_CB, 1);
                cb_pop_front(REDUCE_SCATTER_FINAL_FREE_CB, 1);
            }
            for (uint32_t operation_index = 0; operation_index < args.num_operations; ++operation_index) {
                const uint32_t operation = args.operation(operation_index);
                const OperationKind kind = static_cast<OperationKind>(operation & 0xFF);
                const uint32_t index = operation >> 8;
                if (kind == OperationKind::DownPass) {
                    const uint32_t down_pass = index;
                    const uint32_t pass_width = args.down_pass_width(down_pass);
                    reconfig_data_format(cb_w, cb_h_gathered);
                    matmul_block_init_at<shared_expert_fidelity>(
                        cb_h_gathered, cb_w, /*transpose=*/0, pass_width, row_tiles_per_core, max_column_pairs);
                    uint32_t h_column = 0;
                    tile_regs_acquire();
                    for (uint32_t step = 0; step < down_k_steps; ++step) {
                        cb_wait_front(cb_w, w_slot_tiles);
                        for (uint32_t k = 0; k < down_k_step_tiles; ++k) {
                            const uint32_t k_tile = step * down_k_step_tiles + k;
                            while (k_tile >= args.h_column_first_k_tile(h_column + 1)) {
                                ++h_column;
                            }
                            if (h_column >= h_columns_waited) {
                                h_columns_waited = h_column + 1;
                                cb_wait_front(cb_h_gathered, h_columns_waited * row_tiles_per_core * max_column_pairs);
                            }
                            const uint32_t h_tile = h_column * row_tiles_per_core * max_column_pairs + k_tile -
                                                    args.h_column_first_k_tile(h_column);
                            matmul_block_at<shared_expert_fidelity>(
                                cb_h_gathered,
                                cb_w,
                                h_tile,
                                k * pass_width,
                                0,
                                /*transpose=*/false,
                                pass_width,
                                row_tiles_per_core,
                                max_column_pairs);
                        }
                        cb_pop_front(cb_w, w_slot_tiles);
                    }
                    tile_regs_commit();
                    tile_regs_wait();
                    for (uint32_t row = 0; row < row_tiles_per_core; ++row) {
                        const uint32_t base_word = args.down_pass_row_base(down_pass, row);
                        const uint32_t out_cb =
                            (base_word >> 31) ? REDUCE_SCATTER_HOLD_CB : REDUCE_SCATTER_FINAL_BLOCK_CB;
                        const uint32_t base = base_word & 0x7FFFFFFF;
                        pack_reconfig_data_format(out_cb);
                        for (uint32_t tile = 0; tile < pass_width; ++tile) {
                            pack_tile<true>(row * pass_width + tile, out_cb, base + tile);
                        }
                    }
                    tile_regs_release();
                    // (pass done: the UNPACK of a later finish may read its held tiles)
                    cb_reserve_back(REDUCE_SCATTER_PASS_SYNC_CB, 1);
                    cb_push_back(REDUCE_SCATTER_PASS_SYNC_CB, 1);
                    ++down_passes_done;
                } else if (kind == OperationKind::HandOff) {
                    cb_reserve_back(REDUCE_SCATTER_FINAL_COUNTER_CB, 1);
                    cb_push_back(REDUCE_SCATTER_FINAL_COUNTER_CB, 1);
                } else if (kind == OperationKind::DispatchPrep) {
                    // dispatch prep: my rows of the chip's owned tokens (x own / the latent) untilized into
                    // DISPATCH_PREP_OUT_CB, a unit (n tiles of one row tile, contiguous in L1 at DISPATCH_PREP_UNIT_CB
                    // + offset) at a time; the NCRISC writes its rows into the dispatch buffer. After the last pass:
                    // under the reduce-scatter's waits for arrivals.
                    DeviceZoneScopedN("SE-PREP");
                    const uint32_t prep_table = index;
#ifdef DISPATCH_PREP_LAST
                    // (with chunks: every chunk's units in the last chunk's op - no prep writes under the W re-reads)
                    const uint32_t units_per_chunk =
                        chunk + 1 == live_chunks ? args.prep_units_per_chunk(prep_table) : 0;
                    const uint32_t first_unit = 0, end_unit = live_chunks * units_per_chunk;
#else
                    const uint32_t units_per_chunk = args.prep_units_per_chunk(prep_table);
                    const uint32_t first_unit = chunk * units_per_chunk, end_unit = (chunk + 1) * units_per_chunk;
#endif
                    uint32_t unit_cb_read_pointer = 0;
                    UNPACK((unit_cb_read_pointer = get_local_cb_interface(DISPATCH_PREP_UNIT_CB).fifo_rd_ptr));
                    [[maybe_unused]] uint32_t latent_chunks_ready = 0;  // (chunks' latent arrivals waited for)
                    if (units_per_chunk) {
                        // (the last pass left srcA at the weights' format)
                        reconfig_data_format_srca(DISPATCH_PREP_UNIT_CB);
                        pack_reconfig_data_format(DISPATCH_PREP_OUT_CB);
                        pack_untilize_init<DISPATCH_PREP_BLOCK_COLUMN_TILES, DISPATCH_PREP_FULL_COLUMN_TILES>(
                            DISPATCH_PREP_UNIT_CB, DISPATCH_PREP_OUT_CB);
                    }
                    for (uint32_t unit = first_unit; unit < end_unit; ++unit) {
                        const uint32_t unit_offset = args.prep_unit_offset(prep_table, unit),
                                       unit_flags = args.prep_unit_flags(prep_table, unit);
                        const uint32_t unit_tiles = unit_flags & 0xFFFF;
#ifdef LATENT_READY_CB
                        // (a unit of the latent's arrivals: once they all landed)
                        if ((unit_flags >> 16) && latent_chunks_ready < unit / units_per_chunk + 1) {
                            latent_chunks_ready = unit / units_per_chunk + 1;
                            cb_wait_front(LATENT_READY_CB, latent_chunks_ready);
                        }
#endif
                        cb_reserve_back(DISPATCH_PREP_OUT_CB, DISPATCH_PREP_FULL_COLUMN_TILES);
                        for (uint32_t block = 0; block * DISPATCH_PREP_BLOCK_COLUMN_TILES < unit_tiles; ++block) {
                            // (the unit's block in L1: the CB's read pointer is in 16 B words)
                            UNPACK(
                                (get_local_cb_interface(DISPATCH_PREP_UNIT_CB).fifo_rd_ptr =
                                     unit_cb_read_pointer + (unit_offset + block * DISPATCH_PREP_BLOCK_COLUMN_TILES *
                                                                               moe_block_tiles::kBf16TileBytes) /
                                                                16));
                            pack_untilize_block<DISPATCH_PREP_BLOCK_COLUMN_TILES, DISPATCH_PREP_FULL_COLUMN_TILES>(
                                DISPATCH_PREP_UNIT_CB, 1, DISPATCH_PREP_OUT_CB, block);
                        }
                        cb_push_back(DISPATCH_PREP_OUT_CB, DISPATCH_PREP_FULL_COLUMN_TILES);
                    }
                    if (units_per_chunk) {
                        pack_untilize_uninit(DISPATCH_PREP_OUT_CB);
                    }
                    UNPACK((get_local_cb_interface(DISPATCH_PREP_UNIT_CB).fifo_rd_ptr = unit_cb_read_pointer));
                } else {  // (OperationKind::Finish) entry `index`: its held partial + the arrival(s)
                    const ReduceScatterEntry entry = args.entry(index);
                    // (the arrivals' zone offsets, tiles - every slot before them pushed with them)
                    const bool has_forward = entry.mode & 1, has_backward = entry.mode & 2, is_output = entry.mode & 4;
                    cb_wait_front(REDUCE_SCATTER_PASS_SYNC_CB, down_passes_done);
                    if (has_forward) {
                        cb_wait_front(REDUCE_SCATTER_ARRIVAL_FORWARD_CB, entry.forward_slot + entry.tiles);
                    }
                    if (has_backward) {
                        cb_wait_front(REDUCE_SCATTER_ARRIVAL_BACKWARD_CB, entry.backward_slot + entry.tiles);
                    }
                    const uint32_t first_arrival_cb =
                        has_forward ? REDUCE_SCATTER_ARRIVAL_FORWARD_CB : REDUCE_SCATTER_ARRIVAL_BACKWARD_CB;
                    const uint32_t first_arrival_slot = has_forward ? entry.forward_slot : entry.backward_slot;
                    const uint32_t out_cb =
                        is_output ? REDUCE_SCATTER_OUTPUT_FORWARD_CB : REDUCE_SCATTER_FINAL_BLOCK_CB;
                    pack_reconfig_data_format(out_cb);
                    for (uint32_t first = 0, tiles = 0; first < entry.tiles; first += tiles) {
                        const uint32_t limit =
                            is_output ? SHARED_EXPERT_DEST_TILES - output_tiles_pushed % SHARED_EXPERT_DEST_TILES
                                      : SHARED_EXPERT_DEST_TILES;
                        tiles = entry.tiles - first < limit ? entry.tiles - first : limit;
                        if (is_output) {
                            output_tiles_pushed += tiles;
                        }
                        tile_regs_acquire();
                        reconfig_data_format(REDUCE_SCATTER_HOLD_CB, first_arrival_cb);
                        add_init(REDUCE_SCATTER_HOLD_CB, first_arrival_cb);
                        for (uint32_t j = 0; j < tiles; ++j) {
                            add_tiles(
                                REDUCE_SCATTER_HOLD_CB,
                                first_arrival_cb,
                                entry.hold_offset + first + j,
                                first_arrival_slot + first + j,
                                j);
                        }
                        if (has_forward && has_backward) {  // (the second arrival: DEST -> srcA at fp32 width, + bwd)
                            reconfig_data_format(REDUCE_SCATTER_HOLD_CB, REDUCE_SCATTER_ARRIVAL_BACKWARD_CB);
                            add_reuse_dest_init<EltwiseBinaryReuseDestType::DEST_TO_SRCA>(
                                REDUCE_SCATTER_ARRIVAL_BACKWARD_CB);
                            for (uint32_t j = 0; j < tiles; ++j) {
                                add_reuse_dest_tiles<EltwiseBinaryReuseDestType::DEST_TO_SRCA>(
                                    REDUCE_SCATTER_ARRIVAL_BACKWARD_CB, entry.backward_slot + first + j, j);
                            }
                        }
                        tile_regs_commit();
                        if (is_output) {
                            cb_reserve_back(REDUCE_SCATTER_OUTPUT_FORWARD_CB, tiles);
                        }
                        tile_regs_wait();
                        for (uint32_t j = 0; j < tiles; ++j) {
                            if (is_output) {
                                pack_tile(j, REDUCE_SCATTER_OUTPUT_FORWARD_CB);
                            } else {
                                pack_tile<true>(j, REDUCE_SCATTER_FINAL_BLOCK_CB, entry.final_block_offset + first + j);
                            }
                        }
                        tile_regs_release();
                        if (is_output) {
                            cb_push_back(REDUCE_SCATTER_OUTPUT_FORWARD_CB, tiles);
                        }
                    }
                    if (!is_output) {
                        cb_reserve_back(REDUCE_SCATTER_FINAL_COUNTER_CB, 1);
                        cb_push_back(REDUCE_SCATTER_FINAL_COUNTER_CB, 1);
                    }
                }
            }
            cb_pop_front(cb_h_gathered, h_gathered_tiles);
            // (the chunk's syncs and arrival zones popped: the next chunk's - or the next row block's - from their
            // start)
            if constexpr (row_chunks > 1 || row_block::BLOCKS > 1) {
                cb_wait_front(REDUCE_SCATTER_PASS_SYNC_CB, down_passes_done);
                cb_pop_front(REDUCE_SCATTER_PASS_SYNC_CB, down_passes_done);
                uint32_t arrival_directions = 0;
                for (uint32_t entry = 0; entry < REDUCE_SCATTER_ENTRIES; ++entry) {
                    arrival_directions |= args.entry_mode(entry);  // (the entries: TP of them)
                }
                if (arrival_directions & 1) {
                    cb_wait_front(REDUCE_SCATTER_ARRIVAL_FORWARD_CB, REDUCE_SCATTER_ZONE_TILES_FORWARD);
                    cb_pop_front(REDUCE_SCATTER_ARRIVAL_FORWARD_CB, REDUCE_SCATTER_ZONE_TILES_FORWARD);
                }
                if (arrival_directions & 2) {
                    cb_wait_front(REDUCE_SCATTER_ARRIVAL_BACKWARD_CB, REDUCE_SCATTER_ZONE_TILES_BACKWARD);
                    cb_pop_front(REDUCE_SCATTER_ARRIVAL_BACKWARD_CB, REDUCE_SCATTER_ZONE_TILES_BACKWARD);
                }
            }
        }
    }
    if constexpr (row_block::BLOCKS > 1) {
#ifdef LATENT_CB
        // (row blocks: the skipped chunks' latent pages as padding - the BRISC pops the whole region)
        if (live_chunks < row_chunks) {
            cb_reserve_back(LATENT_CB, (row_chunks - live_chunks) * LATENT_PAGES);
            cb_push_back(LATENT_CB, (row_chunks - live_chunks) * LATENT_PAGES);
        }
#endif
#ifdef LATENT_READY_CB
        // (row blocks: the NCRISC's page a chunk, filled up to live_chunks - the next block's from 0)
        cb_wait_front(LATENT_READY_CB, live_chunks);
        cb_pop_front(LATENT_READY_CB, live_chunks);
#endif
    }
}

// This core's args (SharedExpertComputeArgs), once: the views' places in the full list
SharedExpertComputeArgs read_shared_expert_compute_args() {
    constexpr uint32_t h_column_table_at = gate_up_passes + down_passes;
    constexpr uint32_t operations_at = h_column_table_at + engine_columns + 1;  // (past the shared expert's tiles)
    const uint32_t num_operations = moe_block_args::arg(operations_at);
    const uint32_t down_pass_table_at = operations_at + 1 + num_operations;
    const uint32_t entries_at = down_pass_table_at + down_passes * (1 + row_tiles_per_core);
    return SharedExpertComputeArgs{
        moe_block_args::arg_pointer(0),
        moe_block_args::arg_pointer(0),
        moe_block_args::arg_pointer(h_column_table_at),
        num_operations,
        moe_block_args::arg_pointer(operations_at + 1),
        moe_block_args::arg_pointer(down_pass_table_at),
        moe_block_args::arg_pointer(entries_at),
#ifdef SKIP_PADDING_CHUNKS
        moe_block_args::common(SKIP_PADDING_ARGS_OFFSET),
#endif
    };
}

void kernel_main() {
    moe_block_args::load();  // (this core's args: the page the BRISC hands over)
    const SharedExpertComputeArgs args = read_shared_expert_compute_args();
    // (once: every reconfiguration after this is an explicit *_init / reconfig_data_format call)
    compute_kernel_hw_startup<SrcOrder::Reverse>(cb_x, cb_w, cb_h_local);
    uint32_t output_tiles_pushed = 0;
    for (uint32_t block = 0; block < row_block::BLOCKS; ++block) {
        row_block::begin(block);
        process_row_block(args, output_tiles_pushed);
        row_block::end(block);
    }
}
