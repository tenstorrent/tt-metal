// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

// The shared expert's matmul cores' compute kernel (one kernel a column group): the router's logits matmul, then per
// row chunk gate / up with the activation (h), down + the reduce-scatter's adds, and the dispatch prep's untilize.
#include "ttnn/cpp/ttnn/operations/experimental/moe_block/moe_head/device/kernels/row_block.hpp"
#include "ttnn/cpp/ttnn/operations/experimental/moe_block/moe_head/device/kernels/shared_expert.hpp"
#include "ttnn/cpp/ttnn/operations/experimental/moe_block/moe_head/device/kernels/head_shared_words.hpp"
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
// named compile-time arg "expert_activation", its constants the expert_activation_*_bits named args
#include "ttnn/cpp/ttnn/operations/experimental/moe_block/common/kernels/expert_activation.hpp"
constexpr expert_activation::Kind activation =
    static_cast<expert_activation::Kind>(get_named_compile_time_arg_val("expert_activation"));

// Named compile-time args (head_engine_program.cpp engine(), head_named_args.hpp; the engine's three kernels take the
// same list of the shared expert's values)
namespace names = moe_head_named_args;
// (row tiles of a chunk this core computes)
constexpr uint32_t row_tiles_per_core = get_named_compile_time_arg_val(names::kRowTilesPerCore);
constexpr uint32_t k_step_tiles = get_named_compile_time_arg_val(names::kKStepTiles);  // (K tiles a gate / up step)
// (gate / up passes; latent-down passes after)
constexpr uint32_t gate_up_passes = get_named_compile_time_arg_val(names::kGateUpPasses);
constexpr uint32_t row_chunks = get_named_compile_time_arg_val(names::kRowChunks);
// (h tiles a row of a core, at most)
constexpr uint32_t max_column_pairs = get_named_compile_time_arg_val(names::kMaxColumnPairs);
constexpr uint32_t down_k_step_tiles = get_named_compile_time_arg_val(names::kDownKStepTiles);  // (K tiles a down step)
constexpr uint32_t down_k_steps = get_named_compile_time_arg_val(names::kDownKSteps);
constexpr uint32_t down_passes = get_named_compile_time_arg_val(names::kDownPasses);
// (a W slot: the tiles a step pops)
constexpr uint32_t w_slot_tiles = get_named_compile_time_arg_val(names::kWSlotTiles);
constexpr uint32_t k_shards = get_named_compile_time_arg_val(names::kKShards);  // (K shards: one a TP chip)
constexpr uint32_t k_steps_per_shard = get_named_compile_time_arg_val(names::kKStepsPerShard);
constexpr uint32_t engine_columns = get_named_compile_time_arg_val(names::kEngineColumns);
// the router's logits matmul (router_iterations of it, 1: one)
constexpr uint32_t router_iterations = get_named_compile_time_arg_val(names::kRouterIterations);
constexpr uint32_t router_k_tiles = get_named_compile_time_arg_val(names::kRouterKTiles);
constexpr uint32_t router_k_chunk_tiles = get_named_compile_time_arg_val(names::kRouterKChunkTiles);
constexpr uint32_t router_core_row_tiles = get_named_compile_time_arg_val(names::kRouterCoreRowTiles);
constexpr uint32_t router_row_tiles_per_group = get_named_compile_time_arg_val(names::kRouterRowTilesPerGroup);
constexpr uint32_t router_expert_tiles_per_column = get_named_compile_time_arg_val(names::kRouterExpertTilesPerColumn);
constexpr uint32_t router_x_cb = get_named_compile_time_arg_val(names::kRouterXCb);
constexpr uint32_t router_w_cb = get_named_compile_time_arg_val(names::kRouterWCb);
constexpr uint32_t router_out_cb = get_named_compile_time_arg_val(names::kRouterOutCb);
// the shared expert's DEST tiles and the K shards' fp32 partials' pages
constexpr uint32_t shared_expert_dest_tiles = get_named_compile_time_arg_val(names::kSharedExpertDestTiles);
constexpr uint32_t shared_expert_accumulator_pages =
    get_named_compile_time_arg_val(names::kSharedExpertAccumulatorPages);
// the reduce-scatter's CBs, zones and entries
constexpr uint32_t reduce_scatter_entries = get_named_compile_time_arg_val(names::kReduceScatterEntries);
constexpr uint32_t reduce_scatter_hold_cb = get_named_compile_time_arg_val(names::kReduceScatterHoldCb);
constexpr uint32_t reduce_scatter_final_block_cb = get_named_compile_time_arg_val(names::kReduceScatterFinalBlockCb);
constexpr uint32_t reduce_scatter_arrival_forward_cb =
    get_named_compile_time_arg_val(names::kReduceScatterArrivalForwardCb);
constexpr uint32_t reduce_scatter_arrival_backward_cb =
    get_named_compile_time_arg_val(names::kReduceScatterArrivalBackwardCb);
constexpr uint32_t reduce_scatter_final_counter_cb =
    get_named_compile_time_arg_val(names::kReduceScatterFinalCounterCb);
constexpr uint32_t reduce_scatter_output_forward_cb =
    get_named_compile_time_arg_val(names::kReduceScatterOutputForwardCb);
constexpr uint32_t reduce_scatter_pass_sync_cb = get_named_compile_time_arg_val(names::kReduceScatterPassSyncCb);
constexpr uint32_t reduce_scatter_final_free_cb = get_named_compile_time_arg_val(names::kReduceScatterFinalFreeCb);
constexpr uint32_t reduce_scatter_zone_tiles_forward =
    get_named_compile_time_arg_val(names::kReduceScatterZoneTilesForward);
constexpr uint32_t reduce_scatter_zone_tiles_backward =
    get_named_compile_time_arg_val(names::kReduceScatterZoneTilesBackward);
// the latent (K3's all-to-all; 1: on): its CBs, pages and the gate / up passes before its down passes
constexpr bool latent_enabled = get_named_compile_time_arg_val(names::kLatentEnabled) != 0;
constexpr uint32_t latent_cb = get_named_compile_time_arg_val(names::kLatentCb);
constexpr uint32_t latent_ready_cb = get_named_compile_time_arg_val(names::kLatentReadyCb);
constexpr uint32_t latent_pages = get_named_compile_time_arg_val(names::kLatentPages);
constexpr uint32_t latent_max_column_tiles = get_named_compile_time_arg_val(names::kLatentMaxColumnTiles);
constexpr uint32_t shared_expert_gate_up_passes = get_named_compile_time_arg_val(names::kSharedExpertGateUpPasses);
// the dispatch prep (the row chunks' units: 1: every chunk's in the last chunk's op)
constexpr bool dispatch_prep_last = get_named_compile_time_arg_val(names::kDispatchPrepLast) != 0;
constexpr uint32_t dispatch_prep_unit_cb = get_named_compile_time_arg_val(names::kDispatchPrepUnitCb);
constexpr uint32_t dispatch_prep_out_cb = get_named_compile_time_arg_val(names::kDispatchPrepOutCb);
constexpr uint32_t dispatch_prep_full_column_tiles =
    get_named_compile_time_arg_val(names::kDispatchPrepFullColumnTiles);
constexpr uint32_t dispatch_prep_block_column_tiles =
    get_named_compile_time_arg_val(names::kDispatchPrepBlockColumnTiles);
// a padded forward: the row chunks up to the last with a real row (my BRISC's word, in ewords)
constexpr bool skip_padding_chunks = get_named_compile_time_arg_val(names::kSkipPaddingChunks) != 0;
constexpr uint32_t skip_padding_args_offset = get_named_compile_time_arg_val(names::kSkipPaddingArgsOffset);
constexpr uint32_t live_chunks_flag_id = get_named_compile_time_arg_val(names::kLiveChunksFlagId);
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
    uint32_t hold_offset;         // its held partial in reduce_scatter_hold_cb
    uint32_t final_block_offset;  // its tiles in reduce_scatter_final_block_cb (relayed)
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
                                                         // row r's pack base (kHold: reduce_scatter_hold_cb)
    const volatile tt_l1_ptr uint32_t* entries;          // [reduce_scatter_entries][6]: ReduceScatterEntry
    // common [skip_padding_args_offset]: the router-phase words (live chunks word; skip_padding_chunks only)
    uint32_t phase_words_address;

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
    constexpr uint32_t expert_tiles = router_expert_tiles_per_column;
    // (router_iterations of them: router_core_row_tiles of my rows each)
    for (uint32_t iteration = 0; iteration < router_iterations; ++iteration) {
        // (a K chunk's expert_tiles x row tiles partials in DEST, then fp32 into router_out_cb with the packer's L1
        // accumulation)
        reconfig_data_format(router_x_cb, router_w_cb);
        pack_reconfig_data_format(router_out_cb);
        // (row groups of router_row_tiles_per_group rows: expert_tiles x router_row_tiles_per_group tiles in DEST - all
        // rows at once where they fit; W_r^T's tiles transposed, one row tile a block)
        matmul_block_init_at<router_logits_fidelity>(
            router_w_cb,
            router_x_cb,
            /*transpose=*/1,
            router_row_tiles_per_group,
            /*row_tiles=*/1,
            router_k_chunk_tiles);
        cb_reserve_back(router_out_cb, expert_tiles * router_core_row_tiles);
        for (uint32_t k_chunk = 0; k_chunk < router_k_tiles / router_k_chunk_tiles; ++k_chunk) {
            cb_wait_front(router_x_cb, router_k_chunk_tiles * router_core_row_tiles);
            cb_wait_front(router_w_cb, expert_tiles * router_k_chunk_tiles);
            pack_reconfig_l1_acc(k_chunk > 0 ? 1 : 0);
            for (uint32_t row_group = 0; row_group < router_core_row_tiles; row_group += router_row_tiles_per_group) {
                tile_regs_acquire();
                for (uint32_t expert = 0; expert < expert_tiles; ++expert) {
                    for (uint32_t k = 0; k < router_k_chunk_tiles; ++k) {
                        matmul_block_at<router_logits_fidelity>(
                            router_w_cb,
                            router_x_cb,
                            expert * router_k_chunk_tiles + k,
                            k * router_core_row_tiles + row_group,
                            expert * router_row_tiles_per_group,
                            /*transpose=*/1,
                            router_row_tiles_per_group,
                            /*row_tiles=*/1,
                            router_k_chunk_tiles);
                    }
                }
                tile_regs_commit();
                tile_regs_wait();
                for (uint32_t expert = 0; expert < expert_tiles; ++expert) {
                    for (uint32_t row = 0; row < router_row_tiles_per_group; ++row) {
                        pack_tile<true>(
                            expert * router_row_tiles_per_group + row,
                            router_out_cb,
                            expert * router_core_row_tiles + row_group + row);
                    }
                }
                tile_regs_release();
            }
            cb_pop_front(router_x_cb, router_k_chunk_tiles * router_core_row_tiles);
            cb_pop_front(router_w_cb, expert_tiles * router_k_chunk_tiles);
        }
        pack_reconfig_l1_acc(0);
        cb_push_back(router_out_cb, expert_tiles * router_core_row_tiles);
        reconfig_data_format(cb_w, cb_x);  // (back to gate / up's formats)
    }
}

// One row block of this kernel's work: kernel_main below runs it once per block (row_block.hpp).
// output_tiles_pushed: reduce_scatter_output_forward_cb tiles pushed over the whole call - its blocks end at multiples
// of shared_expert_dest_tiles (the NCRISC reads it page by page, pack_tile does not wrap): a block's adds continue
// where the last block's left it, as a chunk's do after the last chunk's.
void process_row_block(const SharedExpertComputeArgs& args, uint32_t& output_tiles_pushed) {
    router_logits();
    expert_activation::init<activation>();
    uint32_t live_chunks = row_chunks;
    if constexpr (skip_padding_chunks) {
        // (a padded forward: the row chunks up to the last with a real row - the BRISC's word)
        volatile tt_l1_ptr uint32_t* live_chunks_word =
            reinterpret_cast<volatile tt_l1_ptr uint32_t*>(args.phase_words_address + live_chunks_flag_id);
        while ((live_chunks = *live_chunks_word) == 0) {
            invalidate_l1_cache();
        }
    }
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
                cb_reserve_back(cb_partial_sums, shared_expert_accumulator_pages);
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
                    if constexpr (latent_enabled) {
                        // (a latent-down pass: no activation, the chip's latent columns packed bf16 to latent_cb)
                        if (pass >= shared_expert_gate_up_passes) {
                            pack_reconfig_data_format(latent_cb);
                            for (uint32_t row = 0; row < row_tiles_per_core; ++row) {
                                for (uint32_t tile = 0; tile < pass_width; ++tile) {
                                    pack_tile<true>(
                                        row * pass_width + tile,
                                        latent_cb,
                                        row * latent_max_column_tiles + latent_column_offset + tile);
                                }
                            }
                            tile_regs_release();
                            latent_column_offset += pass_width;
                            continue;
                        }
                    }
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
            if (k_shards > 1 && partial_pages_pushed < shared_expert_accumulator_pages) {
                cb_push_back(cb_partial_sums, shared_expert_accumulator_pages - partial_pages_pushed);
                cb_wait_front(cb_partial_sums, shared_expert_accumulator_pages - partial_pages_pushed);
                cb_pop_front(cb_partial_sums, shared_expert_accumulator_pages - partial_pages_pushed);
            }
            if constexpr (latent_enabled) {
                // (the latent: the BRISC hands it to the ports, the NCRISC writes mine; with chunks a chunk's latent
                // pages, the next chunk's packed past them)
                cb_reserve_back(latent_cb, latent_pages);
                cb_push_back(latent_cb, latent_pages);
            }
        }
        {
            DeviceZoneScopedN("SE-DOWNRS");
            // (down + reduce-scatter) the host's program of ops: a down pass - all of K in DEST over h-gathered and the
            // streamed W_down; row r's tiles packed at its base: into reduce_scatter_hold_cb (fp32: an entry that adds
            // an arrival here) or reduce_scatter_final_block_cb (bf16: a start entry, exchanged as is); a hand-off -
            // entry e's start sub-block is complete (a reduce_scatter_final_counter_cb page: the BRISC's hand-off); a
            // finish - the held partial + the arrival(s) in fp32 DEST, packed bf16 into the final block (relay: handed
            // off) or reduce_scatter_output_forward_cb (fin: y). Partials accumulate fp32 in L1; only the exchanged
            // sub-blocks are bf16.
            // (h-gathered columns waited for: the first pass follows the exchange's rounds)
            uint32_t down_passes_done = 0, h_columns_waited = 0;
            if (row_chunks > 1 && chunk > 0) {  // (the ports read the last chunk's final block: mine to write again -
                                                // the NCRISC's credit)
                cb_wait_front(reduce_scatter_final_free_cb, 1);
                cb_pop_front(reduce_scatter_final_free_cb, 1);
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
                        const uint32_t out_cb = (base_word >> moe_head_down_pass_words::kHoldShift)
                                                    ? reduce_scatter_hold_cb
                                                    : reduce_scatter_final_block_cb;
                        const uint32_t base = base_word & moe_head_down_pass_words::kBaseMask;
                        pack_reconfig_data_format(out_cb);
                        for (uint32_t tile = 0; tile < pass_width; ++tile) {
                            pack_tile<true>(row * pass_width + tile, out_cb, base + tile);
                        }
                    }
                    tile_regs_release();
                    // (pass done: the UNPACK of a later finish may read its held tiles)
                    cb_reserve_back(reduce_scatter_pass_sync_cb, 1);
                    cb_push_back(reduce_scatter_pass_sync_cb, 1);
                    ++down_passes_done;
                } else if (kind == OperationKind::HandOff) {
                    cb_reserve_back(reduce_scatter_final_counter_cb, 1);
                    cb_push_back(reduce_scatter_final_counter_cb, 1);
                } else if (kind == OperationKind::DispatchPrep) {
                    // dispatch prep: my rows of the chip's owned tokens (x own / the latent) untilized into
                    // dispatch_prep_out_cb, a unit (n tiles of one row tile, contiguous in L1 at dispatch_prep_unit_cb
                    // + offset) at a time; the NCRISC writes its rows into the dispatch buffer. After the last pass:
                    // under the reduce-scatter's waits for arrivals.
                    DeviceZoneScopedN("SE-PREP");
                    const uint32_t prep_table = index;
                    // (dispatch_prep_last, with chunks: every chunk's units in the last chunk's op - no prep writes
                    // under the W re-reads)
                    const uint32_t units_per_chunk =
                        dispatch_prep_last ? (chunk + 1 == live_chunks ? args.prep_units_per_chunk(prep_table) : 0)
                                           : args.prep_units_per_chunk(prep_table);
                    const uint32_t first_unit = dispatch_prep_last ? 0 : chunk * units_per_chunk,
                                   end_unit = dispatch_prep_last ? live_chunks * units_per_chunk
                                                                 : (chunk + 1) * units_per_chunk;
                    uint32_t unit_cb_read_pointer = 0;
                    UNPACK((unit_cb_read_pointer = get_local_cb_interface(dispatch_prep_unit_cb).fifo_rd_ptr));
                    [[maybe_unused]] uint32_t latent_chunks_ready = 0;  // (chunks' latent arrivals waited for)
                    if (units_per_chunk) {
                        // (the last pass left srcA at the weights' format)
                        reconfig_data_format_srca(dispatch_prep_unit_cb);
                        pack_reconfig_data_format(dispatch_prep_out_cb);
                        pack_untilize_init<dispatch_prep_block_column_tiles, dispatch_prep_full_column_tiles>(
                            dispatch_prep_unit_cb, dispatch_prep_out_cb);
                    }
                    for (uint32_t unit = first_unit; unit < end_unit; ++unit) {
                        const uint32_t unit_offset = args.prep_unit_offset(prep_table, unit),
                                       unit_flags = args.prep_unit_flags(prep_table, unit);
                        const uint32_t unit_tiles = unit_flags & 0xFFFF;
                        // (latent_enabled: a unit of the latent's arrivals: once they all landed)
                        if (latent_enabled && (unit_flags >> 16) && latent_chunks_ready < unit / units_per_chunk + 1) {
                            latent_chunks_ready = unit / units_per_chunk + 1;
                            cb_wait_front(latent_ready_cb, latent_chunks_ready);
                        }
                        cb_reserve_back(dispatch_prep_out_cb, dispatch_prep_full_column_tiles);
                        for (uint32_t block = 0; block * dispatch_prep_block_column_tiles < unit_tiles; ++block) {
                            // (the unit's block in L1: the CB's read pointer is in 16 B words)
                            UNPACK(
                                (get_local_cb_interface(dispatch_prep_unit_cb).fifo_rd_ptr =
                                     unit_cb_read_pointer + (unit_offset + block * dispatch_prep_block_column_tiles *
                                                                               moe_block_tiles::kBf16TileBytes) /
                                                                16));
                            pack_untilize_block<dispatch_prep_block_column_tiles, dispatch_prep_full_column_tiles>(
                                dispatch_prep_unit_cb, 1, dispatch_prep_out_cb, block);
                        }
                        cb_push_back(dispatch_prep_out_cb, dispatch_prep_full_column_tiles);
                    }
                    if (units_per_chunk) {
                        pack_untilize_uninit(dispatch_prep_out_cb);
                    }
                    UNPACK((get_local_cb_interface(dispatch_prep_unit_cb).fifo_rd_ptr = unit_cb_read_pointer));
                } else {  // (OperationKind::Finish) entry `index`: its held partial + the arrival(s)
                    const ReduceScatterEntry entry = args.entry(index);
                    // (the arrivals' zone offsets, tiles - every slot before them pushed with them)
                    const bool has_forward = entry.mode & 1, has_backward = entry.mode & 2, is_output = entry.mode & 4;
                    cb_wait_front(reduce_scatter_pass_sync_cb, down_passes_done);
                    if (has_forward) {
                        cb_wait_front(reduce_scatter_arrival_forward_cb, entry.forward_slot + entry.tiles);
                    }
                    if (has_backward) {
                        cb_wait_front(reduce_scatter_arrival_backward_cb, entry.backward_slot + entry.tiles);
                    }
                    const uint32_t first_arrival_cb =
                        has_forward ? reduce_scatter_arrival_forward_cb : reduce_scatter_arrival_backward_cb;
                    const uint32_t first_arrival_slot = has_forward ? entry.forward_slot : entry.backward_slot;
                    const uint32_t out_cb =
                        is_output ? reduce_scatter_output_forward_cb : reduce_scatter_final_block_cb;
                    pack_reconfig_data_format(out_cb);
                    for (uint32_t first = 0, tiles = 0; first < entry.tiles; first += tiles) {
                        const uint32_t limit =
                            is_output ? shared_expert_dest_tiles - output_tiles_pushed % shared_expert_dest_tiles
                                      : shared_expert_dest_tiles;
                        tiles = entry.tiles - first < limit ? entry.tiles - first : limit;
                        if (is_output) {
                            output_tiles_pushed += tiles;
                        }
                        tile_regs_acquire();
                        reconfig_data_format(reduce_scatter_hold_cb, first_arrival_cb);
                        add_init(reduce_scatter_hold_cb, first_arrival_cb);
                        for (uint32_t j = 0; j < tiles; ++j) {
                            add_tiles(
                                reduce_scatter_hold_cb,
                                first_arrival_cb,
                                entry.hold_offset + first + j,
                                first_arrival_slot + first + j,
                                j);
                        }
                        if (has_forward && has_backward) {  // (the second arrival: DEST -> srcA at fp32 width, + bwd)
                            reconfig_data_format(reduce_scatter_hold_cb, reduce_scatter_arrival_backward_cb);
                            add_reuse_dest_init<EltwiseBinaryReuseDestType::DEST_TO_SRCA>(
                                reduce_scatter_arrival_backward_cb);
                            for (uint32_t j = 0; j < tiles; ++j) {
                                add_reuse_dest_tiles<EltwiseBinaryReuseDestType::DEST_TO_SRCA>(
                                    reduce_scatter_arrival_backward_cb, entry.backward_slot + first + j, j);
                            }
                        }
                        tile_regs_commit();
                        if (is_output) {
                            cb_reserve_back(reduce_scatter_output_forward_cb, tiles);
                        }
                        tile_regs_wait();
                        for (uint32_t j = 0; j < tiles; ++j) {
                            if (is_output) {
                                pack_tile(j, reduce_scatter_output_forward_cb);
                            } else {
                                pack_tile<true>(j, reduce_scatter_final_block_cb, entry.final_block_offset + first + j);
                            }
                        }
                        tile_regs_release();
                        if (is_output) {
                            cb_push_back(reduce_scatter_output_forward_cb, tiles);
                        }
                    }
                    if (!is_output) {
                        cb_reserve_back(reduce_scatter_final_counter_cb, 1);
                        cb_push_back(reduce_scatter_final_counter_cb, 1);
                    }
                }
            }
            cb_pop_front(cb_h_gathered, h_gathered_tiles);
            // (the chunk's syncs and arrival zones popped: the next chunk's - or the next row block's - from their
            // start)
            if constexpr (row_chunks > 1 || row_block::BLOCKS > 1) {
                cb_wait_front(reduce_scatter_pass_sync_cb, down_passes_done);
                cb_pop_front(reduce_scatter_pass_sync_cb, down_passes_done);
                uint32_t arrival_directions = 0;
                for (uint32_t entry = 0; entry < reduce_scatter_entries; ++entry) {
                    arrival_directions |= args.entry_mode(entry);  // (the entries: TP of them)
                }
                if (arrival_directions & 1) {
                    cb_wait_front(reduce_scatter_arrival_forward_cb, reduce_scatter_zone_tiles_forward);
                    cb_pop_front(reduce_scatter_arrival_forward_cb, reduce_scatter_zone_tiles_forward);
                }
                if (arrival_directions & 2) {
                    cb_wait_front(reduce_scatter_arrival_backward_cb, reduce_scatter_zone_tiles_backward);
                    cb_pop_front(reduce_scatter_arrival_backward_cb, reduce_scatter_zone_tiles_backward);
                }
            }
        }
    }
    if constexpr (latent_enabled && row_block::BLOCKS > 1) {
        // (row blocks: the skipped chunks' latent pages as padding - the BRISC pops the whole region)
        if (live_chunks < row_chunks) {
            cb_reserve_back(latent_cb, (row_chunks - live_chunks) * latent_pages);
            cb_push_back(latent_cb, (row_chunks - live_chunks) * latent_pages);
        }
        // (row blocks: the NCRISC's page a chunk, filled up to live_chunks - the next block's from 0)
        cb_wait_front(latent_ready_cb, live_chunks);
        cb_pop_front(latent_ready_cb, live_chunks);
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
        skip_padding_chunks ? moe_block_args::common(skip_padding_args_offset) : 0u,
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
