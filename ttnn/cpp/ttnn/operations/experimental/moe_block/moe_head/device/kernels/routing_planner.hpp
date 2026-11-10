// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

// The routing exchange's DRAM tensors and its planner (routing_planner::plan), parametrized by the ROUTING_PLANNER_*
// defines.
//   * routing_accessors (every routing kernel: the planner's and the exchange's senders);
//   * ROUTING_PLANNER (the host defines it on the kernels that run a planner instance: router_input_feeder,
//     router_output_writer, routing_planner_core): the padded chunk's args and metadata read (PADDED_CHUNK), then the
//     planner - routing_planner::read_planner_args once in kernel_main, routing_planner::plan per row block. Row blocks
//     in one program (ROUTING_PLANNER_PER_BLOCK, row_block.hpp): one planner, its tokens' layout and instance bases
//     picked by the block (an earlier block's plans no token: ROUTING_PLANNER_IDLE_*). ROUTING_PLANNER_NONE: no
//     instance on this kernel's cores - the same names, no planner code.
#pragma once

#include <cstdint>
#include "api/tensor/tensor_accessor.h"

// The routing kernels' DRAM tensors' accessor args, after the kernel's own compile-time args from
// ROUTING_ACCESSORS_COMPILE_TIME_ARGS_OFFSET on (head_chip_program.hpp: ChipProgram::routing_accessors), in this order.
// The kernels pass their own page sizes (the plan's 4 KB, a slot, a counts row), which may differ from the tensor's.
namespace routing_accessors {
constexpr auto plan = TensorAccessorArgs<ROUTING_ACCESSORS_COMPILE_TIME_ARGS_OFFSET>();
constexpr auto counts = TensorAccessorArgs<plan.next_compile_time_args_offset()>();
constexpr auto regions = TensorAccessorArgs<counts.next_compile_time_args_offset()>();
constexpr auto token_index = TensorAccessorArgs<regions.next_compile_time_args_offset()>();
constexpr auto table = TensorAccessorArgs<token_index.next_compile_time_args_offset()>();
constexpr auto slots = TensorAccessorArgs<table.next_compile_time_args_offset()>();  // the exchange's DRAM slots
constexpr auto start = TensorAccessorArgs<slots.next_compile_time_args_offset()>();  // a padded chunk's actual_start
constexpr auto end = TensorAccessorArgs<start.next_compile_time_args_offset()>();    // its actual_end
}  // namespace routing_accessors

#if defined(ROUTING_PLANNER)
// (every kernel's args: moe_block_args, read once in kernel_main; row_block::is_last for the per-block planner)
#include "ttnn/cpp/ttnn/operations/experimental/moe_block/moe_head/device/kernels/row_block.hpp"

#if defined(PADDED_CHUNK)
#include "ttnn/cpp/ttnn/operations/experimental/moe_block/moe_head/device/kernels/padded_rows.hpp"
// PADDED_CHUNK (a padded forward), read on device every call (a trace replays with the values of the moment):
//  * moe_head(..., actual_start=, actual_end=): one chunk, main's chunked-prefill metadata scalars - two one-element
//    uint32 DRAM tensors, the chunk's first real token's absolute position and one past its last (common [2], [3]);
//  * PADDED_FORWARD_METADATA, moe_head(..., metadata=): the forward's metadata words (common/prefill/prefill_chunk.py,
//    one uint32 DRAM tensor, common [2]) - PADDED_FORWARD_CHUNKS (forward_chunks) chunks packed in the forward: at 1
//    (slot, start, end), above {num_chunks, 0} then PADDED_FORWARD_CHUNKS (slot, start, end) records (past num_chunks
//    unused, empty).
// Each SP row holds every chunk's rows_per_chunk = tokens_per_sp_row / PADDED_FORWARD_CHUNKS rows back to back, each
// chunk laid out as main's KV-pad-aware rotated block-cyclic layout over the SP rows (moe_padding_config's writer,
// update_padded_kv_cache): an SP row's real rows of chunk k are a prefix of its block, their count main's closed form.
// Within the SP row, TP chip p owns rows p S .. (p + 1) S (S: owner_rows). A token of this chip is real when its row is
// (PaddedRows::real): the planner counts, stages and plans only those, and the route words of the rest are zero - a
// padding token is not routed, dispatched, computed or combined.
// The router / planner kernels' common args [2] .. [11] (head_router_program.cpp ChipProgram::router), read once in
// kernel_main (read_router_padding_args(moe_block_args::common_pointer(2))):
struct RouterPaddingArgs {
    PaddedForwardArgs forward;       // [2] .. [6]: the scalars' / metadata's addresses, my SP row, SP rows, its tokens
    uint32_t first_row;              // [7] my first row in my SP row (p S; with row blocks the block's first row + p S)
    uint32_t owner_rows;             // [8] S: the rows a TP chip owns in a row block of the SP row
    uint32_t block_rows;             // [9] a row block's rows of the SP row
    uint32_t blocks;                 // [10] the row blocks
    uint32_t first_block_first_row;  // [11] the first row block's first row of mine
};
FORCE_INLINE RouterPaddingArgs read_router_padding_args(const volatile tt_l1_ptr uint32_t* args) {
    return RouterPaddingArgs{read_padded_forward_args(args), args[5], args[6], args[7], args[8], args[9]};
}
// issue only - noc_async_read_barrier + invalidate_l1_cache before reading them: the metadata words into metadata_l1,
// or actual_start's 16 B into metadata_l1 and actual_end's into metadata_l1 + 64 (metadata_l1: padded_metadata_slot)
FORCE_INLINE void padded_metadata_read(const PaddedForwardArgs& args, uint32_t metadata_l1) {
#ifdef PADDED_FORWARD_METADATA
    constexpr uint32_t bytes = PADDED_FORWARD_CHUNKS == 1 ? 16 : (4 * (2 + 3 * PADDED_FORWARD_CHUNKS) + 15) / 16 * 16;
    noc_async_read(
        TensorAccessor(routing_accessors::start, args.metadata_address, 64).get_noc_addr(0), metadata_l1, bytes);
#else
    noc_async_read(
        TensorAccessor(routing_accessors::start, args.metadata_address, 64).get_noc_addr(0), metadata_l1, 16);
    noc_async_read(
        TensorAccessor(routing_accessors::end, args.actual_end_address, 64).get_noc_addr(0), metadata_l1 + 64, 16);
#endif
}
#endif  // PADDED_CHUNK

#if defined(ROUTING_PLANNER_NONE)
// (no planner instance on this kernel's cores: no planner code)
namespace routing_planner {
struct PlannerArgs {};
FORCE_INLINE PlannerArgs read_planner_args(const volatile tt_l1_ptr uint32_t*, const volatile tt_l1_ptr uint32_t*) {
    return PlannerArgs{};
}
FORCE_INLINE void plan(const PlannerArgs&, uint32_t) {}
}  // namespace routing_planner
#else
#include "api/debug/kernel_profiler.hpp"
#include "ttnn/cpp/ttnn/operations/experimental/moe_block/common/kernels/combine_plan_words.hpp"

namespace routing_planner {

// The model and the exchange (the ROUTING_PLANNER_* defines: head_router_program.cpp planner_defines)
constexpr uint32_t top_k = ROUTING_PLANNER_TOP_K, experts_per_chip = ROUTING_PLANNER_EXPERTS_PER_CHIP;
constexpr uint32_t sp_rows = ROUTING_PLANNER_SP_ROWS, tp_columns = ROUTING_PLANNER_TP_COLUMNS,
                   num_chips = sp_rows * tp_columns;
constexpr uint32_t accumulators = ROUTING_PLANNER_ACCUMULATORS;        // (token t's home accumulator: t % accumulators)
constexpr uint32_t links = ROUTING_PLANNER_LINKS;                      // (fabric links a direction)
constexpr uint32_t tokens_per_page = ROUTING_PLANNER_TOKENS_PER_PAGE;  // (routing rows a DRAM page)
constexpr uint32_t instances = ROUTING_PLANNER_INSTANCES;              // (planner instances of the chip)
constexpr uint32_t global_experts = num_chips * experts_per_chip;
constexpr uint32_t header_bytes = ROUTING_PLANNER_HEADER_BYTES;                          // (a packet's header, padded)
constexpr uint32_t expert_count_block_bytes = ROUTING_PLANNER_EXPERT_COUNT_BLOCK_BYTES;  // (per-expert uint16 counts)
constexpr uint32_t instance_block_bytes = ROUTING_PLANNER_INSTANCE_BLOCK_BYTES;  // (an instance's expert prefixes)
constexpr uint32_t slot_bytes = ROUTING_PLANNER_SLOT_BYTES;                      // (a DRAM slot: a packet's capacity)
// (a token's routing row: top_k uint16 expert ids, then top_k bf16 weights; a page of them, its L1 stride)
constexpr uint32_t row_bytes = 4 * top_k, page_bytes = tokens_per_page * row_bytes,
                   page_stride = (page_bytes + 63) / 64 * 64;
constexpr uint32_t owned_accumulators_max = (accumulators + instances - 1) / instances;  // accumulators a core owns
constexpr uint32_t align_64(uint32_t value) { return (value + 63) / 64 * 64; }
// (destinations an instance owns the header totals of: destination % instances == instance)
constexpr uint32_t owned_destinations_max = (num_chips + instances - 1) / instances;
// The tokens' layout of an instance's region (offsets from its base): tokens (ROUTING_PLANNER_TOKENS),
// rows_capacity / entries_capacity (an instance's entries, worst case). Row blocks in one program
// (ROUTING_PLANNER_PER_BLOCK): an earlier block's planner (ROUTING_PLANNER_IDLE_*: no token, a small region) or the
// last block's - the block picks at run time; else compile-time.
struct PlannerLayout {
    uint32_t tokens, rows_capacity, entries_capacity;
    uint32_t instance_rows;    // an instance's rows: its entries' worst case + an expert's padding, within the capacity
    uint32_t pages;            // the routing rows' pages
    uint32_t home_tokens_max;  // my home tokens at most
    uint32_t routing_rows_offset;     // the routing rows' pages (page_stride each)
    uint32_t instance_totals_offset;  // every instance's per-destination totals [instances][num_chips]
    uint32_t my_totals_offset;        // my per-destination entry totals [num_chips]
    uint32_t prefix_offset;           // my per-expert prefixes a destination (instance_block_bytes each)
    uint32_t staging_offset;          // my entries sorted by destination and expert
    uint32_t headers_offset;          // every source's packet header (header_bytes each)
    uint32_t entries_offset;          // my experts' entries read from the packets
    uint32_t runs_offset;             // per (source, slice): where its run of my experts' entries landed
    uint32_t expert_token_offset;     // per (my expert, token) counts, then rows (when local memory is too small)
    uint32_t token_index_offset;      // my rows' token index
    uint32_t table_offset;            // my rows' table (2 words a row)
    uint32_t want_offset;             // my home accumulators' want rows (128 B each); the padded metadata's scratch
    uint32_t partials_offset;         // every instance's n_inj / n_ret partials (64 B each, instance 0's)
    uint32_t plan_offset;             // the plan's head words (instance 0's: n_inj, n_ret, n_local)
    uint32_t counts_offset;           // my experts' counts (instance 0's, a 64 B span)
    uint32_t regions_offset;          // my experts' regions (instance 0's, a 64 B span)
    uint32_t rows_block_offset;       // the plan's rows block (16 words, instance 0's)
    uint32_t expert_counts_offset;    // my per-expert counts of each destination (expert_count_block_bytes each)
    uint32_t count_slices_offset;     // the instances' count slices of my destinations, summed into the totals
    uint32_t header_totals_offset;    // my destinations' packet header totals (expert_count_block_bytes each)
    constexpr PlannerLayout(uint32_t layout_tokens, uint32_t layout_rows_capacity, uint32_t layout_entries_capacity) :
        tokens(layout_tokens),
        rows_capacity(layout_rows_capacity),
        entries_capacity(layout_entries_capacity),
        instance_rows(
            layout_entries_capacity + 32 * experts_per_chip < layout_rows_capacity
                ? layout_entries_capacity + 32 * experts_per_chip
                : layout_rows_capacity),
        pages(layout_tokens / tokens_per_page),
        home_tokens_max(owned_accumulators_max * ((layout_tokens + accumulators - 1) / accumulators)),
        routing_rows_offset(0),
        instance_totals_offset(align_64(pages * page_stride)),
        my_totals_offset(instance_totals_offset + align_64(4 * instances * num_chips)),
        prefix_offset(my_totals_offset + align_64(4 * num_chips)),
        staging_offset(prefix_offset + num_chips * instance_block_bytes),
        headers_offset(staging_offset + align_64(4 * home_tokens_max * top_k + 16 * num_chips)),
        entries_offset(headers_offset + num_chips * header_bytes),
        runs_offset(entries_offset + align_64(4 * entries_capacity + 128 * num_chips * instances)),
        expert_token_offset(runs_offset + align_64(4 * num_chips * instances)),
        token_index_offset(expert_token_offset + align_64(2 * experts_per_chip * tokens)),
        table_offset(token_index_offset + align_64(4 * instance_rows)),
        want_offset(table_offset + align_64(8 * instance_rows)),
        partials_offset(want_offset + align_64(128 * owned_accumulators_max)),
        plan_offset(partials_offset + instances * 64),
        counts_offset(plan_offset + moe_block_plan_words::kPlanHeadBytes),
        regions_offset(counts_offset + align_64(4 * (experts_per_chip + 32))),
        rows_block_offset(regions_offset + align_64(4 * (experts_per_chip + 32))),
        expert_counts_offset(rows_block_offset + 64),
        count_slices_offset(expert_counts_offset + num_chips * expert_count_block_bytes),
        header_totals_offset(count_slices_offset + owned_destinations_max * instances * expert_count_block_bytes) {}
};
// local-memory counters: per global expert, later per (my expert, token) if they fit
constexpr uint32_t local_counter_words = 1024;
static_assert(global_experts <= local_counter_words, "local counters: global experts up to 1024");
static_assert(
    ROUTING_PLANNER_TOKENS <= 2048 && experts_per_chip <= 64,
    "an entry: token (11 bits) | expert (6 bits) | weight (15 bits)");
#ifdef ROUTING_PLANNER_PER_BLOCK
static_assert(
    ROUTING_PLANNER_IDLE_TOKENS <= ROUTING_PLANNER_TOKENS, "an earlier block's planner: fewer tokens than the last's");
#endif
static_assert(
    accumulators <= moe_block_plan_words::kAccMax && num_chips <= moe_block_plan_words::kMaxChips,
    "the plan's want[accumulators][num_chips]");
constexpr auto routing_rows_accessor_args = TensorAccessorArgs<ROUTING_PLANNER_ACCESSOR_COMPILE_TIME_ARGS_OFFSET>();
constexpr auto slots_accessor_args = TensorAccessorArgs<routing_rows_accessor_args.next_compile_time_args_offset()>();

// The fabric's first hop from this chip to each chip (the host asks the fabric: MoEBlockResources::first_hops; the
// combine's ports follow the same table): direction index 0 X+, 1 X-, 2 Y+, 3 Y-, 16 a word, 2 bits each
constexpr uint32_t first_hop_words = (num_chips + 15) / 16;
constexpr uint32_t link_port_words = 8;  // (the port of each (direction, link): direction 2 + link)

// The planner's args (head_router_program.cpp planner_args), read once in kernel_main: the scalars into the fields,
// the tables read in place (`words`: the args from the first, moe_block_args::arg_pointer) through the accessors.
//   [0] the routing rows' address, [1] the exchange slots', [2] / [3] the arrival counters (even / odd call),
//   [4] the call counter word, [5] counts', [6] regions', [7] token index', [8] table's, [9] the plan's address,
//   [10] link_port_words ports, then first_hop_words first-hop words, my instance, (x, y) of every instance's core,
//   every instance's region base, exchange semaphore, go semaphore, (ROUTING_PLANNER_PER_BLOCK) every earlier-block
//   instance's region base, per destination chip (x, y, packet buffer), per served port (x, y, semaphore), the ports'
//   ready semaphore (the sender's, skipped), my go, parts, exchange semaphores.
struct PlannerArgs {
    static constexpr uint32_t link_ports_at = 10;
    static constexpr uint32_t first_hops_at = link_ports_at + link_port_words;
    static constexpr uint32_t instance_at = first_hops_at + first_hop_words;
    static constexpr uint32_t instance_cores_at = instance_at + 1;
    static constexpr uint32_t instance_bases_at = instance_cores_at + 2 * instances;
    static constexpr uint32_t instance_exchange_semaphores_at = instance_bases_at + instances;
    static constexpr uint32_t instance_go_semaphores_at = instance_exchange_semaphores_at + instances;
#ifdef ROUTING_PLANNER_PER_BLOCK
    static constexpr uint32_t idle_instance_bases_at = instance_go_semaphores_at + instances;
    static constexpr uint32_t destinations_at = idle_instance_bases_at + instances;
#else
    static constexpr uint32_t destinations_at = instance_go_semaphores_at + instances;
#endif
    static constexpr uint32_t ports_at = destinations_at + 3 * num_chips;

    const volatile tt_l1_ptr uint32_t* words = nullptr;
    uint32_t routing_rows_address = 0, slots_address = 0, calls_word_address = 0;
    uint32_t counts_address = 0, regions_address = 0, token_index_address = 0, table_address = 0, plan_address = 0;
    uint32_t instance = 0;  // my instance
    // this chip's index, its first expert, its fabric ports (the index of its first local-entry destination) and the
    // sender ports it serves: chip values - ROUTING_PLANNER_MY_CHIP / _PORTS / _NUM_PORTS read this chip's common args
    // (head_program.cpp chip_values), so they are run-time values, read here once
    uint32_t my_chip = 0, my_first_expert = 0, chip_ports = 0, served_ports = 0;
    // (the semaphores: past the served ports and the ports' ready semaphore)
    uint32_t semaphores_at() const { return ports_at + 3 * served_ports + 1; }
    uint32_t go_semaphore = 0, parts_semaphore = 0, exchange_semaphore = 0;
#if defined(PADDED_CHUNK)
    RouterPaddingArgs padding{};
#endif
    // the arrival counter of a call of this parity (every chip's packet counted)
    uint32_t arrival_counter(uint32_t parity) const { return words[2 + parity]; }
    // the port of a (first-hop direction, link) pair: direction 2 + link
    uint32_t link_port(uint32_t index) const { return words[link_ports_at + index]; }
    // the first hop toward a chip (direction index)
    uint32_t first_hop(uint32_t chip) const { return (words[first_hops_at + chip / 16] >> (2 * (chip % 16))) & 3u; }
    uint32_t instance_x(uint32_t other) const { return words[instance_cores_at + 2 * other]; }
    uint32_t instance_y(uint32_t other) const { return words[instance_cores_at + 2 * other + 1]; }
    uint32_t instance_base(uint32_t other) const { return words[instance_bases_at + other]; }
    uint32_t instance_exchange_semaphore(uint32_t other) const {
        return words[instance_exchange_semaphores_at + other];
    }
    uint32_t instance_go_semaphore(uint32_t other) const { return words[instance_go_semaphores_at + other]; }
#ifdef ROUTING_PLANNER_PER_BLOCK
    uint32_t idle_instance_base(uint32_t other) const { return words[idle_instance_bases_at + other]; }
#endif
    // per destination chip: its port core (x, y) and its packet buffer's address there
    uint32_t destination_x(uint32_t chip) const { return words[destinations_at + 3 * chip]; }
    uint32_t destination_y(uint32_t chip) const { return words[destinations_at + 3 * chip + 1]; }
    uint32_t destination_buffer(uint32_t chip) const { return words[destinations_at + 3 * chip + 2]; }
    // the served port cores (x, y) and their semaphore
    uint32_t port_x(uint32_t port) const { return words[ports_at + 3 * port]; }
    uint32_t port_y(uint32_t port) const { return words[ports_at + 3 * port + 1]; }
    uint32_t port_semaphore(uint32_t port) const { return words[ports_at + 3 * port + 2]; }
};
// (kernel_main, once) planner_args: the planner's first arg (moe_block_args::arg_pointer); padding_args: common arg [2]
// (moe_block_args::common_pointer(2) - read under PADDED_CHUNK only)
FORCE_INLINE PlannerArgs read_planner_args(
    const volatile tt_l1_ptr uint32_t* planner_args, [[maybe_unused]] const volatile tt_l1_ptr uint32_t* padding_args) {
    PlannerArgs args;
    args.words = planner_args;
    args.routing_rows_address = planner_args[0];
    args.slots_address = planner_args[1];
    args.calls_word_address = planner_args[4];
    args.counts_address = planner_args[5];
    args.regions_address = planner_args[6];
    args.token_index_address = planner_args[7];
    args.table_address = planner_args[8];
    args.plan_address = planner_args[9];
    args.instance = planner_args[PlannerArgs::instance_at];
    args.my_chip = ROUTING_PLANNER_MY_CHIP;
    args.my_first_expert = args.my_chip * experts_per_chip;
    args.chip_ports = ROUTING_PLANNER_PORTS;
    args.served_ports = ROUTING_PLANNER_NUM_PORTS;
    args.go_semaphore = planner_args[args.semaphores_at()];
    args.parts_semaphore = planner_args[args.semaphores_at() + 1];
    args.exchange_semaphore = planner_args[args.semaphores_at() + 2];
#if defined(PADDED_CHUNK)
    args.padding = read_router_padding_args(padding_args);
#endif
    return args;
}

FORCE_INLINE void plan(const PlannerArgs& args, uint32_t region_base) {
#ifdef ROUTING_PLANNER_PER_BLOCK
    const bool idle = !row_block::is_last();  // (an earlier row block's: no token)
    const PlannerLayout layout =
        idle ? PlannerLayout(
                   ROUTING_PLANNER_IDLE_TOKENS,
                   ROUTING_PLANNER_IDLE_ROWS_CAPACITY,
                   ROUTING_PLANNER_IDLE_ENTRIES_CAPACITY)
             : PlannerLayout(ROUTING_PLANNER_TOKENS, ROUTING_PLANNER_ROWS_CAPACITY, ROUTING_PLANNER_ENTRIES_CAPACITY);
    // (row blocks in one program: an earlier block's planner instances have their own bases - the args past the go
    // semaphores; the last block's (every block's tokens) are the instance bases. An earlier block's instance works in
    // its own region too: the host passes the last block's base as region_base, and that region may overlap the
    // router's expert-id tiles, which every later block reads)
    auto instance_region = [&](uint32_t other) __attribute__((always_inline)) {
        return idle ? args.idle_instance_base(other) : args.instance_base(other);
    };
    if (idle) {
        region_base = args.idle_instance_base(args.instance);
    }
#else
#ifdef ROUTING_PLANNER_IDLE
    constexpr bool idle = true;  // (an earlier row block's own program: no token)
#else
    constexpr bool idle = false;
#endif
    constexpr PlannerLayout layout(
        ROUTING_PLANNER_TOKENS, ROUTING_PLANNER_ROWS_CAPACITY, ROUTING_PLANNER_ENTRIES_CAPACITY);
    auto instance_region = [&](uint32_t other) __attribute__((always_inline)) { return args.instance_base(other); };
#endif
    const uint32_t instance = args.instance;
    // (the RISC's L1 data cache - off by default: every L1 load is then ~10+ cycles; write-through, so only NoC
    // arrivals need the invalidations this kernel already does; off again before returning, for the firmware)
    set_l1_data_cache<true>();

    if (!idle) {  // (an earlier row block's planner plans no token: no rows to read)
        const auto routing_rows = TensorAccessor(routing_rows_accessor_args, args.routing_rows_address, page_bytes);
        for (uint32_t page = 0; page < layout.pages; ++page) {
            noc_async_read(
                routing_rows.get_noc_addr(page),
                region_base + layout.routing_rows_offset + page * page_stride,
                page_bytes);
        }
    }
#if defined(PADDED_CHUNK)
    // (a padded chunk's actual_start / actual_end, into want_offset: local scratch of step 4 - nothing writes it
    // before)
    uint32_t metadata_l1 = 0;
    static_assert(align_64(128 * owned_accumulators_max) >= 128, "the scalars' slots: 64 B aligned in want, 128 B");
    if (!idle) {
        metadata_l1 = padded_metadata_slot(region_base + layout.want_offset);
        padded_metadata_read(args.padding.forward, metadata_l1);
    }
#endif
    volatile tt_l1_ptr uint32_t* calls = reinterpret_cast<volatile tt_l1_ptr uint32_t*>(args.calls_word_address);
    const uint32_t parity = calls[0] & 1;
    calls[0] = calls[0] + 1;
    alignas(4) uint16_t local_counters[local_counter_words];  // (local memory: entries per global expert, then cursors;
                                                              // later per (my expert, token))
    for (uint32_t i = 0; i < global_experts; ++i) {
        local_counters[i] = 0;
    }
    noc_async_read_barrier();
    invalidate_l1_cache();
    // (row blocks, an earlier block: no token planned - the exchange runs empty, the chips' barrier between blocks;
    // the last block plans every block's tokens)
    const uint32_t planned_tokens = idle ? 0 : layout.tokens;
#if defined(PADDED_CHUNK)
    // (a padded forward: my owned token t - block t / owner_rows's, row first_block_first_row + (t / owner_rows)
    // block_rows + t % owner_rows of my SP row - is planned only when its row is real; the rest are padding, unplanned)
    PaddedRows padded_rows{};
    if (!idle) {
        padded_rows = padded_rows_of(metadata_l1, args.padding.forward);
    }
    const RouterPaddingArgs& padding = args.padding;
    auto token_real = [&](uint32_t token) __attribute__((always_inline)) {
        return padded_rows.real(
            padding.first_block_first_row + (token / padding.owner_rows) * padding.block_rows +
            token % padding.owner_rows);
    };
#else
    auto token_real = [](uint32_t) __attribute__((always_inline)) { return true; };
#endif
    // (1) my home tokens (t % accumulators owned by me): my entries per global expert
    auto token_row = [&](uint32_t token) __attribute__((always_inline)) {
        return reinterpret_cast<const uint32_t*>(
            region_base + layout.routing_rows_offset + (token / tokens_per_page) * page_stride +
            (token % tokens_per_page) * row_bytes);
    };
    uint32_t my_entries = 0;
    for (uint32_t accumulator = instance; accumulator < accumulators; accumulator += instances) {
        for (uint32_t token = accumulator; token < planned_tokens; token += accumulators) {
            if (!token_real(token)) {
                continue;
            }
            const uint32_t* row = token_row(token);
            for (uint32_t pair = 0; pair < top_k / 2; ++pair) {
                const uint32_t expert_pair = row[pair];
                ++local_counters[expert_pair & 0xFFFF];
                ++local_counters[expert_pair >> 16];
            }
            my_entries += top_k;
        }
    }
    // (2) my per-destination totals to every planner core
    uint32_t* my_totals = reinterpret_cast<uint32_t*>(region_base + layout.my_totals_offset);
    for (uint32_t destination = 0; destination < num_chips; ++destination) {
        uint32_t sum = 0;
        for (uint32_t expert = 0; expert < experts_per_chip; ++expert) {
            sum += local_counters[destination * experts_per_chip + expert];
        }
        my_totals[destination] = sum;
    }
    for (uint32_t other = 0; other < instances; ++other) {
        noc_async_write(
            region_base + layout.my_totals_offset,
            get_noc_addr(
                args.instance_x(other),
                args.instance_y(other),
                instance_region(other) + layout.instance_totals_offset + instance * num_chips * 4),
            num_chips * 4);
    }
    {  // my per-expert counts of destination d to its owner instance d % instances (slot `instance`): the owner sums
       // the instances' slices into the packet header's totals - the sender then only sends
        uint16_t* expert_counts = reinterpret_cast<uint16_t*>(region_base + layout.expert_counts_offset);
        for (uint32_t destination = 0; destination < num_chips; ++destination) {
            for (uint32_t expert = 0; expert < expert_count_block_bytes / 2; ++expert) {
                expert_counts[destination * (expert_count_block_bytes / 2) + expert] =
                    expert < experts_per_chip ? local_counters[destination * experts_per_chip + expert] : 0;
            }
        }
        for (uint32_t destination = 0, owner = 0; destination < num_chips;
             ++destination, owner = owner + 1 == instances ? 0 : owner + 1) {
            noc_async_write(
                region_base + layout.expert_counts_offset + destination * expert_count_block_bytes,
                get_noc_addr(
                    args.instance_x(owner),
                    args.instance_y(owner),
                    instance_region(owner) + layout.count_slices_offset +
                        ((destination / instances) * instances + instance) * expert_count_block_bytes),
                expert_count_block_bytes);
        }
    }
    noc_async_write_barrier();
    for (uint32_t other = 0; other < instances; ++other) {
        if (other != instance) {
            noc_semaphore_inc(
                get_noc_addr(args.instance_x(other), args.instance_y(other), args.instance_exchange_semaphore(other)),
                1);
        }
    }
    noc_semaphore_wait(reinterpret_cast<volatile tt_l1_ptr uint32_t*>(args.exchange_semaphore), instances - 1);
    *reinterpret_cast<volatile tt_l1_ptr uint32_t*>(args.exchange_semaphore) = 0;
    invalidate_l1_cache();
    // (3) my entries sorted by global expert into staging (each destination's run aligned as its place in the packet,
    // mod 16), straight into the port cores' packet buffers
    const uint32_t* instance_totals = reinterpret_cast<const uint32_t*>(region_base + layout.instance_totals_offset);
    {
        uint32_t packet_offset[num_chips], staged_at[num_chips];
        uint32_t staging_pointer = region_base + layout.staging_offset;
        for (uint32_t destination = 0; destination < num_chips; ++destination) {
            uint32_t entries_before = 0;  // (the earlier instances' entries for this destination)
            for (uint32_t other = 0; other < instance; ++other) {
                entries_before += instance_totals[other * num_chips + destination];
            }
            packet_offset[destination] = header_bytes + 4 * entries_before;
            staging_pointer += (packet_offset[destination] - staging_pointer) & 15u;
            staged_at[destination] = staging_pointer;
            uint16_t* prefix =
                reinterpret_cast<uint16_t*>(region_base + layout.prefix_offset + destination * instance_block_bytes);
            uint32_t count = 0, staging_word = (staging_pointer - (region_base + layout.staging_offset)) / 4;
            for (uint32_t expert = 0; expert < experts_per_chip; ++expert) {
                prefix[expert] = static_cast<uint16_t>(count);
                const uint32_t entries = local_counters[destination * experts_per_chip + expert];
                local_counters[destination * experts_per_chip + expert] =
                    static_cast<uint16_t>(staging_word);  // (cursor: word index in the staging)
                staging_word += entries;
                count += entries;
            }
            prefix[experts_per_chip] = static_cast<uint16_t>(count);
            staging_pointer += 4 * count;
        }
        uint32_t* staging = reinterpret_cast<uint32_t*>(region_base + layout.staging_offset);
        for (uint32_t accumulator = instance; accumulator < accumulators; accumulator += instances) {
            for (uint32_t token = accumulator; token < planned_tokens; token += accumulators) {
                if (!token_real(token)) {
                    continue;
                }
                const uint32_t* row = token_row(token);
                for (uint32_t pair = 0; pair < top_k / 2;
                     ++pair) {  // (token | expert << 11 | weight's 15 low bits << 17)
                    const uint32_t expert_pair = row[pair], weight_pair = row[top_k / 2 + pair];
                    const uint32_t first_global_expert = expert_pair & 0xFFFF, second_global_expert = expert_pair >> 16;
                    staging[local_counters[first_global_expert]++] =
                        token | ((first_global_expert % experts_per_chip) << 11) | ((weight_pair & 0x7FFF) << 17);
                    staging[local_counters[second_global_expert]++] =
                        token | ((second_global_expert % experts_per_chip) << 11) | ((weight_pair >> 16) << 17);
                }
            }
        }
        for (uint32_t destination = instance; destination < num_chips;
             destination += instances) {  // (my destinations' header totals: the instances' counts, two 16-bit lanes
                                          // a word - a lane's sum <= the destination's entries <= tokens top_k)
            const uint32_t* slice_counts = reinterpret_cast<const uint32_t*>(
                region_base + layout.count_slices_offset +
                (destination / instances) * instances * expert_count_block_bytes);
            uint32_t* totals = reinterpret_cast<uint32_t*>(
                region_base + layout.header_totals_offset + (destination / instances) * expert_count_block_bytes);
            for (uint32_t i = 0; i < expert_count_block_bytes / 4; ++i) {
                uint32_t sum = 0;
                for (uint32_t slice = 0; slice < instances; ++slice) {
                    sum += slice_counts[slice * (expert_count_block_bytes / 4) + i];
                }
                totals[i] = sum;
            }
        }
        for (uint32_t destination = 0; destination < num_chips; ++destination) {
            const uint32_t x = args.destination_x(destination), y = args.destination_y(destination);
            const uint32_t packet_buffer = args.destination_buffer(destination);  // (the port's CB 0: the same base)
            if (my_totals[destination]) {
                noc_async_write(
                    staged_at[destination],
                    get_noc_addr(x, y, packet_buffer + packet_offset[destination]),
                    4 * my_totals[destination]);
            }
            noc_async_write(
                region_base + layout.prefix_offset + destination * instance_block_bytes,
                get_noc_addr(x, y, packet_buffer + 64 + expert_count_block_bytes + instance * instance_block_bytes),
                instance_block_bytes);
            noc_inline_dw_write(get_noc_addr(x, y, packet_buffer + 4 * instance), my_totals[destination]);
            if (destination % instances == instance) {
                noc_async_write(
                    region_base + layout.header_totals_offset + (destination / instances) * expert_count_block_bytes,
                    get_noc_addr(x, y, packet_buffer + 64),
                    expert_count_block_bytes);
            }
        }
        noc_async_write_barrier();
        for (uint32_t port = 0; port < args.served_ports; ++port) {
            noc_semaphore_inc(get_noc_addr(args.port_x(port), args.port_y(port), args.port_semaphore(port)), 1);
        }
    }
    // (4) while the packets fly: my home tokens' rows (local: I am their home) - n_local / want / n_ret; want rows
    // straight to the plan, n_local to instance 0
    uint32_t returns_by_port[8] = {0, 0, 0, 0, 0, 0, 0, 0};  // (n_ret)
    {
        uint8_t return_port[num_chips * links];  // n_ret's port: the first hop toward a source, its link
        for (uint32_t source = 0; source < num_chips; ++source) {
            for (uint32_t link = 0; link < links; ++link) {
                return_port[source * links + link] =
                    source == args.my_chip ? 0
                                           : static_cast<uint8_t>(args.link_port(args.first_hop(source) * 2 + link));
            }
        }
        uint32_t* want_rows = reinterpret_cast<uint32_t*>(region_base + layout.want_offset);
        const auto plan_accessor =
            TensorAccessor(routing_accessors::plan, args.plan_address, moe_block_plan_words::kPlanBytes);
        for (uint32_t i = 0, accumulator = instance; accumulator < accumulators; ++i, accumulator += instances) {
            uint32_t want[num_chips], local_entries = 0;
            for (uint32_t source = 0; source < num_chips; ++source) {
                want[source] = 0;
            }
            for (uint32_t token = accumulator; token < planned_tokens; token += accumulators) {
                if (!token_real(token)) {
                    continue;
                }
                const uint32_t* row = token_row(token);
                for (uint32_t pair = 0; pair < top_k / 2; ++pair) {
                    const uint32_t expert_pair = row[pair];
                    const uint32_t first_source = (expert_pair & 0xFFFF) / experts_per_chip,
                                   second_source = (expert_pair >> 16) / experts_per_chip;
                    if (first_source == args.my_chip) {
                        ++local_entries;
                    } else {
                        ++want[first_source];
                        ++returns_by_port
                            [return_port[first_source * links + (accumulator + first_source + args.my_chip) % links]];
                    }
                    if (second_source == args.my_chip) {
                        ++local_entries;
                    } else {
                        ++want[second_source];
                        ++returns_by_port
                            [return_port[second_source * links + (accumulator + second_source + args.my_chip) % links]];
                    }
                }
            }
            for (uint32_t source = 0; source < moe_block_plan_words::kMaxChips; ++source) {
                want_rows[i * 32 + source] = source < num_chips ? want[source] : 0;
            }
            noc_async_write(
                region_base + layout.want_offset + i * 128,
                plan_accessor.get_noc_addr(0) +
                    (moe_block_plan_words::kPlanWant + accumulator * moe_block_plan_words::kMaxChips) * 4,
                128);
            noc_inline_dw_write(
                get_noc_addr(
                    args.instance_x(0),
                    args.instance_y(0),
                    instance_region(0) + layout.plan_offset + (moe_block_plan_words::kPlanNloc + accumulator) * 4),
                local_entries);
        }
    }
    // (5) every packet of the call: instance 0 waits, then wakes the others
    {
        DeviceZoneScopedN("SE-RPWAIT");
        if (instance == 0) {
            const uint32_t arrival_counter = args.arrival_counter(parity);
            noc_semaphore_wait_min(reinterpret_cast<volatile tt_l1_ptr uint32_t*>(arrival_counter), num_chips);
            noc_semaphore_inc(get_noc_addr(arrival_counter), 0u - num_chips);
            for (uint32_t other = 1; other < instances; ++other) {
                noc_semaphore_inc(
                    get_noc_addr(args.instance_x(other), args.instance_y(other), args.instance_go_semaphore(other)), 1);
            }
        } else {
            noc_semaphore_wait(reinterpret_cast<volatile tt_l1_ptr uint32_t*>(args.go_semaphore), 1);
            *reinterpret_cast<volatile tt_l1_ptr uint32_t*>(args.go_semaphore) = 0;
        }
    }
    // (6) the headers (every expert's total -> regions; my experts: a contiguous range balanced by rows), then every
    // packet's entries
    const auto slots = TensorAccessor(slots_accessor_args, args.slots_address, slot_bytes);
    // (a packet's source: other chips' land in my DRAM slot parity num_chips + source through the fabric router's NOC1
    // endpoint - I run on NOC1, the same port. My own chip's packet is written into its slot by its port core's sender
    // on that core's NoC - possibly NOC0, the other port - so I read it from the sender's packet buffer in L1 instead:
    // final once its arrival is counted, and not rewritten before every instance is past this call (the step-2
    // exchange of the next))
    auto packet_source = [&](uint32_t source) -> uint64_t {
        return source == args.my_chip ? get_noc_addr(
                                            args.destination_x(args.my_chip),
                                            args.destination_y(args.my_chip),
                                            args.destination_buffer(args.my_chip))
                                      : slots.get_noc_addr(parity * num_chips + source);
    };
    for (uint32_t source = 0; source < num_chips; ++source) {
        noc_async_read(
            packet_source(source), region_base + layout.headers_offset + source * header_bytes, header_bytes);
    }
    noc_async_read_barrier();
    invalidate_l1_cache();
    uint32_t expert_totals[experts_per_chip], expert_regions[experts_per_chip], source_has_entries[num_chips];
    // (every expert's total over the headers: two 16-bit lanes a word - a lane's sum <= num_chips tokens, one entry a
    // token)
    static_assert(num_chips * ROUTING_PLANNER_TOKENS < 65536, "packed expert totals");
    constexpr uint32_t packed_total_words = (experts_per_chip + 1) / 2;
    uint32_t packed_totals[packed_total_words];
    for (uint32_t i = 0; i < packed_total_words; ++i) {
        packed_totals[i] = 0;
    }
    for (uint32_t source = 0; source < num_chips; ++source) {
        const uint32_t* header_totals = reinterpret_cast<const uint32_t*>(
            region_base + layout.headers_offset + source * header_bytes +
            64);  // (experts_per_chip uint16: expert_count_block_bytes >= 2 packed_total_words words)
        for (uint32_t i = 0; i < packed_total_words; ++i) {
            packed_totals[i] += header_totals[i];
        }
    }
    for (uint32_t expert = 0; expert < experts_per_chip; ++expert) {
        expert_totals[expert] = (packed_totals[expert / 2] >> (16 * (expert & 1))) & 0xFFFF;
    }
    uint32_t region_end = 0, total_entries = 0;  // (the chip's rows, its experts' regions padded to 32; its entries)
    for (uint32_t expert = 0; expert < experts_per_chip; ++expert) {
        total_entries += expert_totals[expert];
        expert_regions[expert] = region_end;
        region_end += (expert_totals[expert] + 31) / 32 * 32;
    }
    // my experts: from boundary `instance` to instance + 1 - boundary j the expert boundary nearest j total_entries /
    // instances entries (whole experts: the nearest, not the next - an instance's work is its entries)
    auto expert_boundary = [&](uint32_t boundary) __attribute__((always_inline)) -> uint32_t {
        if (boundary == 0) {
            return 0;
        }
        if (boundary >= instances) {
            return experts_per_chip;
        }
        for (uint32_t expert = 0, entries_before = 0; expert < experts_per_chip; ++expert) {
            const uint32_t entries_through = entries_before + expert_totals[expert];
            if (entries_through * instances >= boundary * total_entries) {
                return 2 * boundary * total_entries <= (entries_before + entries_through) * instances ? expert
                                                                                                      : expert + 1;
            }
            entries_before = entries_through;
        }
        return experts_per_chip;
    };
    const uint32_t first_expert = expert_boundary(instance), end_expert = expert_boundary(instance + 1);
    const uint32_t first_row = first_expert < experts_per_chip ? expert_regions[first_expert] : region_end,
                   end_row = end_expert < experts_per_chip ? expert_regions[end_expert] : region_end;
    // my experts' entries only: per source, per slice, its run of my experts (a slice is sorted by expert) - each read
    // keeps its source's 64 B alignment; run (source, slice)'s entry x at runs[source instances + slice] + 4 x. An
    // instance's entries are at most total_entries / instances + one expert's (the boundaries are the nearest expert
    // edges): entries_capacity, the build's worst case - packed routing included (the rest of the chip's entries are
    // other instances')
    uint32_t* runs = reinterpret_cast<uint32_t*>(region_base + layout.runs_offset);
    {
        uint32_t entries_read = 0, read_pointer = region_base + layout.entries_offset;
        for (uint32_t source = 0; source < num_chips; ++source) {
            const uint32_t header = region_base + layout.headers_offset + source * header_bytes;
            const uint32_t* slice_totals = reinterpret_cast<const uint32_t*>(header);
            source_has_entries[source] = 0;
            for (uint32_t slice = 0, slice_first = 0; slice < instances; slice_first += slice_totals[slice], ++slice) {
                const uint16_t* prefix = reinterpret_cast<const uint16_t*>(
                    header + 64 + expert_count_block_bytes + slice * instance_block_bytes);
                const uint32_t first_entry = slice_first + prefix[first_expert],
                               entries = prefix[end_expert] - prefix[first_expert];
                runs[source * instances + slice] = 0;
                if (!entries || entries_read + entries > layout.entries_capacity) {
                    continue;  // (past the worst case: dropped - region_end > rows_capacity or the plan's over flag)
                }
                const uint32_t source_offset = header_bytes + 4 * first_entry, aligned_offset = source_offset & ~63u,
                               length = (source_offset - aligned_offset + 4 * entries + 63) & ~63u;
                noc_async_read(packet_source(source) + aligned_offset, read_pointer, length);
                runs[source * instances + slice] = read_pointer + (source_offset - aligned_offset) - 4 * first_entry;
                source_has_entries[source] = 1;
                read_pointer += length;
                entries_read += entries;
            }
        }
    }
    // my experts' counts / regions (instance 0: the 64-byte span of the page that holds them, the rest of the span
    // other chips' experts: zero) - now, under the entries' reads and the fill
    const uint32_t span_first = (args.my_first_expert * 4) & ~63u,
                   span_end = ((args.my_first_expert + experts_per_chip) * 4 + 63) & ~63u;
    if (instance == 0) {
        uint32_t* counts = reinterpret_cast<uint32_t*>(region_base + layout.counts_offset);
        uint32_t* regions = reinterpret_cast<uint32_t*>(region_base + layout.regions_offset);
        for (uint32_t i = 0; i < (span_end - span_first) / 4; ++i) {
            counts[i] = regions[i] = 0;
        }
        for (uint32_t expert = 0; expert < experts_per_chip; ++expert) {
            counts[args.my_first_expert + expert - span_first / 4] = expert_totals[expert];
            regions[args.my_first_expert + expert - span_first / 4] = expert_regions[expert];
        }
        noc_async_write(
            region_base + layout.counts_offset,
            TensorAccessor(routing_accessors::counts, args.counts_address, global_experts * 4).get_noc_addr(0) +
                span_first,
            span_end - span_first);
        noc_async_write(
            region_base + layout.regions_offset,
            TensorAccessor(routing_accessors::regions, args.regions_address, global_experts * 4).get_noc_addr(0) +
                span_first,
            span_end - span_first);
    }
    const uint32_t capped_end_row = end_row < layout.rows_capacity ? end_row : layout.rows_capacity,
                   written_end_row = capped_end_row < first_row + layout.instance_rows
                                         ? capped_end_row
                                         : first_row + layout.instance_rows;
    const uint32_t my_experts = end_expert - first_expert;
    uint16_t* expert_token_rows =
        my_experts * layout.tokens <= local_counter_words
            ? local_counters
            : reinterpret_cast<uint16_t*>(region_base + layout.expert_token_offset);  // per (my expert, token)
    {  // (32-bit stores: expert_token_rows is 4-byte aligned, tokens even)
        uint32_t* expert_token_words = reinterpret_cast<uint32_t*>(expert_token_rows);
        for (uint32_t i = 0; i < my_experts * layout.tokens / 2; ++i) {
            expert_token_words[i] = 0;
        }
    }
    noc_async_read_barrier();
    invalidate_l1_cache();
    // my experts' entries: sources in order, each source's slices in order, each slice's run of my experts (one
    // contiguous range: a slice is sorted by expert); an entry carries its token and expert
    // rows: per expert, ordered by (token, home, k): count per (expert, token), prefix, then each entry its row
    for (uint32_t source = 0; source < num_chips; ++source) {
        if (!source_has_entries[source]) {
            continue;
        }
        const uint32_t header = region_base + layout.headers_offset + source * header_bytes;
        const uint32_t* slice_totals = reinterpret_cast<const uint32_t*>(header);
        for (uint32_t slice = 0, slice_first = 0; slice < instances; slice_first += slice_totals[slice], ++slice) {
            if (!runs[source * instances + slice]) {
                continue;
            }
            const uint32_t* run_entries = reinterpret_cast<const uint32_t*>(runs[source * instances + slice]);
            const uint16_t* prefix = reinterpret_cast<const uint16_t*>(
                header + 64 + expert_count_block_bytes + slice * instance_block_bytes);
            const uint32_t end_entry = slice_first + prefix[end_expert];
            for (uint32_t entry = slice_first + prefix[first_expert]; entry < end_entry; ++entry) {
                const uint32_t packed = run_entries[entry], expert = ((packed >> 11) & 63) - first_expert,
                               token = packed & 0x7FF;
                if (expert >= my_experts || token >= layout.tokens) {
                    continue;  // (a slice's run holds my experts' entries only: never - bounds the counts on bad data)
                }
                ++expert_token_rows[expert * layout.tokens + token];
            }
        }
    }
    for (uint32_t expert = first_expert; expert < end_expert; ++expert) {
        uint16_t* token_rows = expert_token_rows + (expert - first_expert) * layout.tokens;
        uint32_t next_row = expert_regions[expert] - first_row;
        for (uint32_t token = 0; token < layout.tokens; ++token) {
            const uint32_t count = token_rows[token];
            token_rows[token] = static_cast<uint16_t>(next_row);
            next_row += count;
        }
    }
    uint32_t* token_index = reinterpret_cast<uint32_t*>(region_base + layout.token_index_offset);
    uint32_t* table_rows = reinterpret_cast<uint32_t*>(region_base + layout.table_offset);
    uint32_t injections_by_port[8] = {0, 0, 0, 0, 0, 0, 0, 0};  // (n_inj)
    for (uint32_t source = 0; source < num_chips; ++source) {
        if (!source_has_entries[source]) {
            continue;
        }
        const uint32_t header = region_base + layout.headers_offset + source * header_bytes;
        const uint32_t* slice_totals = reinterpret_cast<const uint32_t*>(header);
        const uint32_t source_tokens = source * layout.tokens, link_rotation = args.my_chip + source;
        uint32_t home_port[links];  // (the port toward home `source`, per link)
        for (uint32_t link = 0; link < links; ++link) {
            home_port[link] = source == args.my_chip ? 0 : args.link_port(args.first_hop(source) * 2 + link);
        }
        for (uint32_t slice = 0, slice_first = 0; slice < instances; slice_first += slice_totals[slice], ++slice) {
            if (!runs[source * instances + slice]) {
                continue;
            }
            const uint32_t* run_entries = reinterpret_cast<const uint32_t*>(runs[source * instances + slice]);
            const uint16_t* prefix = reinterpret_cast<const uint16_t*>(
                header + 64 + expert_count_block_bytes + slice * instance_block_bytes);
            const uint32_t end_entry = slice_first + prefix[end_expert];
            for (uint32_t entry = slice_first + prefix[first_expert]; entry < end_entry; ++entry) {
                const uint32_t packed = run_entries[entry], token = packed & 0x7FF, expert = (packed >> 11) & 63;
                if (expert - first_expert >= my_experts || token >= layout.tokens) {
                    continue;  // (as the count above)
                }
                const uint32_t row = expert_token_rows[(expert - first_expert) * layout.tokens + token]++;
                if (row >= layout.instance_rows || first_row + row >= layout.rows_capacity) {
                    continue;  // (past the worst case only: never)
                }
                const uint32_t accumulator = token % accumulators;
                uint32_t destination;
                if (source == args.my_chip) {
                    destination = args.chip_ports + accumulator;
                } else {
                    destination = home_port[(accumulator + link_rotation) % links];
                    ++injections_by_port[destination];
                }
                token_index[row] = source_tokens + token;
                table_rows[2 * row] = destination | ((packed >> 17) << 16);
                table_rows[2 * row + 1] = source | (token << 8) | (expert << 24);
            }
        }
    }
    for (uint32_t expert = first_expert; expert < end_expert; ++expert) {  // (padding rows: zero)
        const uint32_t next_region = expert + 1 < experts_per_chip ? expert_regions[expert + 1] : region_end,
                       padding_end = next_region < written_end_row ? next_region : written_end_row;
        for (uint32_t row = expert_regions[expert] + expert_totals[expert]; row < padding_end; ++row) {
            token_index[row - first_row] = 0;
            table_rows[2 * (row - first_row)] = 0;
            table_rows[2 * (row - first_row) + 1] = 0;
        }
    }
    if (written_end_row > first_row) {
        const auto table_accessor = TensorAccessor(routing_accessors::table, args.table_address, 256);
        noc_async_write(
            region_base + layout.token_index_offset,
            TensorAccessor(routing_accessors::token_index, args.token_index_address, layout.rows_capacity * 4)
                    .get_noc_addr(0) +
                first_row * 4,
            (written_end_row - first_row) * 4);
        for (uint32_t page = first_row / 32; page < written_end_row / 32; ++page) {
            noc_async_write(
                region_base + layout.table_offset + (page - first_row / 32) * 256,
                table_accessor.get_noc_addr(page),
                256);
        }
    }
    // (7) n_inj / n_ret partials to instance 0; instance 0 writes counts / regions, the plan's head words (n_inj,
    // n_ret, n_local) and its rows block (combine_plan_words.hpp)
    {
        uint32_t* partials = reinterpret_cast<uint32_t*>(region_base + layout.partials_offset + instance * 64);
        for (uint32_t i = 0; i < 8; ++i) {
            partials[i] = injections_by_port[i];
            partials[8 + i] = returns_by_port[i];
        }
        if (instance != 0) {
            noc_async_write(
                region_base + layout.partials_offset + instance * 64,
                get_noc_addr(
                    args.instance_x(0),
                    args.instance_y(0),
                    instance_region(0) + layout.partials_offset + instance * 64),
                64);
            noc_async_write_barrier();
            noc_semaphore_inc(get_noc_addr(args.instance_x(0), args.instance_y(0), args.parts_semaphore), 1);
        }
    }
    if (instance == 0) {
        noc_async_write_barrier();  // (my own n_local inline writes, into my plan words)
        noc_semaphore_wait(reinterpret_cast<volatile tt_l1_ptr uint32_t*>(args.parts_semaphore), instances - 1);
        *reinterpret_cast<volatile tt_l1_ptr uint32_t*>(args.parts_semaphore) = 0;
        invalidate_l1_cache();
        uint32_t* plan_head =
            reinterpret_cast<uint32_t*>(region_base + layout.plan_offset);  // (n_local[]: the owners')
        for (uint32_t i = 0; i < moe_block_plan_words::kPlanNloc; ++i) {
            plan_head[i] = 0;
        }
        for (uint32_t accumulator = accumulators;
             accumulator < moe_block_plan_words::kPlanWant - moe_block_plan_words::kPlanNloc;
             ++accumulator) {
            plan_head[moe_block_plan_words::kPlanNloc + accumulator] = 0;  // (no accumulator: no owner writes it)
        }
        for (uint32_t other = 0; other < instances; ++other) {
            const uint32_t* other_partials =
                reinterpret_cast<const uint32_t*>(region_base + layout.partials_offset + other * 64);
            for (uint32_t i = 0; i < 16; ++i) {
                plan_head[i] += other_partials[i];
            }
        }
        uint32_t* rows_block =
            reinterpret_cast<uint32_t*>(region_base + layout.rows_block_offset);  // (the plan's rows block: 16 words)
        for (uint32_t i = 0; i < 16; ++i) {
            rows_block[i] = 0;
        }
        rows_block[8] = region_end;
        rows_block[9] = region_end > layout.rows_capacity ? 1 : 0;
        const uint64_t plan_noc_address =
            TensorAccessor(routing_accessors::plan, args.plan_address, moe_block_plan_words::kPlanBytes)
                .get_noc_addr(0);
        noc_async_write(region_base + layout.plan_offset, plan_noc_address, moe_block_plan_words::kPlanHeadBytes);
        noc_async_write(
            region_base + layout.rows_block_offset, plan_noc_address + moe_block_plan_words::kPlanRowsBlock * 4, 64);
    }
    noc_async_write_barrier();
    noc_async_atomic_barrier();
    set_l1_data_cache<false>();
}
}  // namespace routing_planner
#endif  // ROUTING_PLANNER_NONE
#endif  // ROUTING_PLANNER
