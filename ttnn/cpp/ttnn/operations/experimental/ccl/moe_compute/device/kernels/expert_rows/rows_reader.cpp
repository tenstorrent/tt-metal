// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

// expert rows, reader (RISCV_0) of one core of one expert group.
// 1. Routing, built once per call over the program_cores cores of the program: core i resolves the tokens
//    [i tokens / program_cores, (i + 1) tokens / program_cores) with moe_compute's mapping rules (local experts = ids
//    the device's own mapping row maps to itself, ascending; a (token, k) is local when the token's source row maps the
//    id to this device; a token that names one expert at several k gives one entry per k, as moe_compute lists them)
//    and sends its local entries (local expert, token, score, k) to the root core (core 0). The root counting-sorts
//    them by local expert in core order (= token order) into the routing table: per local expert the routed rows in
//    token order, at most row_cap (the expert's e_t page and double-buffer half; later entries are dropped), and jobs
//    of at most job_rows rows of one expert, dealt to the expert groups: with one row tile per job every job has the
//    same predicted time and job j of the expert order runs on group j % expert_groups (the table stays in expert
//    order); with several each job goes to the group with the earliest predicted finish and the jobs are listed group
//    by group. Every other core reads the table back. The root also writes the table block to DRAM for the placement
//    program.
// 2. Streams this core's share of each job's expert weights in compute order gate/up(0), gate/up(1), down(0),
//    gate/up(2), down(1), ...: gate/up = this core's W0/W1 column groups, down = its W2 output groups, both in
//    moe_compute's prepared layout (the K padding rows at each group's end are not read): a W0/W1 group is read from
//    the bank pieces of the (layer, expert) stream it lies in (a ring position's slice can go on in the next bank), a
//    W2 group from this ring position's own bank; a half block-column and a two-tile last output group hold 2 tiles per
//    K row. A unit is what the compute holds at once: one group's run for a job of one row tile; for a job of several
//    (row_tiles > 1) one chunk of one group, gate/up chunk-major (chunk 0 of every column group, then chunk 1, ...),
//    down group-major, each unit padded to the same number of CB blocks.
#include <stdint.h>
#include "api/dataflow/dataflow_api.h"
#include "api/dataflow/circular_buffer.h"
#include "api/dataflow/endpoints.h"
#include "api/dataflow/noc.h"
#include "api/dataflow/noc_semaphore.h"
#include "api/core_local_mem.h"
#include "api/tensor/noc_traits.h"
#include "expert_rows_common.h"

namespace {

using expert_rows::GroupRun;
using expert_rows::JobRecord;

constexpr uint32_t block_tiles = get_named_compile_time_arg_val("block_tiles");
constexpr uint32_t block_packets = get_named_compile_time_arg_val("block_packets");
constexpr uint32_t weight_tile_bytes = get_named_compile_time_arg_val("weight_tile_bytes");
constexpr uint32_t block_slots = get_named_compile_time_arg_val("block_slots");
constexpr uint32_t blocks_in_flight = get_named_compile_time_arg_val("blocks_in_flight");
constexpr uint32_t tokens = get_named_compile_time_arg_val("tokens");
constexpr uint32_t local_experts = get_named_compile_time_arg_val("local_experts");
constexpr uint32_t top_k = get_named_compile_time_arg_val("top_k");
constexpr uint32_t global_experts = get_named_compile_time_arg_val("global_experts");
constexpr uint32_t index_page_bytes = get_named_compile_time_arg_val("index_page_bytes");
constexpr uint32_t mapping_page_bytes = get_named_compile_time_arg_val("mapping_page_bytes");
constexpr uint32_t tokens_per_source = get_named_compile_time_arg_val("tokens_per_source");
constexpr uint32_t expert_groups = get_named_compile_time_arg_val("expert_groups");
constexpr uint32_t job_rows = get_named_compile_time_arg_val("job_rows");
constexpr uint32_t row_cap = get_named_compile_time_arg_val("row_cap");
constexpr uint32_t grid_w = get_named_compile_time_arg_val("grid_w");
constexpr uint32_t grid_h = get_named_compile_time_arg_val("grid_h");
constexpr uint32_t mask_words = get_named_compile_time_arg_val("mask_words");
constexpr uint32_t sem_entries = get_named_compile_time_arg_val("sem_entries");
constexpr uint32_t sem_table = get_named_compile_time_arg_val("sem_table");
// cb_routing byte offsets (RoutingLayout in moe_expert_rows.cpp)
constexpr uint32_t routing_own_mapping = get_named_compile_time_arg_val("routing_own_mapping");
constexpr uint32_t routing_source_mappings = get_named_compile_time_arg_val("routing_source_mappings");
constexpr uint32_t routing_token_ids = get_named_compile_time_arg_val("routing_token_ids");
constexpr uint32_t routing_token_scores = get_named_compile_time_arg_val("routing_token_scores");
constexpr uint32_t routing_ctl = get_named_compile_time_arg_val("routing_ctl");
constexpr uint32_t routing_active_experts = get_named_compile_time_arg_val("routing_active_experts");
constexpr uint32_t routing_expert_rows = get_named_compile_time_arg_val("routing_expert_rows");
constexpr uint32_t routing_expert_first_rows = get_named_compile_time_arg_val("routing_expert_first_rows");
constexpr uint32_t routing_table = get_named_compile_time_arg_val("routing_table");
constexpr uint32_t routing_table_counts = get_named_compile_time_arg_val("routing_table_counts");
constexpr uint32_t routing_table_offsets = get_named_compile_time_arg_val("routing_table_offsets");
constexpr uint32_t routing_row_tokens = get_named_compile_time_arg_val("routing_row_tokens");
constexpr uint32_t routing_entry_experts = get_named_compile_time_arg_val("routing_entry_experts");
constexpr uint32_t routing_entry_tokens = get_named_compile_time_arg_val("routing_entry_tokens");
constexpr uint32_t routing_entry_scores = get_named_compile_time_arg_val("routing_entry_scores");
constexpr uint32_t routing_entry_k = get_named_compile_time_arg_val("routing_entry_k");
constexpr uint32_t routing_table_bytes = get_named_compile_time_arg_val("routing_table_bytes");
constexpr uint32_t routing_jobs = get_named_compile_time_arg_val("routing_jobs");
constexpr uint32_t routing_entry_areas = get_named_compile_time_arg_val("routing_entry_areas");
// weights in moe_compute's prepared layout (tiles, PreparedLayout in moe_expert_rows.cpp): the stride of a ring
// position's W0/W1 column groups and a bank piece of W0/W1 per (layer, expert); the stride of its W2 output groups
// and its W2 per (layer, expert)
constexpr uint32_t w0_w1_group_tiles = get_named_compile_time_arg_val("w0_w1_group_tiles");
constexpr uint32_t w0_w1_piece_tiles = get_named_compile_time_arg_val("w0_w1_piece_tiles");
constexpr uint32_t w2_group_tiles = get_named_compile_time_arg_val("w2_group_tiles");
constexpr uint32_t w2_slice_tiles = get_named_compile_time_arg_val("w2_slice_tiles");
constexpr uint32_t row_tiles = get_named_compile_time_arg_val("row_tiles");
constexpr uint32_t chunk_tiles = get_named_compile_time_arg_val("chunk_tiles");
constexpr uint32_t chunk_blocks = get_named_compile_time_arg_val("chunk_blocks");
constexpr uint32_t w2_chunk_rows = get_named_compile_time_arg_val("w2_chunk_rows");
constexpr uint32_t hidden_tiles = get_named_compile_time_arg_val("hidden_tiles");
constexpr uint32_t intermediate_tiles = get_named_compile_time_arg_val("intermediate_tiles");
constexpr bool has_bias = get_named_compile_time_arg_val("has_bias") == 1;
constexpr uint32_t ctl_words = get_named_compile_time_arg_val("ctl_words");
// row_tiles > 1: the planner's predicted time of a job (host-scaled cycles): its weight stream while the expert
// groups share DRAM, the busiest core's matmuls and partial reloads per row tile of a held job, and those of a job of
// one row tile
constexpr uint32_t deal_stream = get_named_compile_time_arg_val("deal_stream");
constexpr uint32_t deal_row_tile = get_named_compile_time_arg_val("deal_row_tile");
constexpr uint32_t deal_single = get_named_compile_time_arg_val("deal_single");

constexpr uint32_t bias_rows = has_bias ? 1 : 0;  // the bias K row of the prepared weights
// jobs of several row tiles: held units, padded passes, per-job row tiles
constexpr bool weight_stationary = row_tiles > 1;
// what the other cores read of the ctl block (expert_rows::ctl_bytes)
constexpr uint32_t ctl_bytes =
    weight_stationary ? expert_rows::align_up(expert_rows::ctl_bytes(expert_groups), expert_rows::l1_word_bytes)
                      : expert_rows::ctl_header_bytes;

constexpr uint32_t block_bytes = block_tiles * weight_tile_bytes;
constexpr uint32_t packet_tiles = expert_rows::div_up(block_tiles, block_packets);  // tiles per NoC packet of a block
static_assert(packet_tiles * weight_tile_bytes <= NOC_MAX_BURST_SIZE, "a block's packets fit a NoC burst each");
static_assert(
    top_k >= 1 && top_k <= 32, "top_k in 1..32 (kMaxTopK in moe_expert_rows.cpp): one u16 per k on the stack");
constexpr uint32_t index_page_entries = index_page_bytes / expert_rows::u16_bytes;
constexpr uint16_t not_local = 0xFFFFu;

constexpr uint32_t single_chunk_tiles = expert_rows::single_chunk_tiles(chunk_tiles, row_tiles, hidden_tiles);
constexpr uint32_t gate_up_chunks = expert_rows::div_up(hidden_tiles, chunk_tiles);
constexpr uint32_t single_chunks = expert_rows::div_up(hidden_tiles, single_chunk_tiles);
constexpr uint32_t down_chunks = expert_rows::div_up(intermediate_tiles, w2_chunk_rows);
constexpr uint32_t unit_blocks = weight_stationary ? chunk_blocks : 1;
// one transaction id per block in flight (ids 1..NOC_MAX_TRANSACTION_ID; blocks_in_flight blocks are in flight at most)
constexpr uint32_t transaction_ids = block_slots < NOC_MAX_TRANSACTION_ID ? block_slots : NOC_MAX_TRANSACTION_ID;
static_assert(blocks_in_flight < transaction_ids, "blocks_in_flight < transaction_ids: one id per block in flight");

// common args: source count, mapping row of each source, virtual x / y of the grid, mask of the program's cores, bank
// of each W0/W1 piece (shard); the index of the first arg of each list
struct CommonArgs {
    uint32_t virtual_x_arg, virtual_y_arg, all_cores_mask_arg, piece_banks_arg;
};

FORCE_INLINE CommonArgs common_args() {
    const uint32_t virtual_x_arg = get_common_arg_val<uint32_t>(0) + 1;
    const uint32_t virtual_y_arg = virtual_x_arg + grid_w;
    const uint32_t all_cores_mask_arg = virtual_y_arg + grid_h;
    return {virtual_x_arg, virtual_y_arg, all_cores_mask_arg, all_cores_mask_arg + mask_words};
}

FORCE_INLINE uint32_t source_mapping_row(uint32_t source) { return get_common_arg_val<uint32_t>(1 + source); }

// The tokens this core routes, [first, end), and the sources (mapping rows) they come from.
struct TokenRange {
    uint32_t first, end;
    uint32_t first_source, sources;
};

FORCE_INLINE TokenRange token_range(uint32_t core_index, uint32_t program_cores) {
    const uint32_t first = core_index * tokens / program_cores, end = (core_index + 1) * tokens / program_cores;
    const uint32_t first_source = end > first ? first / tokens_per_source : 0;
    const uint32_t sources = end > first ? (end - 1) / tokens_per_source - first_source + 1 : 0;
    return {first, end, first_source, sources};
}

// Entry area of core i (routing_entry_areas, expert_rows::entry_area_bytes): u32 count, then u16 expert[capacity],
// token[capacity], score[capacity], k_index[capacity].
struct EntryArea {
    uint32_t addr;
    tt_l1_ptr uint16_t* expert;
    tt_l1_ptr uint16_t* token;
    tt_l1_ptr uint16_t* score;
    tt_l1_ptr uint16_t* k_index;
    FORCE_INLINE uint32_t count() const { return *reinterpret_cast<const tt_l1_ptr uint32_t*>(addr); }
};

FORCE_INLINE EntryArea entry_area(uint32_t routing, uint32_t core, uint32_t area_bytes, uint32_t capacity) {
    const uint32_t addr = routing + routing_entry_areas + core * area_bytes;
    tt_l1_ptr uint16_t* expert = reinterpret_cast<tt_l1_ptr uint16_t*>(addr + sizeof(uint32_t));
    return {addr, expert, expert + capacity, expert + 2 * capacity, expert + 3 * capacity};
}

// The device's own mapping row, the mapping rows of this core's sources, and its tokens' expert ids and scores.
template <typename MappingAccessor, typename IndicesAccessor, typename ScoresAccessor>
FORCE_INLINE void read_routing_inputs(
    const Noc& noc,
    const MappingAccessor& mapping,
    const IndicesAccessor& indices,
    const ScoresAccessor& scores,
    uint32_t routing,
    const TokenRange& range,
    uint32_t device_id) {
    constexpr uint32_t mapping_bytes = global_experts * expert_rows::u16_bytes;
    noc.async_read(
        mapping, CoreLocalMem<uint32_t>(routing + routing_own_mapping), mapping_bytes, {.page_id = device_id}, {});
    for (uint32_t source = 0; source < range.sources; ++source) {
        noc.async_read(
            mapping,
            CoreLocalMem<uint32_t>(routing + routing_source_mappings + source * mapping_bytes),
            mapping_bytes,
            {.page_id = source_mapping_row(range.first_source + source)},
            {});
    }
    for (uint32_t token = range.first; token < range.end; ++token) {
        noc.async_read(
            indices,
            CoreLocalMem<uint32_t>(routing + routing_token_ids + (token - range.first) * index_page_bytes),
            index_page_bytes,
            {.page_id = token},
            {});
        noc.async_read(
            scores,
            CoreLocalMem<uint32_t>(routing + routing_token_scores + (token - range.first) * index_page_bytes),
            index_page_bytes,
            {.page_id = token},
            {});
    }
    noc.async_read_barrier();
}

// The mapping rows in place: the own row becomes each global expert's local expert (not_local unless the device maps
// it to itself), each source's row the local expert of the experts that source sends to this device.
FORCE_INLINE void map_to_local_experts(uint32_t routing, uint32_t sources, uint32_t device_id) {
    // this RISC alone works on the scratch after the reads: plain pointers
    tt_l1_ptr uint16_t* local_expert = reinterpret_cast<tt_l1_ptr uint16_t*>(routing + routing_own_mapping);
    uint32_t next_local_expert = 0;
    for (uint32_t expert = 0; expert < global_experts; ++expert) {
        local_expert[expert] = local_expert[expert] == device_id ? next_local_expert++ : not_local;
    }
    for (uint32_t source = 0; source < sources; ++source) {
        tt_l1_ptr uint16_t* source_expert = reinterpret_cast<tt_l1_ptr uint16_t*>(
            routing + routing_source_mappings + source * global_experts * expert_rows::u16_bytes);
        for (uint32_t expert = 0; expert < global_experts; ++expert) {
            source_expert[expert] = source_expert[expert] == device_id ? local_expert[expert] : not_local;
        }
    }
}

// This core's local entries (local expert, token, score, k) in token order, then k order, into its entry area.
FORCE_INLINE void collect_local_entries(uint32_t routing, const EntryArea& area, const TokenRange& range) {
    const tt_l1_ptr uint16_t* token_ids = reinterpret_cast<const tt_l1_ptr uint16_t*>(routing + routing_token_ids);
    const tt_l1_ptr uint16_t* token_scores =
        reinterpret_cast<const tt_l1_ptr uint16_t*>(routing + routing_token_scores);
    uint32_t entry_count = 0;
    for (uint32_t token = range.first; token < range.end; ++token) {
        const tt_l1_ptr uint16_t* source_expert = reinterpret_cast<const tt_l1_ptr uint16_t*>(
            routing + routing_source_mappings +
            (token / tokens_per_source - range.first_source) * global_experts * expert_rows::u16_bytes);
        const tt_l1_ptr uint16_t* ids = token_ids + (token - range.first) * index_page_entries;
        uint16_t experts[top_k];
        for (uint32_t k_index = 0; k_index < top_k; ++k_index) {
            const uint32_t id = ids[k_index];
            experts[k_index] = id < global_experts ? source_expert[id] : not_local;
        }
        for (uint32_t k_index = 0; k_index < top_k; ++k_index) {
            if (experts[k_index] < local_experts) {
                area.expert[entry_count] = experts[k_index];
                area.token[entry_count] = token;
                area.score[entry_count] = token_scores[(token - range.first) * index_page_entries + k_index];
                area.k_index[entry_count] = k_index;
                ++entry_count;
            }
        }
    }
    *reinterpret_cast<tt_l1_ptr uint32_t*>(area.addr) = entry_count;
    asm volatile("" ::: "memory");  // the area's stores are issued before a NoC command reads it
}

FORCE_INLINE void send_entries_to_root(
    const Noc& noc, const EntryArea& area, uint32_t area_bytes, uint32_t root_x, uint32_t root_y) {
    noc.async_write(
        CoreLocalMem<uint32_t>(area.addr),
        UnicastEndpoint{},
        area_bytes,
        {},
        {.noc_x = root_x, .noc_y = root_y, .addr = area.addr});
    noc.async_write_barrier();
    Semaphore<>(sem_entries).up(noc, root_x, root_y, 1);
    noc.async_atomic_barrier();
}

// The root's view of the routing scratch: per local expert its routed rows (count, then fill cursor) and first row,
// the local experts with rows ascending, the routed rows' tokens and the job table.
struct RoutingTable {
    tt_l1_ptr uint16_t* expert_rows;
    tt_l1_ptr uint16_t* expert_first_row;
    tt_l1_ptr uint16_t* active_experts;
    tt_l1_ptr uint16_t* row_tokens;
    tt_l1_ptr JobRecord* jobs;
    uint32_t active = 0;  // local experts with rows
    uint32_t routed_rows = 0;
};

// Rows per local expert over every core's entries, at most row_cap each.
FORCE_INLINE void count_expert_rows(
    RoutingTable& table, uint32_t routing, uint32_t program_cores, uint32_t area_bytes, uint32_t capacity) {
    for (uint32_t expert = 0; expert < local_experts; ++expert) {
        table.expert_rows[expert] = 0;
    }
    for (uint32_t core = 0; core < program_cores; ++core) {
        const EntryArea area = entry_area(routing, core, area_bytes, capacity);
        const uint32_t entry_count = area.count();
        for (uint32_t entry = 0; entry < entry_count; ++entry) {
            if (table.expert_rows[area.expert[entry]] < row_cap) {
                table.expert_rows[area.expert[entry]] = table.expert_rows[area.expert[entry]] + 1;
            }
        }
    }
}

// The local experts with rows ascending and their first rows; each expert's count becomes its fill cursor.
FORCE_INLINE void lay_out_experts(RoutingTable& table) {
    for (uint32_t expert = 0; expert < local_experts; ++expert) {
        if (table.expert_rows[expert]) {
            table.active_experts[table.active++] = expert;
            table.expert_first_row[expert] = table.routed_rows;
            table.routed_rows += table.expert_rows[expert];
            table.expert_rows[expert] = 0;
        }
    }
}

// Areas in core order = tokens ascending within an expert: each expert's rows in token order, and the same entries in
// token order for the table (routing_entry_*). Entries past an expert's row_cap are dropped, as in count_expert_rows.
FORCE_INLINE void fill_rows_and_entries(
    RoutingTable& table, uint32_t routing, uint32_t program_cores, uint32_t area_bytes, uint32_t capacity) {
    tt_l1_ptr uint16_t* entry_experts = reinterpret_cast<tt_l1_ptr uint16_t*>(routing + routing_entry_experts);
    tt_l1_ptr uint16_t* entry_tokens = reinterpret_cast<tt_l1_ptr uint16_t*>(routing + routing_entry_tokens);
    tt_l1_ptr uint16_t* entry_scores = reinterpret_cast<tt_l1_ptr uint16_t*>(routing + routing_entry_scores);
    tt_l1_ptr uint16_t* entry_k = reinterpret_cast<tt_l1_ptr uint16_t*>(routing + routing_entry_k);
    uint32_t table_entry = 0;
    for (uint32_t core = 0; core < program_cores; ++core) {
        const EntryArea area = entry_area(routing, core, area_bytes, capacity);
        const uint32_t entry_count = area.count();
        for (uint32_t entry = 0; entry < entry_count; ++entry) {
            const uint32_t expert = area.expert[entry];
            if (table.expert_rows[expert] == row_cap) {
                continue;
            }
            table.row_tokens[table.expert_first_row[expert] + table.expert_rows[expert]] = area.token[entry];
            table.expert_rows[expert] = table.expert_rows[expert] + 1;
            entry_experts[table_entry] = expert;
            entry_tokens[table_entry] = area.token[entry];
            entry_scores[table_entry] = area.score[entry];
            entry_k[table_entry] = area.k_index[entry];
            ++table_entry;
        }
    }
}

// One row tile per job: every expert's rows in jobs of at most job_rows, in expert order. Returns the job count.
FORCE_INLINE uint32_t list_jobs_in_expert_order(const RoutingTable& table) {
    uint32_t count = 0;
    for (uint32_t active_slot = 0; active_slot < table.active; ++active_slot) {
        const uint32_t expert = table.active_experts[active_slot];
        for (uint32_t first = 0; first < table.expert_rows[expert]; first += job_rows) {
            JobRecord& job = table.jobs[count];
            job.expert = expert;
            job.first_row = table.expert_first_row[expert] + first;
            job.rows = table.expert_rows[expert] - first < job_rows ? table.expert_rows[expert] - first : job_rows;
            ++count;
        }
    }
    return count;
}

// A job's deal, recorded in the entry areas (read by then) until the jobs are listed group by group.
struct Deal {
    uint16_t first_row;
    uint16_t expert;
    uint16_t rows;
    uint16_t group;
};

// Deals jobs to the expert groups (weight_stationary): each to the group with the earliest predicted finish, on a tie
// the one with the least matmul work (stream-bound jobs tie), then the lower group; at most ctl_words - 1 per group
// (cb_ctl's list; expert_groups (ctl_words - 1) is at least the most jobs of a call).
struct JobDealer {
    tt_l1_ptr Deal* deals;
    uint32_t finish[expert_groups], work[expert_groups], group_job_counts[expert_groups];

    explicit JobDealer(tt_l1_ptr Deal* deals) : deals(deals) {
#pragma GCC unroll 32
        for (uint32_t group = 0; group < expert_groups; ++group) {
            finish[group] = 0;
            work[group] = 0;
            group_job_counts[group] = 0;
        }
    }

    void deal(uint32_t job, uint32_t expert, uint32_t first_row, uint32_t rows) {
        const uint32_t job_row_tiles = expert_rows::row_tiles_of(rows);
        const uint32_t job_work = job_row_tiles > 1 ? job_row_tiles * deal_row_tile : deal_single;
        uint32_t best_group = expert_groups, best_finish = 0xFFFFFFFF, best_work = 0xFFFFFFFF;
#pragma GCC unroll 32
        for (uint32_t group = 0; group < expert_groups; ++group) {
            const uint32_t group_finish = finish[group], group_work = work[group];
            if (group_job_counts[group] + 1 < ctl_words &&
                (group_finish < best_finish || (group_finish == best_finish && group_work < best_work))) {
                best_group = group;
                best_finish = group_finish;
                best_work = group_work;
            }
        }
        finish[best_group] += job_work > deal_stream ? job_work : deal_stream;
        work[best_group] += job_work;
        ++group_job_counts[best_group];
        Deal& dealt = deals[job];
        dealt.first_row = first_row;
        dealt.expert = expert;
        dealt.rows = rows;
        dealt.group = best_group;
    }
};

// Several row tiles per job (weight_stationary): jobs of more row tiles are dealt first (every expert's full jobs of
// row_tiles row tiles as the experts are walked, then the experts' last partial jobs by row tiles, from bucket lists).
// Each group then runs its jobs in expert order, so that jobs of many and of few row tiles run at the same time on
// different groups. The deal is recorded per job (expert order) in the entry areas (read by now), then the jobs are
// written group by group (group g's from group_first_job[g]). Returns the job count.
FORCE_INLINE uint32_t deal_jobs(const RoutingTable& table, uint32_t routing, tt_l1_ptr uint16_t* group_first_job) {
    static_assert(
        expert_groups <= 32, "expert_groups <= 32: the dealer's group loops are unrolled, its state on the stack");
    constexpr uint32_t list_end = 0xFFFF;
    // until the jobs are written: bucket lists of the experts with a partial job, that job's index
    tt_l1_ptr uint16_t* bucket_next = reinterpret_cast<tt_l1_ptr uint16_t*>(routing + routing_jobs);
    tt_l1_ptr uint16_t* partial_job = bucket_next + table.active;
    tt_l1_ptr Deal* deals = reinterpret_cast<tt_l1_ptr Deal*>(routing + routing_entry_areas);
    uint32_t bucket_head[row_tiles + 1];
    for (uint32_t bucket = 0; bucket <= row_tiles; ++bucket) {
        bucket_head[bucket] = list_end;
    }
    JobDealer dealer(deals);
    uint32_t count = 0;
    for (uint32_t active_slot = 0; active_slot < table.active; ++active_slot) {
        const uint32_t expert = table.active_experts[active_slot], rows = table.expert_rows[expert];
        const uint32_t first = table.expert_first_row[expert];
        const uint32_t full = rows / job_rows, remainder = rows - full * job_rows;
        for (uint32_t full_job = 0; full_job < full; ++full_job) {
            dealer.deal(count++, expert, first + full_job * job_rows, job_rows);
        }
        if (remainder) {
            const uint32_t tiles = expert_rows::row_tiles_of(remainder);
            bucket_next[active_slot] = bucket_head[tiles];
            bucket_head[tiles] = active_slot;
            partial_job[active_slot] = count++;
        }
    }
    for (uint32_t tiles = row_tiles; tiles >= 1; --tiles) {
        for (uint32_t active_slot = bucket_head[tiles]; active_slot != list_end;
             active_slot = bucket_next[active_slot]) {
            const uint32_t expert = table.active_experts[active_slot], rows = table.expert_rows[expert];
            const uint32_t remainder = rows % job_rows;
            dealer.deal(partial_job[active_slot], expert, table.expert_first_row[expert] + rows - remainder, remainder);
        }
    }
    group_first_job[0] = 0;
    for (uint32_t group = 0; group < expert_groups; ++group) {
        group_first_job[group + 1] = group_first_job[group] + dealer.group_job_counts[group];
        dealer.group_job_counts[group] = group_first_job[group];  // write cursor
    }
    for (uint32_t job = 0; job < count; ++job) {
        const Deal dealt = deals[job];
        JobRecord& record = table.jobs[dealer.group_job_counts[dealt.group]++];
        record.expert = dealt.expert;
        record.first_row = dealt.first_row;
        record.rows = dealt.rows;
    }
    return count;
}

// The ctl block (job count, active experts, routed rows) and the table head for the placement program: routed rows,
// per-expert counts and first rows.
FORCE_INLINE void write_table_head(
    const RoutingTable& table, uint32_t routing, volatile tt_l1_ptr uint32_t* ctl, uint32_t table_jobs) {
    ctl[0] = table_jobs;
    ctl[1] = table.active;
    ctl[2] = table.routed_rows;
    *reinterpret_cast<tt_l1_ptr uint32_t*>(routing + routing_table) = table.routed_rows;
    tt_l1_ptr uint16_t* head_expert_rows = reinterpret_cast<tt_l1_ptr uint16_t*>(routing + routing_table_counts);
    tt_l1_ptr uint16_t* head_expert_first_row = reinterpret_cast<tt_l1_ptr uint16_t*>(routing + routing_table_offsets);
    uint32_t first_row = 0;
    for (uint32_t expert = 0; expert < local_experts; ++expert) {
        head_expert_rows[expert] = table.expert_rows[expert];
        head_expert_first_row[expert] = first_row;
        first_row += table.expert_rows[expert];
    }
    asm volatile("" ::: "memory");  // the table's stores are issued before NoC commands send it
}

// Every other core of the program: the table is ready.
FORCE_INLINE void signal_table_ready(const Noc& noc, const CommonArgs& common, uint32_t root_x, uint32_t root_y) {
    Semaphore<> table_ready(sem_table);
    for (uint32_t word = 0; word < mask_words; ++word) {
        uint32_t bits = get_common_arg_val<uint32_t>(common.all_cores_mask_arg + word);
        while (bits) {
            const uint32_t bit = __builtin_ctz(bits);
            bits &= bits - 1;
            const uint32_t core = word * 32 + bit;
            const uint32_t x = get_common_arg_val<uint32_t>(common.virtual_x_arg + core / grid_h);
            const uint32_t y = get_common_arg_val<uint32_t>(common.virtual_y_arg + core % grid_h);
            if (x != root_x || y != root_y) {
                table_ready.up(noc, x, y, 1);
            }
        }
    }
    noc.async_atomic_barrier();
}

// The root core: the routing table from every core's entries, the ready signal, and the table block to DRAM. Returns
// the job count.
template <typename TableAccessor>
FORCE_INLINE uint32_t build_routing_table(
    const Noc& noc,
    const TableAccessor& table_block,
    uint32_t routing,
    uint32_t program_cores,
    uint32_t area_bytes,
    uint32_t capacity,
    const CommonArgs& common,
    uint32_t root_x,
    uint32_t root_y) {
    volatile tt_l1_ptr uint32_t* ctl = reinterpret_cast<volatile tt_l1_ptr uint32_t*>(routing + routing_ctl);
    tt_l1_ptr uint16_t* group_first_job =
        reinterpret_cast<tt_l1_ptr uint16_t*>(routing + routing_ctl + expert_rows::ctl_header_bytes);
    RoutingTable table{
        .expert_rows = reinterpret_cast<tt_l1_ptr uint16_t*>(routing + routing_expert_rows),
        .expert_first_row = reinterpret_cast<tt_l1_ptr uint16_t*>(routing + routing_expert_first_rows),
        .active_experts = reinterpret_cast<tt_l1_ptr uint16_t*>(routing + routing_active_experts),
        .row_tokens = reinterpret_cast<tt_l1_ptr uint16_t*>(routing + routing_row_tokens),
        .jobs = reinterpret_cast<tt_l1_ptr JobRecord*>(routing + routing_jobs)};
    Semaphore<>(sem_entries).wait_min(program_cores - 1);
    invalidate_l1_cache();
    count_expert_rows(table, routing, program_cores, area_bytes, capacity);
    lay_out_experts(table);
    fill_rows_and_entries(table, routing, program_cores, area_bytes, capacity);
    uint32_t table_jobs;
    if constexpr (weight_stationary) {
        table_jobs = deal_jobs(table, routing, group_first_job);
    } else {
        table_jobs = list_jobs_in_expert_order(table);
    }
    write_table_head(table, routing, ctl, table_jobs);
    signal_table_ready(noc, common, root_x, root_y);
    // drained before this kernel ends
    noc.async_write(
        CoreLocalMem<uint32_t>(routing + routing_table), table_block, routing_table_bytes, {}, {.page_id = 0});
    return table_jobs;
}

// Every other core: the ctl block, the jobs and the routed rows' tokens from the root. Returns the job count.
FORCE_INLINE uint32_t read_routing_table(const Noc& noc, uint32_t routing, uint32_t root_x, uint32_t root_y) {
    volatile tt_l1_ptr uint32_t* ctl = reinterpret_cast<volatile tt_l1_ptr uint32_t*>(routing + routing_ctl);
    Semaphore<>(sem_table).wait_min(1);
    noc.async_read(
        UnicastEndpoint{},
        CoreLocalMem<uint32_t>(routing + routing_ctl),
        ctl_bytes,
        {.noc_x = root_x, .noc_y = root_y, .addr = routing + routing_ctl},
        {});
    noc.async_read_barrier();
    const uint32_t table_jobs = ctl[0];
    const uint32_t routed_rows = ctl[2];
    if (table_jobs) {
        noc.async_read(
            UnicastEndpoint{},
            CoreLocalMem<uint32_t>(routing + routing_jobs),
            expert_rows::align_up(table_jobs * expert_rows::job_record_bytes, expert_rows::l1_word_bytes),
            {.noc_x = root_x, .noc_y = root_y, .addr = routing + routing_jobs},
            {});
        noc.async_read(
            UnicastEndpoint{},
            CoreLocalMem<uint32_t>(routing + routing_row_tokens),
            expert_rows::align_up(routed_rows * expert_rows::u16_bytes, expert_rows::l1_word_bytes),
            {.noc_x = root_x, .noc_y = root_y, .addr = routing + routing_row_tokens},
            {});
        noc.async_read_barrier();
    }
    return table_jobs;
}

// This group's jobs by local index: table jobs group, group + expert_groups, ... with one row tile per job, else as
// dealt (from group_first_job[group]).
// count: the group's jobs.
struct GroupJobs : expert_rows::GroupJobs<weight_stationary, expert_groups, const tt_l1_ptr JobRecord*> {
    uint32_t count;
};

FORCE_INLINE GroupJobs group_jobs(uint32_t routing, uint32_t group, uint32_t table_jobs) {
    const tt_l1_ptr uint16_t* group_first_job =
        reinterpret_cast<const tt_l1_ptr uint16_t*>(routing + routing_ctl + expert_rows::ctl_header_bytes);
    const auto jobs = expert_rows::group_jobs<weight_stationary, expert_groups>(
        reinterpret_cast<const tt_l1_ptr JobRecord*>(routing + routing_jobs), group_first_job, group);
    // the group's jobs: from the ctl block when dealt, else its share of the table's jobs
    const uint32_t count = weight_stationary
                               ? group_first_job[group + 1] - jobs.first
                               : (table_jobs > group ? (table_jobs - group + expert_groups - 1) / expert_groups : 0);
    return {jobs, count};
}

// The compute's ctl tile: this group's job count and (weight_stationary) the row tiles of each of its jobs.
FORCE_INLINE void publish_group_jobs(CircularBuffer& cb_ctl, const GroupJobs& jobs) {
    cb_ctl.reserve_back(1);
    volatile tt_l1_ptr uint32_t* group_ctl = reinterpret_cast<volatile tt_l1_ptr uint32_t*>(cb_ctl.get_write_ptr());
    group_ctl[0] = jobs.count;
    if constexpr (weight_stationary) {
        for (uint32_t job = 0; job < jobs.count && job + 1 < ctl_words; ++job) {
            group_ctl[1 + job] = expert_rows::row_tiles_of(jobs.rows(job));
        }
    }
    cb_ctl.push_back(1);
}

// This core's W0/W1 column groups and W2 output groups (runtime args); the last of each may be half wide.
struct CoreGroups {
    uint32_t w0_w1_first_tile;  // of this core's groups in the (layer, expert) stream
    uint32_t column_groups;
    uint32_t first_output_group;
    uint32_t output_groups;
    bool half_column;  // the last column group is a half block-column
    bool half_output;  // the last output group is two tiles wide
    FORCE_INLINE GroupRun columns() const { return {column_groups, half_column}; }
    FORCE_INLINE GroupRun outputs() const { return {output_groups, half_output}; }
};

enum class Projection : bool { GateUp = false, Down = true };

// K rows of chunk `chunk` of `chunks` chunks of chunk_rows of the k_rows rows (the bias row ends the last)
FORCE_INLINE uint32_t chunk_k_rows(uint32_t chunk, uint32_t chunk_rows, uint32_t chunks, uint32_t k_rows) {
    return (chunk + 1 < chunks ? chunk_rows : k_rows - chunk * chunk_rows) + (chunk + 1 == chunks ? bias_rows : 0);
}

FORCE_INLINE uint32_t unit_padding(uint32_t blocks) { return (unit_blocks - blocks % unit_blocks) % unit_blocks; }

// CB blocks of a one-row-tile pass before its padding
inline uint32_t single_pass_blocks(const CoreGroups& groups, Projection projection) {
    uint32_t blocks = 0;
    if (projection == Projection::Down) {
        for (uint32_t output_group = 0; output_group < groups.output_groups; ++output_group) {
            blocks += expert_rows::run_blocks(
                groups.outputs().width(output_group), intermediate_tiles + bias_rows, block_tiles);
        }
        return blocks;
    }
    for (uint32_t column_group = 0; column_group < groups.column_groups; ++column_group) {
        for (uint32_t chunk = 0; chunk < single_chunks; ++chunk) {
            blocks += expert_rows::run_blocks(
                groups.columns().width(column_group),
                chunk_k_rows(chunk, single_chunk_tiles, single_chunks, hidden_tiles),
                block_tiles);
        }
    }
    return blocks;
}

// Pass pass_index of the stream: gate/up(0), gate/up(1), down(0), gate/up(2), down(1), ..., down(jobs - 1)
struct Pass {
    uint32_t job;
    Projection projection;
};

inline Pass pass_at(uint32_t pass_index, uint32_t passes, uint32_t job_count) {
    if (pass_index == 0) {
        return {0, Projection::GateUp};
    }
    if (pass_index == passes - 1) {
        return {job_count - 1, Projection::Down};
    }
    const uint32_t job = (pass_index + 1) >> 1;
    return (pass_index & 1) ? Pass{job, Projection::GateUp} : Pass{job - 1, Projection::Down};
}

// CB blocks of the whole stream: held passes of whole units, one-row-tile passes padded to whole units when held
// units exist.
FORCE_INLINE uint32_t stream_blocks(const GroupJobs& jobs, const CoreGroups& groups, uint32_t passes) {
    uint32_t total = 0;
    for (uint32_t pass_index = 0; pass_index < passes; ++pass_index) {
        const Pass pass = pass_at(pass_index, passes, jobs.count);
        if (jobs.held(pass.job)) {
            total += (pass.projection == Projection::Down ? groups.output_groups * down_chunks
                                                          : groups.column_groups * gate_up_chunks) *
                     chunk_blocks;
        } else {
            const uint32_t blocks = single_pass_blocks(groups, pass.projection);
            total += blocks + (weight_stationary ? unit_padding(blocks) : 0);
        }
    }
    return total;
}

// The weight stream's position: the pass, its unit and the unit's block, and where the unit's next tiles are in DRAM
// (byte address in its bank, that bank, the W0/W1 piece and the tiles to its end; a W2 run never leaves its bank).
struct StreamCursor {
    const GroupJobs& jobs;
    const CoreGroups& groups;
    uint32_t passes;
    uint32_t w0_w1_addr, w2_addr;
    uint32_t bank;             // this ring position's bank (its W2)
    uint32_t piece_banks_arg;  // common arg index of the first W0/W1 piece's bank
    // where the unit's next tiles are
    uint32_t addr = 0, read_bank, piece = 0, piece_left = 0;
    // the pass and its job's expert, its unit of units, the unit's block of blocks and tiles left to read
    uint32_t pass = 0, expert = 0, unit = 0, units = 0, block = 0, blocks = 0, left = 0;
    Projection projection = Projection::GateUp;
    bool held = false;

    StreamCursor(
        const GroupJobs& jobs,
        const CoreGroups& groups,
        uint32_t passes,
        uint32_t w0_w1_addr,
        uint32_t w2_addr,
        uint32_t bank,
        uint32_t piece_banks_arg) :
        jobs(jobs),
        groups(groups),
        passes(passes),
        w0_w1_addr(w0_w1_addr),
        w2_addr(w2_addr),
        bank(bank),
        piece_banks_arg(piece_banks_arg),
        read_bank(bank) {}

    // Unit unit_index of the open pass: read cursor, real tiles, CB blocks.
    void locate_unit(uint32_t unit_index, uint32_t& tiles) {
        if (projection == Projection::GateUp) {
            const uint32_t chunk_rows = held ? chunk_tiles : single_chunk_tiles,
                           chunks = held ? gate_up_chunks : single_chunks;
            const uint32_t column_group = unit_index % groups.column_groups, chunk = unit_index / groups.column_groups;
            const uint32_t width = groups.columns().width(column_group);
            // the unit's first tile in the (layer, expert) stream
            const uint32_t unit_first_tile =
                groups.w0_w1_first_tile + column_group * w0_w1_group_tiles + width * chunk * chunk_rows;
            piece = unit_first_tile / w0_w1_piece_tiles;
            piece_left = (piece + 1) * w0_w1_piece_tiles - unit_first_tile;
            read_bank = get_common_arg_val<uint32_t>(piece_banks_arg + piece);
            addr = w0_w1_addr +
                   (expert * w0_w1_piece_tiles + unit_first_tile - piece * w0_w1_piece_tiles) * weight_tile_bytes;
            tiles = width * chunk_k_rows(chunk, chunk_rows, chunks, hidden_tiles);
        } else {
            const uint32_t chunk_rows = held ? w2_chunk_rows : intermediate_tiles, chunks = held ? down_chunks : 1;
            const uint32_t output_group = unit_index / chunks, chunk = unit_index % chunks;
            const uint32_t width = groups.outputs().width(output_group);
            read_bank = bank;
            piece_left = 0xFFFFFFFF;
            addr = w2_addr + (expert * w2_slice_tiles + (groups.first_output_group + output_group) * w2_group_tiles +
                              width * chunk * chunk_rows) *
                                 weight_tile_bytes;
            tiles = width * chunk_k_rows(chunk, chunk_rows, chunks, intermediate_tiles);
        }
        blocks = held ? chunk_blocks : expert_rows::div_up(tiles, block_tiles);
    }

    // Unit `units` of a padded pass carries no data.
    void open_unit() {
        uint32_t tiles = 0;
        if (unit < units) {
            locate_unit(unit, tiles);
        } else {
            blocks = unit_padding(single_pass_blocks(groups, projection));
        }
        left = tiles;
        block = 0;
    }

    void open_pass() {
        const Pass opened = pass_at(pass, passes, jobs.count);
        projection = opened.projection;
        expert = jobs.expert(opened.job);
        held = jobs.held(opened.job);
        units = projection == Projection::GateUp ? groups.column_groups * (held ? gate_up_chunks : single_chunks)
                                                 : (held ? groups.output_groups * down_chunks : groups.output_groups);
        unit = 0;
        open_unit();
    }

    // units of the open pass, plus the padding unit of a one-row-tile pass when held units exist
    uint32_t last_unit() const { return units + (weight_stationary && !held ? 1 : 0); }

    // On to the next block that is part of the stream.
    void advance() {
        while (block == blocks) {
            if (++unit < last_unit()) {
                open_unit();
            } else if (++pass < passes) {
                open_pass();
            } else {
                return;
            }
        }
    }
};

// The bank and packet size the read command buffer is set to (one-packet read state), changed only when a packet
// needs others. The weight stream keeps dataflow_api's stateful reads: Noc sets the transaction id only with the read
// state (set_async_read_state), and this stream changes the id every block while it keeps the bank and packet size.
struct ReadState {
    uint32_t bank = 0, tiles = 0, bank_lo = 0;
    FORCE_INLINE void set(uint32_t new_bank, uint32_t new_tiles) {
        const uint64_t bank_noc = get_noc_addr_from_bank_id<true>(new_bank, 0);
        noc_async_read_one_packet_set_state<true>(bank_noc, new_tiles * weight_tile_bytes, 0);
        bank_lo = (uint32_t)bank_noc;
        bank = new_bank;
        tiles = new_tiles;
    }
};

// `tiles` tiles of the cursor's unit to `dst` under one transaction id; a packet ends where the cursor's bank piece
// does.
FORCE_INLINE void read_block(StreamCursor& cursor, ReadState& state, uint32_t dst, uint32_t tiles, uint32_t trid) {
    noc_async_read_set_trid(trid);
    for (uint32_t done = 0; done < tiles;) {
        if (cursor.piece_left == 0) {  // the stream goes on at this (layer, expert)'s start in the next bank
            cursor.read_bank = get_common_arg_val<uint32_t>(cursor.piece_banks_arg + ++cursor.piece);
            cursor.addr -= w0_w1_piece_tiles * weight_tile_bytes;
            cursor.piece_left = w0_w1_piece_tiles;
        }
        uint32_t packet = tiles - done < packet_tiles ? tiles - done : packet_tiles;
        packet = packet < cursor.piece_left ? packet : cursor.piece_left;
        if (cursor.read_bank != state.bank || packet != state.tiles) {
            state.set(cursor.read_bank, packet);
        }
        noc_async_read_one_packet_with_state_with_trid(
            state.bank_lo, cursor.addr, dst + done * weight_tile_bytes, trid);
        done += packet;
        cursor.addr += packet * weight_tile_bytes;
        cursor.piece_left -= packet;
    }
}

// The whole stream into cb_w: blocks_in_flight blocks in flight at most, one transaction id each; a block is pushed
// once its reads landed.
FORCE_INLINE void stream_weights(
    const Noc& noc, CircularBuffer& cb_w, StreamCursor& cursor, uint32_t total_blocks, uint32_t bank) {
    const uint32_t weight_slots = cb_w.get_write_ptr();
    ReadState state;
    state.set(bank, packet_tiles);
    cursor.open_pass();
    cursor.advance();
    uint32_t issued = 0, pushed = 0;
    while (pushed < total_blocks) {
        if (issued < total_blocks && issued - pushed < blocks_in_flight &&
            cb_w.pages_reservable_at_back(block_tiles * (issued - pushed + 1))) {
            const uint32_t tiles = cursor.left < block_tiles ? cursor.left : block_tiles;
            if (tiles) {  // a unit's padding blocks carry no data
                read_block(
                    cursor,
                    state,
                    weight_slots + (issued % block_slots) * block_bytes,
                    tiles,
                    issued % transaction_ids + 1);
            }
            ++issued;
            cursor.left -= tiles;
            ++cursor.block;
            cursor.advance();
        }
        if (pushed < issued && noc.is_read_trid_flushed(pushed % transaction_ids + 1)) {
            cb_w.push_back(block_tiles);
            ++pushed;
        }
    }
    noc_async_read_set_trid(0);
}

}  // namespace

void kernel_main() {
    CircularBuffer cb_w(get_named_compile_time_arg_val("cb_w"));
    CircularBuffer cb_routing(get_named_compile_time_arg_val("cb_routing"));
    CircularBuffer cb_ctl(get_named_compile_time_arg_val("cb_ctl"));
    constexpr auto indices_args = TensorAccessorArgs<0>();
    constexpr auto scores_args = TensorAccessorArgs<indices_args.next_compile_time_args_offset()>();
    constexpr auto mapping_args = TensorAccessorArgs<scores_args.next_compile_time_args_offset()>();
    constexpr auto table_args = TensorAccessorArgs<mapping_args.next_compile_time_args_offset()>();

    uint32_t arg = 0;
    const uint32_t bank = get_arg_val<uint32_t>(arg++);  // this ring position's bank (its W2)
    const uint32_t indices_addr = get_arg_val<uint32_t>(arg++);
    const uint32_t scores_addr = get_arg_val<uint32_t>(arg++);
    const uint32_t mapping_addr = get_arg_val<uint32_t>(arg++);
    const uint32_t table_addr = get_arg_val<uint32_t>(arg++);
    const uint32_t w0_w1_addr = get_arg_val<uint32_t>(arg++);
    const uint32_t w2_addr = get_arg_val<uint32_t>(arg++);
    const uint32_t w0_w1_first_tile = get_arg_val<uint32_t>(arg++);
    const uint32_t column_groups = get_arg_val<uint32_t>(arg++);
    const uint32_t first_output_group = get_arg_val<uint32_t>(arg++), output_groups = get_arg_val<uint32_t>(arg++);
    const uint32_t group = get_arg_val<uint32_t>(arg++);
    const uint32_t core_index = get_arg_val<uint32_t>(arg++), program_cores = get_arg_val<uint32_t>(arg++);
    const uint32_t root_x = get_arg_val<uint32_t>(arg++), root_y = get_arg_val<uint32_t>(arg++);
    const uint32_t device_id = get_arg_val<uint32_t>(arg++);
    const bool half_column = get_arg_val<uint32_t>(arg++) == 1;
    const bool half_output = get_arg_val<uint32_t>(arg++) == 1;
    const CommonArgs common = common_args();
    const bool root = core_index == 0;

    // ---- routing ----
    const TokenRange range = token_range(core_index, program_cores);
    // entries per token at most: one per k
    const uint32_t capacity = ((tokens + program_cores - 1) / program_cores) * top_k;  // per entry area
    const uint32_t area_bytes = expert_rows::entry_area_bytes(capacity);
    const Noc noc;
    cb_routing.reserve_back(1);
    const uint32_t routing = cb_routing.get_write_ptr();
    const auto mapping = TensorAccessor(mapping_args, mapping_addr, mapping_page_bytes);
    const auto indices = TensorAccessor(indices_args, indices_addr, index_page_bytes);
    const auto scores = TensorAccessor(scores_args, scores_addr, index_page_bytes);
    read_routing_inputs(noc, mapping, indices, scores, routing, range, device_id);
    map_to_local_experts(routing, range.sources, device_id);
    const EntryArea area = entry_area(routing, core_index, area_bytes, capacity);
    collect_local_entries(routing, area, range);
    if (!root) {
        send_entries_to_root(noc, area, area_bytes, root_x, root_y);
    }
    const auto table_block = TensorAccessor(table_args, table_addr, routing_table_bytes);
    const uint32_t table_jobs =
        root ? build_routing_table(
                   noc, table_block, routing, program_cores, area_bytes, capacity, common, root_x, root_y)
             : read_routing_table(noc, routing, root_x, root_y);
    asm volatile("" ::: "memory");  // every table store above is issued before the CB is pushed
    cb_routing.push_back(1);
    const GroupJobs jobs = group_jobs(routing, group, table_jobs);
    publish_group_jobs(cb_ctl, jobs);
    if (jobs.count == 0) {
        noc.async_write_barrier();  // the root's table block
        return;
    }

    // ---- weight stream: passes gate/up(0), gate/up(1), down(0), gate/up(2), down(1), ..., down(jobs - 1) ----
    const CoreGroups groups{
        .w0_w1_first_tile = w0_w1_first_tile,
        .column_groups = column_groups,
        .first_output_group = first_output_group,
        .output_groups = output_groups,
        .half_column = half_column,
        .half_output = half_output};
    const uint32_t passes = 2 * jobs.count;
    StreamCursor cursor(jobs, groups, passes, w0_w1_addr, w2_addr, bank, common.piece_banks_arg);
    stream_weights(noc, cb_w, cursor, stream_blocks(jobs, groups, passes), bank);
    noc.async_write_barrier();  // the root's table block
}
