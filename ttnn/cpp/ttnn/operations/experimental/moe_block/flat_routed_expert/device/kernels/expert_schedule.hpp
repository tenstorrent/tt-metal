// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0
//
// The call's expert schedule: which of the chip's local experts run this call, in which order, in which pieces. Every
// data-movement kernel of the op builds it on device from the routing's token counts; they all read the same inputs
// and run the same code, so every core builds the same schedule and the roles stay in step without a host round trip.
// A data-movement kernel hands it to the compute kernel on its core as the schedule page (schedule_meta.hpp).
//
// ---- Inputs ----
//
// Three uint32 row-major DRAM tensors the routing writes on device each call, one page each:
//
//   tensor             | shape  | word i
//   -------------------+--------+-----------------------------------------------------------------------------------
//   counts             | [1, G] | tokens routed to global expert i this call (G: the model's routed experts)
//   regions            | [1, G] | the first row of global expert i's tokens in the dispatch buffer (a multiple of 32)
//   global_expert_ids  | [1, E] | the global id of this chip's local expert i (E: the chip's experts)
//
// and kScheduleArgs runtime args (from the index the kernel passes to load_schedule; the same on every core):
//
//   arg | contents
//   ----+--------------------------------------------------------------------------------------------------------------
//   0   | counts address
//   1   | regions address
//   2   | bytes of a counts / regions row (4 G)
//   3   | lowest token count served: an expert with fewer tokens is skipped (weights included)
//   4   | highest token count served: an expert with more tokens is skipped
//   5   | global_expert_ids address
//   6   | bytes of global_expert_ids (4 E)
//
// Example: G = 8, E = 4, chip 1 holds global experts 4..7, 128 rows per sub-block:
//
//   global_expert_ids = [  4,   5,   6,   7]
//   counts            = [.., .., .., .., 300,   0,  40,   7]
//   regions           = [.., .., .., ..,   0, 320, 320, 384]
//
// Local expert 0 has 300 tokens in dispatch rows 0..299, local expert 1 none, local expert 2 has 40 tokens in rows
// 320..359 and local expert 3 has 7 tokens in rows 384..390.
//
// ---- The schedule ----
//
// A list of entries; an entry is a run of sub-blocks (rows_per_sub_block token rows each, an expert's last one short)
// of one local expert. An entry's gate/up weights come from a load: one expert's gate/up weights, streamed into one
// region of the gate/up cores' weight ring and freed after the load's last use. Without pinning every expert with
// tokens is one entry and one load, most sub-blocks first (ties: local expert order): the call's tail is the last
// expert's compute once its weights have landed, so the expert with the fewest sub-blocks goes last. The example:
//
//   entry | expert | first row | tokens | sub-blocks | load | load's last use
//   ------+--------+-----------+--------+------------+------+----------------
//   0     | 0      | 0         | 300    | 3          | 0    | yes
//   1     | 2      | 320       | 40     | 1          | 1    | yes
//   2     | 3      | 384       | 7      | 1          | 2    | yes
//
// Every role walks the entries in this order: the x path, the gate/up and down compute, the y writes.
//
// ---- Pinning (pin_chunk_sub_blocks != 0) ----
//
// With a skewed routing one hot expert holds most of the tokens and the others a few each. An expert of a few rows is
// bound by its weight stream: it has little compute to hide its weight reads behind, while the hot expert's compute
// leaves the weight stream idle. Pinning slices the hot expert into chunks and interleaves the cold experts between
// them, so each cold expert's weights stream in while a hot chunk computes:
//
//   plain:   H H H H H H | c1 | c2 | c3      (c1..c3 each wait for their weights, nothing to overlap them with)
//   pinned:  H H | c1 | H H | c2 | H H | c3  (c1's weights stream in while the first H chunk computes, and so on)
//
// The hot expert's gate/up weights are one load that stays in its ring region over all its chunks (pinned there);
// the cold experts' loads cycle through the other regions. Experts too large to be weight-bound go after the last
// chunk. Down weights are not pinned: each chunk reads the hot expert's down weights again, unless
// down_weights_per_load. Pinning applies only when the hot expert has at least pin_min_ratio times the other experts'
// mean sub-blocks (below that the re-reads cost more than the overlap saves), and is dropped when the ring regions
// cannot be assigned (below).
//
// ---- Ring regions ----
//
// Loads land in the gate/up weight ring in load order, one region each. The ring's receivers grant slots by count and
// the compute frees a load's slots after its last use, so load l lands in the region freed by the (l - R)-th load to
// retire (R: gate_up_ring_regions), and that retirement must come before load l's first use, or the schedule would
// deadlock. The plain order always satisfies it (load l in region l % R); a pinned order that does not is dropped.
#pragma once

#include <cstdint>

#include "api/dataflow/dataflow_api.h"
#include "flat_accessors.hpp"
#include "flat_options.hpp"
#include "schedule_meta.hpp"
#include "ttnn/cpp/ttnn/operations/experimental/moe_block/common/kernels/moe_block_args.hpp"

// the schedule's runtime args (the table above)
constexpr uint32_t kScheduleArgs = 7;
// row tiles in a full sub-block
constexpr uint32_t kSubBlockRowTiles = flat_options::rows_per_sub_block / 32;

// CB 7 is the schedule's scratch, four buffers of schedule_scratch_bytes: the BRISC's two, then the NCRISC's two
// (a RISC reads the counts and ids rows into its first buffer, the regions row into its second).
enum class ScratchOwner { Brisc, Ncrisc };
inline uint32_t schedule_scratch(ScratchOwner owner) {
    return get_write_ptr(tt::CBIndex::c_7) +
           (owner == ScratchOwner::Ncrisc ? 2 * flat_options::schedule_scratch_bytes : 0);
}

struct ExpertSchedule {
    uint32_t entries = 0;           // schedule entries
    uint32_t total_sub_blocks = 0;  // sub-blocks over all entries
    // per entry
    uint32_t first_row[schedule_page::kMaxEntries];   // its first row in the dispatch buffer (a multiple of 32)
    uint32_t tokens[schedule_page::kMaxEntries];      // its tokens
    uint16_t sub_blocks[schedule_page::kMaxEntries];  // its sub-blocks
    uint8_t expert[schedule_page::kMaxEntries];       // its local expert
    uint8_t load[schedule_page::kMaxEntries];         // its gate/up weight load
    uint8_t last_use[schedule_page::kMaxEntries];     // 1: the load's last use
    // per load, in stream order
    uint32_t loads = 0;
    uint8_t load_expert[flat_options::max_experts];  // its local expert
    uint8_t load_region[flat_options::max_experts];  // its ring region
    uint32_t max_tokens = 0;                         // the most tokens of an entry
};

// Row tiles holding tokens in entry `entry`'s last sub-block.
inline uint32_t last_sub_block_row_tiles(const ExpertSchedule& schedule, uint32_t entry) {
    const uint32_t rows = schedule.tokens[entry] - (schedule.sub_blocks[entry] - 1u) * flat_options::rows_per_sub_block;
    return (rows + 31) / 32;
}

// Row tiles holding tokens in the schedule's sub-block `sub_block` (counted over all entries, in schedule order): a
// full sub-block's for all but an entry's last. The x path ships only those; the compute computes only those.
inline uint32_t sub_block_row_tiles(const ExpertSchedule& schedule, uint32_t sub_block) {
    for (uint32_t entry = 0; entry < schedule.entries; ++entry) {
        if (sub_block < schedule.sub_blocks[entry]) {
            return sub_block + 1u == schedule.sub_blocks[entry] ? last_sub_block_row_tiles(schedule, entry)
                                                                : kSubBlockRowTiles;
        }
        sub_block -= schedule.sub_blocks[entry];
    }
    return kSubBlockRowTiles;
}

// The loads' ring regions from the order the loads retire in (see Ring regions); false if some load would wait on a
// retirement that comes after its first use.
inline bool assign_ring_regions(ExpertSchedule& schedule) {
    constexpr uint32_t regions = flat_options::gate_up_ring_regions;
    constexpr uint8_t kNone = 0xFF;
    // (8-bit: entries < kMaxEntries < 255; the RISC's local memory holds the stack, the schedule and the pin copy)
    uint8_t retire_entry[flat_options::max_experts], retire_load[flat_options::max_experts];
    uint8_t first_use[flat_options::max_experts];
    uint32_t retired = 0;
    for (uint32_t load = 0; load < schedule.loads; ++load) {
        first_use[load] = kNone;
    }
    for (uint32_t entry = 0; entry < schedule.entries; ++entry) {
        if (first_use[schedule.load[entry]] == kNone) {
            first_use[schedule.load[entry]] = entry;
        }
        if (schedule.last_use[entry]) {
            retire_entry[retired] = entry;
            retire_load[retired++] = schedule.load[entry];
        }
    }
    for (uint32_t load = 0; load < schedule.loads; ++load) {
        if (load < regions) {
            schedule.load_region[load] = load;
            continue;
        }
        if (retire_entry[load - regions] >= first_use[load]) {
            return false;
        }
        schedule.load_region[load] = schedule.load_region[retire_load[load - regions]];
    }
    return true;
}

// Pinning (see above): the hot expert in chunks, the cold experts between them. Leaves the plain order when pinning
// does not apply.
inline void pin_hot_expert(ExpertSchedule& schedule) {
    if constexpr (flat_options::pin_chunk_sub_blocks == 0) {
        return;
    }
    // (1 when off: the code below is compiled either way)
    constexpr uint32_t chunk_sub_blocks = flat_options::pin_chunk_sub_blocks ? flat_options::pin_chunk_sub_blocks : 1;
    // an expert of at most this many sub-blocks is weight-bound: worth hiding under a hot chunk
    constexpr uint32_t kColdSubBlocks = 2;
    if (schedule.entries < 2 || schedule.entries > flat_options::max_experts) {
        return;
    }
    uint32_t hot = 0;
    for (uint32_t entry = 1; entry < schedule.entries; ++entry) {
        hot = schedule.sub_blocks[entry] > schedule.sub_blocks[hot] ? entry : hot;
    }
    const uint32_t hot_sub_blocks = schedule.sub_blocks[hot];
    if constexpr (flat_options::pin_min_ratio > 0) {
        uint32_t others = 0;
        for (uint32_t entry = 0; entry < schedule.entries; ++entry) {
            others += entry == hot ? 0 : schedule.sub_blocks[entry];
        }
        if (hot_sub_blocks * (schedule.entries - 1) < flat_options::pin_min_ratio * others) {
            return;
        }
    }
    // the other experts in their new order: the cold ones first (they go between the hot chunks), then the larger ones
    uint8_t others[flat_options::max_experts];
    uint32_t cold = 0, other_count = 0;
    for (uint32_t pass = 0; pass < 2; ++pass) {
        for (uint32_t entry = 0; entry < schedule.entries; ++entry) {
            if (entry != hot && (schedule.sub_blocks[entry] <= kColdSubBlocks) == (pass == 0)) {
                others[other_count++] = entry;
                cold += pass == 0;
            }
        }
    }
    uint32_t chunks = hot_sub_blocks / chunk_sub_blocks;
    chunks = chunks < cold + 1 ? chunks : cold + 1;  // one chunk per cold expert, and one more
    if (chunks < 2) {
        return;
    }
    // the plain entries (at most max_experts: a whole ExpertSchedule copy overflows the RISC stack at 64 experts)
    struct {
        uint32_t entries, first_row[flat_options::max_experts], tokens[flat_options::max_experts];
        uint16_t sub_blocks[flat_options::max_experts];
        uint8_t expert[flat_options::max_experts];
    } plain;
    plain.entries = schedule.entries;
    for (uint32_t entry = 0; entry < schedule.entries; ++entry) {
        plain.first_row[entry] = schedule.first_row[entry];
        plain.tokens[entry] = schedule.tokens[entry];
        plain.sub_blocks[entry] = schedule.sub_blocks[entry];
        plain.expert[entry] = schedule.expert[entry];
    }
    constexpr uint32_t rows_per_sub_block = flat_options::rows_per_sub_block;
    uint32_t placed = 0, hot_done = 0, next_other = 0;
    auto put =
        [&](uint32_t source, uint32_t first_row, uint32_t tokens, uint32_t sub_blocks, uint32_t load, bool last_use) {
            schedule.expert[placed] = plain.expert[source];
            schedule.first_row[placed] = first_row;
            schedule.tokens[placed] = tokens;
            schedule.sub_blocks[placed] = sub_blocks;
            schedule.load[placed] = load;
            schedule.last_use[placed] = last_use;
            ++placed;
        };
    // the hot expert is load 0; other expert i (in the new order) is load 1 + i
    auto put_other = [&](uint32_t i) {
        const uint32_t source = others[i];
        put(source, plain.first_row[source], plain.tokens[source], plain.sub_blocks[source], 1 + i, true);
    };
    for (uint32_t chunk = 0; chunk < chunks; ++chunk) {
        const uint32_t sub_blocks = hot_sub_blocks / chunks + (chunk < hot_sub_blocks % chunks ? 1 : 0);
        const uint32_t tokens_left = plain.tokens[hot] - hot_done * rows_per_sub_block;
        const uint32_t tokens =
            tokens_left < sub_blocks * rows_per_sub_block ? tokens_left : sub_blocks * rows_per_sub_block;
        put(hot, plain.first_row[hot] + hot_done * rows_per_sub_block, tokens, sub_blocks, 0, chunk + 1 == chunks);
        hot_done += sub_blocks;
        if (next_other < other_count) {
            put_other(next_other++);
        }
    }
    while (next_other < other_count) {
        put_other(next_other++);
    }
    schedule.entries = placed;
    schedule.load_expert[0] = plain.expert[hot];
    for (uint32_t i = 0; i < other_count; ++i) {
        schedule.load_expert[1 + i] = plain.expert[others[i]];
    }
    if (!assign_ring_regions(schedule)) {  // back to the plain order
        schedule.entries = plain.entries;
        for (uint32_t entry = 0; entry < plain.entries; ++entry) {
            schedule.first_row[entry] = plain.first_row[entry];
            schedule.tokens[entry] = plain.tokens[entry];
            schedule.sub_blocks[entry] = plain.sub_blocks[entry];
            schedule.expert[entry] = plain.expert[entry];
            schedule.load[entry] = entry;
            schedule.last_use[entry] = 1;
            schedule.load_expert[entry] = plain.expert[entry];
        }
    }
}

// The plain schedule from the local experts' token counts alone (tokens_of(e) / first_row_of(e): local expert e's
// count and first dispatch row), before pinning.
template <typename TokensOf, typename FirstRowOf>
inline void build_schedule(
    ExpertSchedule& schedule,
    uint32_t local_experts,
    uint32_t lowest_tokens,
    uint32_t highest_tokens,
    TokensOf tokens_of,
    FirstRowOf first_row_of) {
    constexpr uint32_t rows_per_sub_block = flat_options::rows_per_sub_block;
    schedule.entries = 0;
    schedule.total_sub_blocks = 0;
    schedule.max_tokens = 0;
    for (uint32_t expert = 0; expert < local_experts; ++expert) {
        const uint32_t tokens = tokens_of(expert);
        if (tokens == 0 || tokens < lowest_tokens || tokens > highest_tokens) {
            continue;
        }
        const uint32_t entry = schedule.entries++;
        schedule.expert[entry] = expert;
        schedule.tokens[entry] = tokens;
        schedule.first_row[entry] = first_row_of(expert);
        schedule.sub_blocks[entry] = (tokens + rows_per_sub_block - 1) / rows_per_sub_block;
        schedule.load[entry] = entry;
        schedule.last_use[entry] = 1;
        schedule.load_expert[entry] = expert;
        schedule.total_sub_blocks += schedule.sub_blocks[entry];
        schedule.max_tokens = tokens > schedule.max_tokens ? tokens : schedule.max_tokens;
    }
    schedule.loads = schedule.entries;
    // most sub-blocks first (ties: local expert order; an insertion sort, stable)
    for (uint32_t entry = 1; entry < schedule.entries; ++entry) {
        for (uint32_t at = entry; at > 0 && schedule.sub_blocks[at] > schedule.sub_blocks[at - 1]; --at) {
            const uint32_t first_row = schedule.first_row[at], tokens = schedule.tokens[at];
            const uint16_t sub_blocks = schedule.sub_blocks[at];
            const uint8_t expert = schedule.expert[at];
            schedule.first_row[at] = schedule.first_row[at - 1];
            schedule.tokens[at] = schedule.tokens[at - 1];
            schedule.sub_blocks[at] = schedule.sub_blocks[at - 1];
            schedule.expert[at] = schedule.expert[at - 1];
            schedule.load_expert[at] = schedule.expert[at];
            schedule.first_row[at - 1] = first_row;
            schedule.tokens[at - 1] = tokens;
            schedule.sub_blocks[at - 1] = sub_blocks;
            schedule.expert[at - 1] = expert;
            schedule.load_expert[at - 1] = expert;
        }
    }
}

// Reads the inputs (runtime args from `args`, see Inputs) into `scratch` (this RISC's two buffers of CB 7:
// schedule_scratch) and builds the schedule: the plain order, pinned when it applies, its ring regions.
template <uint32_t local_experts>
inline void load_schedule(ExpertSchedule& schedule, uint32_t args, uint32_t scratch) {
    static_assert(local_experts <= flat_options::max_experts);
    constexpr uint32_t buffer_bytes = flat_options::schedule_scratch_bytes;
    const uint32_t counts_address = moe_block_args::arg(args), regions_address = moe_block_args::arg(args + 1);
    const uint32_t row_bytes = moe_block_args::arg(args + 2);
    const uint32_t lowest_tokens = moe_block_args::arg(args + 3), highest_tokens = moe_block_args::arg(args + 4);
    const uint32_t ids_address = moe_block_args::arg(args + 5), ids_bytes = moe_block_args::arg(args + 6);
    const uint32_t row_read_bytes = (row_bytes + 63) / 64 * 64;
    // the ids after the counts row (host: schedule_scratch_bytes holds both, each rounded up to 64 B)
    const uint32_t ids_l1 = scratch + row_read_bytes;
    const auto counts_reader = TensorAccessor(flat_acc::counts, counts_address, row_bytes);
    const auto regions_reader = TensorAccessor(flat_acc::regions, regions_address, row_bytes);
    const auto ids_reader = TensorAccessor(flat_acc::ids, ids_address, ids_bytes);
    noc_async_read(counts_reader.get_noc_addr(0), scratch, row_read_bytes);
    noc_async_read(regions_reader.get_noc_addr(0), scratch + buffer_bytes, row_read_bytes);
    noc_async_read(ids_reader.get_noc_addr(0), ids_l1, (ids_bytes + 63) / 64 * 64);
    noc_async_read_barrier();
    volatile tt_l1_ptr uint32_t* counts = reinterpret_cast<volatile tt_l1_ptr uint32_t*>(scratch);
    volatile tt_l1_ptr uint32_t* regions = reinterpret_cast<volatile tt_l1_ptr uint32_t*>(scratch + buffer_bytes);
    volatile tt_l1_ptr uint32_t* ids = reinterpret_cast<volatile tt_l1_ptr uint32_t*>(ids_l1);
    build_schedule(
        schedule,
        local_experts,
        lowest_tokens,
        highest_tokens,
        [&](uint32_t expert) { return counts[ids[expert]]; },
        [&](uint32_t expert) { return regions[ids[expert]]; });
    pin_hot_expert(schedule);
    assign_ring_regions(schedule);  // (the plain order: always assignable)
}

// Hands the schedule to this core's compute kernel: one page of `page_cb` (schedule_meta.hpp).
inline void publish_schedule(const ExpertSchedule& schedule, uint32_t page_cb) {
    static_assert(schedule_page::kWords * 4 <= flat_options::schedule_page_bytes, "CB 6 page (schedule_page_bytes)");
    cb_reserve_back(page_cb, 1);
    volatile tt_l1_ptr uint32_t* page = reinterpret_cast<volatile tt_l1_ptr uint32_t*>(get_write_ptr(page_cb));
    page[schedule_page::kEntries] = schedule.entries;
    page[schedule_page::kTotalSubBlocks] = schedule.total_sub_blocks;
    for (uint32_t entry = 0; entry < schedule.entries; ++entry) {
        const uint32_t load = schedule.load[entry];
        page[schedule_page::kSubBlocks + entry] = schedule.sub_blocks[entry];
        page[schedule_page::kLoadWord + entry] =
            schedule_page::load_word(load, schedule.load_region[load], schedule.last_use[entry] != 0);
        page[schedule_page::kLastRowTiles + entry] = last_sub_block_row_tiles(schedule, entry);
    }
    cb_push_back(page_cb, 1);
}
