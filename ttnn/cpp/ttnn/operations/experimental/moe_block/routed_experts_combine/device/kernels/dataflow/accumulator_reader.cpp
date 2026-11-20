// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

// accumulator_reader: feeds an accumulator's compute every row homed on it, local and from every other chip
// Runs on: combine accumulators (alone, or as the accumulator role of combine_port_or_accumulator_reader.cpp),
//          RISCV_1 (NCRISC), NoC 0 (combine dm_kernel: RISCV_1 is pinned to NoC 0)
// Responsibilities:
//  - reads the device plan (DRAM) and the landed rows over NoC 0: the L1 hand-off ring the flat expert's y writers
//    fill, the N - 1 per-source landing rings in its L1 (landing on the flat expert cores: in their L1), trailer-polled
//  - pushes each row + its gate weight + a meta page to the compute (cb_row / cb_w / cb_meta), then an end marker
//  - returns a credit (or shared_landing: a slot grant) per taken row to the home port facing its source
//  (credits_to_return_offset / MISC)
//  - zeroes this core's state per call; accumulator 0 is the chip's zero gate (combine_zero_gate.hpp)

// Accumulator reader: its local rows (the L1 hand-off ring) and its N - 1 rings (one per source chip, each up to this
// call's count in the device plan); a consumed slot is one credit, counted on the port that faces the row's source on
// this accumulator's link to it (that port sends the credits back). Each row goes to the compute with its gate weight.
// It also zeroes this core's state for the call (combine_zero_gate.hpp); accumulator 0's reader is its chip's gate.
// shared_landing (the shared landing pool): no rings - the pool's slots granted to sources by id (grants_per_source
// grants outstanding a source at most), each grant written into the home port's grant ring (grant_ids_offset: per
// (source, accumulator) grants_per_source words, slot | tag << 24) and counted (credits_to_return_offset / MRTOT, as
// the credits are with rings): the port forwards them in its snapshots. Taken slots go back on the free list and to the
// sources with rows still to grant.

#include "ttnn/cpp/ttnn/operations/experimental/moe_block/common/kernels/moe_block_args.hpp"
#include <cstdint>
#include "api/dataflow/dataflow_api.h"
#include "combine_zero_gate.hpp"
#include "combine_pool.hpp"
#include "combine_formats.hpp"
#include "combine_options.hpp"
#include "../combine_names.hpp"
#include "ttnn/cpp/ttnn/operations/experimental/moe_block/common/kernels/combine_plan_words.hpp"

// Compile-time args: all named (combine_names.hpp), the same whether this role runs alone or in a two-role kernel;
// the device plan's accessor args are the only positional ones
constexpr uint32_t cb_row = get_named_compile_time_arg_val(combine_names::kRowCb);
constexpr uint32_t cb_meta = get_named_compile_time_arg_val(combine_names::kMetaCb);
constexpr uint32_t cb_tab = get_named_compile_time_arg_val(combine_names::kAccumulatorTableCb);
constexpr uint32_t row_bytes = get_named_compile_time_arg_val(combine_names::kRowBytes);
constexpr uint32_t SLOT = get_named_compile_time_arg_val(combine_names::kLandingSlotBytes);
// landing pool slots (one ring per source, carved per call)
constexpr uint32_t landing_slots_per_accumulator =
    get_named_compile_time_arg_val(combine_names::kLandingSlotsPerAccumulator);
constexpr uint32_t A = get_named_compile_time_arg_val(combine_names::kAccumulatorCount);
constexpr uint32_t accumulators_padded = (A + 3) / 4 * 4;
constexpr uint32_t row_tiles = get_named_compile_time_arg_val(combine_names::kRowTiles);
constexpr uint32_t N = get_named_compile_time_arg_val(combine_names::kChipCount);
constexpr uint32_t L = get_named_compile_time_arg_val(combine_names::kFabricLinks);
constexpr uint32_t credits_to_return_offset = get_named_compile_time_arg_val(combine_names::kCreditsToReturnOffset),
                   MISC = get_named_compile_time_arg_val(combine_names::kMiscWordsOffset),
                   RINGS = get_named_compile_time_arg_val(combine_names::kRingsOffset),
                   LAYO = get_named_compile_time_arg_val(combine_names::kRingLayoutsToForwardOffset);
constexpr uint32_t cb_w = get_named_compile_time_arg_val(combine_names::kWeightCb);
constexpr uint32_t MTAKEN = combine_formats::kMiscTaken, MRTOT = combine_formats::kMiscRowsConsumed;
// MISC words (combine_formats.hpp): the local ring's landed[] array and hand-off base; kMiscPortLanded /
// kMiscPortHandoffBase on an accumulator's own core, kMiscAccLandedOnPort / kMiscAccHandoffBaseOnPort when ports
// share it
constexpr uint32_t ALAND = get_named_compile_time_arg_val(combine_names::kLocalRingLandedWord);
constexpr uint32_t ALBASE = get_named_compile_time_arg_val(combine_names::kLocalRingHandoffBaseWord);
// the local ring (arena-relative)
constexpr uint32_t LOC_OFF = get_named_compile_time_arg_val(combine_names::kLocalRingOffset);
constexpr uint32_t WORDS_OFF = get_named_compile_time_arg_val(combine_names::kYWriterWordsOffset);
constexpr uint32_t BAR_SLOT = get_named_compile_time_arg_val(combine_names::kBarrierCallSlot);
// (the device plan)
constexpr auto plan_args =
    TensorAccessorArgs<get_named_compile_time_arg_val(combine_names::kAccumulatorReaderPlanAccessorFirstArg)>();
constexpr uint32_t LASTPB =
    row_bytes;  // where the row's trailer starts (formats::kTrailerBytes; one fabric packet a row)
// The shared landing (combine_options.hpp): slots granted to sources by id, at most grants_per_source outstanding a
// source, the ids written at grant_ids_offset in the home port's state
constexpr bool shared_landing = combine_options::shared_landing;
constexpr uint32_t grants_per_source = combine_options::shared_landing_grants_per_source;
constexpr uint32_t grant_ids_offset = combine_options::shared_landing_grant_ids_offset;
static_assert(
    !shared_landing || landing_slots_per_accumulator <= 1024,
    "shared landing: slot ids 16-bit, the owners / free list in CB_TAB (acc_tab_bytes)");
// Landing on the flat expert cores (CombineLanding::FlatCores): the rings' slots there - ring slot i is pool slot
// g = accumulator_index x landing_slots_per_accumulator + i of the flat expert cores' free arena (combine_pool.hpp; its
// args after the ports' x, y). A row landed once its trailer reads so (32 B reads, several slots a source at once);
// taking it clears the trailer there before the credit goes back (trailers zeroed each call). landing_reads_in_flight
// row reads go at once.
constexpr bool landing_on_flat_expert_cores = combine_options::landing_on_flat_expert_cores;
constexpr uint32_t landing_reads_in_flight = combine_options::landing_reads_in_flight;

namespace {
namespace formats = combine_formats;

// A row landed in a slot once both arrival flags of its trailer are set (combine_formats.hpp: the row's trailer).
FORCE_INLINE bool row_landed(volatile tt_l1_ptr uint32_t* trailer) {
    return trailer[formats::kTrailerFlagLow] != 0 && trailer[formats::kTrailerFlagHigh] != 0;
}
// A slot in this core's L1 reads empty again once its trailer's flags are cleared.
FORCE_INLINE void mark_slot_empty(volatile tt_l1_ptr uint32_t* trailer) {
    trailer[formats::kTrailerFlagLow] = 0;
    trailer[formats::kTrailerFlagHigh] = 0;
}
// A slot on another core (a flat-expert pool slot) reads empty again once its whole trailer is zeroed: written from
// this core's hardware zeros region.
FORCE_INLINE void mark_remote_slot_empty(uint64_t slot_noc_addr) {
    noc_async_write(MEM_ZEROS_BASE, slot_noc_addr + LASTPB, formats::kTrailerBytes);
}
// The compute's index of a row's token: tokens are dealt to a chip's A accumulators round-robin (token % A), so this
// accumulator holds the token as its token / A.
FORCE_INLINE uint32_t local_token(volatile tt_l1_ptr uint32_t* trailer) {
    return formats::header_token(trailer[formats::kTrailerHeader]) / A;
}
}  // namespace

#ifdef COMBINE_ROLE_FUNCTION_NAME  // (one role of a two-role kernel: combine_port_or_accumulator_reader / _writer.cpp)
void COMBINE_ROLE_FUNCTION_NAME() {
#else
void kernel_main() {
    moe_block_args::load();  // (a two-role kernel loads the core's args once, then picks this chip's role)
#endif
    size_t a = 0;
    const uint32_t area = moe_block_args::arg(a++);  // the arena (same address on every core and chip)
    const uint32_t plan = moe_block_args::arg(a++);
    const uint32_t accumulator_index = moe_block_args::arg(a++);
    const uint32_t my_chip = moe_block_args::arg(a++);  // this chip's index (its own ring is not there)
    const uint32_t barrier_addr = moe_block_args::arg(a++);
    const uint32_t gate_x = moe_block_args::arg(a++), gate_y = moe_block_args::arg(a++);
    // this call's zeroing: the state block and every landing slot's trailer (a slot is taken when its trailer flags
    // are set)
    const uint32_t call = rec::next_call(barrier_addr, BAR_SLOT);
    rec::zero_l1(area, RINGS);
    const uint32_t pool_a = 7 + 2 * N;  // (landing on the flat expert cores: the pool's args, after the ports' x, y)
    auto pool_noc = [&](uint32_t i) -> uint64_t {  // my ring slot i's NoC address on its flat expert core
        return rec_pool_noc(pool_a, area, SLOT, accumulator_index * landing_slots_per_accumulator + i);
    };
    if constexpr (landing_on_flat_expert_cores) {  // (no row in any of my slots: their trailers zeroed)
        for (uint32_t i = 0; i < landing_slots_per_accumulator; ++i) {
            mark_remote_slot_empty(pool_noc(i));
        }
        noc_async_write_barrier();
    } else {
        rec::zero_l1_strided(area + RINGS + LASTPB, formats::kTrailerBytes, SLOT, landing_slots_per_accumulator);
    }
    rec::count_in(gate_x, gate_y, barrier_addr);
    uint32_t want[N], port_x[N], port_y[N];
    for (uint32_t k = 0; k < N; ++k) {
        port_x[k] = moe_block_args::arg(a++);
        port_y[k] = moe_block_args::arg(a++);
    }
    if constexpr (landing_on_flat_expert_cores) {
        a += moe_block_args::arg(a);
    }
    {
        const uint32_t is_gate = moe_block_args::arg(a++), zero_count_ins = moe_block_args::arg(a++);
        const uint32_t n_cores = moe_block_args::arg(a++);
        if (is_gate) {
            rec::release(barrier_addr, call, zero_count_ins, n_cores, a);
        }
        a += 2 * n_cores;
    }
    const uint32_t local_ring_slots = moe_block_args::arg(a++), writer_count = moe_block_args::arg(a++);
    const uint32_t ldid = moe_block_args::arg(a++);
    const uint32_t local_ring_base = area + LOC_OFF, lwords = area + WORDS_OFF;
    const uint32_t writer_args = a;  // (the y writers' x, y)
    {  // (the compute's tokens, its first meta page - active flag | tokens: no compute args, they differ by chip)
        cb_reserve_back(cb_meta, 1);
        reinterpret_cast<volatile tt_l1_ptr uint32_t*>(get_write_ptr(cb_meta))[0] =
            0x80000000u | moe_block_args::arg(writer_args + 2 * writer_count);
        cb_push_back(cb_meta, 1);
    }
    cb_reserve_back(cb_tab, 1);
    const uint32_t scratch =
        (get_write_ptr(cb_tab) + 63) & ~63u;  // (CB_TAB: scratch - the plan's words, the free list)
    uint32_t n_local = 0;
    {                                                 // (the device plan: n_local and want[] of this accumulator)
        namespace plan_words = moe_block_plan_words;  // (n_local[kAccMax] at scratch, want[accumulator_index] at
                                                      // scratch + 128: below grant_queue at scratch + 256)
        const uint64_t plan_noc = TensorAccessor(plan_args, plan, plan_words::kPlanBytes).get_noc_addr(0);
        noc_async_read(plan_noc + plan_words::kPlanNloc * 4, scratch, 4 * plan_words::kAccMax);  // n_local[kAccMax]
        noc_async_read(
            plan_noc + (plan_words::kPlanWant + accumulator_index * plan_words::kMaxChips) * 4,
            scratch + 4 * plan_words::kAccMax,
            128);  // want[accumulator_index]
        noc_async_read_barrier();
        n_local = reinterpret_cast<volatile tt_l1_ptr uint32_t*>(scratch)[accumulator_index];
        for (uint32_t k = 0; k < N; ++k) {
            want[k] =
                k == my_chip ? 0 : reinterpret_cast<volatile tt_l1_ptr uint32_t*>(scratch + 4 * plan_words::kAccMax)[k];
        }
    }
    uint32_t n_rows = n_local;
    for (uint32_t k = 0; k < N; ++k) {
        n_rows += want[k];
    }
    rec::wait_released(
        barrier_addr, call);  // (this chip's ports and writers zeroed: layouts, credits and grants may go out)
    volatile tt_l1_ptr uint32_t* misc = reinterpret_cast<volatile tt_l1_ptr uint32_t*>(area + MISC);
    volatile tt_l1_ptr uint32_t* taken = misc + MTAKEN;  // per ring (source chip), cumulative
    // the landing pool, carved for this call: every source with rows here gets one slot, the rest in proportion to its
    // rows (at most its rows); each source's ring (tag << 24 | base << 12 | depth) goes to the home port that returns
    // its credits, which forwards it to the source (the source's rows wait for it)
    const uint32_t calls_ = misc[formats::kMiscAccCalls] + 1;
    misc[formats::kMiscAccCalls] = calls_;
    const uint32_t tag = (calls_ - 1) % 255 + 1;
    uint32_t rbase[N], rdep[N], taken_at_start[N];
    {
        uint32_t sources_with_rows = 0, total_want = 0;
        for (uint32_t k = 0; k < N; ++k) {
            if (k != my_chip && want[k]) {
                ++sources_with_rows;
                total_want += want[k];
            }
        }
        // half the extra slots in equal shares, half in proportion to the rows (all proportional deepened the hot
        // rings past their links' rate and starved the rest: Zipf-spread combine +14%); largest remainders hand out
        // every slot (floors alone left near-equal shares at one slot each)
        const uint32_t equal_share = sources_with_rows
                                         ? (landing_slots_per_accumulator - sources_with_rows) / 2 / sources_with_rows
                                         : 0,
                       extra = landing_slots_per_accumulator - sources_with_rows - equal_share * sources_with_rows;
        uint32_t share_remainder[N], left = extra;
        for (uint32_t k = 0; k < N; ++k) {
            rdep[k] = 0;
            share_remainder[k] = 0;
            if (k != my_chip && want[k]) {
                const uint64_t q = static_cast<uint64_t>(extra) * want[k];
                rdep[k] = 1 + equal_share + static_cast<uint32_t>(q / total_want);
                share_remainder[k] = static_cast<uint32_t>(q % total_want);
                left -= rdep[k] - 1 - equal_share;
            }
        }
        for (; left; --left) {
            uint32_t best_source = N;
            for (uint32_t k = 0; k < N; ++k) {
                if (rdep[k] && (best_source == N || share_remainder[k] > share_remainder[best_source])) {
                    best_source = k;
                }
            }
            if (best_source == N) {
                break;
            }
            ++rdep[best_source];
            share_remainder[best_source] = 0;
        }
        uint32_t b = 0;
        for (uint32_t k = 0; k < N; ++k) {
            uint32_t d = rdep[k] < want[k] ? rdep[k] : want[k];  // (no more slots than rows)
            if constexpr (shared_landing) {
                d = d < grants_per_source
                        ? d
                        : grants_per_source;  // (the first grants: the carve's shares, grants_per_source at most)
            }
            rbase[k] = b;
            rdep[k] = d;
            b += d;
            taken_at_start[k] = taken[k];
            if constexpr (!shared_landing) {
                if (k != my_chip && port_x[k]) {
                    noc_inline_dw_write(
                        get_noc_addr(
                            port_x[k], port_y[k], area + LAYO + 4 * (k * accumulators_padded + accumulator_index)),
                        formats::ring_layout(tag, rbase[k], d));
                }
            }
        }
    }
    // shared: each slot's owner (source, 0xFF free) and the free list in CB_TAB's scratch past the plan's words; per
    // source the grants outstanding and issued this call
    // (CB_TAB from scratch: the plan's words, the grant queues at 256 (2 N grants_per_source B), the trailer probes at
    // 512 (<= 1 KB), the owners at 1600 (landing_slots_per_accumulator B), the free list after them
    // (2 landing_slots_per_accumulator B) - acc_tab_bytes)
    volatile tt_l1_ptr uint8_t* own = reinterpret_cast<volatile tt_l1_ptr uint8_t*>(scratch + 1600);
    volatile tt_l1_ptr uint16_t* fstack =
        reinterpret_cast<volatile tt_l1_ptr uint16_t*>(scratch + 1600 + ((landing_slots_per_accumulator + 3u) & ~3u));
    // a source's granted slots in grant order - it fills them in that order (one port, one path), so its oldest
    // outstanding grant, at (issued - outstanding) % grants_per_source, is the only one to check
    volatile tt_l1_ptr uint16_t* grant_queue = reinterpret_cast<volatile tt_l1_ptr uint16_t*>(scratch + 256);
    static_assert(2 * N * grants_per_source <= 256, "shared landing: the grant queues in CB_TAB");
    // (landing on the flat expert cores: each source's next slots' trailers, 32 B a slot - past the shared landing's
    // grant queues)
    const uint32_t probe = scratch + 512;
    uint32_t gout[N], giss[N], nfree = 0;
    auto grant = [&](uint32_t k, uint32_t slot) {  // (the id into the home port's grant ring, then its count)
        const uint32_t sequence = taken_at_start[k] + giss[k];
        noc_inline_dw_write(
            get_noc_addr(
                port_x[k],
                port_y[k],
                area + grant_ids_offset +
                    4 * ((k * accumulators_padded + accumulator_index) * grants_per_source +
                         sequence % grants_per_source)),
            formats::grant_word(sequence, slot));
        noc_semaphore_inc(
            get_noc_addr(
                port_x[k],
                port_y[k],
                area + credits_to_return_offset + 4 * (k * accumulators_padded + accumulator_index)),
            1);
        noc_semaphore_inc(get_noc_addr(port_x[k], port_y[k], area + MISC + 4 * (MRTOT + k)), 1);
        own[slot] = static_cast<uint8_t>(k);
        grant_queue[k * grants_per_source + giss[k] % grants_per_source] = static_cast<uint16_t>(slot);
        ++giss[k];
        ++gout[k];
    };
    // a freed slot to the source with rows still to grant (want - outstanding), fewest outstanding first
    auto regrant = [&]() {
        while (nfree) {
            uint32_t best_source = N;
            for (uint32_t k = 0; k < N; ++k) {
                if (k == my_chip || gout[k] >= grants_per_source || want[k] <= gout[k]) {
                    continue;
                }
                if (best_source == N || gout[k] < gout[best_source] ||
                    (gout[k] == gout[best_source] && want[k] - gout[k] > want[best_source] - gout[best_source])) {
                    best_source = k;
                }
            }
            if (best_source == N) {
                break;
            }
            grant(best_source, fstack[--nfree]);
        }
    };
    if constexpr (shared_landing) {
        for (uint32_t i = 0; i < landing_slots_per_accumulator; ++i) {
            own[i] = 0xFF;
        }
        for (uint32_t k = 0; k < N; ++k) {
            gout[k] = giss[k] = 0;
        }
        for (uint32_t k = 0; k < N; ++k) {
            if (k == my_chip || !port_x[k]) {
                continue;
            }
            for (uint32_t j = 0; j < rdep[k]; ++j) {
                grant(k, rbase[k] + j);
            }
        }
        for (uint32_t i = landing_slots_per_accumulator; i-- > 0;) {
            if (own[i] == 0xFF) {
                fstack[nfree++] = static_cast<uint16_t>(i);
            }
        }
        regrant();  // (the slots the first grants left: to whoever has rows still to grant)
    }
    auto ring_slot = [&](uint32_t k, uint32_t taken_index) {
        return area + RINGS + (rbase[k] + (taken_index - taken_at_start[k]) % rdep[k]) * SLOT;
    };
    auto push_row = [&](uint32_t slot) {  // a row in this core's L1: its gate weight (element 0), the row, its token
        volatile tt_l1_ptr uint32_t* trailer = reinterpret_cast<volatile tt_l1_ptr uint32_t*>(slot + LASTPB);
        cb_reserve_back(cb_w, 1);
        reinterpret_cast<volatile tt_l1_ptr uint32_t*>(get_write_ptr(cb_w))[0] =
            formats::trailer_weight_bits(trailer[formats::kTrailerWeight]);
        cb_push_back(cb_w, 1);
        cb_reserve_back(cb_row, row_tiles);
        cb_reserve_back(cb_meta, 1);
        noc_async_read(get_noc_addr(slot), get_write_ptr(cb_row), row_bytes);
        reinterpret_cast<volatile tt_l1_ptr uint32_t*>(get_write_ptr(cb_meta))[0] = local_token(trailer);
        noc_async_read_barrier();
        cb_push_back(cb_row, row_tiles);
        cb_push_back(cb_meta, 1);
    };
    uint32_t done = 0, local_taken = 0;
    // L1 hand-off: local rows land in a ring of local_ring_slots slots at local_ring_base (the y writers,
    // y_row_major.hpp combine_handoff), granted from here
    volatile tt_l1_ptr uint32_t* landed = misc + ALAND;  // (combine_formats.hpp: the MISC block)
    const uint32_t lbase = misc[ALBASE];
    uint32_t lgranted = 0;
    auto lgrant_to = [&](uint32_t g) {
        for (uint32_t w = 0; w < writer_count; ++w) {
            noc_semaphore_inc(
                get_noc_addr(
                    moe_block_args::arg(writer_args + 2 * w),
                    moe_block_args::arg(writer_args + 1 + 2 * w),
                    lwords + 4 * ldid),
                g - lgranted);
        }
        lgranted = g;
    };
    if (n_local) {
        lgrant_to(n_local < local_ring_slots ? n_local : local_ring_slots);
    }
    auto take_local = [&]() -> bool {  // the next local row (L1 hand-off ring), if every writer landed it
        uint32_t m = 0xFFFFFFFF;
        for (uint32_t w = 0; w < writer_count; ++w) {
            const uint32_t v = landed[w] - lbase;
            m = v < m ? v : m;
        }
        if (local_taken >= m) {
            return false;
        }
        const uint32_t slot = local_ring_base + (local_taken % local_ring_slots) * SLOT;
        push_row(slot);
        ++local_taken;
        ++done;
        const uint32_t g = local_taken + local_ring_slots < n_local ? local_taken + local_ring_slots : n_local;
        // (grants in batches of up to 4 - at most the ring: a batch larger than local_ring_slots never fills, and the
        // writers stall)
        if (g > lgranted && (g - lgranted >= (local_ring_slots < 4 ? local_ring_slots : 4) || g == n_local)) {
            lgrant_to(g);
        }
        return true;
    };
    auto take_ring = [&](uint32_t k) -> bool {  // the next row of source k's ring, if it landed
        const uint32_t taken_index = taken[k];
        const uint32_t s = ring_slot(k, taken_index);
        volatile tt_l1_ptr uint32_t* trailer = reinterpret_cast<volatile tt_l1_ptr uint32_t*>(s + LASTPB);
        if (!row_landed(trailer)) {
            return false;
        }
        push_row(s);
        mark_slot_empty(trailer);
        taken[k] = taken_index + 1;
        --want[k];
        ++done;
        noc_semaphore_inc(
            get_noc_addr(
                port_x[k],
                port_y[k],
                area + credits_to_return_offset + 4 * (k * accumulators_padded + accumulator_index)),
            1);
        noc_semaphore_inc(get_noc_addr(port_x[k], port_y[k], area + MISC + 4 * (MRTOT + k)), 1);
        return true;
    };
    // landing on the flat expert cores: the landed rows of the probed slots, landing_reads_in_flight row reads in
    // flight (the row CB's depth); taking a row clears its trailer there (the caller fences those writes before the
    // credits go back)
    // The slots probed per source at once: each probe reads one slot's trailer (formats::kTrailerBytes) into CB_TAB's
    // probe area, N x slots_probed_per_source trailers in all - fewer per source the more sources there are, so they
    // fit
    constexpr uint32_t slots_probed_per_source = N > 16 ? 1 : (N > 8 ? 2 : 4);
    static_assert(N * slots_probed_per_source * formats::kTrailerBytes <= 1024, "the trailer probes: 1 KB of CB_TAB");
    uint32_t inflight = 0;
    auto flush_rows = [&]() {
        noc_async_read_barrier();
        for (uint32_t i = 0; i < inflight; ++i) {  // (a row at a time: a push wraps the CB only at its exact end)
            cb_push_back(cb_row, row_tiles);
            cb_push_back(cb_meta, 1);
        }
        inflight = 0;
    };
    auto take_pool = [&](uint32_t k, volatile tt_l1_ptr uint32_t* trailer) {
        const uint32_t taken_index = taken[k];
        const uint64_t slot_noc = pool_noc(rbase[k] + (taken_index - taken_at_start[k]) % rdep[k]);
        cb_reserve_back(cb_w, 1);
        reinterpret_cast<volatile tt_l1_ptr uint32_t*>(get_write_ptr(cb_w))[0] =
            formats::trailer_weight_bits(trailer[formats::kTrailerWeight]);
        cb_push_back(cb_w, 1);
        cb_reserve_back(cb_row, row_tiles * (inflight + 1));
        cb_reserve_back(cb_meta, inflight + 1);
        uint32_t row_write_ptr =
            get_write_ptr(cb_row) + inflight * row_tiles * get_local_cb_interface(cb_row).fifo_page_size;
        if (row_write_ptr >= get_local_cb_interface(cb_row).fifo_limit) {
            row_write_ptr -= get_local_cb_interface(cb_row).fifo_size;
        }
        uint32_t meta_write_ptr = get_write_ptr(cb_meta) + inflight * get_local_cb_interface(cb_meta).fifo_page_size;
        if (meta_write_ptr >= get_local_cb_interface(cb_meta).fifo_limit) {
            meta_write_ptr -= get_local_cb_interface(cb_meta).fifo_size;
        }
        noc_async_read(slot_noc, row_write_ptr, row_bytes);  // the row
        mark_remote_slot_empty(slot_noc);  // (the caller waits for both before the slot can be written again)
        reinterpret_cast<volatile tt_l1_ptr uint32_t*>(meta_write_ptr)[0] = local_token(trailer);
        taken[k] = taken_index + 1;
        --want[k];
        ++done;
        if (++inflight == landing_reads_in_flight) {
            flush_rows();
        }
    };
    // shared, landing on the flat expert cores: source k's oldest granted slot i on its flat expert core (its trailer
    // probed into trailer): the row read, its trailer cleared there, the slot free (granted again only once the caller
    // has fenced the clears)
    auto take_pool_slot = [&](uint32_t i, uint32_t k, volatile tt_l1_ptr uint32_t* trailer) {
        const uint64_t slot_noc = pool_noc(i);
        cb_reserve_back(cb_w, 1);
        reinterpret_cast<volatile tt_l1_ptr uint32_t*>(get_write_ptr(cb_w))[0] =
            formats::trailer_weight_bits(trailer[formats::kTrailerWeight]);
        cb_push_back(cb_w, 1);
        cb_reserve_back(cb_row, row_tiles * (inflight + 1));
        cb_reserve_back(cb_meta, inflight + 1);
        uint32_t row_write_ptr =
            get_write_ptr(cb_row) + inflight * row_tiles * get_local_cb_interface(cb_row).fifo_page_size;
        if (row_write_ptr >= get_local_cb_interface(cb_row).fifo_limit) {
            row_write_ptr -= get_local_cb_interface(cb_row).fifo_size;
        }
        uint32_t meta_write_ptr = get_write_ptr(cb_meta) + inflight * get_local_cb_interface(cb_meta).fifo_page_size;
        if (meta_write_ptr >= get_local_cb_interface(cb_meta).fifo_limit) {
            meta_write_ptr -= get_local_cb_interface(cb_meta).fifo_size;
        }
        noc_async_read(slot_noc, row_write_ptr, row_bytes);  // the row
        mark_remote_slot_empty(slot_noc);  // (the caller waits for both before the slot can be written again)
        reinterpret_cast<volatile tt_l1_ptr uint32_t*>(meta_write_ptr)[0] = local_token(trailer);
        own[i] = 0xFF;
        fstack[nfree++] = static_cast<uint16_t>(i);
        taken[k] = taken[k] + 1;
        --want[k];
        --gout[k];
        ++done;
        if (++inflight == landing_reads_in_flight) {
            flush_rows();
        }
    };
    auto take_slot = [&](uint32_t i) -> bool {  // shared: slot i's row, if it landed
        const uint32_t s = area + RINGS + i * SLOT;
        volatile tt_l1_ptr uint32_t* trailer = reinterpret_cast<volatile tt_l1_ptr uint32_t*>(s + LASTPB);
        if (!row_landed(trailer)) {
            return false;
        }
        push_row(s);
        mark_slot_empty(trailer);
        const uint32_t k = own[i];
        own[i] = 0xFF;
        fstack[nfree++] = static_cast<uint16_t>(i);
        taken[k] = taken[k] + 1;
        --want[k];
        --gout[k];
        ++done;
        return true;
    };
    while (done < n_rows) {
        invalidate_l1_cache();
        while (local_taken < n_local) {
            if (!take_local()) {
                break;
            }
        }
        if constexpr (shared_landing && landing_on_flat_expert_cores) {  // (each source's oldest
                                                                         // slots_probed_per_source grants probed at
                                                                         // once, the landed rows taken)
            uint32_t probe_count[N];
            for (uint32_t k = 0; k < N; ++k) {
                const uint32_t n = gout[k] < slots_probed_per_source ? gout[k] : slots_probed_per_source;
                probe_count[k] = n;
                for (uint32_t j = 0; j < n; ++j) {
                    noc_async_read(
                        pool_noc(grant_queue[k * grants_per_source + (giss[k] - gout[k] + j) % grants_per_source]) +
                            LASTPB,
                        probe + formats::kTrailerBytes * (k * slots_probed_per_source + j),
                        formats::kTrailerBytes);
                }
            }
            noc_async_read_barrier();
            bool any = false;
            for (uint32_t k = 0; k < N; ++k) {
                for (uint32_t j = 0; j < probe_count[k]; ++j) {
                    volatile tt_l1_ptr uint32_t* trailer = reinterpret_cast<volatile tt_l1_ptr uint32_t*>(
                        probe + formats::kTrailerBytes * (k * slots_probed_per_source + j));
                    if (!row_landed(trailer)) {
                        break;
                    }
                    take_pool_slot(
                        grant_queue[k * grants_per_source + (giss[k] - gout[k]) % grants_per_source], k, trailer);
                    any = true;
                }
            }
            if (any) {  // (the slots granted again once their rows are read and their trailers cleared)
                if (inflight) {
                    flush_rows();
                }
                noc_async_write_barrier();
                regrant();
            }
        } else if constexpr (shared_landing) {  // (each source's oldest grant only; freed slots granted again at once)
            bool any = false;
            for (uint32_t k = 0; k < N; ++k) {
                while (gout[k] &&
                       take_slot(grant_queue[k * grants_per_source + (giss[k] - gout[k]) % grants_per_source])) {
                    any = true;
                }
            }
            if (any) {
                regrant();
            }
        } else if constexpr (landing_on_flat_expert_cores) {  // (each source's next slots_probed_per_source slots
                                                              // probed at once, then the landed rows taken)
            uint32_t probe_count[N];
            for (uint32_t k = 0; k < N; ++k) {
                uint32_t n = want[k] < slots_probed_per_source ? want[k] : slots_probed_per_source;
                n = n < rdep[k] ? n : rdep[k];
                probe_count[k] = n;
                for (uint32_t j = 0; j < n; ++j) {
                    noc_async_read(
                        pool_noc(rbase[k] + (taken[k] + j - taken_at_start[k]) % rdep[k]) + LASTPB,
                        probe + formats::kTrailerBytes * (k * slots_probed_per_source + j),
                        formats::kTrailerBytes);
                }
            }
            noc_async_read_barrier();
            uint32_t landed_count[N], any = 0;
            for (uint32_t k = 0; k < N; ++k) {
                landed_count[k] = 0;
                for (uint32_t j = 0; j < probe_count[k]; ++j) {
                    volatile tt_l1_ptr uint32_t* trailer = reinterpret_cast<volatile tt_l1_ptr uint32_t*>(
                        probe + formats::kTrailerBytes * (k * slots_probed_per_source + j));
                    if (!row_landed(trailer)) {
                        break;
                    }
                    take_pool(k, trailer);
                    ++landed_count[k];
                }
                any |= landed_count[k];
            }
            if (any) {  // (credits once the rows are read and their trailers cleared: the sources write the slots
                        // again)
                if (inflight) {
                    flush_rows();
                }
                noc_async_write_barrier();
                for (uint32_t k = 0; k < N; ++k) {
                    if (landed_count[k]) {
                        noc_semaphore_inc(
                            get_noc_addr(
                                port_x[k],
                                port_y[k],
                                area + credits_to_return_offset + 4 * (k * accumulators_padded + accumulator_index)),
                            landed_count[k]);
                        noc_semaphore_inc(
                            get_noc_addr(port_x[k], port_y[k], area + MISC + 4 * (MRTOT + k)), landed_count[k]);
                    }
                }
            }
        } else {
            for (uint32_t k = 0; k < N; ++k) {
                if (want[k]) {
                    take_ring(k);
                }
            }
        }
    }
    cb_reserve_back(cb_meta, 1);  // (the compute's end marker: it does not know the row count)
    reinterpret_cast<volatile tt_l1_ptr uint32_t*>(get_write_ptr(cb_meta))[0] = 0xFFFFFFFF;
    cb_push_back(cb_meta, 1);
    misc[ALBASE] = lbase + n_local;
    noc_async_atomic_barrier();
}
