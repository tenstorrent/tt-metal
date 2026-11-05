// SPDX-FileCopyrightText: 2026 Tenstorrent USA, Inc.
// SPDX-License-Identifier: Apache-2.0

// nomic_moe_route_dispatch reader (RISCV_0 / NOC0), every core of the 11x10 grid.
//
// Token core (group g, member c; x shard resident in L1):
//   [W sender of an even slice: DRAM read + multicast, first] -> publish_x_block (round 1, matmul) -> zero the
//   combine_weights tiles -> receive_weight_slice (even members; the writer receives odd ones) -> publish x (round 2,
//   untilize; not with whole-piece push) -> exchange_partials_block -> select_block (reader's tokens; the writer
//   selects the rest) -> leader: count row to the coordinator -> count table -> assign_rows_block -> push_rows_block
//   (the reader's tokens; rows assigned while pushing) -> [ARRIVAL_PER_BLOCK: arrival increments on the owners] ->
//   [global: landing increment] -> leader: write_combine_index_block -> write_combine_weights_block (the member's
//   tiles, every other one when the writer holds tokens)
// Every core: [coordinator: table multicast, expert_counts] -> global: [coordinator: landing barrier + go multicast]
//   -> wait go -> publish the owned blocks of cb_gathered_rows to compute. ARRIVAL_PER_BLOCK: publish each owned
//   block once its arrival count is complete, issuing the member's combine_weights tiles in between.
//
// Raw NoC for the intra-group partial exchange, the count gather and the row pushes: the Mcast helper is
// one-to-many (op_design.md, Helpers considered and rejected). select_block / assign_rows_block are scalar RISC code.

#include <cstdint>
#include <string_view>

#include "api/core_local_mem.h"
#include "api/dataflow/circular_buffer.h"
#include "api/dataflow/dataflow_api.h"
#include "api/dataflow/endpoints.h"
#include "api/dataflow/noc.h"
#include "api/dataflow/noc_semaphore.h"
#include "api/tensor/noc_traits.h"
#include "ttnn/cpp/ttnn/kernel_lib/mcast/kernel/mcast_args.hpp"
#include "ttnn/cpp/ttnn/kernel_lib/perf_instrumentation.hpp"
#include "nomic_moe_route_dispatch_push.hpp"

using namespace dataflow_kernel_lib;
using namespace route_dispatch;

#ifdef ARRIVAL_PER_BLOCK  // host define (ARRIVAL_PER_BLOCK knob): per-owner counted arrival, no global barrier
constexpr bool arrival_per_block = true;
#else
constexpr bool arrival_per_block = false;
#endif

namespace {

// Common runtime args: 4 addresses, 4 page sizes, then the NoC coordinates of the grid cores.
constexpr uint32_t kAddrArg = 0;
constexpr uint32_t kPageArg = 4;
constexpr uint32_t kCoordsArg = 8;

}  // namespace

void kernel_main() {
    constexpr uint32_t ht = get_named_compile_time_arg_val("hidden_tiles");
    constexpr uint32_t rps = get_named_compile_time_arg_val("rows_per_shard");
    constexpr uint32_t G = get_named_compile_time_arg_val("hidden_split");
    constexpr uint32_t w = get_named_compile_time_arg_val("block_width_tiles");
    constexpr uint32_t k = get_named_compile_time_arg_val("pieces_per_row");
    constexpr uint32_t num_cores = get_named_compile_time_arg_val("num_cores");
    constexpr uint32_t push_bytes_in_flight = get_named_compile_time_arg_val("push_bytes_in_flight");
    constexpr uint32_t push_stagger_tokens = get_named_compile_time_arg_val("push_stagger_tokens");
    constexpr uint32_t push_cmd_bufs = get_named_compile_time_arg_val("push_cmd_bufs");
    constexpr bool push_gather = get_named_compile_time_arg_val("push_gather") != 0;
    constexpr uint32_t stage_tiles = get_named_compile_time_arg_val("stage_tiles");
    static_assert(push_gather || stage_tiles == ht, "a member stages its own slice unless it gathers whole pieces");
    constexpr uint32_t coordinator = get_named_compile_time_arg_val("coordinator");
    constexpr uint32_t rows = get_named_compile_time_arg_val("rows");
    constexpr uint32_t num_experts = get_named_compile_time_arg_val("num_experts");
    constexpr uint32_t top_k = get_named_compile_time_arg_val("top_k");
    constexpr uint32_t weight_tiles = get_named_compile_time_arg_val("weight_tiles");
    constexpr uint32_t w_sender_top = get_named_compile_time_arg_val("w_sender_top");
    constexpr uint32_t w_slice_nocs = get_named_compile_time_arg_val("w_slice_nocs");
    constexpr uint32_t num_groups = get_named_compile_time_arg_val("num_groups");
    constexpr uint32_t t_pad = get_named_compile_time_arg_val("t_pad");
    constexpr uint32_t drain_offset = get_named_compile_time_arg_val("drain_offset");
    constexpr uint32_t counts_offset = get_named_compile_time_arg_val("counts_offset");
    constexpr uint32_t table_row_bytes = get_named_compile_time_arg_val("table_row_bytes");
    constexpr uint32_t cb_x_tiles = get_named_compile_time_arg_val("cb_x_tiles");
    constexpr uint32_t cb_router_weight = get_named_compile_time_arg_val("cb_router_weight");
    constexpr uint32_t cb_own_partial = get_named_compile_time_arg_val("cb_own_partial");
    constexpr uint32_t cb_partials = get_named_compile_time_arg_val("cb_partials");
    constexpr uint32_t cb_scores = get_named_compile_time_arg_val("cb_scores");
    constexpr uint32_t cb_x_rows = get_named_compile_time_arg_val("cb_x_rows");
    constexpr uint32_t cb_slot_rows = get_named_compile_time_arg_val("cb_slot_rows");
    constexpr uint32_t cb_gathered_rows = get_named_compile_time_arg_val("cb_gathered_rows");
    constexpr uint32_t cb_reader_scratch = get_named_compile_time_arg_val("cb_reader_scratch");
    constexpr uint32_t cb_select_go = get_named_compile_time_arg_val("cb_select_go");
    constexpr uint32_t cb_writer_counts = get_named_compile_time_arg_val("cb_writer_counts");
    static_assert(top_k == 2 && num_experts == kNumExperts, "the selection is specialized on the model's E and top-k");
    constexpr bool x_interleaved = get_named_compile_time_arg_val("x_interleaved") != 0;
    static_assert(G >= 1 && G <= kMaxSlices, "the hidden axis is whole (interleaved x) or split into 2 or 3 slices");
    static_assert(!x_interleaved || G == 1, "an interleaved x core holds whole rows");
    constexpr uint32_t shard_tiles = rps * ht;
    constexpr uint32_t tokens_per_shard = rps * kTile;
    static_assert(
        kHeaderWords <= get_named_compile_time_arg_val("region_words"), "slot-page header overflows its host size");
    static_assert(
        kHandshakeWords * 4 <= get_named_compile_time_arg_val("handshake_page_bytes"), "handshake page too small");
    constexpr uint32_t packed_row_words = table_row_bytes / 4;

    constexpr auto weight_args = TensorAccessorArgs<0>();
    constexpr auto combine_index_args = TensorAccessorArgs<weight_args.next_compile_time_args_offset()>();
    constexpr auto combine_weights_args = TensorAccessorArgs<combine_index_args.next_compile_time_args_offset()>();
    constexpr auto expert_counts_args = TensorAccessorArgs<combine_weights_args.next_compile_time_args_offset()>();
    [[maybe_unused]] constexpr auto x_args = TensorAccessorArgs<expert_counts_args.next_compile_time_args_offset()>();

    const uint32_t core_idx = get_arg_val<uint32_t>(0);
    const bool is_token = get_arg_val<uint32_t>(1) != 0;
    const uint32_t member = get_arg_val<uint32_t>(2);
    const uint32_t group = get_arg_val<uint32_t>(3);
    const uint32_t first_token = get_arg_val<uint32_t>(4);
    const uint32_t core_tokens = get_arg_val<uint32_t>(5);
    const uint32_t num_blocks = get_arg_val<uint32_t>(6);
    [[maybe_unused]] const uint32_t x_addr = get_arg_val<uint32_t>(7);
    [[maybe_unused]] const uint32_t x_page = get_arg_val<uint32_t>(8);
    const uint32_t reader_tokens = get_arg_val<uint32_t>(9);  // the reader RISC's tokens [0, reader_tokens)
    // This member writes combine_weights tiles (host: the leader, or every member under WEIGHTS_SPREAD).
    const bool writes_weights = get_arg_val<uint32_t>(10) != 0;
    constexpr uint32_t kPushArgs = 11;
    const bool stages = get_arg_val<uint32_t>(kPushArgs + kPushArgPieces) != 0;

    const auto weight = TensorAccessor(
        weight_args, get_common_arg_val<uint32_t>(kAddrArg + 0), get_common_arg_val<uint32_t>(kPageArg + 0));
    const auto combine_index = TensorAccessor(
        combine_index_args, get_common_arg_val<uint32_t>(kAddrArg + 1), get_common_arg_val<uint32_t>(kPageArg + 1));
    const auto combine_weights = TensorAccessor(
        combine_weights_args, get_common_arg_val<uint32_t>(kAddrArg + 2), get_common_arg_val<uint32_t>(kPageArg + 2));
    const auto expert_counts = TensorAccessor(
        expert_counts_args, get_common_arg_val<uint32_t>(kAddrArg + 3), get_common_arg_val<uint32_t>(kPageArg + 3));

    Noc noc;
    CircularBuffer x_tiles(cb_x_tiles);
    CircularBuffer router_weight(cb_router_weight);
    CircularBuffer own_partial(cb_own_partial);
    CircularBuffer partials(cb_partials);
    CircularBuffer scores(cb_scores);
    CircularBuffer x_rows(cb_x_rows);
    CircularBuffer slot_rows(cb_slot_rows);
    CircularBuffer gathered(cb_gathered_rows);
    CircularBuffer select_go(cb_select_go);
    CircularBuffer weight_tiles_cb(get_named_compile_time_arg_val("cb_weight_tiles"));
    const uint32_t weight_base_tiles = weight_tiles_cb.get_write_ptr();
    CircularBuffer writer_counts_cb(cb_writer_counts);
    const uint32_t weight_base = router_weight.get_write_ptr();
    const uint32_t partials_base = partials.get_write_ptr();
    const uint32_t gathered_base = gathered.get_write_ptr();
    const uint32_t scratch_base = CircularBuffer(cb_reader_scratch).get_write_ptr();
    volatile tt_l1_ptr uint32_t* table = reinterpret_cast<volatile tt_l1_ptr uint32_t*>(scratch_base);
    volatile tt_l1_ptr uint32_t* drain_word =
        reinterpret_cast<volatile tt_l1_ptr uint32_t*>(scratch_base + drain_offset);
    Semaphore<> sem_partials(get_named_compile_time_arg_val("sem_partials"));
    Semaphore<> sem_counts(get_named_compile_time_arg_val("sem_counts"));
    constexpr uint32_t sem_arrived = get_named_compile_time_arg_val("sem_arrived");
    constexpr uint32_t arrival_signals = get_named_compile_time_arg_val("arrival_signals");
    const uint32_t coordinator_coord = get_common_arg_val<uint32_t>(kCoordsArg + coordinator);

    const bool is_coordinator = core_idx == coordinator;
    const bool is_weight_sender = core_idx <= w_sender_top && core_idx + G > w_sender_top;
    const uint32_t sender_slice = w_sender_top - core_idx;

    // ---- W slice (reader-sent slices): issued before anything else ----
    if (is_weight_sender) {
        MaybeDeviceZoneScope("rd_w_send");
        deliver_weight_slice<G, ht, w_slice_nocs, kReaderRisc>(noc, weight, weight_base, sender_slice);
    }

    // ---- publish_x_block, round 1 (matmul) ----
    // Interleaved x (R2): the core's tile rows read in one burst.
    const uint32_t core_tiles = (core_tokens + kTile - 1) / kTile;
    if (is_token) {
        x_tiles.reserve_back(shard_tiles);
        if constexpr (x_interleaved) {
            const auto x = TensorAccessor(x_args, x_addr, x_page);
            const uint32_t dst = x_tiles.get_write_ptr();
            const uint32_t first_page = first_token / kTile * ht;
            for (uint32_t tile = 0; tile < core_tiles * ht; ++tile) {
                noc.async_read(
                    x,
                    CoreLocalMem<uint32_t>(dst + tile * kBf16TileBytes),
                    kBf16TileBytes,
                    {.page_id = first_page + tile},
                    {});
            }
            noc.async_read_barrier();
        }
        x_tiles.push_back(shard_tiles);
    }

    volatile tt_l1_ptr uint32_t* slot_page = nullptr;

    // write_combine_weights_block (write_weight_tiles): a member's tiles weights_first, + weights_step, ...; when the
    // writer holds tokens it takes every other one of the member's tiles, starting with the first.
    constexpr uint32_t weights_stride = get_named_compile_time_arg_val("weights_stride");
    const bool writer_has_tokens = core_tokens > reader_tokens;
    const uint32_t weights_first = (weights_stride > 1 ? member : 0) + (writer_has_tokens ? weights_stride : 0);
    const uint32_t weights_step = writer_has_tokens ? 2 * weights_stride : weights_stride;
    const uint32_t weight_writes = is_token && writes_weights ? top_k * core_tiles : 0;

    // exchange_counts (table half): the coordinator waits for every group's row and multicasts the table; the
    // other token cores receive it. A coordinator that is a token core sends after its own count row.
    const auto receive_or_send_table = [&](bool token_core) {
        constexpr auto table_channel = McastArgs<
            get_named_compile_time_arg_val("table_ct_offset"),
            get_named_compile_time_arg_val("table_rt_offset")>();
        if (is_coordinator) {
            MaybeDeviceZoneScope("rd_table_send");
            sem_counts.wait(num_groups);
            table_channel.sender(noc).send(scratch_base, scratch_base, num_groups * table_row_bytes);
        } else if (token_core) {
            MaybeDeviceZoneScope("rd_wait_table");
            table_channel.receiver(noc).receive();
        }
    };
    // Totals and the slots of the groups before `group`. A packed sum is at most 2T < 65536 a half: no carry.
    const auto scan_table = [&](uint32_t group_of, uint32_t(&totals)[kNumExperts], uint32_t(&before)[kNumExperts]) {
        uint32_t packed_total[4] = {};
        uint32_t packed_before[4] = {};
        for (uint32_t g = 0; g < num_groups; ++g) {
            const volatile tt_l1_ptr uint32_t* row = table + g * packed_row_words;
            uint32_t words[4];
#pragma GCC unroll 4
            for (uint32_t word = 0; word < packed_row_words; ++word) {
                words[word] = row[word];
            }
#pragma GCC unroll 4
            for (uint32_t word = 0; word < packed_row_words; ++word) {
                packed_total[word] += words[word];
                if (g < group_of) {
                    packed_before[word] += words[word];
                }
            }
        }
        for (uint32_t expert = 0; expert < kNumExperts; ++expert) {
            const uint32_t shift = (expert % 2) * 16;
            totals[expert] = (packed_total[expert / 2] >> shift) & 0xFFFF;
            before[expert] = (packed_before[expert / 2] >> shift) & 0xFFFF;
        }
    };

    if (is_token) {
        // ---- combine_weights tiles (members that write them), in the W-fetch shadow: zero now, column 0 is written
        // by select_block ----
        if (writes_weights) {
            MaybeDeviceZoneScope("rd_consts");
            noc.async_write_zeros(weight_tiles_cb, top_k * rps * kFp32TileBytes, {.offset_bytes = 0});
            noc.write_zeros_l1_barrier();
        }

        // ---- receive_weight_slice: slice `member` (faces only, at its compact slot; the writer receives NOC1 slices)
        // ----
        {
            MaybeDeviceZoneScope("rd_wait_w");
            receive_weight_slice<G, w_slice_nocs, kReaderRisc>(
                noc, router_weight, weight_tiles, member, is_weight_sender && sender_slice == member % w_slice_nocs);
        }

        // ---- exchange_partials_block: faces 0 and 2 (expert columns 0..15) of this member's region of cb_partials
        // (compute packed it there) to the same region on the HIGHER members, on NOC0 (east / south: short hops). The
        // writer sends to the lower members on NOC1: a westward NOC0 write wraps the whole row ring, and with every
        // group's members finishing together those wraps congest each other. ----
        {
            MaybeDeviceZoneScope("rd_wait_own_partial");
            own_partial.wait_front(1);
        }
        // ---- publish_x_block, round 2 (untilize, after the softmax): the matmul popped round 1 ----
        if constexpr (!push_gather) {
            x_tiles.reserve_back(shard_tiles);
            x_tiles.push_back(shard_tiles);
        }
        const uint32_t first_member_core = group * G;
        {
            MaybeDeviceZoneScope("rd_xchg_send");
            send_partial_faces<rps>(
                noc, partials_base + member * rps * kFp32TileBytes, kCoordsArg + first_member_core, member + 1, G);
            noc.async_write_barrier();
            for (uint32_t peer = member + 1; peer < G; ++peer) {
                const uint32_t coord = get_common_arg_val<uint32_t>(kCoordsArg + first_member_core + peer);
                sem_partials.up(noc, coord_x(coord), coord_y(coord), 1);
            }
        }
        own_partial.pop_front(1);
        {
            MaybeDeviceZoneScope("rd_wait_peers");
            sem_partials.wait(G - 1);
        }
        partials.reserve_back(G * rps);
        partials.push_back(G * rps);

        // ---- select_block: the reader's tokens here, the writer's on the writer RISC ----
        slot_rows.reserve_back(1);
        slot_page = reinterpret_cast<volatile tt_l1_ptr uint32_t*>(slot_rows.get_write_ptr());
        // R3: tile-padding tokens of a ragged last tile row are never selected; their combine_index is 0.
        for (uint32_t t = core_tokens; t < core_tiles * kTile; ++t) {
            slot_page[t] = 0;
            slot_page[tokens_per_shard + t] = 0;
        }
        volatile tt_l1_ptr uint32_t* weight_words =
            writes_weights ? reinterpret_cast<volatile tt_l1_ptr uint32_t*>(weight_base_tiles) : nullptr;
        const uint32_t reader_hi = core_tokens < reader_tokens ? core_tokens : reader_tokens;
        const bool writer_selects = core_tokens > reader_tokens;
        if (writer_selects) {
            select_go.reserve_back(1);
            volatile tt_l1_ptr uint32_t* go = reinterpret_cast<volatile tt_l1_ptr uint32_t*>(select_go.get_write_ptr());
            go[kGoSlotPage] = reinterpret_cast<uint32_t>(slot_page);
            go[kGoWeights] = reinterpret_cast<uint32_t>(weight_words);
            select_go.push_back(1);
        }
        uint64_t reader_counts;
        {
            // occupancy: includes the per-tile rd_wait_scores waits (the code tiles are published one at a time)
            MaybeDeviceZoneScope("rd_select");
            reader_counts = select_tokens<rps>(
                reinterpret_cast<const volatile tt_l1_ptr uint32_t*>(scores.get_read_ptr()),
                slot_page,
                weight_words,
                0,
                reader_hi,
                [&](uint32_t tile) {
                    MaybeDeviceZoneScope("rd_wait_scores");
                    scores.wait_front(tile + 1);
                });
        }
        uint64_t writer_counts = 0;
        if (writer_selects) {
            MaybeDeviceZoneScope("rd_wait_writer_counts");
            writer_counts_cb.wait_front(1);
            const volatile tt_l1_ptr uint32_t* words =
                reinterpret_cast<const volatile tt_l1_ptr uint32_t*>(writer_counts_cb.get_read_ptr());
            writer_counts = (uint64_t{words[1]} << 32) | words[0];
            writer_counts_cb.pop_front(1);
        }
        scores.wait_front(rps);
        scores.pop_front(rps);
        uint32_t counts[kNumExperts];
        uint32_t reader_count[kNumExperts];
        for (uint32_t expert = 0; expert < kNumExperts; ++expert) {
            reader_count[expert] = count_of(reader_counts, expert);
            counts[expert] = reader_count[expert] + count_of(writer_counts, expert);
        }

        // ---- exchange_counts (gather half): the leader's row of 8 u16 counts to the coordinator ----
        if (member == 0) {
            MaybeDeviceZoneScope("rd_count_send");
            volatile tt_l1_ptr uint32_t* own_row = table + group * packed_row_words;
            for (uint32_t word = 0; word < packed_row_words; ++word) {
                own_row[word] = counts[2 * word] | (counts[2 * word + 1] << 16);
            }
            drain_stores(drain_word);
            const uint32_t row_addr = scratch_base + group * table_row_bytes;
            noc.async_write(
                CoreLocalMem<uint32_t>(row_addr),
                UnicastEndpoint{},
                table_row_bytes,
                {},
                {.noc_x = coord_x(coordinator_coord), .noc_y = coord_y(coordinator_coord), .addr = row_addr});
            noc.async_write_barrier();
            sem_counts.up(noc, coord_x(coordinator_coord), coord_y(coordinator_coord), 1);
        }

        // ---- count table: gathered on the coordinator, multicast to the grid ----
        receive_or_send_table(true);

        // ---- assign_rows_block: region starts and each RISC's first row per expert (rows assigned while pushing) ----
        uint32_t totals[kNumExperts];
        uint32_t before[kNumExperts];
        uint32_t reader_base[kNumExperts];
        volatile tt_l1_ptr uint32_t* header = slot_page + top_k * tokens_per_shard;
        {
            MaybeDeviceZoneScope("rd_assign");
            scan_table(group, totals, before);
            uint32_t region_start = 0;
            for (uint32_t expert = 0; expert < kNumExperts; ++expert) {
                reader_base[expert] = region_start + before[expert];
                header[kHeaderStarts + expert] = region_start;
                header[kHeaderTotals + expert] = totals[expert];
                header[kHeaderWriterBase + expert] = reader_base[expert] + reader_count[expert];
                region_start += round_up_to_tile(totals[expert]);
            }
            header[kHeaderEnd] = region_start;
        }
        uint32_t staging = 0;
        if (stages) {
            MaybeDeviceZoneScope("rd_wait_xrows");
            x_rows.wait_front(rps * stage_tiles);
            staging = x_rows.get_read_ptr();
        }
        header[kHeaderStaging] = staging;
        drain_stores(drain_word);
        slot_rows.push_back(1);

        // ---- push_rows_block: the reader's tokens [0, reader_tokens) ----
        // Push units: 2 * group (the reader's tokens), 2 * group + 1 (the writer's, taken here when it has none).
        {
            MaybeDeviceZoneScope("rd_push");
            push_rows_block<
                stage_tiles,
                tokens_per_shard,
                RowPusher<w, k, num_cores, push_bytes_in_flight, push_cmd_bufs>>(
                kPushArgs,
                gathered_base,
                kCoordsArg,
                slot_page,
                reader_base,
                staging,
                0,
                reader_hi,
                rows,
                2 * group,
                writer_selects ? 1 : 2,
                2 * num_groups,
                group * push_stagger_tokens,
                member == 0);
        }
        if (arrival_per_block && stages) {
            noc.async_write_barrier();
            signal_arrivals<k, num_cores>(
                noc,
                kPushArgs,
                kCoordsArg,
                sem_arrived,
                header,
                reader_base,
                reader_count,
                rows,
                2 * group,
                writer_selects ? 1 : 2,
                2 * num_groups,
                reader_hi);
        }
        const auto write_own_index = [&]() {
            if (member == 0) {
                drain_stores(drain_word);
                // The RISC with the core's last tokens also writes the zeroed tile-padding tokens (R3).
                write_combine_index<tokens_per_shard>(
                    noc,
                    combine_index,
                    reinterpret_cast<uint32_t>(slot_page),
                    first_token,
                    0,
                    writer_selects ? reader_hi : core_tiles * kTile);
            }
        };
        if constexpr (arrival_per_block) {
            write_own_index();
        } else {
            {
                MaybeDeviceZoneScope("rd_push_barrier");
                noc.async_write_barrier();
            }
            Semaphore<>(sem_arrived).up(noc, coord_x(coordinator_coord), coord_y(coordinator_coord), 1);
            // ---- write_combine_index_block, write_combine_weights_block: after the landing increment, so the
            // landing barrier waits on the L1 pushes only, not on DRAM writes queued behind other cores' ----
            write_own_index();
            if (weight_writes > 0) {
                MaybeDeviceZoneScope("rd_weights_issue");
                drain_stores(drain_word);
                write_weight_tiles<rps, t_pad>(
                    noc, combine_weights, weight_base_tiles, first_token, core_tiles, weights_first, weights_step);
            }
        }
    } else {
        receive_or_send_table(false);
    }

    if (is_coordinator) {
        uint32_t totals[kNumExperts];
        uint32_t before[kNumExperts];
        scan_table(0, totals, before);
        volatile tt_l1_ptr uint32_t* staged =
            reinterpret_cast<volatile tt_l1_ptr uint32_t*>(scratch_base + counts_offset);
        for (uint32_t expert = 0; expert < kNumExperts; ++expert) {
            staged[expert] = totals[expert];
        }
        drain_stores(drain_word);
        noc.async_write(
            CoreLocalMem<uint32_t>(scratch_base + counts_offset), expert_counts, kNumExperts * 4, {}, {.page_id = 0});
    }

#ifdef ARRIVAL_PER_BLOCK
    // ---- counted arrival: each owned block goes to the tilize as soon as all its rows have landed. In between,
    // write_combine_weights_block issues one tile at a time: its DRAM writes overlap the tail of the pushes
    // without holding up a complete block. ----
    if (weight_writes > 0) {
        drain_stores(drain_word);
    }
    for (uint32_t block = 0, write = weights_first; block < num_blocks || write < weight_writes;) {
        if (block < num_blocks && Semaphore<>(sem_arrived + block).value() == kTile * w) {
            gathered.reserve_back(w);
            gathered.push_back(w);
            ++block;
        } else if (write < weight_writes) {
            write_weight_tiles<rps, t_pad>(  // one tile
                noc,
                combine_weights,
                weight_base_tiles,
                first_token,
                core_tiles,
                write,
                weight_writes);
            write += weights_step;
        }
    }
#else  // the go channel is attached only for the global barrier
    // ---- land_barrier: every pushing RISC has landed; the coordinator releases the tilize ----
    constexpr auto go_channel =
        McastArgs<get_named_compile_time_arg_val("go_ct_offset"), get_named_compile_time_arg_val("go_rt_offset")>();
    {
        MaybeDeviceZoneScope("rd_wait_go");
        if (is_coordinator) {
            Semaphore<>(sem_arrived).wait(arrival_signals);
            go_channel.sender(noc).send_signal();
        } else {
            go_channel.receiver(noc).receive_signal();
        }
    }
    if (num_blocks > 0) {
        gathered.reserve_back(num_blocks * w);
        gathered.push_back(num_blocks * w);
    }
#endif
    MaybeDeviceZoneScope("rd_final_barrier");
    noc.async_write_barrier();
    noc.async_atomic_barrier();
}
