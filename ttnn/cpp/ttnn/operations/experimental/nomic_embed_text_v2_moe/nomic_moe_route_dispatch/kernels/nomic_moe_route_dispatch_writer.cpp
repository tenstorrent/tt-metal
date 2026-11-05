// SPDX-FileCopyrightText: 2026 Tenstorrent USA, Inc.
// SPDX-License-Identifier: Apache-2.0

// nomic_moe_route_dispatch writer (RISCV_1 / NOC1), token cores. The core's tokens are split between the two DM
// RISCs at reader_tokens; the writer takes [reader_tokens, core_tokens):
//   [whole-piece push: gather_stage_block from the group peers' x shards, on NOC1 off the W multicasts] ->
//   [odd member: receive_weight_slice] -> select_block for its tokens (from cb_scores, after the reader's cb_select_go
//   page) -> its per-expert counts to the reader (cb_writer_counts) -> wait cb_slot_rows (region header, untilized
//   shard address) -> push_rows_block -> [ARRIVAL_PER_BLOCK: arrival increments on the owners] -> [global: one landing
//   increment on the coordinator] -> leader: its combine_index segments -> its share of the member's combine_weights
//   tiles (NOC1) -> write barrier.
// Sender of an odd W slice (non-token core): DRAM read + multicast on NOC1, so two slices travel at once.
// Every core, DRAM `dispatched` (regime R4): write_dispatched_block for each owned output block (shard
//   core_idx + b * num_cores) as compute tilizes it into the streaming cb_dispatched, one flush per block.

#include <cstdint>

#include "api/dataflow/circular_buffer.h"
#include "api/dataflow/dataflow_api.h"
#include "api/dataflow/noc.h"
#include "api/dataflow/noc_semaphore.h"
#include "ttnn/cpp/ttnn/kernel_lib/perf_instrumentation.hpp"
#include "nomic_moe_route_dispatch_push.hpp"

using namespace route_dispatch;

#ifdef ARRIVAL_PER_BLOCK  // host define (ARRIVAL_PER_BLOCK knob): per-owner counted arrival, no global barrier
constexpr bool arrival_per_block = true;
#else
constexpr bool arrival_per_block = false;
#endif

void kernel_main() {
    constexpr uint32_t ht = get_named_compile_time_arg_val("hidden_tiles");
    constexpr uint32_t rps = get_named_compile_time_arg_val("rows_per_shard");
    constexpr uint32_t w = get_named_compile_time_arg_val("block_width_tiles");
    constexpr uint32_t k = get_named_compile_time_arg_val("pieces_per_row");
    constexpr uint32_t num_cores = get_named_compile_time_arg_val("num_cores");
    constexpr uint32_t push_bytes_in_flight = get_named_compile_time_arg_val("push_bytes_in_flight");
    constexpr uint32_t push_stagger_tokens = get_named_compile_time_arg_val("push_stagger_tokens");
    constexpr uint32_t push_cmd_bufs = get_named_compile_time_arg_val("push_cmd_bufs");
    constexpr uint32_t stage_tiles = get_named_compile_time_arg_val("stage_tiles");
    constexpr bool push_gather = get_named_compile_time_arg_val("push_gather") != 0;
    constexpr uint32_t G = get_named_compile_time_arg_val("hidden_split");
    constexpr uint32_t num_groups = get_named_compile_time_arg_val("num_groups");
    constexpr uint32_t rows = get_named_compile_time_arg_val("rows");
    constexpr uint32_t cb_slot_rows = get_named_compile_time_arg_val("cb_slot_rows");
    constexpr uint32_t sem_arrived = get_named_compile_time_arg_val("sem_arrived");
    constexpr uint32_t cb_scores = get_named_compile_time_arg_val("cb_scores");
    constexpr uint32_t cb_select_go = get_named_compile_time_arg_val("cb_select_go");
    constexpr uint32_t cb_writer_counts = get_named_compile_time_arg_val("cb_writer_counts");
    constexpr bool out_dram = get_named_compile_time_arg_val("out_dram") != 0;
    constexpr uint32_t hidden_tiles_total = get_named_compile_time_arg_val("hidden_tiles_total");
    static_assert(w * k == hidden_tiles_total, "the k pieces of w tiles cover the output tile row");
    constexpr auto combine_index_args = TensorAccessorArgs<0>();
    [[maybe_unused]] constexpr auto dispatched_args =
        TensorAccessorArgs<combine_index_args.next_compile_time_args_offset()>();
    constexpr auto weight_args = TensorAccessorArgs<dispatched_args.next_compile_time_args_offset()>();
    constexpr uint32_t w_sender_top = get_named_compile_time_arg_val("w_sender_top");
    constexpr uint32_t w_slice_nocs = get_named_compile_time_arg_val("w_slice_nocs");
    constexpr uint32_t cb_gathered_rows = get_named_compile_time_arg_val("cb_gathered_rows");
    constexpr uint32_t tokens_per_shard = rps * kTile;
    static_assert(
        kHeaderWords <= get_named_compile_time_arg_val("region_words"), "slot-page header overflows its host size");
    static_assert(
        kHandshakeWords * 4 <= get_named_compile_time_arg_val("handshake_page_bytes"), "handshake page too small");

    const bool is_token = get_arg_val<uint32_t>(0) != 0;
    const uint32_t member = get_arg_val<uint32_t>(1);
    const uint32_t first_token = get_arg_val<uint32_t>(2);
    const uint32_t core_tokens = get_arg_val<uint32_t>(3);
    const uint32_t combine_index_addr = get_arg_val<uint32_t>(4);
    const uint32_t combine_index_page = get_arg_val<uint32_t>(5);
    const uint32_t core_idx = get_arg_val<uint32_t>(6);
    [[maybe_unused]] const uint32_t num_blocks = get_arg_val<uint32_t>(7);
    [[maybe_unused]] const uint32_t dispatched_addr = get_arg_val<uint32_t>(8);
    [[maybe_unused]] const uint32_t dispatched_page = get_arg_val<uint32_t>(9);
    const uint32_t reader_tokens = get_arg_val<uint32_t>(10);
    const uint32_t group = get_arg_val<uint32_t>(11);
    [[maybe_unused]] const uint32_t x_addr = get_arg_val<uint32_t>(12);
    const bool writes_weights = get_arg_val<uint32_t>(13) != 0;
    constexpr uint32_t kPushArgs = 14;

    Noc noc;
    constexpr uint32_t kCoordsArg = 0;
    // Common runtime args: the NoC coordinates of the grid cores, then W's address and page size.

    // ---- W slice (writer-sent slices, NOC1): a sender is never a token core ----
    if constexpr (w_slice_nocs > 1) {
        if (core_idx <= w_sender_top && core_idx + G > w_sender_top) {
            MaybeDeviceZoneScope("wr_w_send");
            const auto weight = TensorAccessor(
                weight_args,
                get_common_arg_val<uint32_t>(kCoordsArg + num_cores),
                get_common_arg_val<uint32_t>(kCoordsArg + num_cores + 1));
            deliver_weight_slice<G, ht, w_slice_nocs, kWriterRisc>(
                noc,
                weight,
                CircularBuffer(get_named_compile_time_arg_val("cb_router_weight")).get_write_ptr(),
                w_sender_top - core_idx);
        }
    }
    if (is_token) {
        const auto combine_index = TensorAccessor(combine_index_args, combine_index_addr, combine_index_page);
        CircularBuffer slot_rows(cb_slot_rows);
        const uint32_t gathered_base = CircularBuffer(cb_gathered_rows).get_write_ptr();

        // ---- receive_weight_slice (writer-sent slices), inside the stage gather's read latency when it has one ----
        bool weight_received = false;
        const auto receive_member_weight = [&]() {
            MaybeDeviceZoneScope("wr_wait_w");
            CircularBuffer router_weight(get_named_compile_time_arg_val("cb_router_weight"));
            receive_weight_slice<G, w_slice_nocs, kWriterRisc>(
                noc, router_weight, get_named_compile_time_arg_val("weight_tiles"), member, false);
        };

        // ---- gather_stage_block (whole-piece push): the staged tile range of every tile row, from this core's and
        // its group peers' resident x shards (same address on every core). On NOC1, off the W multicast's NOC; the
        // untilize consumes it after the softmax. ----
        if constexpr (push_gather) {
            const uint32_t stage_lo = get_arg_val<uint32_t>(kPushArgs + kPushArgStageLo);
            if (get_arg_val<uint32_t>(kPushArgs + kPushArgPieces) != 0) {
                MaybeDeviceZoneScope("wr_stage_gather");
                CircularBuffer stage_tiles_cb(get_named_compile_time_arg_val("cb_stage_tiles"));
                const uint32_t core_tiles = (core_tokens + kTile - 1) / kTile;
                stage_tiles_cb.reserve_back(rps * stage_tiles);
                const uint32_t dst = stage_tiles_cb.get_write_ptr();
                for (uint32_t peer = 0; peer < G; ++peer) {
                    const uint32_t lo = stage_lo > peer * ht ? stage_lo : peer * ht;
                    const uint32_t hi =
                        stage_lo + stage_tiles < (peer + 1) * ht ? stage_lo + stage_tiles : (peer + 1) * ht;
                    if (lo >= hi) {
                        continue;
                    }
                    const uint32_t peer_coord = get_common_arg_val<uint32_t>(kCoordsArg + group * G + peer);
                    for (uint32_t tile_row = 0; tile_row < core_tiles; ++tile_row) {
                        noc.async_read(
                            UnicastEndpoint{},
                            CoreLocalMem<uint32_t>(dst + (tile_row * stage_tiles + lo - stage_lo) * kBf16TileBytes),
                            (hi - lo) * kBf16TileBytes,
                            {.noc_x = coord_x(peer_coord),
                             .noc_y = coord_y(peer_coord),
                             .addr = x_addr + (tile_row * ht + lo - peer * ht) * kBf16TileBytes},
                            {});
                    }
                }
                receive_member_weight();
                {
                    MaybeDeviceZoneScope("wr_stage_barrier");
                    noc.async_read_barrier();
                }
                stage_tiles_cb.push_back(rps * stage_tiles);
                weight_received = true;
            }
        }

        if (!weight_received) {
            receive_member_weight();
        }

        // ---- exchange_partials_block, westward half: this member's partial faces to the LOWER members on NOC1 (the
        // reader sends to the higher ones on NOC0) ----
        {
            CircularBuffer own_partial_west(get_named_compile_time_arg_val("cb_own_partial_west"));
            {
                MaybeDeviceZoneScope("wr_wait_own_partial");
                own_partial_west.wait_front(1);
            }
            MaybeDeviceZoneScope("wr_xchg_send");
            const uint32_t region = CircularBuffer(get_named_compile_time_arg_val("cb_partials")).get_write_ptr() +
                                    member * rps * kFp32TileBytes;
            send_partial_faces<rps>(noc, region, kCoordsArg + group * G, 0, member);
            noc.async_write_barrier();
            Semaphore<> sem_partials(get_named_compile_time_arg_val("sem_partials"));
            for (uint32_t peer = 0; peer < member; ++peer) {
                const uint32_t coord = get_common_arg_val<uint32_t>(kCoordsArg + group * G + peer);
                sem_partials.up(noc, coord_x(coord), coord_y(coord), 1);
            }
            own_partial_west.pop_front(1);
        }

        volatile tt_l1_ptr uint32_t* slot_page = nullptr;
        if (core_tokens > reader_tokens) {
            // ---- select_block: the writer's tokens, into the slot page the reader reserved ----
            CircularBuffer select_go(cb_select_go);
            CircularBuffer writer_counts(cb_writer_counts);
            {
                MaybeDeviceZoneScope("wr_wait_select_go");
                select_go.wait_front(1);
            }
            const volatile tt_l1_ptr uint32_t* go =
                reinterpret_cast<const volatile tt_l1_ptr uint32_t*>(select_go.get_read_ptr());
            volatile tt_l1_ptr uint32_t* page = reinterpret_cast<volatile tt_l1_ptr uint32_t*>(go[kGoSlotPage]);
            volatile tt_l1_ptr uint32_t* weight_words = reinterpret_cast<volatile tt_l1_ptr uint32_t*>(go[kGoWeights]);
            select_go.pop_front(1);
            uint64_t counts;
            {
                // occupancy: includes the per-tile wr_wait_scores waits. The writer only peeks at cb_scores (no pop):
                // the reader is its consumer and pops it after these counts arrive.
                MaybeDeviceZoneScope("wr_select");
                CircularBuffer scores(cb_scores);
                counts = select_tokens<rps>(
                    reinterpret_cast<const volatile tt_l1_ptr uint32_t*>(scores.get_read_ptr()),
                    page,
                    weight_words,
                    reader_tokens,
                    core_tokens,
                    [&](uint32_t tile) {
                        MaybeDeviceZoneScope("wr_wait_scores");
                        scores.wait_front(tile + 1);
                    });
            }
            writer_counts.reserve_back(1);
            volatile tt_l1_ptr uint32_t* words =
                reinterpret_cast<volatile tt_l1_ptr uint32_t*>(writer_counts.get_write_ptr());
            words[0] = static_cast<uint32_t>(counts);
            words[1] = static_cast<uint32_t>(counts >> 32);
            drain_stores(words + 2);  // slot and weight stores land before the reader reads them over the NoC
            writer_counts.push_back(1);

            // ---- push_rows_block: the writer's tokens ----
            {
                MaybeDeviceZoneScope("wr_wait_slots");
                slot_rows.wait_front(1);
            }
            volatile tt_l1_ptr uint32_t* header = page + 2 * tokens_per_shard;
            uint32_t base[kNumExperts];
            for (uint32_t expert = 0; expert < kNumExperts; ++expert) {
                base[expert] = header[kHeaderWriterBase + expert];
            }
            const uint32_t staging = header[kHeaderStaging] + reader_tokens * stage_tiles * kTileRowBytes;
            {
                MaybeDeviceZoneScope("wr_push");
                push_rows_block<
                    stage_tiles,
                    tokens_per_shard,
                    RowPusher<w, k, num_cores, push_bytes_in_flight, push_cmd_bufs>>(
                    kPushArgs,
                    gathered_base,
                    kCoordsArg,
                    page,
                    base,
                    staging,
                    reader_tokens,
                    core_tokens,
                    rows,
                    2 * group + 1,
                    1,
                    2 * num_groups,
                    group * push_stagger_tokens,
                    member == 0);
            }
            if constexpr (arrival_per_block) {
                noc.async_write_barrier();
                uint32_t writer_count[kNumExperts];
                for (uint32_t expert = 0; expert < kNumExperts; ++expert) {
                    writer_count[expert] = count_of(counts, expert);
                }
                signal_arrivals<k, num_cores>(
                    noc,
                    kPushArgs,
                    kCoordsArg,
                    sem_arrived,
                    header,
                    base,
                    writer_count,
                    rows,
                    2 * group + 1,
                    1,
                    2 * num_groups,
                    core_tokens - reader_tokens);
            }
            drain_stores(words + 2);
            slot_page = page;
            {
                MaybeDeviceZoneScope("wr_push_barrier");
                noc.async_write_barrier();
            }
            slot_rows.pop_front(1);
        }
        if constexpr (!arrival_per_block) {  // the landing increment, also from a writer without tokens
            const uint32_t coordinator_coord =
                get_common_arg_val<uint32_t>(kCoordsArg + get_named_compile_time_arg_val("coordinator"));
            Semaphore<>(sem_arrived).up(noc, coord_x(coordinator_coord), coord_y(coordinator_coord), 1);
        }
        // ---- write_combine_index_block, write_combine_weights_block (the writer's share, on NOC1): after the
        // landing increment. The slot page and the weight tiles are not rewritten after the reader's single publish.
        // ----
        if (slot_page != nullptr) {
            if (member == 0) {
                write_combine_index<tokens_per_shard>(
                    noc,
                    combine_index,
                    reinterpret_cast<uint32_t>(slot_page),
                    first_token,
                    reader_tokens,
                    round_up_to_tile(core_tokens));  // with the zeroed tile-padding tokens (R3)
            }
            if (writes_weights) {
                MaybeDeviceZoneScope("wr_weights_issue");
                constexpr uint32_t stride = get_named_compile_time_arg_val("weights_stride");
                constexpr auto weights_args = TensorAccessorArgs<weight_args.next_compile_time_args_offset()>();
                const auto combine_weights = TensorAccessor(
                    weights_args,
                    get_common_arg_val<uint32_t>(kCoordsArg + num_cores + 2),
                    get_common_arg_val<uint32_t>(kCoordsArg + num_cores + 3));
                write_weight_tiles<rps, get_named_compile_time_arg_val("t_pad")>(
                    noc,
                    combine_weights,
                    CircularBuffer(get_named_compile_time_arg_val("cb_weight_tiles")).get_write_ptr(),
                    first_token,
                    round_up_to_tile(core_tokens) / kTile,
                    stride > 1 ? member : 0,
                    2 * stride);
            }
            MaybeDeviceZoneScope("wr_final_barrier");
            noc.async_write_barrier();
        }
        noc.async_atomic_barrier();
    }

    // ---- write_dispatched_block: each owned block's w tiles to their DRAM pages, one flush per block ----
    if constexpr (out_dram) {
        MaybeDeviceZoneScope("wr_dram_out");
        const auto dispatched = TensorAccessor(dispatched_args, dispatched_addr, dispatched_page);
        CircularBuffer out_blocks(get_named_compile_time_arg_val("cb_dispatched"));
        for (uint32_t block = 0; block < num_blocks; ++block) {
            const uint32_t shard = core_idx + block * num_cores;
            const uint32_t first_page = (shard / k) * hidden_tiles_total + (shard % k) * w;
            out_blocks.wait_front(w);
            const uint32_t src = out_blocks.get_read_ptr();
            for (uint32_t tile = 0; tile < w; ++tile) {
                noc.async_write(
                    CoreLocalMem<uint32_t>(src + tile * kBf16TileBytes),
                    dispatched,
                    kBf16TileBytes,
                    {},
                    {.page_id = first_page + tile});
            }
            noc.async_writes_flushed();
            out_blocks.pop_front(w);
        }
        noc.async_write_barrier();
    }
}
