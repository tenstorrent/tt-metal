// SPDX-FileCopyrightText: 2026 Tenstorrent USA, Inc.
// SPDX-License-Identifier: Apache-2.0

// nomic_moe_route_dispatch reader (RISCV_0 / NOC0), every core of the 11x10 grid.
//
// Token core (group g, member c; x shard resident in L1):
//   [W sender of an even slice: DRAM read + multicast, first] -> publish_x_block (round 1, matmul) -> zero the
//   combine_weights tiles -> receive_weight_slice (even members; the writer receives odd ones) -> publish x (round 2,
//   untilize; not with whole-piece push) -> exchange_partials_block -> select_block (reader's tokens; the writer
//   selects the rest) -> leader: count row to the coordinator (posted) -> publish the slot page to the writer ->
//   count table -> assign_rows_block (table scan split with a selecting writer) -> push_rows_block
//   (the reader's tokens; rows assigned while pushing; posted writes) -> its combine_weights tiles' weights ->
//   leader: write_combine_index_block ->
//   write_combine_weights_block (the member's tiles, every other one when the writer holds tokens)
// Every core: [coordinator: table multicast, expert_counts] -> owner_land: wait until the push writes of this core's
//   owned blocks have landed here (posted-write slave counter, no global barrier) -> publish the owned blocks of
//   cb_gathered_rows to compute.
//
// Raw NoC for the intra-group partial exchange, the count gather (posted rows counted on the coordinator's
// NIU_SLV_POSTED_WR_REQ_RECEIVED: no kernel_lib helper counts landed posted writes; replacing write barrier +
// Semaphore::up per row measured 8x512 24.0 -> 23.4 us, Perf R3.3 r33_table_assign) and the row pushes: the Mcast
// helper is one-to-many (op_design.md, Helpers considered and rejected). select_block / assign_rows_block are scalar
// RISC code.

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
    const uint32_t coordinator_coord = get_common_arg_val<uint32_t>(kCoordsArg + coordinator);
    // Posted-write slave counts before any push lands here: pushes start only after the count table, thousands of
    // cycles after every core's launch.
    uint32_t land_base = owner_land_count();

    const bool is_coordinator = core_idx == coordinator;
    const bool is_weight_sender = core_idx <= w_sender_top && core_idx + G > w_sender_top;
    const uint32_t sender_slice = w_sender_top - core_idx;

    // ---- W slice (reader-sent slices): issued before anything else ----
    if (is_weight_sender) {
        MaybeDeviceZoneScope("rd_w_send");
        deliver_weight_slice<G, ht, w_slice_nocs, kReaderRisc>(
            noc, weight, weight_base, sender_slice, get_named_compile_time_arg_val("sem_w_local"));
    }

    // Zero the landed-pass counters before round 1 lets compute start: peers send code flags once our partials
    // reached them, which the writer can do while this RISC is still sending W.
    const uint32_t code_flags_base = CircularBuffer(get_named_compile_time_arg_val("cb_code_flags")).get_write_ptr();
    if constexpr (G > 1) {
        if (is_token) {
            init_code_flags(reinterpret_cast<volatile tt_l1_ptr uint32_t*>(code_flags_base), 2 * rps);
        }
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
            {
                MaybeDeviceZoneScope("rd_table_rows");
                // The count rows are the only posted writes landing here before the table goes out (pushes start
                // after it); a leader coordinator's own row is already in place.
                const uint32_t rows_in = num_groups - (token_core && member == 0 ? 1 : 0);
                while (owner_land_count() - land_base < rows_in) {
                }
                invalidate_l1_cache();
                land_base += rows_in;
            }
            table_channel.sender(noc).send(scratch_base, scratch_base, num_groups * table_row_bytes);
        } else if (token_core) {
            MaybeDeviceZoneScope("rd_wait_table");
            table_channel.receiver(noc).receive();
        }
    };
    static_assert(packed_row_words == 4, "a count-table row is 8 u16 counts");

    if (is_token) {
        const HalfSplit split = half_split(reader_tokens, core_tokens);
        // ---- combine_weights tiles (members that write them), in the W-fetch shadow: zero now, column 0 is filled
        // in the count-table wait ----
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

        // ---- exchange_partials_block, receive side: the writer sends each own partial tile to every peer as compute
        // packs it, followed by a landed-tile count on this member's counter there. A tile goes to compute (G pages of
        // cb_partials) once it is packed here and every peer's count has reached it. ----
        {
            MaybeDeviceZoneScope("rd_wait_own_partial");
            own_partial.wait_front(1);
        }
        // ---- publish_x_block, round 2 (untilize, after the softmax): reserve_back waits for the whole matmul ----
        if constexpr (!push_gather) {
            x_tiles.reserve_back(shard_tiles);
            x_tiles.push_back(shard_tiles);
        }
        const uint32_t first_member_core = group * G;
        {
            MaybeDeviceZoneScope("rd_wait_peers");
            partials.reserve_back(G * rps);
            for (uint32_t tile = 1; tile <= rps; ++tile) {
                own_partial.wait_front(tile);
                for (uint32_t contributor = 0; contributor < G; ++contributor) {
                    if (contributor != member) {
                        Semaphore<>(kSemPartials[contributor]).wait_min(tile);
                    }
                }
                partials.push_back(G);
            }
            own_partial.pop_front(rps);
        }

        // ---- code all-gather: this member's code passes go to the higher members and this core (the writer sends to
        // the lower ones), from inside the select's per-half waits ----
        CodePump<G> code_pump(
            get_named_compile_time_arg_val("cb_code_stage"),
            scores.get_read_ptr(),
            code_flags_base,
            member,
            kCoordsArg + first_member_core,
            member,
            G,
            split);
        const volatile tt_l1_ptr uint32_t* code_flags =
            reinterpret_cast<const volatile tt_l1_ptr uint32_t*>(code_flags_base);

        // ---- select_block: the reader's tokens here, the writer's on the writer RISC ----
        slot_rows.reserve_back(1);
        slot_page = reinterpret_cast<volatile tt_l1_ptr uint32_t*>(slot_rows.get_write_ptr());
        // R3: tile-padding tokens of a ragged last tile row are never selected; their combine_index is 0.
        for (uint32_t t = core_tokens; t < core_tiles * kTile; ++t) {
            slot_page[t] = 0;
            slot_page[tokens_per_shard + t] = 0;
        }
        const uint32_t reader_hi = core_tokens < reader_tokens ? core_tokens : reader_tokens;
        const bool writer_selects = core_tokens > reader_tokens;
        if (writer_selects) {
            select_go.reserve_back(1);
            volatile tt_l1_ptr uint32_t* go = reinterpret_cast<volatile tt_l1_ptr uint32_t*>(select_go.get_write_ptr());
            go[kGoSlotPage] = reinterpret_cast<uint32_t>(slot_page);
            select_go.push_back(1);
        }
        uint64_t reader_counts;
        {
            // occupancy: includes the per-tile rd_wait_scores waits (the code tiles are published one at a time)
            MaybeDeviceZoneScope("rd_select");
            reader_counts = select_tokens<rps>(
                reinterpret_cast<const volatile tt_l1_ptr uint32_t*>(scores.get_read_ptr()),
                slot_page,
                0,
                reader_hi,
                [&](uint32_t half) {
                    MaybeDeviceZoneScope("rd_wait_scores");
                    if constexpr (G > 1) {
                        wait_code_half<G>(noc, code_pump, code_flags, half, split);
                    } else {
                        scores.wait_front(half / 2 + 1);
                    }
                });
        }
        if constexpr (G > 1) {
            MaybeDeviceZoneScope("rd_code_finish");
            code_pump.finish(noc);
        }
        const uint32_t code_base = scores.get_read_ptr();  // the code rows stay put after the G == 1 pop below
        uint64_t writer_counts = 0;
        if (writer_selects) {
            MaybeDeviceZoneScope("rd_wait_writer_counts");
            writer_counts_cb.wait_front(1);
            const volatile tt_l1_ptr uint32_t* words =
                reinterpret_cast<const volatile tt_l1_ptr uint32_t*>(writer_counts_cb.get_read_ptr());
            writer_counts = (uint64_t{words[1]} << 32) | words[0];
            writer_counts_cb.pop_front(1);
        }

        // ---- exchange_counts (gather half): the leader's row of 8 u16 counts to the coordinator, first thing after
        // the counts: straight from the two RISCs' packed byte counts (an expert's count over a core is at most its 128
        // tokens, so no byte lane carries) ----
        if (member == 0) {
            MaybeDeviceZoneScope("rd_count_send");
            const uint64_t packed = reader_counts + writer_counts;
            const uint32_t lo = static_cast<uint32_t>(packed);
            const uint32_t hi = static_cast<uint32_t>(packed >> 32);
            volatile tt_l1_ptr uint32_t* own_row = table + group * packed_row_words;
            own_row[0] = (lo & 0xFF) | ((lo & 0xFF00) << 8);
            own_row[1] = ((lo >> 16) & 0xFF) | ((lo >> 24) << 16);
            own_row[2] = (hi & 0xFF) | ((hi & 0xFF00) << 8);
            own_row[3] = ((hi >> 16) & 0xFF) | ((hi >> 24) << 16);
            drain_stores(drain_word);
            const uint32_t row_addr = scratch_base + group * table_row_bytes;
            // Posted: the coordinator counts the rows on its posted-write slave counter (no ack, no atomic).
            if (!is_coordinator) {
                noc.async_write<NocOptions::POSTED>(
                    CoreLocalMem<uint32_t>(row_addr),
                    UnicastEndpoint{},
                    table_row_bytes,
                    {},
                    {.noc_x = coord_x(coordinator_coord), .noc_y = coord_y(coordinator_coord), .addr = row_addr});
            }
        }
        if constexpr (G == 1) {
            scores.wait_front(rps);
            scores.pop_front(rps);
        }
        // ---- assign_rows_block, part 1: publish the slot page before the table wait (staging address, the reader's
        // counts, the table address), so a selecting writer scans the landed table itself; the table-landed flag
        // follows the receive (Perf R3.3) ----
        volatile tt_l1_ptr uint32_t* header = slot_page + top_k * tokens_per_shard;
        uint32_t staging = 0;
        if (stages) {
            MaybeDeviceZoneScope("rd_wait_xrows");
            x_rows.wait_front(rps * stage_tiles);
            staging = x_rows.get_read_ptr();
        }
        header[kHeaderStaging] = staging;
        header[kHeaderReaderCounts] = static_cast<uint32_t>(reader_counts);
        header[kHeaderReaderCounts + 1] = static_cast<uint32_t>(reader_counts >> 32);
        header[kHeaderTable] = scratch_base;
        header[kHeaderTableFlag] = 0;
        header[kHeaderScanReader + kScanFlag] = 0;
        header[kHeaderScanWriter + kScanFlag] = 0;
        drain_stores(drain_word);
        slot_rows.push_back(1);
        // ---- count table: gathered on the coordinator, multicast to the grid ----
        receive_or_send_table(true);
        header[kHeaderTableFlag] = 1;

        // ---- assign_rows_block, part 2: region starts and the reader's first row per expert. With a selecting
        // writer each RISC scans half the table rows and they swap packed partial sums (scan_rows_packed). ----
        uint32_t reader_base[kNumExperts];
        {
            MaybeDeviceZoneScope("rd_assign");
            uint32_t totals[kNumExperts];
            uint32_t before[kNumExperts];
            const tt_l1_ptr uint32_t* rows_words = const_cast<const tt_l1_ptr uint32_t*>(table);
            volatile tt_l1_ptr uint32_t* mine = header + kHeaderScanReader;
            scan_rows_packed(rows_words, 0, writer_selects ? num_groups / 2 : num_groups, group, mine);
            if (writer_selects) {
                mine[kScanFlag] = 1;
                const volatile tt_l1_ptr uint32_t* theirs = header + kHeaderScanWriter;
                wait_flag(theirs + kScanFlag);
                combine_scans(mine, theirs, totals, before);
            } else {
                combine_scans(mine, static_cast<const uint32_t*>(nullptr), totals, before);
            }
            assign_regions(header, totals, before, 0, reader_base);
        }

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
        // ---- write_combine_index_block, write_combine_weights_block: right after the pushes, in the other cores'
        // push shadow (the owners prove landing themselves; nothing here waits on the pushes) ----
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
        // combine_weights column 0 of the reader's tiles: off the table -> push chain (Perf R3.3)
        if (weight_writes > 0) {
            MaybeDeviceZoneScope("rd_weights_fill");
            fill_weight_tiles<rps>(code_base, weight_base_tiles, core_tiles, core_tokens, weights_first, weights_step);
        }
        if (weight_writes > 0) {
            MaybeDeviceZoneScope("rd_weights_issue");
            drain_stores(drain_word);
            write_weight_tiles<rps, t_pad>(
                noc, combine_weights, weight_base_tiles, first_token, core_tiles, weights_first, weights_step);
        }
    } else {
        receive_or_send_table(false);
    }

    if (is_coordinator) {
        uint32_t sums[2 * 4];
        uint32_t totals[kNumExperts];
        uint32_t before[kNumExperts];
        scan_rows_packed(const_cast<const tt_l1_ptr uint32_t*>(table), 0, num_groups, 0, sums);
        combine_scans(sums, static_cast<const uint32_t*>(nullptr), totals, before);
        volatile tt_l1_ptr uint32_t* staged =
            reinterpret_cast<volatile tt_l1_ptr uint32_t*>(scratch_base + counts_offset);
        for (uint32_t expert = 0; expert < kNumExperts; ++expert) {
            staged[expert] = totals[expert];
        }
        drain_stores(drain_word);
        noc.async_write(
            CoreLocalMem<uint32_t>(scratch_base + counts_offset), expert_counts, kNumExperts * 4, {}, {.page_id = 0});
    }

    // ---- owner_land: this core's blocks are complete once 32 * num_blocks push writes have landed here ----
    if (num_blocks > 0) {
#if !defined(ABLATE_PUSH) && !defined(ABLATE_PUSH_WRITES)
        MaybeDeviceZoneScope("rd_wait_land");
        while (owner_land_count() - land_base < kTile * num_blocks) {
        }
#endif
        gathered.reserve_back(num_blocks * w);
        gathered.push_back(num_blocks * w);
    }
    MaybeDeviceZoneScope("rd_final_barrier");
    noc.async_write_barrier();
    noc.async_atomic_barrier();
    noc.async_writes_flushed<NocOptions::POSTED>();  // the pushes' data has left this core's L1
}
