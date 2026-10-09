// SPDX-FileCopyrightText: 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

/*
 * Per-worker writer for the fused Wan2.2 distributed RMSNorm AG (forwarder model).
 *
 * The worker holds NO fabric connection — its forwarder core does. Per row the
 * worker:
 *   1. Takes compute's row-0 transposed stat tile (stats_transposed_local_cb)
 *      and NoC-writes its 128 B stick (two contiguous 64 B face-rows packed
 *      contiguous) into its forwarder's packet_buf[round%2] + slot*128 B, then
 *      increments the forwarder's fwd_arrival_sem.
 *   2. Waits on its own go-sem (forwarder sets it once that round's ring gather
 *      has landed in this chip's DRAM scratch).
 *   3. Reads its ring_size gathered sticks from DRAM (page(d, forwarder, round)
 *      + slot*128 B for each device d) into ROW 0 of stats_transposed_gathered_cb
 *      tiles, and pushes them to compute (which FPU-adds + transpose_wh_dest).
 *   4. Drains the row's output_cb tiles to the output tensor.
 *
 * Also populates compute's reduce-scalar / epsilon / trans_mat CBs up front
 * (shared helper) so the reader starts the input read ASAP.
 *
 * is_tp_1 (ring==1 / per_head_norm) never reaches this kernel — that path keeps
 * stats local in compute and uses the plain drain-only writer.
 */

#include <cstdint>

#include "api/dataflow/dataflow_api.h"
#include "api/dataflow/noc.h"
#include "api/dataflow/circular_buffer.h"
#include "api/dataflow/noc_semaphore.h"
#include "api/dataflow/endpoints.h"
#include "api/tensor/noc_traits.h"
#include "api/core_local_mem.h"
#include "tt_metal/fabric/hw/inc/noc_addr.h"
#include "dit_rmsnorm_scalar_setup.hpp"
#include "tools/profiler/kernel_profiler.hpp"

constexpr std::uint32_t output_cb = get_compile_time_arg_val(0);
constexpr std::uint32_t num_tile_cols = get_compile_time_arg_val(1);
constexpr std::uint32_t block_size = get_compile_time_arg_val(2);
constexpr std::uint32_t stats_transposed_local_cb = get_compile_time_arg_val(3);
constexpr std::uint32_t stats_transposed_gathered_cb = get_compile_time_arg_val(4);
constexpr std::uint32_t ring_size = get_compile_time_arg_val(5);
constexpr std::uint32_t head_dim_tiles = get_compile_time_arg_val(6);
constexpr std::uint32_t total_num_tile_rows = get_compile_time_arg_val(7);
constexpr std::uint32_t max_rounds = get_compile_time_arg_val(8);              // pages per (device,forwarder)
constexpr std::uint32_t stick_bytes = get_compile_time_arg_val(9);             // 128
constexpr std::uint32_t num_chunks_per_device = get_compile_time_arg_val(10);  // num_forwarders*max_rounds
// Shared packet CB (created on the whole core grid -> uniform L1 addr, so this
// worker's CircularBuffer(packet_cb).get_write_ptr() == the forwarder core's
// packet base) and grid-uniform sync sem ids.
constexpr std::uint32_t packet_cb = get_compile_time_arg_val(11);
constexpr std::uint32_t arrival_sem_id = get_compile_time_arg_val(12);
constexpr std::uint32_t go_sem_id = get_compile_time_arg_val(13);
// Tile row-0 layout (post transpose_wh): face_00 row0 = bytes [0,64), face_01
// row0 = bytes [1024,1088). 32 fp32 = 128 B real data per stat tile.
constexpr std::uint32_t kFaceRowBytes = 64u;
constexpr std::uint32_t kFace01Off = 1024u;
// Stats transported per token-tile: 1 for RMSNorm (sum-of-squares), 2 for Welford
// LayerNorm (mean, variance). The physical stick is num_stats * 128 B; each stat is
// one 128 B packed row-0 stick (two 64 B face-rows). num_stats==1 -> RMS layout.
constexpr std::uint32_t kStatBytes = 128u;
static_assert(stick_bytes % kStatBytes == 0, "stick_bytes must be a whole multiple of the 128 B packed stat stick");
constexpr std::uint32_t num_stats = stick_bytes / kStatBytes;

// Scalar/eps/trans_mat population args (after the output + dram accessors).
constexpr auto output_args = TensorAccessorArgs<14>();
constexpr auto stats_dram_args = TensorAccessorArgs<output_args.next_compile_time_args_offset()>();
constexpr std::uint32_t SCB = stats_dram_args.next_compile_time_args_offset();
constexpr std::uint32_t w_sum_cb = get_compile_time_arg_val(SCB + 0);
constexpr std::uint32_t w_avg_cb = get_compile_time_arg_val(SCB + 1);
constexpr std::uint32_t w_eps_cb = get_compile_time_arg_val(SCB + 2);
constexpr std::uint32_t w_transmat_cb = get_compile_time_arg_val(SCB + 3);
constexpr std::uint32_t w_reduce_factor = get_compile_time_arg_val(SCB + 4);
constexpr std::uint32_t w_eps_bits = get_compile_time_arg_val(SCB + 5);
constexpr std::uint32_t w_fuse_rope = get_compile_time_arg_val(SCB + 6);
constexpr auto w_transmat_args = TensorAccessorArgs<SCB + 7>();
// Broadcast [1,H] gamma read on BRISC at kernel start (the reader then skips it): the writer is
// idle until PRE ends, so gamma lands during the input read without sharing NCRISC's queue.
constexpr std::uint32_t WCB = w_transmat_args.next_compile_time_args_offset();
constexpr std::uint32_t writer_reads_weight = get_compile_time_arg_val(WCB + 0);
constexpr std::uint32_t w_weight_cb = get_compile_time_arg_val(WCB + 1);
constexpr std::uint32_t w_weight_tiles = get_compile_time_arg_val(WCB + 2);
constexpr auto w_weight_args = TensorAccessorArgs<WCB + 3>();
// Path-aware dual-NoC drain (Blackhole, DRAM-interleaved output, kernels in DM_DYNAMIC_NOC). The drain's
// default NoC1 routes north then west, so the hot links are the westward rows next to the DRAM columns,
// loaded by every core's wrap-around traffic. A destination whose NoC0 (east, then south) path is short goes
// out on NoC0 on every other visit to its bank, unloading those links without crowding NoC0's own rows.
constexpr std::uint32_t dual_noc_drain = get_compile_time_arg_val(w_weight_args.next_compile_time_args_offset());
// Two-wave column split: this worker owns one column half of its tile-row. Its stick sits in the page as
// fp32 tile row 0 (face_00 row 0 at the slot offset, face_01 row 0 at +1024), and the row's two halves
// are adjacent 64 B slots, so one gather_pair_stride-spaced read per device brings both halves' sticks.
constexpr std::uint32_t col_split = get_compile_time_arg_val(w_weight_args.next_compile_time_args_offset() + 1);
constexpr std::uint32_t gathered_cb_pages = get_compile_time_arg_val(w_weight_args.next_compile_time_args_offset() + 2);
constexpr std::uint32_t gather_pair_stride =
    get_compile_time_arg_val(w_weight_args.next_compile_time_args_offset() + 3);
constexpr std::uint32_t kPairReadBytes = kFace01Off + 2u * kFaceRowBytes;

void kernel_main() {
    size_t arg_idx = 0;
    const std::uint32_t output_addr = get_common_arg_val<std::uint32_t>(0);
    const std::uint32_t tile_row_start = get_arg_val<std::uint32_t>(arg_idx++);
    const std::uint32_t tile_row_end = get_arg_val<std::uint32_t>(arg_idx++);
    const std::uint32_t transformation_mat_addr = get_common_arg_val<std::uint32_t>(1);
    const std::uint32_t stats_dram_addr = get_common_arg_val<std::uint32_t>(2);
    // Forwarder core NoC coords (which core to write the stick to / inc arrival),
    // plus this worker's per-core forwarder group + slot (runtime, differs per core).
    const std::uint32_t fwd_x = get_arg_val<std::uint32_t>(arg_idx++);
    const std::uint32_t fwd_y = get_arg_val<std::uint32_t>(arg_idx++);
    const std::uint32_t my_forwarder_index = get_arg_val<std::uint32_t>(arg_idx++);
    const std::uint32_t my_slot = get_arg_val<std::uint32_t>(arg_idx++);
    // Column offset of this worker's half (0 without the split), the stick's byte offset in the forwarder
    // packet, the row pair's byte offset in each device page, and the arrival increment (1 << 16*wave).
    const std::uint32_t col_offset = get_arg_val<std::uint32_t>(arg_idx++);
    const std::uint32_t stick_off = get_arg_val<std::uint32_t>(arg_idx++);
    const std::uint32_t pair_off = get_arg_val<std::uint32_t>(arg_idx++);
    const std::uint32_t arrival_inc = get_arg_val<std::uint32_t>(arg_idx++);
    const std::uint32_t weight_addr = get_common_arg_val<std::uint32_t>(3);

    Noc noc;
    // Second NoC for the dual-NoC drain (only used when dual_noc_drain; the writer's own NoC is 1).
    Noc noc_alt(static_cast<std::uint8_t>(1u - noc.get_noc_id()));

    CircularBuffer cb_packet(packet_cb);
    CircularBuffer cb_output(output_cb);
    CircularBuffer cb_stats_local(stats_transposed_local_cb);
    CircularBuffer cb_stats_gathered(stats_transposed_gathered_cb);

    // Grid-uniform: my own packet_cb base == the forwarder's packet base, and both
    // semaphores resolve to the same L1 offset on me and on the forwarder.
    const std::uint32_t fwd_packet_buf_addr = cb_packet.get_write_ptr();
    const std::uint32_t packet_slot_bytes = cb_packet.get_tile_size();  // unit_packet_bytes (per round%2 slot)
    Semaphore<> fwd_arrival_sem(arrival_sem_id);
    Semaphore<> go_sem(go_sem_id);

    const std::uint32_t output_tile_bytes = cb_output.get_tile_size();
    const auto output_accessor = TensorAccessor(output_args, output_addr);
    const std::uint32_t output_page_bytes = output_accessor.get_aligned_page_size();
    const auto stats_dram = TensorAccessor(stats_dram_args, stats_dram_addr);
    const std::uint32_t gathered_tile_bytes = cb_stats_gathered.get_tile_size();
    const std::uint32_t stat_tile_bytes = cb_stats_local.get_tile_size();

    // Populate compute's scalar/eps/trans_mat CBs before anything else.
    dit_rmsnorm_generate_scalars_and_transmat<
        w_sum_cb,
        w_avg_cb,
        w_eps_cb,
        w_transmat_cb,
        w_reduce_factor,
        static_cast<bool>(w_fuse_rope)>(w_eps_bits, TensorAccessor(w_transmat_args, transformation_mat_addr));

    // ---- push my num_stats sticks for `round` into the forwarder's packet_buf, inc arrival ----
    // Each stat occupies a 128 B sub-stick at dst + s*128 (mean then variance for
    // LayerNorm); the worker contributes one slot (my_slot*stick_bytes) regardless.
    auto push_stick = [&](std::uint32_t round) {
        DeviceZoneScopedN("W_PUSH");
        cb_stats_local.wait_front(num_stats);
        const std::uint32_t src0 = cb_stats_local.get_read_ptr();
        const std::uint32_t dst = fwd_packet_buf_addr + (round & 1u) * packet_slot_bytes + stick_off;
        constexpr std::uint32_t f01_dst_off = (col_split != 0) ? kFace01Off : kFaceRowBytes;
        // The forwarder's packet buffer is a grid-uniform CB, so its address is our own.
        UnicastEndpoint fwd_core;
        for (std::uint32_t s = 0; s < num_stats; s++) {
            const std::uint32_t src = src0 + s * stat_tile_bytes;
            const std::uint32_t sub = dst + s * kStatBytes;
            noc.async_write(  // face_00 row0
                CoreLocalMem<std::uint32_t>(src),
                fwd_core,
                kFaceRowBytes,
                {},
                {.noc_x = fwd_x, .noc_y = fwd_y, .addr = sub});
            noc.async_write(  // face_01 row0
                CoreLocalMem<std::uint32_t>(src + kFace01Off),
                fwd_core,
                kFaceRowBytes,
                {},
                {.noc_x = fwd_x, .noc_y = fwd_y, .addr = sub + f01_dst_off});
        }
        // Arrival handshake: the stick must be visible in the forwarder's packet buffer
        // *before* the forwarder sees the count go up. The writes and the inc leave this core
        // on the same NoC and VC (NOC_UNICAST_WRITE_VC) for the same destination, so they are
        // delivered in order once the writes have left the NIU (the fabric's flush=true fused
        // write+inc relies on the same ordering). No write-ack round trip before the inc, and
        // no atomic-ack wait here: the atomic barrier runs once at kernel end. Flushed also
        // means the stat CB slot has been read out, so it can be popped.
        noc.async_writes_flushed();
        fwd_arrival_sem.up(noc, fwd_x, fwd_y, arrival_inc);
        cb_stats_local.pop_front(num_stats);
    };

    bool first_stick_pushed = false;

    // Gamma face_00 row 0 + face_01 row 0 per tile (the rest of a [1,H] TILE page is padding),
    // streamed to compute in chunks of NUM_DRAM_BANKS pages: each chunk's reads carry their own
    // read trid, and a chunk is pushed kGammaLookahead chunks behind the issue front, so compute's
    // x*gamma pre-pass (cumulative per-block weight_cb wait) starts as soon as PRE ends instead
    // of after the last gamma page lands (the issue loop is ~50 ns/read, ~5.5 us at H=7168/4).
    // Within a chunk each worker starts at a different page (one page per bank per chunk), so
    // the workers don't all hit the same DRAM bank at once.
    if constexpr (writer_reads_weight != 0) {
        DeviceZoneScopedN("W_GAMMA");
        CircularBuffer cb_weight(w_weight_cb);
        const auto weight_accessor = TensorAccessor(w_weight_args, weight_addr);
        const std::uint32_t weight_tile_bytes = cb_weight.get_tile_size();
        const std::uint32_t weight_datum_bytes = weight_tile_bytes / 1024u;    // bf16 2 B / fp32 4 B
        const std::uint32_t weight_face_row_bytes = 16u * weight_datum_bytes;  // FACE_WIDTH
        const std::uint32_t weight_face_bytes = 256u * weight_datum_bytes;     // FACE_HW
        constexpr std::uint32_t kGammaChunk = NUM_DRAM_BANKS;
        constexpr std::uint32_t kGammaTrids = 4u;  // read trids 1..4 on this RISC's NoC (BRISC only)
        constexpr std::uint32_t kGammaLookahead = 2u;
        static_assert(kGammaLookahead < kGammaTrids, "lookahead must not reuse an in-flight trid");
        constexpr std::uint32_t kFaceRowMaxBytes = 64u;  // fp32 face row; one NoC packet
        constexpr std::uint32_t num_gamma_chunks = (w_weight_tiles + kGammaChunk - 1u) / kGammaChunk;
        const std::uint32_t rot = tile_row_start % kGammaChunk;
        // Don't let the gamma issue loop gate the AG start: if PRE has already pushed the
        // first row's stat, push the stick now (write flush only, the reads stay in flight).
        auto poll_stick = [&]() {
            if (!first_stick_pushed && tile_row_start < tile_row_end &&
                cb_stats_local.pages_available_at_front(num_stats)) {
                push_stick(0);
                first_stick_pushed = true;
            }
        };
        cb_weight.reserve_back(w_weight_tiles);
        const std::uint32_t weight_base = cb_weight.get_write_ptr();
        for (std::uint32_t k = 0; k < num_gamma_chunks + kGammaLookahead; k++) {
            if (k < num_gamma_chunks) {
                const std::uint32_t chunk_start = k * kGammaChunk;
                const std::uint32_t chunk_pages =
                    (w_weight_tiles - chunk_start >= kGammaChunk) ? kGammaChunk : (w_weight_tiles - chunk_start);
                // Sticky tag: every read below (plain one-packet reads) is counted under this trid.
                noc_async_read_set_trid(1u + (k % kGammaTrids), noc.get_noc_id());
                std::uint32_t j = rot % chunk_pages;
                for (std::uint32_t n = 0; n < chunk_pages; n++) {
                    const std::uint32_t w_page = chunk_start + j;
                    const std::uint32_t w_dst = weight_base + w_page * weight_tile_bytes;
                    noc.async_read<NocOptions::DEFAULT, kFaceRowMaxBytes>(
                        weight_accessor,
                        CoreLocalMem<std::uint32_t>(w_dst),
                        weight_face_row_bytes,
                        {.page_id = col_offset + w_page},
                        {});
                    noc.async_read<NocOptions::DEFAULT, kFaceRowMaxBytes>(
                        weight_accessor,
                        CoreLocalMem<std::uint32_t>(w_dst + weight_face_bytes),
                        weight_face_row_bytes,
                        {.page_id = col_offset + w_page, .offset_bytes = weight_face_bytes},
                        {});
                    j = (j + 1 == chunk_pages) ? 0u : (j + 1);
                    poll_stick();
                }
            }
            if (k >= kGammaLookahead) {
                const std::uint32_t kb = k - kGammaLookahead;
                const std::uint32_t kb_start = kb * kGammaChunk;
                const std::uint32_t kb_pages =
                    (w_weight_tiles - kb_start >= kGammaChunk) ? kGammaChunk : (w_weight_tiles - kb_start);
                noc.async_read_barrier<NocOptions::TXN_ID>({.trid = 1u + (kb % kGammaTrids)});
                cb_weight.push_back(kb_pages);
            }
            poll_stick();
        }
        // Restore the default read trid for the plain reads below.
        noc_async_read_set_trid(0, noc.get_noc_id());
    }

    std::uint32_t go_target = 0;
    for (std::uint32_t tile_row = tile_row_start; tile_row < tile_row_end; tile_row++) {
        const std::uint32_t round = tile_row - tile_row_start;

        // ---- 1. push my sticks (unless already pushed while the gamma reads were in flight) ----
        if (!first_stick_pushed || round != 0) {
            push_stick(round);
        }

        // ---- 2. wait for the forwarder's go (this round's ring gather landed) ----
        {
            DeviceZoneScopedN("W_AGWAIT");
            go_target += 1;
            go_sem.wait_min(go_target);
        }

        // ---- 3. read num_stats*ring gathered sticks from DRAM into ROW 0 of gathered tiles ----
        // Device-major, stat-minor order: gathered tile (d*num_stats + s). For LayerNorm
        // this yields interleaved [mean_d, var_d] per device, as combine_welford_partials wants.
        if constexpr (col_split != 0) {
            // One read per device: [pair_off, pair_off + 1152) holds both halves' face_00 rows 0 (64 B
            // apart) and, 1024 B on, both face_01 rows 0. Landed at d * gather_pair_stride, half h's
            // stick is the fp32 tile row 0 at 64 B page d * pair_pages + h (compute indexes it there).
            cb_stats_gathered.reserve_back(gathered_cb_pages);
            const std::uint32_t gbase = cb_stats_gathered.get_write_ptr();
            for (std::uint32_t d = 0; d < ring_size; d++) {
                const std::uint32_t page_idx = d * num_chunks_per_device + my_forwarder_index * max_rounds + round;
                noc.async_read(
                    stats_dram,
                    CoreLocalMem<std::uint32_t>(gbase + d * gather_pair_stride),
                    kPairReadBytes,
                    {.page_id = page_idx, .offset_bytes = pair_off},
                    {});
            }
            noc.async_read_barrier();
            cb_stats_gathered.push_back(gathered_cb_pages);
        } else {
            cb_stats_gathered.reserve_back(num_stats * ring_size);
            const std::uint32_t gbase = cb_stats_gathered.get_write_ptr();
            for (std::uint32_t d = 0; d < ring_size; d++) {
                const std::uint32_t page_idx = d * num_chunks_per_device + my_forwarder_index * max_rounds + round;
                for (std::uint32_t s = 0; s < num_stats; s++) {
                    const std::uint32_t tile_dst = gbase + (d * num_stats + s) * gathered_tile_bytes;
                    const std::uint32_t src_off = my_slot * stick_bytes + s * kStatBytes;
                    noc.async_read(  // -> face_00 row0
                        stats_dram,
                        CoreLocalMem<std::uint32_t>(tile_dst),
                        kFaceRowBytes,
                        {.page_id = page_idx, .offset_bytes = src_off},
                        {});
                    noc.async_read(  // -> face_01 row0
                        stats_dram,
                        CoreLocalMem<std::uint32_t>(tile_dst + kFace01Off),
                        kFaceRowBytes,
                        {.page_id = page_idx, .offset_bytes = src_off + kFaceRowBytes},
                        {});
                }
            }
            noc.async_read_barrier();
            cb_stats_gathered.push_back(num_stats * ring_size);
        }

        // ---- 4. drain this row's output_cb tiles ----
        // Per-block wait + pop (NOT a cumulative wait with a single end-of-row pop):
        // under block_major_post the factory sizes output_cb to just 2*block_size
        // (block-local), NOT the whole row, so a cumulative output_cb.wait_front(
        // 3*block_size...) could never be satisfied — compute can't push a 3rd block
        // into a 2-block CB it never popped → deadlock (this is why is_tp_1 wide, which
        // uses the per-block drain-only writer, worked while TP>1 wide hung). Compute
        // pushes block_size-padded slots per col-block; wait/pop the full block, but
        // only NoC-write the valid tiles. Matches the drain-only writer's drain loop.
        {
            DeviceZoneScopedN("W_DRAIN");
            // Banks 0-3 sit in the x=0 DRAM column and banks 4-7 in the x=9 column (BH channel == bank).
            // The short eastward (NoC0) column is x=9 for cores left of it and x=0 (via the 16->0 wrap)
            // for cores right of it.
            const bool left_of_dram_col = my_x[0] < 9u;
            for (std::uint32_t col_tile = 0; col_tile < num_tile_cols; col_tile += block_size) {
                const std::uint32_t tiles_in_block =
                    ((num_tile_cols - col_tile) >= block_size) ? block_size : (num_tile_cols - col_tile);
                cb_output.wait_front(block_size);
                std::uint32_t rd = cb_output.get_read_ptr();
                for (std::uint32_t i = 0; i < tiles_in_block; i++) {
                    const std::uint32_t c = col_offset + col_tile + i;
                    const std::uint32_t h = c / head_dim_tiles;
                    const std::uint32_t t_col = c - h * head_dim_tiles;
                    const std::uint32_t out_idx =
                        h * total_num_tile_rows * head_dim_tiles + tile_row * head_dim_tiles + t_col;
                    bool use_alt = false;
                    if constexpr (dual_noc_drain != 0) {
                        const std::uint32_t bank = out_idx % NUM_DRAM_BANKS;
                        const bool east_col_bank = (bank >= NUM_DRAM_BANKS / 2) == left_of_dram_col;
                        use_alt = east_col_bank && (((out_idx / NUM_DRAM_BANKS) & 1u) == 0u);
                    }
                    // Posted (no write ack): the drain's per-core rate is capped downstream of issue
                    // (r03-b02-a03 / r03-b03-a03), so drop the ack round trip each 2 KB tile carries.
                    if (use_alt) {
                        noc_alt.async_write<NocOptions::POSTED>(
                            CoreLocalMem<std::uint32_t>(rd),
                            output_accessor,
                            output_page_bytes,
                            {},
                            {.page_id = out_idx});
                    } else {
                        noc.async_write<NocOptions::POSTED>(
                            CoreLocalMem<std::uint32_t>(rd),
                            output_accessor,
                            output_page_bytes,
                            {},
                            {.page_id = out_idx});
                    }
                    rd += output_tile_bytes;
                }
                // Departed from L1 before compute may reuse the slots.
                noc.async_writes_flushed<NocOptions::POSTED>();
                if constexpr (dual_noc_drain != 0) {
                    noc_alt.async_writes_flushed<NocOptions::POSTED>();
                }
                cb_output.pop_front(block_size);
            }
        }
    }
    noc.async_write_barrier();
    noc.async_writes_flushed<NocOptions::POSTED>();
    noc.async_atomic_barrier();  // the arrival incs (not waited on per push)
    if constexpr (dual_noc_drain != 0) {
        noc_alt.async_write_barrier();
        noc_alt.async_writes_flushed<NocOptions::POSTED>();
    }
    // Reset the go-sem for the next invocation. Trace replay re-runs this kernel without
    // re-running host-side semaphore init, so leaving a stale non-zero count here would let
    // the next replay's very first go_sem.wait_min(1) fall straight through.
    go_sem.set(0);
}
