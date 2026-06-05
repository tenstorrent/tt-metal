// SPDX-FileCopyrightText: 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

/*
 * Multi-worker MUX writer for the fused Wan2.2 distributed RMSNorm op (TP>1).
 *
 * Same logical job as `wan_rmsnorm_fused_writer.cpp`'s TP>1 path: ring-AG the
 * per-row stats this worker owns, then drain output_cb to DRAM. The
 * difference: this kernel uses the fabric MUX (`WorkerToFabricMuxSender`)
 * instead of opening a dedicated `FabricConnectionManager` channel per
 * worker. That lets many worker cores per chip share a small number of MUX
 * cores (and thus a small number of physical fabric channels), so we can
 * lift the previous single-AG-core restriction and parallelize the compute
 * across N worker cores per chip.
 *
 * Per-worker layout:
 *   - Worker owns rows [tile_row_start, tile_row_end).
 *   - Worker connects to up to two MUX cores: one for forward fabric, one for
 *     backward (whichever direction(s) has a valid neighbor in this topology).
 *   - For each row: local NoC write to my own stats_gathered slot, plus a
 *     fabric multicast fused write+atomic_inc in each valid direction.
 *   - Then waits for `num_tile_rows * (ring_size - 1)` atomic incs on the
 *     per-worker GlobalSemaphore (each remote chip's matching worker writes
 *     to my sem once per row).
 *   - Pushes the gathered stats batch to compute, then drains output_cb.
 *
 * Termination protocol (mirrors all_gather_async's minimal_default_writer):
 *   - Each worker disconnects from its MUX(es).
 *   - The "termination master" worker (one per MUX) waits for (num_mux_clients - 1)
 *     ack incs on a local sem, then sends the graceful-terminate signal to
 *     the MUX. Non-master workers signal the master via that same sem.
 */

#include <cstdint>

#include "api/dataflow/dataflow_api.h"
#include "tt_metal/fabric/hw/inc/noc_addr.h"
#include "tt_metal/fabric/hw/inc/packet_header_pool.h"
#include "tt_metal/fabric/hw/inc/tt_fabric_mux_interface.hpp"
#include "tt_metal/fabric/hw/inc/linear/api.h"
// Fabric-canonical addrgen: addrgen_detail::get_noc_address() for fabric
// destination addresses (handles the Wormhole DRAM-coord -> noc0 flip the
// fabric EDM expects; plain get_noc_addr() does not).
#include "tt_metal/fabric/hw/inc/linear/addrgen_api.h"
// For populating compute's reduce-scalar / epsilon / trans_mat CBs here (moved
// off the reader so the reader starts input reads ASAP).
#include "ttnn/cpp/ttnn/kernel_lib/reduce_helpers_dataflow.hpp"
#include "ttnn/kernel/dataflow/generate_bcast_scalar.hpp"
#include "tools/profiler/kernel_profiler.hpp"

using namespace tt::tt_fabric::linear::experimental;

// =============================================================================
// Compile-time args (Phase 9 packed-page AG)
// =============================================================================
constexpr uint32_t output_cb = get_compile_time_arg_val(0);
constexpr uint32_t num_tile_cols = get_compile_time_arg_val(1);
constexpr uint32_t block_size = get_compile_time_arg_val(2);
// Compute pushes block_size slots per col-block (LLK packer requirement), so a
// row occupies div_up(num_tile_cols, block_size) * block_size CB slots. output_cb
// is sized to 2 * this, so each row's slots are contiguous (never wrap mid-row).
constexpr uint32_t padded_row_tiles = ((num_tile_cols + block_size - 1u) / block_size) * block_size;
// stats_local_cb : window_size fp32 stat tiles produced by the compute kernel
// (reduce<SUM,REDUCE_ROW> output). Real data lives in COL 0 — 32 fp32 values
// stored at strided offsets within face_00 (col 0 of the top-left face) +
// face_10 (col 0 of the bottom-left face). The other 31 cols are zero by
// LLK design. The writer extracts col 0 via 32 strided L1 loads per tile and
// packs them contiguously into stats_packed_local_cb for a single fabric mcast.
constexpr uint32_t stats_local_cb = get_compile_time_arg_val(3);
// stats_gathered_cb : window_size * ring_size fp32 tiles. After AG, the
// writer scatters the gathered packed bytes back into row 0 of these tiles
// (face_00[0..63] + face_01[0..63] per tile). Compute then transposes each
// tile to col 0 before the existing AVG-reduce chain.
constexpr uint32_t stats_gathered_cb = get_compile_time_arg_val(4);
// stats_packed_local_cb : 2 slots × page_size_bytes (= TILE_HEIGHT *
// window_size * sizeof(float)). Writer-owned scratch; packs row-0 spans
// from `window_size` transposed tiles into one row-major page, fabric
// mcasts it, then advances to the next slot for the next chunk.
constexpr uint32_t stats_packed_local_cb = get_compile_time_arg_val(5);
// stats_packed_gathered_cb : ring_size slots × page_size_bytes. Writer
// reads (ring_size-1) remote-device packed pages from DRAM into this CB;
// the local-device slot is L1-copied from stats_packed_local_cb (skip the
// local-DRAM roundtrip — Phase 1.1 pattern at page granularity).
constexpr uint32_t stats_packed_gathered_cb = get_compile_time_arg_val(6);
constexpr uint32_t ring_size = get_compile_time_arg_val(7);
constexpr uint32_t my_device_index = get_compile_time_arg_val(8);
constexpr uint32_t num_targets_forward = get_compile_time_arg_val(9);
constexpr uint32_t num_targets_backward = get_compile_time_arg_val(10);
constexpr uint32_t chunk_size_rows = get_compile_time_arg_val(11);
constexpr uint32_t num_chunks_per_device = get_compile_time_arg_val(12);
// See wan_rmsnorm_fused_writer.cpp for the output_tile_idx mapping rationale.
constexpr uint32_t head_dim_tiles = get_compile_time_arg_val(13);
constexpr uint32_t total_num_tile_rows = get_compile_time_arg_val(14);

// MUX CT args (5, in canonical order — matches ccl::fabric_mux_connection_ct_args).
// Same for forward and backward MUX (we use the same FabricMuxConfig for both).
constexpr uint8_t fabric_mux_num_buffers_per_channel = get_compile_time_arg_val(15);
constexpr size_t fabric_mux_channel_buffer_size_bytes = get_compile_time_arg_val(16);
constexpr size_t fabric_mux_status_address = get_compile_time_arg_val(17);
constexpr size_t fabric_mux_termination_signal_address = get_compile_time_arg_val(18);
constexpr uint32_t num_mux_clients = get_compile_time_arg_val(19);

constexpr auto output_args = TensorAccessorArgs<20>();
// Packed-page DRAM scratch accessor.
constexpr auto stats_dram_args = TensorAccessorArgs<output_args.next_compile_time_args_offset()>();

// Scalar/eps/trans_mat population args (appended after the accessors so the
// fixed/MUX/accessor CT indices above are untouched). The writer populates
// compute's reduce_scalar_*/epsilon/transformation_mat CBs so the reader can
// start the input read immediately.
constexpr uint32_t SCB_BASE = stats_dram_args.next_compile_time_args_offset();
constexpr uint32_t w_reduce_scalar_sum_cb = get_compile_time_arg_val(SCB_BASE + 0);
constexpr uint32_t w_reduce_scalar_avg_cb = get_compile_time_arg_val(SCB_BASE + 1);
constexpr uint32_t w_epsilon_cb = get_compile_time_arg_val(SCB_BASE + 2);
constexpr uint32_t w_transformation_mat_cb = get_compile_time_arg_val(SCB_BASE + 3);
constexpr uint32_t w_reduce_factor = get_compile_time_arg_val(SCB_BASE + 4);
constexpr uint32_t w_epsilon_bits = get_compile_time_arg_val(SCB_BASE + 5);
constexpr uint32_t w_fuse_rope = get_compile_time_arg_val(SCB_BASE + 6);
constexpr auto w_transmat_args = TensorAccessorArgs<SCB_BASE + 7>();

// Tile layout after transpose_wh: real per-token data sits in row 0 of the
// 32x32 tile = row 0 of face_00 (top-left, byte offsets 0..63) plus row 0 of
// face_01 (top-right, byte offsets 1024..1087). The 960 bytes between those
// two spans are garbage (rest of face_00). face_10/face_11 (bottom row of
// faces) also hold garbage. So real data per tile = 2 × 64 B = 128 B.
constexpr uint32_t kTileFaceRowBytes = 64u;                    // 16 fp32 values
constexpr uint32_t kTileFace01ByteOffset = 1024u;              // start of face_01 in tile
constexpr uint32_t kRowBytesPerTile = 2u * kTileFaceRowBytes;  // 128 — real data per tile-row

// =============================================================================
// Per-direction MUX runtime-arg parsing
// =============================================================================
struct MuxRtArgs {
    bool connection_valid;
    bool is_termination_master;
    uint8_t mux_x;
    uint8_t mux_y;
    uint32_t channel_base_address;
    uint32_t connection_info_address;
    uint32_t connection_handshake_address;
    uint32_t flow_control_address;
    uint32_t buffer_index_address;
    uint8_t channel_id;
    uint32_t termination_sync_addr;
    uint32_t local_fabric_mux_status_addr;
    uint32_t local_flow_control_addr;
    uint32_t local_teardown_addr;
    uint32_t local_buffer_index_addr;
    uint8_t termination_master_noc_x;
    uint8_t termination_master_noc_y;
};

FORCE_INLINE MuxRtArgs parse_mux_rt_args(size_t& arg_idx) {
    MuxRtArgs a;
    a.connection_valid = get_arg_val<uint32_t>(arg_idx++) == 1;
    a.is_termination_master = get_arg_val<uint32_t>(arg_idx++) == 1;
    a.mux_x = static_cast<uint8_t>(get_arg_val<uint32_t>(arg_idx++));
    a.mux_y = static_cast<uint8_t>(get_arg_val<uint32_t>(arg_idx++));
    a.channel_base_address = get_arg_val<uint32_t>(arg_idx++);
    a.connection_info_address = get_arg_val<uint32_t>(arg_idx++);
    a.connection_handshake_address = get_arg_val<uint32_t>(arg_idx++);
    a.flow_control_address = get_arg_val<uint32_t>(arg_idx++);
    a.buffer_index_address = get_arg_val<uint32_t>(arg_idx++);
    a.channel_id = static_cast<uint8_t>(get_arg_val<uint32_t>(arg_idx++));
    a.termination_sync_addr = get_semaphore(get_arg_val<uint32_t>(arg_idx++));
    a.local_fabric_mux_status_addr = get_semaphore(get_arg_val<uint32_t>(arg_idx++));
    a.local_flow_control_addr = get_semaphore(get_arg_val<uint32_t>(arg_idx++));
    a.local_teardown_addr = get_semaphore(get_arg_val<uint32_t>(arg_idx++));
    a.local_buffer_index_addr = get_semaphore(get_arg_val<uint32_t>(arg_idx++));
    a.termination_master_noc_x = static_cast<uint8_t>(get_arg_val<uint32_t>(arg_idx++));
    a.termination_master_noc_y = static_cast<uint8_t>(get_arg_val<uint32_t>(arg_idx++));
    return a;
}

FORCE_INLINE tt::tt_fabric::WorkerToFabricMuxSender<fabric_mux_num_buffers_per_channel> build_mux_sender(
    const MuxRtArgs& a) {
    return tt::tt_fabric::build_connection_to_fabric_endpoint<fabric_mux_num_buffers_per_channel>(
        a.mux_x,
        a.mux_y,
        a.channel_id,
        fabric_mux_num_buffers_per_channel,
        fabric_mux_channel_buffer_size_bytes,
        a.channel_base_address,
        a.connection_info_address,
        a.connection_handshake_address,
        a.flow_control_address,
        a.buffer_index_address,
        a.local_flow_control_addr,
        a.local_teardown_addr,
        a.local_buffer_index_addr);
}

// Signal-and-or-wait termination handshake on a single MUX. Mirrors
// minimal_default_writer.cpp:662-676 — non-masters atomic_inc the master's
// termination_sync sem; the master waits for (num_mux_clients - 1) incs then
// graceful-terminates the MUX. We collapse this into one helper since both
// directions use identical logic.
FORCE_INLINE void terminate_mux_handshake(const MuxRtArgs& a) {
    if (a.is_termination_master) {
        auto* sync_ptr = reinterpret_cast<volatile tt_l1_ptr uint32_t*>(a.termination_sync_addr);
        noc_semaphore_wait(sync_ptr, num_mux_clients - 1);
        tt::tt_fabric::fabric_endpoint_terminate(a.mux_x, a.mux_y, fabric_mux_termination_signal_address);
    } else {
        uint64_t dest_addr =
            safe_get_noc_addr(a.termination_master_noc_x, a.termination_master_noc_y, a.termination_sync_addr, 0);
        noc_semaphore_inc(dest_addr, 1);
        noc_async_atomic_barrier();
    }
}

void kernel_main() {
    // ---------- runtime args ----------
    size_t arg_idx = 0;
    const uint32_t output_addr = get_arg_val<uint32_t>(arg_idx++);
    const uint32_t tile_row_start = get_arg_val<uint32_t>(arg_idx++);
    const uint32_t tile_row_end = get_arg_val<uint32_t>(arg_idx++);
    const uint32_t out_ready_sem_bank_addr = get_arg_val<uint32_t>(arg_idx++);
    // Persistent DRAM stats buffer (Phase 1). The buffer holds
    // [num_workers, chunk_size_rows, ring_size] fp32 tiles. This worker owns
    // tiles [worker_tile_base, worker_tile_base + chunk_size_rows*ring_size).
    const uint32_t stats_dram_addr = get_arg_val<uint32_t>(arg_idx++);
    // First chunk index this worker owns on this chip. Chunk indices are
    // chip-global so all workers share the same DRAM scratch — worker i owns
    // chunks [worker_chunk_base, worker_chunk_base + chunks_in_this_worker).
    const uint32_t worker_chunk_base = get_arg_val<uint32_t>(arg_idx++);
    // trans_mat base address for the writer-side population (only read when
    // w_fuse_rope). Refreshed by override_runtime_arguments on cache hits.
    const uint32_t transformation_mat_addr = get_arg_val<uint32_t>(arg_idx++);

    // Two MUX rt blocks: forward first, then backward. Both blocks present
    // always (set by host), with `connection_valid=false` if that direction
    // has no neighbor in this topology (e.g. line endpoints).
    MuxRtArgs fwd_mux_args = parse_mux_rt_args(arg_idx);
    MuxRtArgs bwd_mux_args = parse_mux_rt_args(arg_idx);

    const uint32_t output_tile_bytes = get_tile_size(output_cb);
    const auto output_accessor = TensorAccessor(output_args, output_addr);
    const auto stats_dram_accessor = TensorAccessor(stats_dram_args, stats_dram_addr);
    const uint32_t num_tile_rows = tile_row_end - tile_row_start;

    // Populate compute's reduce-scalar / epsilon / trans_mat CBs here (moved off
    // the reader so the reader can start the input read ASAP). Done before the
    // MUX handshake so the values are ready by the time compute starts; the
    // trans_mat NoC read uses this writer's own NoC (independent of fabric).
    dataflow_kernel_lib::calculate_and_prepare_reduce_scaler<
        w_reduce_scalar_sum_cb,
        ckernel::PoolType::SUM,
        ckernel::ReduceDim::REDUCE_ROW>();
    dataflow_kernel_lib::calculate_and_prepare_reduce_scaler<
        w_reduce_scalar_avg_cb,
        ckernel::PoolType::AVG,
        ckernel::ReduceDim::REDUCE_ROW,
        w_reduce_factor>();
    generate_bcast_col_scalar(w_epsilon_cb, w_epsilon_bits);
    if constexpr (w_fuse_rope) {
        const auto transformation_mat_accessor = TensorAccessor(w_transmat_args, transformation_mat_addr);
        cb_reserve_back(w_transformation_mat_cb, 1);
        const uint32_t transformation_mat_wr_ptr = get_write_ptr(w_transformation_mat_cb);
        noc_async_read_tile(0, transformation_mat_accessor, transformation_mat_wr_ptr);
        noc_async_read_barrier();
        cb_push_back(w_transformation_mat_cb, 1);
    }

    // ---------- Build MUX senders + start zero-init ----------
    auto fwd_mux_conn = build_mux_sender(fwd_mux_args);
    auto bwd_mux_conn = build_mux_sender(bwd_mux_args);

    const uint32_t stats_tile_bytes = get_tile_size(stats_gathered_cb);

    // Issue the ONE-TIME L1 zero before waiting on fabric endpoints — the
    // direct stores run on this core's RISC-V independently of the fabric
    // status polling, so the ~3 µs zero overlaps the fabric handshake latency.
    cb_reserve_back(stats_gathered_cb, chunk_size_rows * ring_size);
    {
        const uint32_t zero_base = get_write_ptr(stats_gathered_cb);
        volatile tt_l1_ptr uint32_t* zero_tile_words = reinterpret_cast<volatile tt_l1_ptr uint32_t*>(zero_base);
        const uint32_t words_per_tile = stats_tile_bytes / sizeof(uint32_t);
        for (uint32_t i = 0; i < words_per_tile; i++) {
            zero_tile_words[i] = 0u;
        }
    }

    if (fwd_mux_args.connection_valid) {
        tt::tt_fabric::wait_for_fabric_endpoint_ready(
            fwd_mux_args.mux_x,
            fwd_mux_args.mux_y,
            fabric_mux_status_address,
            fwd_mux_args.local_fabric_mux_status_addr);
    }
    if (bwd_mux_args.connection_valid) {
        tt::tt_fabric::wait_for_fabric_endpoint_ready(
            bwd_mux_args.mux_x,
            bwd_mux_args.mux_y,
            fabric_mux_status_address,
            bwd_mux_args.local_fabric_mux_status_addr);
    }
    if (fwd_mux_args.connection_valid) {
        tt::tt_fabric::fabric_client_connect(fwd_mux_conn);
    }
    if (bwd_mux_args.connection_valid) {
        tt::tt_fabric::fabric_client_connect(bwd_mux_conn);
    }

    // ---------- Allocate packet headers ----------
    auto pkt_hdr_forward = PacketHeaderPool::allocate_header();
    auto pkt_hdr_backward = PacketHeaderPool::allocate_header();
    // Each packed page = window_size × 128 B of real data. Last chunk on
    // each chip may have rows_in_chunk < chunk_size_rows; we still allocate
    // and transmit a full page (padding bytes are unread garbage).
    const uint32_t page_size_bytes = chunk_size_rows * kRowBytesPerTile;

    // ---------- ONE-TIME bulk-NoC fill of stats_gathered_cb tiles ----------
    // The first tile was already zeroed above (before the fabric-endpoint
    // wait, so the zero-write overlaps the wait). Now bulk-NoC that zero
    // tile into every other slot. Local→local NoC writes run in parallel
    // with NCRISC; no serial store overhead.
    //
    // Why we zero: the scatter only touches col 0 (face_00 col 0 + face_10
    // col 0) of each tile. reduce<AVG,REDUCE_ROW> in compute reads ALL 32
    // cols per row, so cols 1..31 (and the unused faces 01/11) must be
    // zero or the mean is corrupted. One-time zero suffices because the
    // CB slots cycle and the scatter only ever touches col 0.
    const uint32_t chunk_stats_tiles_init = chunk_size_rows * ring_size;
    if (chunk_stats_tiles_init > 1) {
        const uint32_t zero_base = get_write_ptr(stats_gathered_cb);
        const uint64_t src_noc_addr = safe_get_noc_addr(my_x[0], my_y[0], zero_base, 0);
        for (uint32_t i = 1; i < chunk_stats_tiles_init; i++) {
            const uint32_t dst_addr = zero_base + i * stats_tile_bytes;
            noc_async_read(src_noc_addr, dst_addr, stats_tile_bytes);
        }
        noc_async_read_barrier();
    }
    // Don't push — the chunk loop will fill col 0 of each tile and push then.

    // GlobalSemaphore lives on THIS worker core; remote chips' matching workers
    // atomic_inc it via fabric. Compute its NoC-0 addr for the packet headers
    // and a local L1 pointer for our cumulative wait + reset.
    const uint64_t out_ready_sem_noc_addr_in_pkt = safe_get_noc_addr(my_x[0], my_y[0], out_ready_sem_bank_addr, 0);
    volatile tt_l1_ptr uint32_t* out_ready_sem_ptr =
        reinterpret_cast<volatile tt_l1_ptr uint32_t*>(out_ready_sem_bank_addr);

    // ---------- Chunked AG + output drain (packed-page) ----------
    // Each chunk produces ONE fabric packet (one packed page) instead of
    // chunk_size_rows whole-tile packets — that's the entire point of
    // Phase 9. After AG, we scatter the gathered packed pages back into
    // row 0 of stats_gathered_cb tiles for the compute kernel to transpose.
    // Sem is cumulative across chunks: each chunk contributes (ring_size-1)
    // expected incs (one from each remote device's matching worker).
    uint32_t row_processed = 0;
    uint32_t chunks_processed = 0;
    uint32_t cumulative_expected_incs = 0;
    while (row_processed < num_tile_rows) {
        const uint32_t rows_in_chunk =
            ((row_processed + chunk_size_rows) <= num_tile_rows) ? chunk_size_rows : (num_tile_rows - row_processed);
        const uint32_t chunk_stats_tiles = rows_in_chunk * ring_size;
        const uint32_t chunk_idx_on_device = worker_chunk_base + chunks_processed;

        // ---- Phase A: pack window into staging CB, fabric mcast ONE packet ----
        {
            DeviceZoneScopedN("W_AG");
            // Reserve packed_local slot and packed_gathered slot for THIS chip.
            cb_reserve_back(stats_packed_local_cb, 1);
            const uint32_t packed_local_addr = get_write_ptr(stats_packed_local_cb);

            // Reserve full ring's worth of gathered slots up front; we fill our
            // own slot in this phase and read remote slots in Phase A.5.
            cb_reserve_back(stats_packed_gathered_cb, ring_size);
            const uint32_t packed_gathered_base = get_write_ptr(stats_packed_gathered_cb);
            const uint32_t my_packed_gathered_addr = packed_gathered_base + my_device_index * page_size_bytes;

            // Cumulative wait: writer can start packing row 0 while compute is
            // still producing row 1+. Overlaps writer-pack with compute-pre for
            // chunks with >1 row.
            const uint32_t stats_local_base = get_read_ptr(stats_local_cb);
            for (uint32_t r = 0; r < rows_in_chunk; r++) {
                cb_wait_front(stats_local_cb, r + 1);
                const volatile tt_l1_ptr uint32_t* tile_src =
                    reinterpret_cast<const volatile tt_l1_ptr uint32_t*>(stats_local_base + r * stats_tile_bytes);
                uint32_t* packed_dst = reinterpret_cast<uint32_t*>(packed_local_addr + r * kRowBytesPerTile);
                // Face_00 col 0 (rows 0..15): uint32 indices 0, 16, ..., 240.
                for (uint32_t i = 0; i < 16; i++) {
                    packed_dst[i] = tile_src[i * 16];
                }
                // Face_10 col 0 (rows 16..31): face_10 starts at byte 2048
                // (uint32 idx 512), col 0 at indices 512, 528, ..., 752.
                for (uint32_t i = 0; i < 16; i++) {
                    packed_dst[16 + i] = tile_src[512 + i * 16];
                }
            }
            cb_pop_front(stats_local_cb, rows_in_chunk);

            // No L1 copy of my own page into stats_packed_gathered_cb. The
            // scatter step below reads my own slot directly from packed_local
            // (the source we just packed); other slots come from DRAM reads.
            // Saves a 128 B NoC self-write per chunk.

            // Fabric mcast: the SAME page index lands at the same DRAM address
            // on every chip; my_device_index ensures my data goes to my pages.
            const uint32_t my_dram_page_idx = my_device_index * num_chunks_per_device + chunk_idx_on_device;
            // Fabric destination address MUST be built with the fabric addrgen
            // helper, not plain get_noc_addr(): the latter encodes the DRAM bank
            // using this writer kernel's NOC, whereas the EDM fabric router
            // issues the write on its own NOC. On Wormhole the two NOCs mirror
            // coordinates AND DRAM cores have no virtual coords, so a plain
            // get_noc_addr() resolves on the remote chip to a Tensix core L1 ->
            // "NOC target address overflow" (watcher) -> device hang. The
            // addrgen helper flips DRAM coords into the canonical noc0 system
            // the fabric APIs expect (no-op on Blackhole, which is why this
            // only bit the Wormhole Galaxy). Mirrors all_gather_async's writers.
            const uint64_t dram_dest_noc_addr =
                tt::tt_fabric::linear::addrgen_detail::get_noc_address(stats_dram_accessor, my_dram_page_idx, 0);
            if constexpr (num_targets_forward > 0) {
                if (fwd_mux_args.connection_valid) {
#ifndef WAN_ABL_SKIP_FABRIC
                    fabric_multicast_noc_fused_unicast_with_atomic_inc(
                        &fwd_mux_conn,
                        pkt_hdr_forward,
                        packed_local_addr,
                        page_size_bytes,
                        tt::tt_fabric::NocUnicastAtomicIncFusedCommandHeader{
                            dram_dest_noc_addr, out_ready_sem_noc_addr_in_pkt, 1, false},
                        /*start_distance=*/1,
                        static_cast<uint8_t>(num_targets_forward));
#endif
                }
            }
            if constexpr (num_targets_backward > 0) {
                if (bwd_mux_args.connection_valid) {
#ifndef WAN_ABL_SKIP_FABRIC
                    fabric_multicast_noc_fused_unicast_with_atomic_inc(
                        &bwd_mux_conn,
                        pkt_hdr_backward,
                        packed_local_addr,
                        page_size_bytes,
                        tt::tt_fabric::NocUnicastAtomicIncFusedCommandHeader{
                            dram_dest_noc_addr, out_ready_sem_noc_addr_in_pkt, 1, false},
                        /*start_distance=*/1,
                        static_cast<uint8_t>(num_targets_backward));
#endif
                }
            }
            // packed_local_cb is double-buffered; release this slot so the next
            // chunk can use the other.
            cb_push_back(stats_packed_local_cb, 1);
            cb_pop_front(stats_packed_local_cb, 1);

            cumulative_expected_incs += (ring_size - 1);
#ifndef WAN_ABL_SKIP_FABRIC
            if (cumulative_expected_incs > 0) {
                noc_semaphore_wait_min(out_ready_sem_ptr, cumulative_expected_incs);
            }
#endif
            // Wait for all outstanding NoC operations to complete:
            //  - noc_async_write_barrier: the local L1 copy (and the fabric
            //    sender's NoC read from packed_local_addr, since send_chunk_from_address
            //    is non-blocking and the read is still in flight).
            //  - noc_async_atomic_barrier: any outstanding atomic transactions
            //    from this core's perspective (analogous to the pattern in
            //    all_gather_async/minimal_default_writer.cpp).
            // Without these, the next chunk's reuse of packed_local_cb slots can
            // race with the in-flight fabric reads, and small DRAM writes from
            // the fabric router on the receiver side may not have committed by
            // the time we issue the next-chunk reads.
            noc_async_write_barrier();
            noc_async_atomic_barrier();

            // ---- Phase A.5: Read remote-device pages (skip own — already L1-copied) ----
#ifndef WAN_ABL_SKIP_GATHER_SCATTER
            for (uint32_t d = 0; d < ring_size; d++) {
                if (d == my_device_index) {
                    continue;
                }
                const uint32_t dram_page_idx = d * num_chunks_per_device + chunk_idx_on_device;
                const uint32_t local_slot_addr = packed_gathered_base + d * page_size_bytes;
                noc_async_read_page(dram_page_idx, stats_dram_accessor, local_slot_addr);
            }
            noc_async_read_barrier();
#endif

            // Scatter packed bytes directly into COL 0 of stats_gathered_cb tiles
            // (32 strided fp32 stores per tile). The compute kernel then runs
            // reduce<AVG,REDUCE_ROW> directly without any post-transpose — col 0
            // is the natural "stat tile" position. Saves the compute post
            // transpose pass entirely.
            //
            // Col 0 in tile-storage:
            //   - Face_00 col 0 (rows 0..15): byte offsets 0, 64, 128, ..., 960
            //     = uint32_t indices 0, 16, ..., 240.
            //   - Face_10 col 0 (rows 16..31): face starts at byte 2048 (idx 512),
            //     col 0 indices 512, 528, ..., 752.
            cb_reserve_back(stats_gathered_cb, chunk_stats_tiles);
            const uint32_t stats_gathered_base = get_write_ptr(stats_gathered_cb);
#ifndef WAN_ABL_SKIP_GATHER_SCATTER
            for (uint32_t r = 0; r < rows_in_chunk; r++) {
                for (uint32_t d = 0; d < ring_size; d++) {
                    // Own slot reads from packed_local (no L1 copy needed);
                    // remote slots come from packed_gathered (DRAM-read above).
                    const uint32_t packed_src =
                        (d == my_device_index) ? (packed_local_addr + r * kRowBytesPerTile)
                                               : (packed_gathered_base + d * page_size_bytes + r * kRowBytesPerTile);
                    const uint32_t tile_dst = stats_gathered_base + (r * ring_size + d) * stats_tile_bytes;
                    volatile tt_l1_ptr uint32_t* dst = reinterpret_cast<volatile tt_l1_ptr uint32_t*>(tile_dst);
                    const uint32_t* src = reinterpret_cast<const uint32_t*>(packed_src);
                    // Face_00 col 0 (rows 0..15): 16 strided stores at uint32 stride 16.
                    for (uint32_t i = 0; i < 16; i++) {
                        dst[i * 16] = src[i];
                    }
                    // Face_10 col 0 (rows 16..31): 16 strided stores starting at idx 512.
                    for (uint32_t i = 0; i < 16; i++) {
                        dst[512 + i * 16] = src[16 + i];
                    }
                }
            }
#endif  // WAN_ABL_SKIP_GATHER_SCATTER
            cb_push_back(stats_packed_gathered_cb, ring_size);
            cb_pop_front(stats_packed_gathered_cb, ring_size);

            cb_push_back(stats_gathered_cb, chunk_stats_tiles);
        }  // W_AG

        DeviceZoneScopedN("W_DRAIN");
        // Drain this chunk's output_cb tiles to DRAM. Compute always pushes
        // `block_size` slots per col-block (LLK packer requirement) even when
        // only `tiles_in_block` (clamped) are valid; we pop the full
        // padded_row_tiles per row (incl. over-pushed garbage slots) but only
        // NoC-write the valid tiles. Without popping the padding, multi-chunk
        // overflows output_cb because over-pushed slots never drain.
        for (uint32_t r = 0; r < rows_in_chunk; r++) {
            const uint32_t tile_row = tile_row_start + row_processed + r;
            // Deep drain: issue each col-block's writes as soon as compute
            // pushes it (incremental cb_wait_front keeps within-row overlap),
            // but defer the flush + pop to the END of the row so the whole
            // row's writes pipeline at DRAM depth under ONE flush instead of
            // one per block_size=2 tiles. output_cb is 2 padded rows, so a
            // row's slots are contiguous and compute can fill row r+1 while
            // this row drains.
            uint32_t row_base_rd_ptr = 0;
            uint32_t cumulative_tiles = 0;
            for (uint32_t col_tile = 0; col_tile < num_tile_cols; col_tile += block_size) {
                const uint32_t tiles_in_block =
                    ((num_tile_cols - col_tile) >= block_size) ? block_size : (num_tile_cols - col_tile);
                // Compute pushes block_size slots per col-block regardless of
                // the clamped valid count, so wait for the padded cumulative.
                cumulative_tiles += block_size;
                cb_wait_front(output_cb, cumulative_tiles);
                if (col_tile == 0) {
                    row_base_rd_ptr = get_read_ptr(output_cb);
                }
                uint32_t output_rd_ptr = row_base_rd_ptr + col_tile * output_tile_bytes;
                for (uint32_t i = 0; i < tiles_in_block; i++) {
                    const uint32_t c = col_tile + i;
                    const uint32_t h = c / head_dim_tiles;
                    const uint32_t t_col = c - h * head_dim_tiles;
                    const uint32_t output_tile_idx =
                        h * total_num_tile_rows * head_dim_tiles + tile_row * head_dim_tiles + t_col;
#ifndef WAN_ABL_SKIP_OUTPUT_WRITE
                    noc_async_write_tile(output_tile_idx, output_accessor, output_rd_ptr);
#endif
                    output_rd_ptr += output_tile_bytes;
                }
            }
            // One flush + one pop per row. _flushed (write request committed to
            // NoC, not round-trip ACK) — the L1 source slots are safe to reuse
            // once the requests have left the core; the final _barrier at end of
            // kernel ensures all in-flight writes complete before MUX disconnect.
            noc_async_writes_flushed();
            cb_pop_front(output_cb, padded_row_tiles);
        }

        row_processed += rows_in_chunk;
        chunks_processed += 1;
    }
    // Final barrier — all in-flight output writes must complete before
    // we disconnect from the fabric MUX.
    noc_async_write_barrier();

    // Reset sem for any subsequent invocation.
    noc_semaphore_set(out_ready_sem_ptr, 0);

    // ---------- Disconnect MUX + terminate ----------
    noc_async_write_barrier();
    noc_async_atomic_barrier();

    if (fwd_mux_args.connection_valid) {
        tt::tt_fabric::fabric_client_disconnect(fwd_mux_conn);
        terminate_mux_handshake(fwd_mux_args);
    }
    if (bwd_mux_args.connection_valid) {
        tt::tt_fabric::fabric_client_disconnect(bwd_mux_conn);
        terminate_mux_handshake(bwd_mux_args);
    }
    noc_async_write_barrier();
}
