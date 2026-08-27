// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include "api/dataflow/dataflow_api.h"
#include "api/dataflow/noc.h"
#include "api/dataflow/circular_buffer.h"
#include "api/dataflow/noc_semaphore.h"
#include "ttnn/kernel/dataflow/generate_bcast_scalar.hpp"
#include "ttnn/cpp/ttnn/kernel_lib/reduce_helpers_dataflow.hpp"
#include "dataflow_common.hpp"
#include "exp_fused_op_indexer.hpp"

#ifdef USE_MUX
#include "tt_metal/fabric/hw/inc/tt_fabric_mux_interface.hpp"
#include "tt_metal/fabric/hw/inc/tt_fabric_api.h"
#include "tt_metal/fabric/hw/inc/noc_addr.h"
#include "tt_metal/fabric/hw/inc/linear/addrgen_api.h"
#include "tt_metal/fabric/hw/inc/linear/api.h"
#include "cpp/ttnn/operations/ccl/kernel_common/worker_sync_utils.hpp"
#include "cpp/ttnn/operations/ccl/ccl_host_types.hpp"
using namespace tt::tt_fabric::linear::experimental;
#endif

struct QChunkInfo {
    bool is_joint_q;
    Slice out_slice;
    uint32_t end_seq_tile;
};

// Compute output slice and stats tile range for one Q chunk.
// is_joint_q distinguishes local-sequence Q chunks from joint-context Q chunks,
// which write to different output tensors and have different causal extents.
//
// @param q_chunk             Q chunk index within [0, num_q_chunks) (local then joint)
// @param nb                  Batch index
// @param nq                  Head index
// @param ring_id             Device ID that owns the current ring iteration's KV shard
// @param num_local_q_chunks  Number of Q chunks from the local sequence (joint starts after)
// @param Sq_chunk_t          Q chunk size in tiles
// @param DHt                 Head dimension in tiles
// @param Lt                  Joint (cross-attention context) sequence length in tiles
// @param local_padded_Nt     Per-device padded local sequence length in tiles
inline QChunkInfo get_q_chunk_info(
    const uint32_t q_chunk,
    const uint32_t nb,
    const uint32_t nq,
    const uint32_t ring_id,
    const uint32_t num_local_q_chunks,
    const uint32_t Sq_chunk_t,
    const uint32_t DHt,
    const uint32_t Lt,
    const uint32_t local_padded_Nt) {
    QChunkInfo info;
    info.is_joint_q = q_chunk >= num_local_q_chunks;
    if (info.is_joint_q) {
        const uint32_t joint_out_row_start_tile = (q_chunk - num_local_q_chunks) * Sq_chunk_t;
        info.out_slice = Slice(nb, nq, joint_out_row_start_tile, joint_out_row_start_tile + Sq_chunk_t, 0, DHt);
        info.end_seq_tile = Lt;
    } else {
        const uint32_t out_row_start_tile = q_chunk * Sq_chunk_t;
        info.out_slice = Slice(nb, nq, out_row_start_tile, out_row_start_tile + Sq_chunk_t, 0, DHt);
        info.end_seq_tile = local_padded_Nt * (ring_id + 1);
    }
    return info;
}

void kernel_main() {
    Noc noc;

    constexpr uint32_t B = get_compile_time_arg_val(0);
    constexpr uint32_t NH = get_compile_time_arg_val(1);
    constexpr uint32_t DHt = get_compile_time_arg_val(2);
    constexpr uint32_t Sq_chunk_t = get_compile_time_arg_val(3);
    constexpr uint32_t Sk_chunk_t = get_compile_time_arg_val(4);
    constexpr uint32_t local_padded_N = get_compile_time_arg_val(5);
    constexpr uint32_t local_padded_Nt = get_compile_time_arg_val(6);
    constexpr uint32_t logical_n = get_compile_time_arg_val(7);
    constexpr uint32_t logical_nt = get_compile_time_arg_val(8);
    constexpr uint32_t Lt = get_compile_time_arg_val(9);
    constexpr uint32_t L = get_compile_time_arg_val(10);
    constexpr uint32_t num_local_q_chunks = get_compile_time_arg_val(11);
    constexpr uint32_t num_joint_q_chunks = get_compile_time_arg_val(12);
    constexpr uint32_t num_local_k_chunks = get_compile_time_arg_val(13);
    constexpr uint32_t num_joint_k_chunks = get_compile_time_arg_val(14);
    constexpr uint32_t num_q_chunks = get_compile_time_arg_val(15);
    constexpr uint32_t identity_scalar_packed = get_compile_time_arg_val(16);
    constexpr uint32_t scale_val = get_compile_time_arg_val(17);
    constexpr uint32_t ring_size = get_compile_time_arg_val(18);
    constexpr uint32_t global_n_partial_col = get_compile_time_arg_val(19);
    constexpr uint32_t joint_l_partial_col = get_compile_time_arg_val(20);
    constexpr uint32_t out_subblock_h = get_compile_time_arg_val(22);
    // Sparse frame-block attention: the writer drains the normalized output at each q_chunk's last work
    // iter (compute normalizes cb_out there), not the geometric last ring iter. cb_out is a 2-slot
    // row-group ping-pong, so a mismatch strands compute's row drain -> deadlock.
    constexpr bool sparse_frames_enabled = get_compile_time_arg_val(23) == 1;

    constexpr auto out_args = TensorAccessorArgs<24>();
    constexpr auto joint_out_args = TensorAccessorArgs<out_args.next_compile_time_args_offset()>();
    // stats_args follows joint_out_args but is unused by the writer (stats are only
    // needed for multi-Q accumulator save/restore which this kernel doesn't support).
    // The MUX CT args start after stats_args.
    constexpr auto stats_args_skip = TensorAccessorArgs<joint_out_args.next_compile_time_args_offset()>();

#ifdef USE_MUX
    constexpr uint32_t mux_ct_base = stats_args_skip.next_compile_time_args_offset();
    constexpr uint8_t fabric_mux_num_buffers_per_channel = get_compile_time_arg_val(mux_ct_base + 0);
    constexpr size_t fabric_mux_channel_buffer_size_bytes = get_compile_time_arg_val(mux_ct_base + 1);
    constexpr size_t fabric_mux_status_address = get_compile_time_arg_val(mux_ct_base + 2);
    constexpr size_t fabric_mux_termination_signal_address = get_compile_time_arg_val(mux_ct_base + 3);
    constexpr uint32_t num_mux_clients = get_compile_time_arg_val(mux_ct_base + 4);


    // All-gather CT args (following 5 MUX CT args)
    constexpr uint32_t ag_ct_base = mux_ct_base + 5;
    constexpr uint32_t ag_packet_size_in_pages = get_compile_time_arg_val(ag_ct_base + 0);
    constexpr uint32_t ag_page_size = get_compile_time_arg_val(ag_ct_base + 1);
    constexpr auto ag_gathered_k_args = TensorAccessorArgs<ag_ct_base + 2>();
    constexpr auto ag_gathered_v_args = TensorAccessorArgs<ag_gathered_k_args.next_compile_time_args_offset()>();
    // Local K/V input accessors — sparse fabric writer reads its own shard from local input on ring
    // iter 0 (gathered accessors above cover iter>0). Present for all fabric writers; dense ignores.
    constexpr auto local_k_input_args = TensorAccessorArgs<ag_gathered_v_args.next_compile_time_args_offset()>();
    constexpr auto local_v_input_args = TensorAccessorArgs<local_k_input_args.next_compile_time_args_offset()>();
#endif

    uint32_t argidx = 0;
    const uint32_t out_addr = get_arg_val<uint32_t>(argidx++);
    const uint32_t joint_out_addr = get_arg_val<uint32_t>(argidx++);
    argidx++;  // skip stats_addr (unused — stats only needed for multi-Q accumulator save)
    const uint32_t global_q_start = get_arg_val<uint32_t>(argidx++);
    const uint32_t global_q_end = get_arg_val<uint32_t>(argidx++);
    // Only one Q chunk per core is allowed
    ASSERT(global_q_end - global_q_start <= 1);

    RingSDPAOpIndexer fused_op_indexer = RingSDPAOpIndexer(argidx);

#ifdef USE_MUX

    // Parse fabric MUX client connection RT args (17 values).
    // mux_connection_valid, is_termination_master, mux_x, mux_y,
    // channel_base_address, connection_info_address, connection_handshake_address,
    // flow_control_address, buffer_index_address, channel_credits_stream_id,
    // 5 local semaphores, termination_master_noc_x, termination_master_noc_y
    const bool mux_connection_valid = get_arg_val<uint32_t>(argidx++) == 1;
    const bool is_termination_master = get_arg_val<uint32_t>(argidx++) == 1;
    const uint8_t fabric_mux_x = static_cast<uint8_t>(get_arg_val<uint32_t>(argidx++));
    const uint8_t fabric_mux_y = static_cast<uint8_t>(get_arg_val<uint32_t>(argidx++));
    const uint32_t fabric_mux_channel_base_address = get_arg_val<uint32_t>(argidx++);
    const uint32_t fabric_mux_connection_info_address = get_arg_val<uint32_t>(argidx++);
    const uint32_t fabric_mux_connection_handshake_address = get_arg_val<uint32_t>(argidx++);
    const uint32_t fabric_mux_flow_control_address = get_arg_val<uint32_t>(argidx++);
    const uint32_t fabric_mux_buffer_index_address = get_arg_val<uint32_t>(argidx++);
    const uint8_t fabric_mux_channel_id = static_cast<uint8_t>(get_arg_val<uint32_t>(argidx++));
    const uint32_t termination_sync_sem_id = get_arg_val<uint32_t>(argidx++);
    const uint32_t local_fabric_mux_status_addr = get_semaphore(get_arg_val<uint32_t>(argidx++));
    const uint32_t local_flow_control_addr = get_semaphore(get_arg_val<uint32_t>(argidx++));
    const uint32_t local_teardown_addr = get_semaphore(get_arg_val<uint32_t>(argidx++));
    const uint32_t local_buffer_index_addr = get_semaphore(get_arg_val<uint32_t>(argidx++));
    const uint8_t termination_master_noc_x = static_cast<uint8_t>(get_arg_val<uint32_t>(argidx++));
    const uint8_t termination_master_noc_y = static_cast<uint8_t>(get_arg_val<uint32_t>(argidx++));


    // Build connection object at outer scope (lifetime spans the ring loop and teardown).
    auto mux_conn = tt::tt_fabric::build_connection_to_fabric_endpoint<fabric_mux_num_buffers_per_channel>(
        fabric_mux_x,
        fabric_mux_y,
        fabric_mux_channel_id,
        fabric_mux_num_buffers_per_channel,
        fabric_mux_channel_buffer_size_bytes,
        fabric_mux_channel_base_address,
        fabric_mux_connection_info_address,
        fabric_mux_connection_handshake_address,
        fabric_mux_flow_control_address,
        fabric_mux_buffer_index_address,
        local_flow_control_addr,
        local_teardown_addr,
        local_buffer_index_addr);

    if (mux_connection_valid) {
        tt::tt_fabric::wait_for_fabric_endpoint_ready(
            fabric_mux_x, fabric_mux_y, fabric_mux_status_address, local_fabric_mux_status_addr);
        tt::tt_fabric::fabric_client_connect(mux_conn);
    }

    // ---- MUX writer setup: parsed by all MUX writers with valid connections ----
    volatile tt_l1_ptr PACKET_HEADER_TYPE* pkt_scatter_hdr = nullptr;
    volatile tt_l1_ptr PACKET_HEADER_TYPE* pkt_unicast_hdr = nullptr;
    volatile tt_l1_ptr PACKET_HEADER_TYPE* pkt_hdr_sem_inc = nullptr;
    uint32_t ag_output_Wt = 0, ag_output_Ht = 0;
    uint32_t gathered_k_addr_ag_rt = 0, gathered_v_addr_ag_rt = 0;
    // Local K/V input addresses (sparse fabric writer sources its own shard from dram; unused in dense).
    uint32_t local_k_addr_rt = 0, local_v_addr_rt = 0;
    // Private L1 scratch for the gather-ready poll's noc read destination (never per_link_sem).
    uint32_t poll_scratch_addr = 0;
    // Hoisted so the sparse fabric loop below can wait on this device's per-link gather-ready semaphore.
    uint32_t out_ready_sem_addr = 0;
    uint32_t injector_noc_x = 0, injector_noc_y = 0;
    uint32_t num_muxes_in_direction = 1, my_mux_index = 0;

    if (mux_connection_valid) {
        out_ready_sem_addr = get_arg_val<uint32_t>(argidx++);
        injector_noc_x = get_arg_val<uint32_t>(argidx++);
        injector_noc_y = get_arg_val<uint32_t>(argidx++);
        num_muxes_in_direction = get_arg_val<uint32_t>(argidx++);
        my_mux_index = get_arg_val<uint32_t>(argidx++);
        ag_output_Wt = get_arg_val<uint32_t>(argidx++);
        ag_output_Ht = get_arg_val<uint32_t>(argidx++);
        gathered_k_addr_ag_rt = get_arg_val<uint32_t>(argidx++);
        gathered_v_addr_ag_rt = get_arg_val<uint32_t>(argidx++);
        local_k_addr_rt = get_arg_val<uint32_t>(argidx++);
        local_v_addr_rt = get_arg_val<uint32_t>(argidx++);
        poll_scratch_addr = get_semaphore(get_arg_val<uint32_t>(argidx++));
        // (Removed a vestigial OpSignaler construction here: the program factory pushes no signaler RT
        // args — kWriterFabricArgCount ends at the local K/V addresses — so it read a data-dependent
        // number of unrelated words and, under sparse, misaligned the per-q_chunk work bitmap parsed
        // below, corrupting write_output_now. It was never used to signal anything.)

        pkt_scatter_hdr = PacketHeaderPool::allocate_header();
        pkt_unicast_hdr = PacketHeaderPool::allocate_header();
        pkt_hdr_sem_inc = PacketHeaderPool::allocate_header();

        // Pre-configure scatter write header with max packet size; per-call with_state
        // overrides DstAddrs, ChunkSizes, and PayloadSize for the actual tile count.
        constexpr uint32_t scatter_init_count = ag_packet_size_in_pages >= 2 ? ag_packet_size_in_pages : 2;
        uint64_t dummy_addrs[4] = {0, 0, 0, 0};
        uint16_t scatter_chunk_sizes[3] = {
            static_cast<uint16_t>(ag_page_size),
            static_cast<uint16_t>(ag_page_size),
            static_cast<uint16_t>(ag_page_size)};
        fabric_unicast_noc_scatter_write_set_state<
            UnicastScatterWriteUpdateMask::ChunkSizes | UnicastScatterWriteUpdateMask::PayloadSize>(
            pkt_scatter_hdr, 1,
            NocUnicastScatterCommandHeader(dummy_addrs, scatter_chunk_sizes, scatter_init_count),
            ag_page_size * ag_packet_size_in_pages);

        // Pre-configure unicast write header: payload size is constant
        fabric_unicast_noc_unicast_write_set_state<UnicastWriteUpdateMask::PayloadSize>(
            pkt_unicast_hdr, 1, nullptr, ag_page_size);

        // Pre-configure atomic inc header for signaling the injector core on the next device
        // safe_get_noc_addr() cannot be migrated to Device 2.0 because it creates a cross-chip semaphore address,
        // while the current Device 2.0 Semaphore class only supports same-chip operations.
        const uint64_t injector_out_ready_sem_noc_addr =
            safe_get_noc_addr(injector_noc_x, injector_noc_y, out_ready_sem_addr, 0);
        fabric_unicast_noc_unicast_atomic_inc_set_state<
            UnicastAtomicIncUpdateMask::DstAddr | UnicastAtomicIncUpdateMask::Val |
            UnicastAtomicIncUpdateMask::Flush>(
            pkt_hdr_sem_inc, 1,
            tt::tt_fabric::NocUnicastAtomicIncCommandHeader{injector_out_ready_sem_noc_addr, 1});

        fabric_set_unicast_route<false>(pkt_scatter_hdr, 1);
        fabric_set_unicast_route<false>(pkt_unicast_hdr, 1);
        fabric_set_unicast_route<false>(pkt_hdr_sem_inc, 1);
    }

    const auto gathered_k_writer = TensorAccessor(ag_gathered_k_args, gathered_k_addr_ag_rt);
    const auto gathered_v_writer = TensorAccessor(ag_gathered_v_args, gathered_v_addr_ag_rt);

    // Read generators for the sparse fabric writer's own-fetch: local input shard on ring iter 0, this
    // device's gathered buffer (ring_size shards of local_padded_Nt) on iter>0. Constructed for all
    // fabric writers; only the sparse path issues reads through them.
    const uint32_t writer_padded_Nt = local_padded_Nt * ring_size;
    const auto writer_input_tile_logical = TensorTileShape(B, NH, local_padded_Nt, DHt);
    const auto writer_gathered_tile_logical = TensorTileShape(B, NH, writer_padded_Nt, DHt);
    const auto local_k_fetch_gen =
        PaddedAddrGenerator(TensorAccessor(local_k_input_args, local_k_addr_rt), writer_input_tile_logical);
    const auto local_v_fetch_gen =
        PaddedAddrGenerator(TensorAccessor(local_v_input_args, local_v_addr_rt), writer_input_tile_logical);
    const auto gathered_k_fetch_gen =
        PaddedAddrGenerator(TensorAccessor(ag_gathered_k_args, gathered_k_addr_ag_rt), writer_gathered_tile_logical);
    const auto gathered_v_fetch_gen =
        PaddedAddrGenerator(TensorAccessor(ag_gathered_v_args, gathered_v_addr_ag_rt), writer_gathered_tile_logical);
#endif

    // Per-q_chunk work bitmap (one uint32 per q_chunk; bit `iter` set iff that q_chunk processes any
    // attended K in ring iter `iter`). Present in the RT stream only when sparse is enabled (the host
    // gates it identically). Read LAST — after all fixed args (incl. the fabric MUX/AG block) — so the
    // hash-excluded dynamic semaphore slots stay at their fixed indices; only the total arg count grows.
    uint32_t writer_q_work_bitmap[num_q_chunks] = {};
    if constexpr (sparse_frames_enabled) {
        for (uint32_t q = 0; q < num_q_chunks; ++q) {
            writer_q_work_bitmap[q] = get_arg_val<uint32_t>(argidx++);
        }
    }

    constexpr uint32_t cb_out = tt::CBIndex::c_16;
    constexpr uint32_t cb_mask_in = tt::CBIndex::c_3;
    constexpr uint32_t cb_k_writer_in = tt::CBIndex::c_14;
    constexpr uint32_t cb_v_writer_in = tt::CBIndex::c_15;
    constexpr uint32_t tile_bytes = get_tile_size(cb_out);
    // Per-tile byte sizes for the sparse fabric writer's own-fetch into c_14/c_15.
    constexpr uint32_t k_tile_bytes = get_tile_size(cb_k_writer_in);
    constexpr uint32_t v_tile_bytes = get_tile_size(cb_v_writer_in);

    CircularBuffer cb_k_w(cb_k_writer_in);
    CircularBuffer cb_v_w(cb_v_writer_in);
    const auto out_writer = TensorAccessor(out_args, out_addr);
    const auto joint_out_writer = TensorAccessor(joint_out_args, joint_out_addr);

    const auto output_tile_logical = TensorTileShape(B, NH, local_padded_Nt, DHt);
    const auto joint_tile_logical = TensorTileShape(B, NH, Lt, DHt);

    const auto out_generator = PaddedAddrGenerator(out_writer, output_tile_logical);
    const auto joint_out_generator = PaddedAddrGenerator(joint_out_writer, joint_tile_logical);

    constexpr uint32_t cb_scale_in = tt::CBIndex::c_4;
    constexpr uint32_t cb_col_identity = tt::CBIndex::c_8;
    constexpr uint32_t cb_identity_scale_in = tt::CBIndex::c_5;

    generate_bcast_unary_scalar(CircularBuffer(cb_scale_in), scale_val);
    generate_bcast_col_scalar(CircularBuffer(cb_col_identity), identity_scalar_packed);
    dataflow_kernel_lib::calculate_and_prepare_reduce_scaler<
        cb_identity_scale_in,
        ckernel::PoolType::MAX,
        ckernel::ReduceDim::REDUCE_ROW,
        dataflow_kernel_lib::SUM_AND_MAX_REDUCE_FACTOR>();

    // Lightweight mask: generate all mask tiles once into single CB before the ring loop.
    // Only needed when any K/joint dimension has padding that doesn't fill a chunk.
    constexpr bool local_n_has_padding = local_padded_Nt % Sk_chunk_t != 0;
    constexpr bool global_n_has_padding = logical_n % (Sk_chunk_t * tt::constants::TILE_HEIGHT) != 0;
    constexpr bool joint_has_padding = L > 0 && L % (Sk_chunk_t * tt::constants::TILE_HEIGHT) != 0;
    constexpr bool needs_lightweight_mask = local_n_has_padding || global_n_has_padding || joint_has_padding;
    if constexpr (needs_lightweight_mask) {
        generate_lightweight_mask_tiles<global_n_partial_col, joint_l_partial_col, cb_mask_in>(noc);
    }

    const uint32_t last_active_ring_iter =
        find_last_active_ring_iter(fused_op_indexer.seq, local_padded_Nt, logical_n / tt::constants::TILE_HEIGHT, L);

#ifdef USE_MUX
    // Sparse own-fetch gather-ready. The previous device's writer increments the per-link out-ready
    // semaphore onto this device's injector core (injector_noc_x/y), not this writer, so poll the
    // injector's copy over the local noc. Read into a private scratch, never per_link_sem: if this
    // writer is also the head's reselected injector, reading into per_link_sem races the remote +1.
    const uint64_t injector_gather_sem_noc_addr = get_noc_addr(injector_noc_x, injector_noc_y, out_ready_sem_addr);
    volatile tt_l1_ptr uint32_t* writer_gather_scratch =
        reinterpret_cast<volatile tt_l1_ptr uint32_t*>(poll_scratch_addr);
    uint32_t writer_gather_ready_count = 0;
#endif

    // Output drained after the ring loop, not at the producing iter: under sparse the producing iter
    // differs per core, so an in-loop drain would stall this core's later fabric forwards. Dense
    // produces on the last iter, so its timing is unchanged.
    QChunkInfo pending_qi{};
    bool output_pending = false;

    for (uint32_t ring_iter = 0; ring_iter < ring_size; ++ring_iter) {
        uint32_t ring_id = fused_op_indexer.get_next_ring_id_and_sync();

        const bool do_joint_kv = ring_id == ring_size - 1;
        const uint32_t num_kv_chunks = do_joint_kv ? num_local_k_chunks + num_joint_k_chunks : num_local_k_chunks;

        const uint32_t ring_iter_kv_start_tile = ring_id * local_padded_Nt;
        const uint32_t global_n_tile_id = logical_n / tt::constants::TILE_HEIGHT;
        const bool ring_iter_processes_KV_chunks = ring_iter_kv_start_tile <= global_n_tile_id;
        const bool ring_iter_does_work = ring_iter_processes_KV_chunks || (do_joint_kv && L != 0);
        if (!ring_iter_does_work) {
            continue;
        }

        {
            // Accumulators persist in L1 (single Q-chunk per core).
            // Write final normalized output on last ring iteration.
            const bool is_last_ring_iter = (ring_iter == last_active_ring_iter);

            for (uint32_t global_q_chunk = global_q_start; global_q_chunk < global_q_end; ++global_q_chunk) {
                const uint32_t nb = global_q_chunk / (NH * num_q_chunks);
                const uint32_t nq = (global_q_chunk % (NH * num_q_chunks)) / num_q_chunks;
                const uint32_t q_chunk = global_q_chunk % num_q_chunks;

                const auto qi = get_q_chunk_info(
                    q_chunk, nb, nq, ring_id, num_local_q_chunks, Sq_chunk_t, DHt, Lt, local_padded_Nt);

                // When is the normalized output ready to drain? Compute pushes cb_out (row-by-row)
                // when it finishes this q_chunk's last work iter. Dense: the geometric last active iter.
                // Sparse: the highest set bit of the q_chunk's work bitmap, which can be earlier (edge
                // frames whose window doesn't reach the last-processed frame). The writer must read
                // cb_out on exactly that iter or the 2-slot ping-pong deadlocks. Gather forwarding below
                // still keys off is_last_ring_iter (geometric).
                bool write_output_now = is_last_ring_iter;
                if constexpr (sparse_frames_enabled) {
                    const uint32_t qb = writer_q_work_bitmap[q_chunk];
                    // qb != 0: compute normalizes on this q_chunk's last work iter (highest set bit).
                    // qb == 0: the q_chunk attends no K on any iter (a padded q frame — a real frame
                    // always attends its own), so compute takes the zero-work drain path every iter and
                    // never pushes cb_out. Don't drain here, or the deferred output write blocks forever.
                    write_output_now = (qb != 0u) && (ring_iter == (31u - __builtin_clz(qb)));
                }

#ifdef USE_MUX
                uint32_t KV_chunks_processed_in_iter = 0;
                constexpr uint32_t k_chunk_tiles = Sk_chunk_t * DHt;
                constexpr uint32_t v_chunk_tiles = Sk_chunk_t * DHt;
                if (mux_connection_valid) {
                    const uint32_t rows_per_mux = (Sk_chunk_t + num_muxes_in_direction - 1) / num_muxes_in_direction;
                    const uint32_t my_row_start = my_mux_index * rows_per_mux;
                    const uint32_t my_row_end = std::min(my_row_start + rows_per_mux, (uint32_t)Sk_chunk_t);

                    for (uint32_t k_chunk = 0; k_chunk < num_kv_chunks; ++k_chunk) {
                        const bool kv_chunk_is_joint = k_chunk >= num_local_k_chunks;
                        const uint32_t kv_global_start_tile = local_padded_Nt * ring_id + k_chunk * Sk_chunk_t;
                        const bool kv_chunk_is_beyond_logical_n =
                            !kv_chunk_is_joint && (kv_global_start_tile >= logical_nt);

                        if (kv_chunk_is_beyond_logical_n) {
                            continue;
                        }
                        KV_chunks_processed_in_iter++;

                        const uint32_t gathered_kv_start_tile =
                            ring_iter_kv_start_tile + k_chunk * Sk_chunk_t;
                        const Slice kv_slice(
                            nb, nq, gathered_kv_start_tile, gathered_kv_start_tile + Sk_chunk_t, 0, DHt);
                        const uint32_t end_seq_tile = std::min(logical_nt, local_padded_Nt * (ring_id + 1));

                        const uint32_t bh_offset = (nb * NH + nq) * ag_output_Wt * ag_output_Ht;

                        // Source this chunk's K into c_14. Sparse: own-fetch from dram (local on ring
                        // iter 0, this device's gathered buffer on iter>0, gated on gather-ready) so the
                        // fabric feed never waits on the local compute. Dense: take the reader-filled CB.
                        uint32_t base_k_read_ptr = 0;
                        if constexpr (sparse_frames_enabled) {
                            if (!is_last_ring_iter && !kv_chunk_is_joint) {
                                if (ring_iter > 0) {
                                    writer_gather_ready_count++;
                                    // Poll the injector's semaphore copy (cross-core, local noc) until this
                                    // chunk's row slice has arrived; wait_min can't reach the non-local copy.
                                    while (true) {
                                        noc_async_read(
                                            injector_gather_sem_noc_addr, poll_scratch_addr, sizeof(uint32_t));
                                        noc_async_read_barrier();
                                        if (*writer_gather_scratch >= writer_gather_ready_count) {
                                            break;
                                        }
                                    }
                                }
                                cb_k_w.reserve_back(k_chunk_tiles);
                                base_k_read_ptr = cb_k_w.get_write_ptr();
                                fetch_block(
                                    ring_iter == 0 ? local_k_fetch_gen : gathered_k_fetch_gen,
                                    ring_iter == 0
                                        ? Slice(nb, nq, k_chunk * Sk_chunk_t, k_chunk * Sk_chunk_t + Sk_chunk_t, 0, DHt)
                                        : kv_slice,
                                    ring_iter == 0 ? std::min(logical_nt, local_padded_Nt) : end_seq_tile,
                                    cb_k_writer_in,
                                    base_k_read_ptr,
                                    k_tile_bytes,
                                    true /*transpose*/);
                            }
                        } else {
                            cb_k_w.wait_front(k_chunk_tiles);
                            base_k_read_ptr = cb_k_w.get_read_ptr();
                        }
                        if (!is_last_ring_iter) {
                            if (!kv_chunk_is_joint) {
                                for (uint32_t col = 0; col < DHt; ++col) {
                                    for (uint32_t row = my_row_start; row < my_row_end;
                                         row += ag_packet_size_in_pages) {
                                        uint32_t tiles_in_batch = 0;
                                        uint64_t k_noc_addrs[4] = {0, 0, 0, 0};
                                        for (uint32_t i = 0; i < ag_packet_size_in_pages && row + i < my_row_end; i++) {
                                            if (kv_slice.d2_start + row + i >= end_seq_tile) {
                                                break;
                                            }
                                            const uint32_t dest_id =
                                                bh_offset + (kv_global_start_tile + row + i) * ag_output_Wt + col;
                                            k_noc_addrs[tiles_in_batch] =
                                                tt::tt_fabric::linear::addrgen_detail::get_noc_address(
                                                    gathered_k_writer, dest_id, 0);
                                            tiles_in_batch++;
                                        }
                                        if (tiles_in_batch == 0) {
                                            break;
                                        }
                                        const uint32_t src_l1_addr =
                                            base_k_read_ptr + (row + col * Sk_chunk_t) * ag_page_size;
                                        if (tiles_in_batch == ag_packet_size_in_pages) {
                                            uint16_t k_cs[3] = {
                                                static_cast<uint16_t>(ag_page_size),
                                                static_cast<uint16_t>(ag_page_size),
                                                static_cast<uint16_t>(ag_page_size)};
                                            fabric_unicast_noc_scatter_write_with_state<
                                                UnicastScatterWriteUpdateMask::DstAddrs |
                                                UnicastScatterWriteUpdateMask::ChunkSizes |
                                                UnicastScatterWriteUpdateMask::PayloadSize>(
                                                &mux_conn,
                                                pkt_scatter_hdr,
                                                src_l1_addr,
                                                NocUnicastScatterCommandHeader(k_noc_addrs, k_cs, tiles_in_batch),
                                                ag_page_size * tiles_in_batch);
                                            noc.async_writes_flushed();
                                        } else {
                                            // Partial batch: fall back to per-tile unicast writes to avoid
                                            // variable chunk_count scatter writes which cause non-determinism.
                                            // noc.async_writes_flushed() after each send ensures the previous
                                            // header NOC write completes before pkt_unicast_hdr is modified
                                            // for the next tile (they share the same L1 header).
                                            for (uint32_t i = 0; i < tiles_in_batch; i++) {
                                                fabric_unicast_noc_unicast_write_with_state<
                                                    UnicastWriteUpdateMask::DstAddr>(
                                                    &mux_conn,
                                                    pkt_unicast_hdr,
                                                    src_l1_addr + i * ag_page_size,
                                                    NocUnicastCommandHeader{k_noc_addrs[i]});
                                                noc.async_writes_flushed();
                                            }
                                        }
                                    }
                                }
                            }
                        }
                        if constexpr (!sparse_frames_enabled) {
                            cb_k_w.pop_front(k_chunk_tiles);
                        }

                        // Source this chunk's V into c_15. Gather-ready was already awaited for K above
                        // (one per-chunk signal covers both), so sparse just fetches; dense waits on the CB.
                        uint32_t base_v_read_ptr = 0;
                        if constexpr (sparse_frames_enabled) {
                            if (!is_last_ring_iter && !kv_chunk_is_joint) {
                                cb_v_w.reserve_back(v_chunk_tiles);
                                base_v_read_ptr = cb_v_w.get_write_ptr();
                                fetch_block(
                                    ring_iter == 0 ? local_v_fetch_gen : gathered_v_fetch_gen,
                                    ring_iter == 0
                                        ? Slice(nb, nq, k_chunk * Sk_chunk_t, k_chunk * Sk_chunk_t + Sk_chunk_t, 0, DHt)
                                        : kv_slice,
                                    ring_iter == 0 ? std::min(logical_nt, local_padded_Nt) : end_seq_tile,
                                    cb_v_writer_in,
                                    base_v_read_ptr,
                                    v_tile_bytes,
                                    false /*transpose*/);
                            }
                        } else {
                            cb_v_w.wait_front(v_chunk_tiles);
                            base_v_read_ptr = cb_v_w.get_read_ptr();
                        }
                        if (!is_last_ring_iter) {
                            if (!kv_chunk_is_joint) {
                                for (uint32_t row = my_row_start; row < my_row_end; ++row) {
                                    if (kv_slice.d2_start + row >= end_seq_tile) {
                                        break;
                                    }
                                    for (uint32_t col = 0; col < DHt; col += ag_packet_size_in_pages) {
                                        uint32_t tiles_in_batch = 0;
                                        uint64_t v_noc_addrs[4] = {0, 0, 0, 0};
                                        for (uint32_t i = 0; i < ag_packet_size_in_pages && col + i < DHt; i++) {
                                            const uint32_t dest_id =
                                                bh_offset + (kv_global_start_tile + row) * ag_output_Wt + col + i;
                                            v_noc_addrs[tiles_in_batch] =
                                                tt::tt_fabric::linear::addrgen_detail::get_noc_address(
                                                    gathered_v_writer, dest_id, 0);
                                            tiles_in_batch++;
                                        }
                                        if (tiles_in_batch == 0) {
                                            break;
                                        }
                                        const uint32_t src_l1_addr = base_v_read_ptr + (row * DHt + col) * ag_page_size;
                                        if (tiles_in_batch == ag_packet_size_in_pages) {
                                            uint16_t v_cs[3] = {
                                                static_cast<uint16_t>(ag_page_size),
                                                static_cast<uint16_t>(ag_page_size),
                                                static_cast<uint16_t>(ag_page_size)};
                                            fabric_unicast_noc_scatter_write_with_state<
                                                UnicastScatterWriteUpdateMask::DstAddrs |
                                                UnicastScatterWriteUpdateMask::ChunkSizes |
                                                UnicastScatterWriteUpdateMask::PayloadSize>(
                                                &mux_conn,
                                                pkt_scatter_hdr,
                                                src_l1_addr,
                                                NocUnicastScatterCommandHeader(v_noc_addrs, v_cs, tiles_in_batch),
                                                ag_page_size * tiles_in_batch);
                                            noc.async_writes_flushed();
                                        } else {
                                            // Partial batch: fall back to per-tile unicast writes to avoid
                                            // variable chunk_count scatter writes which cause non-determinism.
                                            // noc.async_writes_flushed() after each send ensures the previous
                                            // header NOC write completes before pkt_unicast_hdr is modified
                                            // for the next tile (they share the same L1 header).
                                            for (uint32_t i = 0; i < tiles_in_batch; i++) {
                                                fabric_unicast_noc_unicast_write_with_state<
                                                    UnicastWriteUpdateMask::DstAddr>(
                                                    &mux_conn,
                                                    pkt_unicast_hdr,
                                                    src_l1_addr + i * ag_page_size,
                                                    NocUnicastCommandHeader{v_noc_addrs[i]});
                                                noc.async_writes_flushed();
                                            }
                                        }
                                    }
                                }
                            }
                        }
                        if constexpr (!sparse_frames_enabled) {
                            cb_v_w.pop_front(v_chunk_tiles);
                        }

                        if (!is_last_ring_iter) {
                            fabric_unicast_noc_unicast_atomic_inc_with_state(&mux_conn, pkt_hdr_sem_inc);
                            noc.async_writes_flushed();
                        }
                    }

                    // Phase-alignment dummy: dense keeps the reader-fed writer CB in lock-step with the
                    // compute CB. Sparse own-fetches c_14/c_15 (no reader feed), so there is nothing to drain.
                    if constexpr (!sparse_frames_enabled) {
                        if (KV_chunks_processed_in_iter % 2 == 0) {
                            cb_k_w.wait_front(k_chunk_tiles);
                            cb_v_w.wait_front(v_chunk_tiles);
                            cb_k_w.pop_front(k_chunk_tiles);
                            cb_v_w.pop_front(v_chunk_tiles);
                        }
                    }
                }
#endif

                // Capture the drain descriptor; deferred to after the ring loop (see pending_qi).
                if (write_output_now) {
                    pending_qi = qi;
                    output_pending = true;
                }
            }
        }
    }

    // Deferred output drain, before the fabric teardown so each writer releases its compute (via
    // cb_out) independently and cannot be trapped at the teardown barrier waiting on this drain.
    if (output_pending) {
        write_block_row_grouped_trid(
            noc,
            pending_qi.is_joint_q ? joint_out_generator : out_generator,
            pending_qi.out_slice,
            pending_qi.end_seq_tile,
            cb_out,
            tile_bytes,
            out_subblock_h,
            /*flush_trid=*/0);
        noc.async_write_barrier();
    }

#ifdef USE_MUX
    if (mux_connection_valid) {
        noc.async_atomic_barrier();
        tt::tt_fabric::fabric_client_disconnect(mux_conn);
        Semaphore<> termination_sync_sem(termination_sync_sem_id);
        if (is_termination_master) {
            termination_sync_sem.wait(num_mux_clients - 1);
            tt::tt_fabric::fabric_endpoint_terminate(fabric_mux_x, fabric_mux_y, fabric_mux_termination_signal_address);
        } else {
            termination_sync_sem.up(noc, termination_master_noc_x, termination_master_noc_y, 1);
            noc.async_atomic_barrier();
        }
    }
#endif
    noc.async_write_barrier();
}
