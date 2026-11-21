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
#include "metadata_scalar_read.hpp"
#include "ttnn/operations/transformer/sdpa/device/kernels/ring_joint_derived_slots.hpp"
#include "ttnn/operations/transformer/sdpa/device/kernels/recipe_ring_cbs.hpp"
#include "exp_fused_op_indexer.hpp"
#include "tt_metal/fabric/hw/inc/tt_fabric_mux_interface.hpp"
#include "tt_metal/fabric/hw/inc/tt_fabric_api.h"
#include "tt_metal/fabric/hw/inc/noc_addr.h"
#include "tt_metal/fabric/hw/inc/linear/addrgen_api.h"
#include "tt_metal/fabric/hw/inc/linear/api.h"

// Exp ring joint SDPA writer (compute: exp_ring_joint_sdpa.cpp, the precision recipes). It generates the recipe's
// reduce scaler and column identity, drains each pass's normalized output, and on the two MUX client columns
// (named arg use_mux) forwards the K/V chunks the reader fetched to the next device over the fabric MUX (the fused
// all-gather). Both writer kernels take the same compile-time args.

namespace ring_joint = ttnn::operations::transformer::sdpa::ring_joint;
using namespace tt::tt_fabric::linear::experimental;

struct QChunkInfo {
    bool is_joint_q;
    Slice out_slice;
    uint32_t end_seq_tile;
};

// Compute the output slice and sequence tile limit for one Q chunk.
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
    constexpr uint32_t local_padded_Nt = get_compile_time_arg_val(5);
    constexpr uint32_t logical_n_ct = get_compile_time_arg_val(6);
    constexpr uint32_t logical_nt_ct = get_compile_time_arg_val(7);
    constexpr uint32_t Lt = get_compile_time_arg_val(8);
    constexpr uint32_t L = get_compile_time_arg_val(9);
    constexpr uint32_t num_local_q_chunks = get_compile_time_arg_val(10);
    constexpr uint32_t num_local_k_chunks = get_compile_time_arg_val(11);
    constexpr uint32_t num_joint_k_chunks = get_compile_time_arg_val(12);
    constexpr uint32_t num_q_chunks = get_compile_time_arg_val(13);
    constexpr uint32_t ring_size = get_compile_time_arg_val(14);
    // Output Q tile rows per drain: the row groups compute normalizes (two rows when the Q chunk is a whole number of
    // pairs, else one).
    constexpr uint32_t out_subblock_h = get_compile_time_arg_val(15);
    // Trace-safe logical_n: when set, logical_n_ct above is only the worst-case placeholder.
    constexpr bool has_logical_n_tensor = get_compile_time_arg_val(16) == 1;
    // The MUX client columns forward K/V over the fabric (the fused all-gather).
    constexpr bool use_mux = get_named_compile_time_arg_val("use_mux") == 1;

    // Sits ahead of the output accessors so the offsets below (and the MUX/AG block chaining off
    // joint_out_args) keep deriving normally in both modes.
    constexpr auto logical_n_args = TensorAccessorArgs<17>();
    constexpr auto out_args = TensorAccessorArgs<logical_n_args.next_compile_time_args_offset()>();
    constexpr auto joint_out_args = TensorAccessorArgs<out_args.next_compile_time_args_offset()>();

    constexpr uint32_t mux_ct_base = joint_out_args.next_compile_time_args_offset();
    constexpr uint8_t fabric_mux_num_buffers_per_channel = get_compile_time_arg_val(mux_ct_base + 0);
    constexpr size_t fabric_mux_channel_buffer_size_bytes = get_compile_time_arg_val(mux_ct_base + 1);
    // The mux sits in program-local L1: its addresses here and in the runtime args are offsets from the map address
    // it publishes in semaphore fabric_mux_l1_base_semaphore_id.
    constexpr uint32_t fabric_mux_l1_base_semaphore_id = get_compile_time_arg_val(mux_ct_base + 2);
    constexpr size_t fabric_mux_termination_signal_offset = get_compile_time_arg_val(mux_ct_base + 3);
    constexpr uint32_t num_mux_clients = get_compile_time_arg_val(mux_ct_base + 4);

    // All-gather CT args (following 5 MUX CT args)
    constexpr uint32_t ag_ct_base = mux_ct_base + 5;
    constexpr uint32_t ag_packet_size_in_pages = get_compile_time_arg_val(ag_ct_base + 0);
    constexpr uint32_t ag_page_size = get_compile_time_arg_val(ag_ct_base + 1);
    constexpr auto ag_gathered_k_args = TensorAccessorArgs<ag_ct_base + 2>();
    constexpr auto ag_gathered_v_args = TensorAccessorArgs<ag_gathered_k_args.next_compile_time_args_offset()>();

    uint32_t argidx = 0;
    const uint32_t out_addr = get_arg_val<uint32_t>(argidx++);
    const uint32_t joint_out_addr = get_arg_val<uint32_t>(argidx++);
    // Head-serial passes: this core owns flat Q chunks q_base + p * q_stride for p in [0, q_count).
    const uint32_t q_base = get_arg_val<uint32_t>(argidx++);
    const uint32_t q_stride = get_arg_val<uint32_t>(argidx++);
    const uint32_t q_count = get_arg_val<uint32_t>(argidx++);
    RingSDPAOpIndexer fused_op_indexer = RingSDPAOpIndexer(argidx);

    // Fabric MUX client connection RT args (17 values, MUX writers only):
    // mux_connection_valid, is_termination_master, mux_x, mux_y,
    // channel_base_address, connection_info_address, connection_handshake_address,
    // flow_control_address, buffer_index_address, channel_credits_stream_id,
    // 5 local semaphores, termination_master_noc_x, termination_master_noc_y
    bool mux_connection_valid = false;
    bool is_termination_master = false;
    uint8_t fabric_mux_x = 0, fabric_mux_y = 0, fabric_mux_channel_id = 0;
    uint8_t termination_master_noc_x = 0, termination_master_noc_y = 0;
    uint32_t termination_sync_sem_id = 0;
    // Connection object at outer scope (lifetime spans the ring loop and teardown).
    tt::tt_fabric::WorkerToFabricMuxSender<fabric_mux_num_buffers_per_channel> mux_conn;
    uint32_t fabric_mux_l1_base = 0;
    if constexpr (use_mux) {
        mux_connection_valid = get_arg_val<uint32_t>(argidx++) == 1;
        is_termination_master = get_arg_val<uint32_t>(argidx++) == 1;
        fabric_mux_x = static_cast<uint8_t>(get_arg_val<uint32_t>(argidx++));
        fabric_mux_y = static_cast<uint8_t>(get_arg_val<uint32_t>(argidx++));
        const uint32_t fabric_mux_channel_base_offset = get_arg_val<uint32_t>(argidx++);
        const uint32_t fabric_mux_connection_info_offset = get_arg_val<uint32_t>(argidx++);
        const uint32_t fabric_mux_connection_handshake_offset = get_arg_val<uint32_t>(argidx++);
        const uint32_t fabric_mux_flow_control_offset = get_arg_val<uint32_t>(argidx++);
        const uint32_t fabric_mux_buffer_index_offset = get_arg_val<uint32_t>(argidx++);
        fabric_mux_channel_id = static_cast<uint8_t>(get_arg_val<uint32_t>(argidx++));
        termination_sync_sem_id = get_arg_val<uint32_t>(argidx++);
        const uint32_t local_fabric_mux_status_addr = get_semaphore(get_arg_val<uint32_t>(argidx++));
        const uint32_t local_flow_control_addr = get_semaphore(get_arg_val<uint32_t>(argidx++));
        const uint32_t local_teardown_addr = get_semaphore(get_arg_val<uint32_t>(argidx++));
        const uint32_t local_buffer_index_addr = get_semaphore(get_arg_val<uint32_t>(argidx++));
        termination_master_noc_x = static_cast<uint8_t>(get_arg_val<uint32_t>(argidx++));
        termination_master_noc_y = static_cast<uint8_t>(get_arg_val<uint32_t>(argidx++));
        if (mux_connection_valid) {
            // Waits for the mux to be ready to accept connections.
            mux_conn = tt::tt_fabric::build_connection_to_program_local_fabric_mux<fabric_mux_num_buffers_per_channel>(
                fabric_mux_x,
                fabric_mux_y,
                fabric_mux_channel_id,
                fabric_mux_num_buffers_per_channel,
                fabric_mux_channel_buffer_size_bytes,
                fabric_mux_l1_base_semaphore_id,
                fabric_mux_channel_base_offset,
                fabric_mux_connection_info_offset,
                fabric_mux_connection_handshake_offset,
                fabric_mux_flow_control_offset,
                fabric_mux_buffer_index_offset,
                local_flow_control_addr,
                local_teardown_addr,
                local_buffer_index_addr,
                local_fabric_mux_status_addr,
                fabric_mux_l1_base);
            tt::tt_fabric::fabric_client_connect(mux_conn);
        }
    }

    // ---- MUX writer setup: parsed by all MUX writers with valid connections ----
    // Rotated packet headers: reusing one header forces a NoC flush per packet (the header's L1
    // must not be rewritten while its send is in flight). Rotating a small pool amortizes that to
    // one flush per kNumFwdHdrs packets — the flush was the dominant serial cost per forwarded
    // chunk. Pool budget: NUM_PACKET_HEADERS/2 = 12 headers per RISC; 8 + 2 + 1 = 11 used.
    constexpr uint32_t kNumScatterHdrs = 8;
    constexpr uint32_t kNumUnicastHdrs = 2;
    volatile tt_l1_ptr PACKET_HEADER_TYPE* pkt_scatter_hdrs[kNumScatterHdrs] = {nullptr};
    volatile tt_l1_ptr PACKET_HEADER_TYPE* pkt_unicast_hdrs[kNumUnicastHdrs] = {nullptr};
    volatile tt_l1_ptr PACKET_HEADER_TYPE* pkt_hdr_sem_inc = nullptr;
    // Per-type sends since the last flush: a header may be rewritten only once every send of it
    // has been flushed, so each type flushes when its whole pool has been used since the last
    // flush. A flush clears everything, so both counters reset together.
    uint32_t scatter_since_flush = 0;
    uint32_t unicast_since_flush = 0;
    uint32_t scatter_hdr_idx = 0;
    uint32_t unicast_hdr_idx = 0;
    uint32_t ag_output_Wt = 0, ag_output_Ht = 0;
    uint32_t gathered_k_addr_ag_rt = 0, gathered_v_addr_ag_rt = 0;
    uint32_t injector_noc_x = 0, injector_noc_y = 0;
    uint32_t num_muxes_in_direction = 1, my_mux_index = 0;
    // Split-head forwarding dedup: when set, this row is the follower of a pair sharing one head —
    // the leader row's clients send byte-identical packets, so this client skips all fabric data
    // and semincs (it still drains its K/V handles to keep the reader's CB pacing unchanged).
    bool dedup_skip_forward = false;

    if (mux_connection_valid) {
        const uint32_t out_ready_sem_addr = get_arg_val<uint32_t>(argidx++);
        injector_noc_x = get_arg_val<uint32_t>(argidx++);
        injector_noc_y = get_arg_val<uint32_t>(argidx++);
        num_muxes_in_direction = get_arg_val<uint32_t>(argidx++);
        my_mux_index = get_arg_val<uint32_t>(argidx++);
        ag_output_Wt = get_arg_val<uint32_t>(argidx++);
        ag_output_Ht = get_arg_val<uint32_t>(argidx++);
        gathered_k_addr_ag_rt = get_arg_val<uint32_t>(argidx++);
        gathered_v_addr_ag_rt = get_arg_val<uint32_t>(argidx++);
        dedup_skip_forward = get_arg_val<uint32_t>(argidx++) == 1;

        for (uint32_t h = 0; h < kNumScatterHdrs; ++h) {
            pkt_scatter_hdrs[h] = PacketHeaderPool::allocate_header();
        }
        for (uint32_t h = 0; h < kNumUnicastHdrs; ++h) {
            pkt_unicast_hdrs[h] = PacketHeaderPool::allocate_header();
        }
        pkt_hdr_sem_inc = PacketHeaderPool::allocate_header();

        // Pre-configure scatter write headers with max packet size; per-call with_state
        // overrides DstAddrs, ChunkSizes, and PayloadSize for the actual tile count.
        constexpr uint32_t scatter_init_count = ag_packet_size_in_pages >= 2 ? ag_packet_size_in_pages : 2;
        uint64_t dummy_addrs[4] = {0, 0, 0, 0};
        uint16_t scatter_chunk_sizes[3] = {
            static_cast<uint16_t>(ag_page_size),
            static_cast<uint16_t>(ag_page_size),
            static_cast<uint16_t>(ag_page_size)};
        for (uint32_t h = 0; h < kNumScatterHdrs; ++h) {
            fabric_unicast_noc_scatter_write_set_state<
                UnicastScatterWriteUpdateMask::ChunkSizes | UnicastScatterWriteUpdateMask::PayloadSize>(
                pkt_scatter_hdrs[h],
                1,
                NocUnicastScatterCommandHeader(dummy_addrs, scatter_chunk_sizes, scatter_init_count),
                ag_page_size * ag_packet_size_in_pages);
        }

        // Pre-configure unicast write headers: payload size is constant
        for (uint32_t h = 0; h < kNumUnicastHdrs; ++h) {
            fabric_unicast_noc_unicast_write_set_state<UnicastWriteUpdateMask::PayloadSize>(
                pkt_unicast_hdrs[h], 1, nullptr, ag_page_size);
        }

        // Pre-configure atomic inc header for signaling the injector core on the next device
        // safe_get_noc_addr() cannot be migrated to Device 2.0 because it creates a cross-chip semaphore address,
        // while the current Device 2.0 Semaphore class only supports same-chip operations.
        const uint64_t injector_out_ready_sem_noc_addr =
            safe_get_noc_addr(injector_noc_x, injector_noc_y, out_ready_sem_addr, 0);
        fabric_unicast_noc_unicast_atomic_inc_set_state<
            UnicastAtomicIncUpdateMask::DstAddr | UnicastAtomicIncUpdateMask::Val | UnicastAtomicIncUpdateMask::Flush>(
            pkt_hdr_sem_inc, 1, tt::tt_fabric::NocUnicastAtomicIncCommandHeader{injector_out_ready_sem_noc_addr, 1});

        for (uint32_t h = 0; h < kNumScatterHdrs; ++h) {
            fabric_set_unicast_route<false>(pkt_scatter_hdrs[h], 1);
        }
        for (uint32_t h = 0; h < kNumUnicastHdrs; ++h) {
            fabric_set_unicast_route<false>(pkt_unicast_hdrs[h], 1);
        }
        fabric_set_unicast_route<false>(pkt_hdr_sem_inc, 1);
    }

    const auto gathered_k_writer = TensorAccessor(ag_gathered_k_args, gathered_k_addr_ag_rt);
    const auto gathered_v_writer = TensorAccessor(ag_gathered_v_args, gathered_v_addr_ag_rt);

    // The recipe CB layout (recipe_ring_cbs.hpp); the MUX writer pops its own handles on the K and V CBs.
    constexpr uint32_t cb_out = recipe_ring_cb::out;
    constexpr uint32_t cb_k_writer_in = recipe_ring_cb::k_writer;
    constexpr uint32_t cb_v_writer_in = recipe_ring_cb::v_writer;
    constexpr uint32_t tile_bytes = get_tile_size(cb_out);

    CircularBuffer cb_k_w(cb_k_writer_in);
    CircularBuffer cb_v_w(cb_v_writer_in);
    const auto out_writer = TensorAccessor(out_args, out_addr);
    const auto joint_out_writer = TensorAccessor(joint_out_args, joint_out_addr);

    const auto output_tile_logical = TensorTileShape(B, NH, local_padded_Nt, DHt);
    const auto joint_tile_logical = TensorTileShape(B, NH, Lt, DHt);

    const auto out_generator = PaddedAddrGenerator(out_writer, output_tile_logical);
    const auto joint_out_generator = PaddedAddrGenerator(joint_out_writer, joint_tile_logical);

    // The recipe's reduce scaler and column identity (as the dense recipe reader, reader_recipe.cpp).
    dataflow_kernel_lib::calculate_and_prepare_reduce_scaler<
        recipe_ring_cb::identity_scale,
        ckernel::PoolType::MAX,
        ckernel::ReduceDim::REDUCE_ROW,
        dataflow_kernel_lib::SUM_AND_MAX_REDUCE_FACTOR>();
    generate_bcast_col_scalar_zeroed(CircularBuffer(recipe_ring_cb::col_identity), 0x3f803f80);

    // Read the live length ONCE, before the ring loop. This kernel is a protocol participant: the MUX forwarding loop
    // below skips chunks on the same predicate as the reader and sends one per-link atomic_inc per forwarded chunk,
    // which is exactly what the reader's injector gate counts. Reader and writer must therefore derive identical
    // values from the same tensor.
    uint32_t logical_n = logical_n_ct;
    uint32_t logical_nt = logical_nt_ct;
    if constexpr (has_logical_n_tensor) {
        // Borrow the output CB's L1 as read scratch: this runs before compute pushes any output into it.
        logical_n = trace_metadata::read_metadata_scalar_u32(
            noc, logical_n_args, get_common_arg_val<uint32_t>(0), CircularBuffer(cb_out).get_write_ptr());
        ASSERT(logical_n >= 1);
        logical_nt = ring_joint::tiles_for(logical_n);
    }

    const uint32_t last_active_ring_iter =
        find_last_active_ring_iter(fused_op_indexer.seq, local_padded_Nt, logical_nt, L);

    // Pass-outer, ring-inner, as the reader and compute: pass p handles head-segment (p * rows + my_row) over the whole
    // ring. The AG forwarding below and the output drain are parameterized per chunk and per pass.
    for (uint32_t pass = 0; pass < q_count; ++pass) {
        RingIdSequencer order = fused_op_indexer.seq;
        for (uint32_t ring_iter = 0; ring_iter < ring_size; ++ring_iter) {
            const uint32_t ring_id = order.get_next_ring_id([](uint32_t, uint32_t) {});

            const bool do_joint_kv = ring_id == ring_size - 1;
            const uint32_t num_kv_chunks = do_joint_kv ? num_local_k_chunks + num_joint_k_chunks : num_local_k_chunks;

            const uint32_t ring_iter_kv_start_tile = ring_id * local_padded_Nt;
            const bool ring_iter_processes_KV_chunks = ring_iter_kv_start_tile < logical_nt;
            const bool ring_iter_does_work = ring_iter_processes_KV_chunks || (do_joint_kv && L != 0);
            if (!ring_iter_does_work) {
                continue;
            }

            {
                // The pass's state stays resident in L1; its last active ring iteration writes the normalized output.
                const bool is_last_ring_iter = (ring_iter == last_active_ring_iter);
                const uint32_t global_q_chunk = q_base + pass * q_stride;
                const uint32_t nb = global_q_chunk / (NH * num_q_chunks);
                const uint32_t nq = (global_q_chunk % (NH * num_q_chunks)) / num_q_chunks;
                const uint32_t q_chunk = global_q_chunk % num_q_chunks;

                const auto qi = get_q_chunk_info(
                    q_chunk, nb, nq, ring_id, num_local_q_chunks, Sq_chunk_t, DHt, Lt, local_padded_Nt);

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

                        const uint32_t gathered_kv_start_tile = ring_iter_kv_start_tile + k_chunk * Sk_chunk_t;
                        const Slice kv_slice(
                            nb, nq, gathered_kv_start_tile, gathered_kv_start_tile + Sk_chunk_t, 0, DHt);
                        const uint32_t end_seq_tile = std::min(logical_nt, local_padded_Nt * (ring_id + 1));

                        const uint32_t bh_offset = (nb * NH + nq) * ag_output_Wt * ag_output_Ht;

                        // Wait for reader to fill K, forward this writer's row slice over fabric.
                        // Positional gate, not last-active: a shard must complete all ring_size-1 hops even
                        // when this device's last ACTIVE iteration is earlier (trailing pad shards);
                        // gating on last-active starves downstream.
                        cb_k_w.wait_front(k_chunk_tiles);
                        if (!dedup_skip_forward && ring_iter != ring_size - 1) {
                            if (!kv_chunk_is_joint) {
                                const uint32_t base_k_read_ptr = cb_k_w.get_read_ptr();
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
                                            if (scatter_since_flush >= kNumScatterHdrs) {
                                                noc.async_writes_flushed();
                                                scatter_since_flush = 0;
                                                unicast_since_flush = 0;
                                            }
                                            fabric_unicast_noc_scatter_write_with_state<
                                                UnicastScatterWriteUpdateMask::DstAddrs |
                                                UnicastScatterWriteUpdateMask::ChunkSizes |
                                                UnicastScatterWriteUpdateMask::PayloadSize>(
                                                &mux_conn,
                                                pkt_scatter_hdrs[scatter_hdr_idx],
                                                src_l1_addr,
                                                NocUnicastScatterCommandHeader(k_noc_addrs, k_cs, tiles_in_batch),
                                                ag_page_size * tiles_in_batch);
                                            scatter_hdr_idx = (scatter_hdr_idx + 1) % kNumScatterHdrs;
                                            scatter_since_flush++;
                                        } else {
                                            // Partial batch: fall back to per-tile unicast writes to avoid
                                            // variable chunk_count scatter writes which cause non-determinism.
                                            for (uint32_t i = 0; i < tiles_in_batch; i++) {
                                                if (unicast_since_flush >= kNumUnicastHdrs) {
                                                    noc.async_writes_flushed();
                                                    scatter_since_flush = 0;
                                                    unicast_since_flush = 0;
                                                }
                                                fabric_unicast_noc_unicast_write_with_state<
                                                    UnicastWriteUpdateMask::DstAddr>(
                                                    &mux_conn,
                                                    pkt_unicast_hdrs[unicast_hdr_idx],
                                                    src_l1_addr + i * ag_page_size,
                                                    NocUnicastCommandHeader{k_noc_addrs[i]});
                                                unicast_hdr_idx = (unicast_hdr_idx + 1) % kNumUnicastHdrs;
                                                unicast_since_flush++;
                                            }
                                        }
                                    }
                                }
                            }
                        }
                        // Flush before pop: the payload sends read straight out of the CB pages.
                        noc.async_writes_flushed();
                        scatter_since_flush = 0;
                        unicast_since_flush = 0;
                        cb_k_w.pop_front(k_chunk_tiles);

                        // Wait for reader to fill V, forward this writer's row slice over fabric
                        cb_v_w.wait_front(v_chunk_tiles);
                        if (!dedup_skip_forward && ring_iter != ring_size - 1) {
                            if (!kv_chunk_is_joint) {
                                const uint32_t base_v_read_ptr = cb_v_w.get_read_ptr();
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
                                            if (scatter_since_flush >= kNumScatterHdrs) {
                                                noc.async_writes_flushed();
                                                scatter_since_flush = 0;
                                                unicast_since_flush = 0;
                                            }
                                            fabric_unicast_noc_scatter_write_with_state<
                                                UnicastScatterWriteUpdateMask::DstAddrs |
                                                UnicastScatterWriteUpdateMask::ChunkSizes |
                                                UnicastScatterWriteUpdateMask::PayloadSize>(
                                                &mux_conn,
                                                pkt_scatter_hdrs[scatter_hdr_idx],
                                                src_l1_addr,
                                                NocUnicastScatterCommandHeader(v_noc_addrs, v_cs, tiles_in_batch),
                                                ag_page_size * tiles_in_batch);
                                            scatter_hdr_idx = (scatter_hdr_idx + 1) % kNumScatterHdrs;
                                            scatter_since_flush++;
                                        } else {
                                            // Partial batch: fall back to per-tile unicast writes to avoid
                                            // variable chunk_count scatter writes which cause non-determinism.
                                            for (uint32_t i = 0; i < tiles_in_batch; i++) {
                                                if (unicast_since_flush >= kNumUnicastHdrs) {
                                                    noc.async_writes_flushed();
                                                    scatter_since_flush = 0;
                                                    unicast_since_flush = 0;
                                                }
                                                fabric_unicast_noc_unicast_write_with_state<
                                                    UnicastWriteUpdateMask::DstAddr>(
                                                    &mux_conn,
                                                    pkt_unicast_hdrs[unicast_hdr_idx],
                                                    src_l1_addr + i * ag_page_size,
                                                    NocUnicastCommandHeader{v_noc_addrs[i]});
                                                unicast_hdr_idx = (unicast_hdr_idx + 1) % kNumUnicastHdrs;
                                                unicast_since_flush++;
                                            }
                                        }
                                    }
                                }
                            }
                        }
                        // Flush before pop: the payload sends read straight out of the CB pages.
                        noc.async_writes_flushed();
                        scatter_since_flush = 0;
                        unicast_since_flush = 0;
                        cb_v_w.pop_front(v_chunk_tiles);

                        // Joint KV is replicated and read locally, never forwarded over fabric, so it
                        // must not bump the per-link forward semaphore the receiver counts only for
                        // non-joint chunks.
                        if (!dedup_skip_forward && ring_iter != ring_size - 1 && !kv_chunk_is_joint) {
                            fabric_unicast_noc_unicast_atomic_inc_with_state(&mux_conn, pkt_hdr_sem_inc);
                            noc.async_writes_flushed();
                        }
                    }
                }

                // On last ring iteration, drain normalized output to DRAM.
                if (is_last_ring_iter) {
                    // Default trid here → pass 0 so per-group flush waits exactly for these writes.
                    write_block_row_grouped_trid(
                        noc,
                        qi.is_joint_q ? joint_out_generator : out_generator,
                        qi.out_slice,
                        qi.end_seq_tile,
                        cb_out,
                        tile_bytes,
                        out_subblock_h,
                        /*flush_trid=*/0);
                    noc.async_write_barrier();
                }
                // The reader's phase-alignment K/V pair, after the output drain: with one K and one V entry (the
                // factory's L1 fallback) the reader can push it only once compute has released the last chunk, which
                // compute does after normalizing into the output CB that this drain empties.
                if (mux_connection_valid && KV_chunks_processed_in_iter % 2 == 0) {
                    cb_k_w.wait_front(k_chunk_tiles);
                    cb_v_w.wait_front(v_chunk_tiles);
                    cb_k_w.pop_front(k_chunk_tiles);
                    cb_v_w.pop_front(v_chunk_tiles);
                }
            }
        }
    }

    if (mux_connection_valid) {
        noc.async_atomic_barrier();
        tt::tt_fabric::fabric_client_disconnect(mux_conn);
        Semaphore<> termination_sync_sem(termination_sync_sem_id);
        if (is_termination_master) {
            termination_sync_sem.wait(num_mux_clients - 1);
            tt::tt_fabric::fabric_endpoint_terminate(
                fabric_mux_x, fabric_mux_y, fabric_mux_l1_base + fabric_mux_termination_signal_offset);
        } else {
            termination_sync_sem.up(noc, termination_master_noc_x, termination_master_noc_y, 1);
            noc.async_atomic_barrier();
        }
    }
    noc.async_write_barrier();
}
