// SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#include "api/dataflow/dataflow_api.h"
#include "ttnn/kernel/dataflow/generate_bcast_scalar.hpp"
#include "ttnn/kernel/dataflow/generate_reduce_scaler.hpp"
#include "dataflow_common.hpp"
#include "fused_op_receiver.hpp"

#ifdef USE_MUX
#include "tt_metal/fabric/hw/inc/tt_fabric_mux_interface.hpp"
#include "tt_metal/fabric/hw/inc/tt_fabric_api.h"
#include "tt_metal/fabric/hw/inc/noc_addr.h"
#include "tt_metal/fabric/hw/inc/linear/addrgen_api.h"
#include "tt_metal/fabric/hw/inc/linear/api.h"
#include "cpp/ttnn/operations/ccl/kernel_common/worker_sync_utils.hpp"
#include "cpp/ttnn/operations/ccl/ccl_host_types.hpp"
#include "api/debug/dprint.h"
using namespace tt::tt_fabric::linear::experimental;
#endif

// Eager-path reader: reads the previous ring iteration's normalized output and LSE from DRAM.
// Used by the non-streaming (old sdpa_ring) path for sigmoid-based inter-iteration merging.
// Pushes output tiles into cb_prev_out (c_7) and LSE tiles into cb_lse_in (c_6).
//
// @param cat_out_generator   Address generator for the output DRAM tensor (local or joint)
// @param stats_writer        TensorAccessor for the stats DRAM tensor
// @param stats_tile_logical  Tile shape of the stats tensor (for address computation)
// @param nb                  Batch index
// @param nq                  Head index
// @param Sq_chunk_t          Q chunk size in tiles
// @param out_slice           Row/col tile range in the output tensor for this Q chunk
// @param end_seq_tile        Last valid sequence tile (for padding-aware reads)
// @param stats_seq_start_tile  First tile row in the stats tensor for this Q chunk
// @param stats_seq_end_tile    One-past-last tile row (clamped to avoid reading past padding)
// @param cb_prev_out         CB to push previous output tiles into (c_7, read by compute)
// @param cb_lse_in           CB to push previous LSE tiles into (c_6, read by compute)
// @param tile_bytes          Output tile size in bytes
// @param stats_tile_bytes    Stats tile size in bytes
template <typename ReaderType, typename TensorAccessorType>
void read_prev_output_and_lse(
    const PaddedAddrGenerator<ReaderType>& cat_out_generator,
    const TensorAccessorType& stats_writer,
    const TensorTileShape& stats_tile_logical,
    const uint32_t nb,
    const uint32_t nq,
    const uint32_t Sq_chunk_t,
    const Slice out_slice,
    const uint32_t end_seq_tile,
    const uint32_t stats_seq_start_tile,
    const uint32_t stats_seq_end_tile,
    const uint32_t cb_prev_out,
    const uint32_t cb_lse_in,
    const uint32_t tile_bytes,
    const uint32_t stats_tile_bytes) {
    // Read previous output for this Q chunk
    read_block(cat_out_generator, out_slice, end_seq_tile, cb_prev_out, tile_bytes, false);

    // Read previous LSE for this Q chunk
    cb_reserve_back(cb_lse_in, Sq_chunk_t);
    uint32_t lse_addr = get_write_ptr(cb_lse_in);
    for (uint32_t i = stats_seq_start_tile; i < stats_seq_end_tile; i++) {
        noc_async_read_tile(stats_tile_logical.id_of(nb, nq, i, 0), stats_writer, lse_addr);
        lse_addr += stats_tile_bytes;
    }
    noc_async_read_barrier();
    cb_push_back(cb_lse_in, Sq_chunk_t);
}

// Deferred-norm reader: restores raw (un-normalized) accumulators from DRAM for multi-Q-chunk.
// Reads output, running max, and running sum — the full online-softmax state needed to continue
// accumulation from a previous ring iteration.
// Pushes into: cb_prev_out (c_7), cb_max_in (c_6), cb_sum_in (c_11).
//
// @param cat_out_generator   Address generator for the output DRAM tensor (local or joint)
// @param stats_writer        TensorAccessor for the stats DRAM tensor
// @param stats_tile_logical  Tile shape of the stats tensor
// @param nb                  Batch index
// @param nq                  Head index
// @param Sq_chunk_t          Q chunk size in tiles
// @param out_slice           Row/col tile range in the output tensor for this Q chunk
// @param end_seq_tile        Last valid sequence tile (for padding-aware reads)
// @param stats_seq_start_tile  First tile row in stats tensor for this Q chunk's max/sum
// @param stats_seq_end_tile    One-past-last tile row (clamped to sequence bounds)
// @param sum_offset          Row offset into stats tensor where sum tiles start (= local_padded_Nt + Lt)
// @param cb_prev_out         CB for previous output tiles (c_7)
// @param cb_max_in           CB for previous max tiles (c_6)
// @param cb_sum_in           CB for previous sum tiles (c_11)
// @param tile_bytes          Output tile size in bytes
// @param stats_tile_bytes    Stats tile size in bytes
template <typename ReaderType, typename TensorAccessorType>
void read_prev_accumulators(
    const PaddedAddrGenerator<ReaderType>& cat_out_generator,
    const TensorAccessorType& stats_writer,
    const TensorTileShape& stats_tile_logical,
    const uint32_t nb,
    const uint32_t nq,
    const uint32_t Sq_chunk_t,
    const Slice out_slice,
    const uint32_t end_seq_tile,
    const uint32_t stats_seq_start_tile,
    const uint32_t stats_seq_end_tile,
    const uint32_t sum_offset,
    const uint32_t cb_prev_out,
    const uint32_t cb_max_in,
    const uint32_t cb_sum_in,
    const uint32_t tile_bytes,
    const uint32_t stats_tile_bytes) {
    // Read previous output
    read_block(cat_out_generator, out_slice, end_seq_tile, cb_prev_out, tile_bytes, false);

    // Read max from stats DRAM (first half)
    cb_reserve_back(cb_max_in, Sq_chunk_t);
    uint32_t max_addr = get_write_ptr(cb_max_in);
    for (uint32_t i = stats_seq_start_tile; i < stats_seq_end_tile; i++) {
        noc_async_read_tile(stats_tile_logical.id_of(nb, nq, i, 0), stats_writer, max_addr);
        max_addr += stats_tile_bytes;
    }
    noc_async_read_barrier();
    cb_push_back(cb_max_in, Sq_chunk_t);

    // Read sum from stats DRAM (second half, offset by sum_offset)
    cb_reserve_back(cb_sum_in, Sq_chunk_t);
    uint32_t sum_addr = get_write_ptr(cb_sum_in);
    for (uint32_t i = stats_seq_start_tile; i < stats_seq_end_tile; i++) {
        noc_async_read_tile(stats_tile_logical.id_of(nb, nq, sum_offset + i, 0), stats_writer, sum_addr);
        sum_addr += stats_tile_bytes;
    }
    noc_async_read_barrier();
    cb_push_back(cb_sum_in, Sq_chunk_t);
}

// Deferred-norm writer: saves raw accumulators to DRAM for multi-Q-chunk between ring iterations.
// Drains output, running max, and running sum from compute CBs to the DRAM tensors.
// Reads from: cb_out (c_16), cb_max_out (c_17), cb_sum_out (c_10).
//
// @param cat_out_generator   Address generator for the output DRAM tensor (local or joint)
// @param stats_writer        TensorAccessor for the stats DRAM tensor
// @param stats_tile_logical  Tile shape of the stats tensor
// @param nb                  Batch index
// @param nq                  Head index
// @param Sq_chunk_t          Q chunk size in tiles
// @param out_slice           Row/col tile range in the output tensor for this Q chunk
// @param end_seq_tile        Last valid sequence tile
// @param stats_seq_start_tile  First tile row in stats tensor for this Q chunk's max/sum
// @param stats_seq_end_tile    One-past-last tile row (clamped to sequence bounds)
// @param sum_offset          Row offset into stats tensor where sum tiles start (= local_padded_Nt + Lt)
// @param cb_out              CB to drain output tiles from (c_16, pushed by compute)
// @param cb_max_out          CB to drain max tiles from (c_17, pushed by compute)
// @param cb_sum_out          CB to drain sum tiles from (c_10, pushed by compute)
// @param tile_bytes          Output tile size in bytes
// @param stats_tile_bytes    Stats tile size in bytes
template <typename ReaderType, typename TensorAccessorType>
void write_accumulators(
    const PaddedAddrGenerator<ReaderType>& cat_out_generator,
    const TensorAccessorType& stats_writer,
    const TensorTileShape& stats_tile_logical,
    const uint32_t nb,
    const uint32_t nq,
    const uint32_t Sq_chunk_t,
    const Slice out_slice,
    const uint32_t end_seq_tile,
    const uint32_t stats_seq_start_tile,
    const uint32_t stats_seq_end_tile,
    const uint32_t sum_offset,
    const uint32_t cb_out,
    const uint32_t cb_max_out,
    const uint32_t cb_sum_out,
    const uint32_t tile_bytes,
    const uint32_t stats_tile_bytes) {
    // Write output
    write_block(cat_out_generator, out_slice, end_seq_tile, cb_out, tile_bytes);

    // Write max to stats DRAM (first half)
    cb_wait_front(cb_max_out, Sq_chunk_t);
    uint32_t max_write_addr = get_read_ptr(cb_max_out);
    for (uint32_t i = stats_seq_start_tile; i < stats_seq_end_tile; i++) {
        noc_async_write_tile(stats_tile_logical.id_of(nb, nq, i, 0), stats_writer, max_write_addr);
        max_write_addr += stats_tile_bytes;
    }
    noc_async_writes_flushed();
    cb_pop_front(cb_max_out, Sq_chunk_t);

    // Write sum to stats DRAM (second half, offset by sum_offset)
    cb_wait_front(cb_sum_out, Sq_chunk_t);
    uint32_t sum_write_addr = get_read_ptr(cb_sum_out);
    for (uint32_t i = stats_seq_start_tile; i < stats_seq_end_tile; i++) {
        noc_async_write_tile(stats_tile_logical.id_of(nb, nq, sum_offset + i, 0), stats_writer, sum_write_addr);
        sum_write_addr += stats_tile_bytes;
    }
    noc_async_writes_flushed();
    cb_pop_front(cb_sum_out, Sq_chunk_t);
}

// Eager-path writer: writes normalized output and LSE to DRAM every ring iteration.
// Used by the non-streaming (old sdpa_ring) path.
// Reads from: cb_out (c_16), cb_lse_out (c_17).
//
// @param cat_out_generator   Address generator for the output DRAM tensor (local or joint)
// @param stats_writer        TensorAccessor for the stats DRAM tensor
// @param stats_tile_logical  Tile shape of the stats tensor
// @param nb                  Batch index
// @param nq                  Head index
// @param Sq_chunk_t          Q chunk size in tiles
// @param out_slice           Row/col tile range in the output tensor for this Q chunk
// @param end_seq_tile        Last valid sequence tile
// @param stats_seq_start_tile  First tile row in stats tensor for this Q chunk's LSE
// @param stats_seq_end_tile    One-past-last tile row (clamped to sequence bounds)
// @param cb_out              CB to drain output tiles from (c_16)
// @param cb_lse_out          CB to drain LSE tiles from (c_17)
// @param tile_bytes          Output tile size in bytes
// @param stats_tile_bytes    Stats tile size in bytes
template <typename ReaderType, typename TensorAccessorType>
void write_output_and_lse(
    const PaddedAddrGenerator<ReaderType>& cat_out_generator,
    const TensorAccessorType& stats_writer,
    const TensorTileShape& stats_tile_logical,
    const uint32_t nb,
    const uint32_t nq,
    const uint32_t Sq_chunk_t,
    const Slice out_slice,
    const uint32_t end_seq_tile,
    const uint32_t stats_seq_start_tile,
    const uint32_t stats_seq_end_tile,
    const uint32_t cb_out,
    const uint32_t cb_lse_out,
    const uint32_t tile_bytes,
    const uint32_t stats_tile_bytes) {
    write_block(cat_out_generator, out_slice, end_seq_tile, cb_out, tile_bytes);

    cb_wait_front(cb_lse_out, Sq_chunk_t);
    uint32_t lse_addr = get_read_ptr(cb_lse_out);
    for (uint32_t i = stats_seq_start_tile; i < stats_seq_end_tile; i++) {
        noc_async_write_tile(stats_tile_logical.id_of(nb, nq, i, 0), stats_writer, lse_addr);
        lse_addr += stats_tile_bytes;
    }
    noc_async_writes_flushed();
    cb_pop_front(cb_lse_out, Sq_chunk_t);
}

struct QChunkInfo {
    bool is_joint_q;
    Slice out_slice;
    uint32_t end_seq_tile;
    uint32_t stats_seq_start_tile;
    uint32_t stats_seq_end_tile;
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
        info.stats_seq_start_tile = local_padded_Nt + (q_chunk - num_local_q_chunks) * Sq_chunk_t;
        info.stats_seq_end_tile = info.stats_seq_start_tile + Sq_chunk_t;
        info.stats_seq_start_tile = std::min(info.stats_seq_start_tile, local_padded_Nt + Lt);
        info.stats_seq_end_tile = std::min(info.stats_seq_end_tile, local_padded_Nt + Lt);
    } else {
        const uint32_t out_row_start_tile = q_chunk * Sq_chunk_t;
        info.out_slice = Slice(nb, nq, out_row_start_tile, out_row_start_tile + Sq_chunk_t, 0, DHt);
        info.end_seq_tile = local_padded_Nt * (ring_id + 1);
        info.stats_seq_start_tile = q_chunk * Sq_chunk_t;
        info.stats_seq_end_tile = info.stats_seq_start_tile + Sq_chunk_t;
        info.stats_seq_start_tile = std::min(info.stats_seq_start_tile, local_padded_Nt);
        info.stats_seq_end_tile = std::min(info.stats_seq_end_tile, local_padded_Nt);
    }
    return info;
}

void kernel_main() {
    constexpr uint32_t B = get_compile_time_arg_val(0);
    constexpr uint32_t NH = get_compile_time_arg_val(1);
    constexpr uint32_t DHt = get_compile_time_arg_val(2);
    constexpr uint32_t Sq_chunk_t = get_compile_time_arg_val(3);
    constexpr uint32_t Sk_chunk_t = get_compile_time_arg_val(4);
    constexpr uint32_t local_padded_N = get_compile_time_arg_val(5);
    constexpr uint32_t local_padded_Nt = get_compile_time_arg_val(6);
    constexpr uint32_t padded_Nt = get_compile_time_arg_val(7);
    constexpr uint32_t logical_n = get_compile_time_arg_val(8);
    constexpr uint32_t logical_nt = get_compile_time_arg_val(9);
    constexpr uint32_t Lt = get_compile_time_arg_val(10);
    constexpr uint32_t L = get_compile_time_arg_val(11);
    constexpr uint32_t num_local_q_chunks = get_compile_time_arg_val(12);
    constexpr uint32_t num_joint_q_chunks = get_compile_time_arg_val(13);
    constexpr uint32_t num_local_k_chunks = get_compile_time_arg_val(14);
    constexpr uint32_t num_joint_k_chunks = get_compile_time_arg_val(15);
    constexpr uint32_t num_q_chunks = get_compile_time_arg_val(16);
    constexpr uint32_t identity_scalar_packed = get_compile_time_arg_val(17);
    constexpr uint32_t scale_val = get_compile_time_arg_val(18);
    constexpr uint32_t ring_size = get_compile_time_arg_val(19);
    constexpr uint32_t global_n_partial_col = get_compile_time_arg_val(20);
    constexpr uint32_t joint_l_partial_col = get_compile_time_arg_val(21);
    constexpr bool use_streaming_compute = get_compile_time_arg_val(22) == 1;

    constexpr auto out_args = TensorAccessorArgs<23>();
    constexpr auto joint_out_args = TensorAccessorArgs<out_args.next_compile_time_args_offset()>();
    constexpr auto stats_args = TensorAccessorArgs<joint_out_args.next_compile_time_args_offset()>();

#ifdef USE_MUX
    constexpr uint32_t mux_ct_base = stats_args.next_compile_time_args_offset();
    constexpr uint8_t fabric_mux_num_buffers_per_channel = get_compile_time_arg_val(mux_ct_base + 0);
    constexpr size_t fabric_mux_channel_buffer_size_bytes = get_compile_time_arg_val(mux_ct_base + 1);
    constexpr size_t fabric_mux_status_address = get_compile_time_arg_val(mux_ct_base + 2);
    constexpr size_t fabric_mux_termination_signal_address = get_compile_time_arg_val(mux_ct_base + 3);
    constexpr uint32_t num_mux_clients = get_compile_time_arg_val(mux_ct_base + 4);


    // All-gather CT args (following 5 MUX CT args)
    constexpr uint32_t ag_ct_base = mux_ct_base + 5;
    constexpr uint32_t ag_device_index = get_compile_time_arg_val(ag_ct_base + 0);
    constexpr uint32_t ag_packet_size_in_pages = get_compile_time_arg_val(ag_ct_base + 1);
    constexpr uint32_t ag_page_size = get_compile_time_arg_val(ag_ct_base + 2);
    constexpr uint32_t ag_pkt_hdr_cb_id = get_compile_time_arg_val(ag_ct_base + 3);
    constexpr uint32_t ag_kv_scratch_cb_id = get_compile_time_arg_val(ag_ct_base + 4);
    constexpr uint32_t ag_num_targets_forward = get_compile_time_arg_val(ag_ct_base + 5);
    constexpr uint32_t ag_num_targets_backward = get_compile_time_arg_val(ag_ct_base + 6);
    constexpr ttnn::ccl::Topology ag_topology =
        static_cast<ttnn::ccl::Topology>(get_compile_time_arg_val(ag_ct_base + 7));
    constexpr auto ag_input_k_args = TensorAccessorArgs<ag_ct_base + 8>();
    constexpr auto ag_input_v_args = TensorAccessorArgs<ag_input_k_args.next_compile_time_args_offset()>();
    constexpr auto ag_gathered_k_args = TensorAccessorArgs<ag_input_v_args.next_compile_time_args_offset()>();
    constexpr auto ag_gathered_v_args = TensorAccessorArgs<ag_gathered_k_args.next_compile_time_args_offset()>();
#endif

    uint32_t argidx = 0;
    const uint32_t out_addr = get_arg_val<uint32_t>(argidx++);
    const uint32_t joint_out_addr = get_arg_val<uint32_t>(argidx++);
    const uint32_t stats_addr = get_arg_val<uint32_t>(argidx++);
    const uint32_t global_q_start = get_arg_val<uint32_t>(argidx++);
    const uint32_t global_q_end = get_arg_val<uint32_t>(argidx++);

    RingSDPAOpReceiver fused_op_receiver = RingSDPAOpReceiver(
        false, /* wait_for_op_signal */
        argidx);

#ifdef USE_MUX
    // push_ring_sdpa_fused_op_rt_args appends 4 values (ring_size, ring_index, direction, semaphore_id).
    // The writer's RingSDPAOpReceiver reads only the 3 ring params (wait_for_op_signal=false),
    // so skip the 1 trailing semaphore ID before reading MUX args.
    argidx += 1;

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
    const uint32_t termination_sync_sem_addr = get_semaphore(get_arg_val<uint32_t>(argidx++));
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
    uint32_t injector_noc_x = 0, injector_noc_y = 0;
    uint32_t num_muxes_in_direction = 1, my_mux_index = 0;

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

        // OpSignaler constructor advances argidx past its RT args
        OpSignaler(argidx);

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

        DPRINT << "injector sem x,y: " << injector_noc_x << "," << injector_noc_y << " addr: " << out_ready_sem_addr << ENDL();
    }

    const auto gathered_k_writer = TensorAccessor(ag_gathered_k_args, gathered_k_addr_ag_rt, ag_page_size);
    const auto gathered_v_writer = TensorAccessor(ag_gathered_v_args, gathered_v_addr_ag_rt, ag_page_size);
#endif

    // c_6/c_17 carry softmax statistics between compute and writer for DRAM round-trips.
    // Aliased by role: cb_max_* for deferred norm, cb_lse_* for eager norm.
    constexpr uint32_t cb_max_in = tt::CBIndex::c_6;  // deferred norm: DRAM → compute (running max)
    constexpr uint32_t cb_lse_in = tt::CBIndex::c_6;  // eager norm: DRAM → compute (LSE)
    constexpr uint32_t cb_prev_out = tt::CBIndex::c_7;
    constexpr uint32_t cb_out = tt::CBIndex::c_16;
    constexpr uint32_t cb_max_out = tt::CBIndex::c_17;  // deferred norm: compute → DRAM (running max)
    constexpr uint32_t cb_lse_out = tt::CBIndex::c_17;  // eager norm: compute → DRAM (LSE)
    constexpr uint32_t cb_mask_in = tt::CBIndex::c_3;
    constexpr uint32_t cb_sum_out = tt::CBIndex::c_10;
    constexpr uint32_t cb_sum_in = tt::CBIndex::c_11;
    constexpr uint32_t cb_k_writer_in = tt::CBIndex::c_14;
    constexpr uint32_t cb_v_writer_in = tt::CBIndex::c_15;
    constexpr uint32_t tile_bytes = get_tile_size(cb_out);
    constexpr uint32_t stats_tile_bytes = get_tile_size(cb_max_in);

    const auto out_writer = TensorAccessor(out_args, out_addr, tile_bytes);
    const auto joint_out_writer = TensorAccessor(joint_out_args, joint_out_addr, tile_bytes);
    const auto stats_writer = TensorAccessor(stats_args, stats_addr, stats_tile_bytes);

    const auto output_tile_logical = TensorTileShape(B, NH, local_padded_Nt, DHt);
    const auto joint_tile_logical = TensorTileShape(B, NH, Lt, DHt);
    // stats tensor is 2× the sequence length: first half stores max (used by both eager and
    // deferred-norm paths), second half stores sum (deferred-norm only).
    const auto stats_tile_logical = TensorTileShape(B, NH, (local_padded_Nt + Lt) * 2, 1);

    const auto out_generator = PaddedAddrGenerator(out_writer, output_tile_logical);
    const auto joint_out_generator = PaddedAddrGenerator(joint_out_writer, joint_tile_logical);

    constexpr uint32_t cb_scale_in = tt::CBIndex::c_4;
    constexpr uint32_t cb_col_identity = tt::CBIndex::c_8;
    constexpr uint32_t cb_identity_scale_in = tt::CBIndex::c_5;

    generate_bcast_unary_scalar(cb_scale_in, scale_val);
    generate_bcast_col_scalar(cb_col_identity, identity_scalar_packed);
    generate_reduce_scaler(cb_identity_scale_in, identity_scalar_packed);

    // Lightweight mask: generate all mask tiles once into single CB before the ring loop.
    // Only needed when any K/joint dimension has padding that doesn't fill a chunk.
    constexpr bool local_n_has_padding = local_padded_Nt % Sk_chunk_t != 0;
    constexpr bool global_n_has_padding = logical_n % (Sk_chunk_t * tt::constants::TILE_HEIGHT) != 0;
    constexpr bool joint_has_padding = L > 0 && L % (Sk_chunk_t * tt::constants::TILE_HEIGHT) != 0;
    constexpr bool needs_lightweight_mask = local_n_has_padding || global_n_has_padding || joint_has_padding;
    if constexpr (needs_lightweight_mask) {
        generate_lightweight_mask_tiles<global_n_partial_col, joint_l_partial_col, cb_mask_in>();
    }

    const uint32_t last_active_ring_iter =
        find_last_active_ring_iter(fused_op_receiver.seq, local_padded_Nt, logical_n / tt::constants::TILE_HEIGHT, L);

    for (uint32_t ring_iter = 0; ring_iter < ring_size; ++ring_iter) {
        uint32_t ring_id = fused_op_receiver.get_next_ring_id_and_sync();

        const bool do_joint_kv = ring_id == ring_size - 1;
        const uint32_t num_kv_chunks = do_joint_kv ? num_local_k_chunks + num_joint_k_chunks : num_local_k_chunks;

        const uint32_t ring_iter_kv_start_tile = ring_id * local_padded_Nt;
        const uint32_t ring_iter_kv_end_tile = ring_iter_kv_start_tile + num_local_k_chunks * Sk_chunk_t;
        const uint32_t global_n_tile_id = logical_n / tt::constants::TILE_HEIGHT;
        const bool ring_iter_processes_KV_chunks = ring_iter_kv_start_tile <= global_n_tile_id;
        const bool ring_iter_does_work = ring_iter_processes_KV_chunks || (do_joint_kv && L != 0);
        if (!ring_iter_does_work) {
            continue;
        }

        /**
        We have 3 possible masks
        - global N mask
        - local N mask
        - joint L mask

        Global N mask:
            - If the logical_n falls within this ring iter's KV range
            - And logical_n length (within local_padded_N) does not divide by K chunk size

        Local N mask
            - If local_padded_N does not divide by K chunk size, the last chunk needs a mask

        Joint L mask
            - If joint length L does not divide by K chunk size, the last chunk needs a mask
        */

        // GLOBAL N MASK
        // Find out if logical_n falls within this ring iter's KV range
        const int32_t global_n_within_ring_iter = logical_n - ring_id * local_padded_N;
        // Note the > and <=. This means there is real length of logical_n within this ring iter.
        const bool global_n_is_within_ring_iter =
            global_n_within_ring_iter > 0 && global_n_within_ring_iter <= (int32_t)local_padded_N;
        const bool global_n_needs_masking = global_n_within_ring_iter % (Sk_chunk_t * tt::constants::TILE_HEIGHT) != 0;
        const bool ring_iter_needs_global_n_mask = global_n_is_within_ring_iter && global_n_needs_masking;

        // LOCAL N MASK
        const bool local_n_needs_masking = local_padded_Nt % Sk_chunk_t != 0;
        // If global N is in the ring iter, it supersedes the local N mask.
        const bool ring_iter_needs_local_n_mask = local_n_needs_masking && !global_n_is_within_ring_iter;

        // JOINT L MASK
        const bool joint_n_needs_masking = L % (Sk_chunk_t * tt::constants::TILE_HEIGHT) != 0;
        const bool ring_iter_needs_joint_n_mask = joint_n_needs_masking && do_joint_kv;

        // Deferred normalization is always paired with streaming compute.
        constexpr bool use_deferred_norm = use_streaming_compute;
        if constexpr (use_deferred_norm) {
            // Deferred norm: accumulates across ring iterations with exponential rescaling.
            // Single Q-chunk: accumulators persist in L1, write final output on last ring_iter.
            // Multi Q-chunk: raw accumulators round-trip through DRAM between ring iterations.
            const bool is_last_ring_iter = (ring_iter == last_active_ring_iter);
            const bool single_q_chunk = (global_q_end - global_q_start == 1);

            for (uint32_t global_q_chunk = global_q_start; global_q_chunk < global_q_end; ++global_q_chunk) {
                const uint32_t nb = global_q_chunk / (NH * num_q_chunks);
                const uint32_t nq = (global_q_chunk % (NH * num_q_chunks)) / num_q_chunks;
                const uint32_t q_chunk = global_q_chunk % num_q_chunks;

                const auto qi = get_q_chunk_info(
                    q_chunk, nb, nq, ring_id, num_local_q_chunks, Sq_chunk_t, DHt, Lt, local_padded_Nt);

                constexpr uint32_t sum_offset = local_padded_Nt + Lt;

                if (!single_q_chunk && ring_iter > 0) {
                    read_prev_accumulators(
                        qi.is_joint_q ? joint_out_generator : out_generator,
                        stats_writer,
                        stats_tile_logical,
                        nb,
                        nq,
                        Sq_chunk_t,
                        qi.out_slice,
                        qi.end_seq_tile,
                        qi.stats_seq_start_tile,
                        qi.stats_seq_end_tile,
                        sum_offset,
                        cb_prev_out,
                        cb_max_in,
                        cb_sum_in,
                        tile_bytes,
                        stats_tile_bytes);
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

                        // Wait for reader to fill K, forward this writer's row slice over fabric
                        cb_wait_front(cb_k_writer_in, k_chunk_tiles);
                        if (!is_last_ring_iter) {
                        if (!kv_chunk_is_joint) {
                            const uint32_t base_k_read_ptr = get_read_ptr(cb_k_writer_in);
                            for (uint32_t col = 0; col < DHt; ++col) {
                                for (uint32_t row = my_row_start; row < my_row_end; row += ag_packet_size_in_pages) {
                                    uint32_t tiles_in_batch = 0;
                                    uint64_t k_noc_addrs[4] = {0, 0, 0, 0};
                                    for (uint32_t i = 0; i < ag_packet_size_in_pages && row + i < my_row_end; i++) {
                                        if (kv_slice.d2_start + row + i >= end_seq_tile) break;
                                        const uint32_t dest_id = bh_offset
                                            + (kv_global_start_tile + row + i) * ag_output_Wt + col;
                                        k_noc_addrs[tiles_in_batch] =
                                            tt::tt_fabric::linear::addrgen_detail::get_noc_address(
                                                gathered_k_writer, dest_id, 0);
                                        tiles_in_batch++;
                                    }
                                    if (tiles_in_batch == 0) break;
                                    const uint32_t src_l1_addr = base_k_read_ptr
                                        + (row + col * Sk_chunk_t) * ag_page_size;
                                    if (tiles_in_batch > 1) {
                                        uint16_t k_cs[3] = {
                                            static_cast<uint16_t>(ag_page_size),
                                            static_cast<uint16_t>(ag_page_size),
                                            static_cast<uint16_t>(ag_page_size)};
                                        fabric_unicast_noc_scatter_write_with_state<
                                            UnicastScatterWriteUpdateMask::DstAddrs |
                                            UnicastScatterWriteUpdateMask::ChunkSizes |
                                            UnicastScatterWriteUpdateMask::PayloadSize>(
                                            &mux_conn, pkt_scatter_hdr, src_l1_addr,
                                            NocUnicastScatterCommandHeader(k_noc_addrs, k_cs, tiles_in_batch),
                                            ag_page_size * tiles_in_batch);
                                    } else {
                                        fabric_unicast_noc_unicast_write_with_state<UnicastWriteUpdateMask::DstAddr>(
                                            &mux_conn, pkt_unicast_hdr, src_l1_addr,
                                            NocUnicastCommandHeader{k_noc_addrs[0]});
                                    }
                                    noc_async_write_barrier();
                                }
                            }
                            noc_async_writes_flushed();
                        }
                        }
                        cb_pop_front(cb_k_writer_in, k_chunk_tiles);

                        // Wait for reader to fill V, forward this writer's row slice over fabric
                        cb_wait_front(cb_v_writer_in, v_chunk_tiles);
                        if (!is_last_ring_iter) {
                        if (!kv_chunk_is_joint) {
                            const uint32_t base_v_read_ptr = get_read_ptr(cb_v_writer_in);
                            for (uint32_t row = my_row_start; row < my_row_end; ++row) {
                                if (kv_slice.d2_start + row >= end_seq_tile) break;
                                for (uint32_t col = 0; col < DHt; col += ag_packet_size_in_pages) {
                                    uint32_t tiles_in_batch = 0;
                                    uint64_t v_noc_addrs[4] = {0, 0, 0, 0};
                                    for (uint32_t i = 0; i < ag_packet_size_in_pages && col + i < DHt; i++) {
                                        const uint32_t dest_id = bh_offset
                                            + (kv_global_start_tile + row) * ag_output_Wt + col + i;
                                        v_noc_addrs[tiles_in_batch] =
                                            tt::tt_fabric::linear::addrgen_detail::get_noc_address(
                                                gathered_v_writer, dest_id, 0);
                                        tiles_in_batch++;
                                    }
                                    if (tiles_in_batch == 0) break;
                                    const uint32_t src_l1_addr = base_v_read_ptr
                                        + (row * DHt + col) * ag_page_size;
                                    if (tiles_in_batch > 1) {
                                        uint16_t v_cs[3] = {
                                            static_cast<uint16_t>(ag_page_size),
                                            static_cast<uint16_t>(ag_page_size),
                                            static_cast<uint16_t>(ag_page_size)};
                                        fabric_unicast_noc_scatter_write_with_state<
                                            UnicastScatterWriteUpdateMask::DstAddrs |
                                            UnicastScatterWriteUpdateMask::ChunkSizes |
                                            UnicastScatterWriteUpdateMask::PayloadSize>(
                                            &mux_conn, pkt_scatter_hdr, src_l1_addr,
                                            NocUnicastScatterCommandHeader(v_noc_addrs, v_cs, tiles_in_batch),
                                            ag_page_size * tiles_in_batch);
                                    } else {
                                        fabric_unicast_noc_unicast_write_with_state<UnicastWriteUpdateMask::DstAddr>(
                                            &mux_conn, pkt_unicast_hdr, src_l1_addr,
                                            NocUnicastCommandHeader{v_noc_addrs[0]});
                                    }
                                    noc_async_write_barrier();
                                }
                            }
                            noc_async_writes_flushed();
                        }
                        }
                        cb_pop_front(cb_v_writer_in, v_chunk_tiles);

                        if (!is_last_ring_iter) {
                            fabric_unicast_noc_unicast_atomic_inc_with_state(&mux_conn, pkt_hdr_sem_inc);
                            noc_async_writes_flushed();
                        }
                    }

                    if (KV_chunks_processed_in_iter % 2 == 0) {
                        cb_wait_front(cb_k_writer_in, k_chunk_tiles);
                        cb_wait_front(cb_v_writer_in, v_chunk_tiles);
                        cb_pop_front(cb_k_writer_in, k_chunk_tiles);
                        cb_pop_front(cb_v_writer_in, v_chunk_tiles);
                    }
                }
#endif

                if (is_last_ring_iter) {
                    write_block(
                        qi.is_joint_q ? joint_out_generator : out_generator,
                        qi.out_slice,
                        qi.end_seq_tile,
                        cb_out,
                        tile_bytes);
                } else if (!single_q_chunk) {
                    write_accumulators(
                        qi.is_joint_q ? joint_out_generator : out_generator,
                        stats_writer,
                        stats_tile_logical,
                        nb,
                        nq,
                        Sq_chunk_t,
                        qi.out_slice,
                        qi.end_seq_tile,
                        qi.stats_seq_start_tile,
                        qi.stats_seq_end_tile,
                        sum_offset,
                        cb_out,
                        cb_max_out,
                        cb_sum_out,
                        tile_bytes,
                        stats_tile_bytes);
                }
            }
            noc_async_write_barrier();
        } else {
            for (uint32_t global_q_chunk = global_q_start; global_q_chunk < global_q_end; ++global_q_chunk) {
                // global_q_chunk is index into `B * NH * num_q_chunks`. Need to get nb, nq, q_chunk from this.
                const uint32_t nb = global_q_chunk / (NH * num_q_chunks);
                const uint32_t nq = (global_q_chunk % (NH * num_q_chunks)) / num_q_chunks;
                const uint32_t q_chunk = global_q_chunk % num_q_chunks;

                const auto qi = get_q_chunk_info(
                    q_chunk, nb, nq, ring_id, num_local_q_chunks, Sq_chunk_t, DHt, Lt, local_padded_Nt);

                // If not on the first iteration, read LSE and previous output chunk.
                // No race condition because writer kernel writes previous output before reading it again
                if (ring_iter > 0) {
                    read_prev_output_and_lse(
                        qi.is_joint_q ? joint_out_generator : out_generator,
                        stats_writer,
                        stats_tile_logical,
                        nb,
                        nq,
                        Sq_chunk_t,
                        qi.out_slice,
                        qi.end_seq_tile,
                        qi.stats_seq_start_tile,
                        qi.stats_seq_end_tile,
                        cb_prev_out,
                        cb_lse_in,
                        tile_bytes,
                        stats_tile_bytes);
                }

                write_output_and_lse(
                    qi.is_joint_q ? joint_out_generator : out_generator,
                    stats_writer,
                    stats_tile_logical,
                    nb,
                    nq,
                    Sq_chunk_t,
                    qi.out_slice,
                    qi.end_seq_tile,
                    qi.stats_seq_start_tile,
                    qi.stats_seq_end_tile,
                    cb_out,
                    cb_lse_out,
                    tile_bytes,
                    stats_tile_bytes);
            }
            noc_async_write_barrier();  // Ensure writes of output and LSE complete before next iteration
        }
    }

#ifdef USE_MUX
    if (mux_connection_valid) {
        DPRINT << "Disconnecting from mux" << ENDL();
        tt::tt_fabric::fabric_client_disconnect(mux_conn);
        if (is_termination_master) {
            DPRINT << "Waiting for termination sync" << ENDL();
            auto* termination_sync_ptr = reinterpret_cast<volatile tt_l1_ptr uint32_t*>(termination_sync_sem_addr);
            noc_semaphore_wait(termination_sync_ptr, num_mux_clients - 1);
            tt::tt_fabric::fabric_endpoint_terminate(fabric_mux_x, fabric_mux_y, fabric_mux_termination_signal_address);
        } else {
            DPRINT << "Sending termination signal" << ENDL();
            uint64_t dest_addr =
                get_noc_addr(termination_master_noc_x, termination_master_noc_y, termination_sync_sem_addr);
            noc_semaphore_inc(dest_addr, 1);
            noc_async_atomic_barrier();
        }
    }
    DPRINT << "Mux done" << ENDL();
#endif
}
