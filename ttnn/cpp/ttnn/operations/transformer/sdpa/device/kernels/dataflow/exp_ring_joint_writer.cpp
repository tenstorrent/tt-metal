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
#include "cpp/ttnn/operations/ccl/kernel_common/worker_sync_utils.hpp"
#include "cpp/ttnn/operations/ccl/ccl_host_types.hpp"
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
    // push_ring_sdpa_fused_op_rt_args always appends 6 values (4 ring params + 2 semaphore IDs).
    // The writer's RingSDPAOpReceiver reads only the 4 ring params (wait_for_op_signal=false),
    // so skip the 2 trailing semaphore IDs before reading MUX args.
    argidx += 2;

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

    // ---- K/V All-Gather setup (termination master only) ----
    // State is declared in the outer scope so it can be accessed from the ring loop below.
    // Phase 1 (send local slice) runs at ring_iter==0; Phase 2 steps (store-and-forward)
    // run one-per-ring_iter thereafter, interleaved with SDPA output writes to avoid deadlock.
    bool do_ag = false;
    uint32_t ag_direction = 0;
    uint32_t ag_input_Wt = 0, ag_input_Ht = 0, ag_output_Wt = 0, ag_output_Ht = 0;
    uint32_t ag_gather_dim = 0, ag_batch_head_count = 0;
    uint32_t ag_tile_id_start = 0, ag_tile_id_end = 0, ag_ring_size = 0;
    uint32_t ag_writes_expected = 0, ag_slice_writes = 0;
    uint32_t kv_scratch_addr = 0;
    volatile tt_l1_ptr PACKET_HEADER_TYPE* pkt_hdr_write = nullptr;
    volatile tt_l1_ptr PACKET_HEADER_TYPE* pkt_hdr_sem = nullptr;
    volatile tt_l1_ptr uint32_t* out_ready_sem_ptr = nullptr;
    OpSignaler op_signaler;
    uint32_t ag_local_gathered_tile_id_start = 0;
    // DRAM base addresses for K/V input and gathered output; updated inside the if-block.
    // TensorAccessors are constructed as temporaries at the call sites to avoid the
    // non-assignable const member restriction in InterleavedAddrGen.
    uint32_t k_addr_ag_rt = 0, v_addr_ag_rt = 0;
    uint32_t gathered_k_addr_ag_rt = 0, gathered_v_addr_ag_rt = 0;

    if (is_termination_master && mux_connection_valid) {
        // Parse AG RT args
        ag_direction = get_arg_val<uint32_t>(argidx++);
        ag_input_Wt = get_arg_val<uint32_t>(argidx++);
        ag_input_Ht = get_arg_val<uint32_t>(argidx++);
        ag_output_Wt = get_arg_val<uint32_t>(argidx++);
        ag_output_Ht = get_arg_val<uint32_t>(argidx++);
        ag_gather_dim = get_arg_val<uint32_t>(argidx++);
        ag_batch_head_count = get_arg_val<uint32_t>(argidx++);
        ag_tile_id_start = get_arg_val<uint32_t>(argidx++);
        ag_tile_id_end = get_arg_val<uint32_t>(argidx++);
        ag_ring_size = get_arg_val<uint32_t>(argidx++);
        const uint32_t out_ready_sem_id = get_arg_val<uint32_t>(argidx++);
        k_addr_ag_rt = get_arg_val<uint32_t>(argidx++);
        v_addr_ag_rt = get_arg_val<uint32_t>(argidx++);
        gathered_k_addr_ag_rt = get_arg_val<uint32_t>(argidx++);
        gathered_v_addr_ag_rt = get_arg_val<uint32_t>(argidx++);

        op_signaler = OpSignaler(argidx);

        // Allocate L1 scratch for one packet of K or V tiles
        cb_reserve_back(ag_kv_scratch_cb_id, ag_packet_size_in_pages);
        kv_scratch_addr = get_write_ptr(ag_kv_scratch_cb_id);
        cb_push_back(ag_kv_scratch_cb_id, ag_packet_size_in_pages);

        // Allocate two packet header slots: one for data writes, one for sem_inc
        cb_reserve_back(ag_pkt_hdr_cb_id, 1);
        const uint32_t pkt_hdr_write_addr = get_write_ptr(ag_pkt_hdr_cb_id);
        cb_push_back(ag_pkt_hdr_cb_id, 1);
        cb_reserve_back(ag_pkt_hdr_cb_id, 1);
        const uint32_t pkt_hdr_sem_addr = get_write_ptr(ag_pkt_hdr_cb_id);
        cb_push_back(ag_pkt_hdr_cb_id, 1);

        pkt_hdr_write = reinterpret_cast<volatile tt_l1_ptr PACKET_HEADER_TYPE*>(pkt_hdr_write_addr);
        pkt_hdr_sem = reinterpret_cast<volatile tt_l1_ptr PACKET_HEADER_TYPE*>(pkt_hdr_sem_addr);

        // Set routing: always 1 hop in direction (store-and-forward handles multi-hop)
        fabric_set_unicast_route<false>(pkt_hdr_write, 1);
        fabric_set_unicast_route<false>(pkt_hdr_sem, 1);

        // Semaphore on this core that remote senders will atomically increment via fabric
        out_ready_sem_ptr = reinterpret_cast<volatile tt_l1_ptr uint32_t*>(get_semaphore(out_ready_sem_id));

        // Destination for sem_inc packets: same physical NOC coordinates on all chips
        const uint64_t remote_out_ready_sem_noc_addr =
            safe_get_noc_addr(my_x[0], my_y[0], get_semaphore(out_ready_sem_id), 0);
        pkt_hdr_sem->to_noc_unicast_atomic_inc(
            tt::tt_fabric::NocUnicastAtomicIncCommandHeader{remote_out_ready_sem_noc_addr, 1});

        // Tile offset of our chip's local slice in the gathered output tensor
        ag_local_gathered_tile_id_start =
            (ag_gather_dim == 3) ? ag_device_index * ag_input_Wt : ag_device_index * ag_input_Ht * ag_input_Wt;

        // Number of additional slices to relay after the local one (mirrors CCL writer)
        if constexpr (ag_topology == ttnn::ccl::Topology::Ring) {
            ag_writes_expected = (ag_direction == 1) ? (ag_num_targets_backward > 0 ? ag_num_targets_backward - 1 : 0)
                                                     : (ag_num_targets_forward > 0 ? ag_num_targets_forward - 1 : 0);
        } else {
            // Linear
            ag_writes_expected =
                (ag_direction == 1 && ag_num_targets_backward > 0)
                    ? ag_num_targets_forward
                    : ((ag_direction == 0 && ag_num_targets_forward > 0) ? ag_num_targets_backward : 0);
        }

        do_ag = true;
    }

    // Lambda: send all tiles for one local K or V tensor to the next device.
    // Reads from local input DRAM, writes to the gathered output tensor on the next device.
    auto send_local_slice = [&](const auto& input_acc, const auto& gathered_acc) {
        uint32_t gathered_bh_start = ag_local_gathered_tile_id_start;
        uint32_t input_bh_start = 0;
        for (uint32_t bh = 0; bh < ag_batch_head_count; ++bh) {
            uint32_t tiles_read = ag_tile_id_start;
            uint32_t pages_in_row = ag_tile_id_start % ag_input_Wt;
            uint32_t input_row_off = (ag_tile_id_start / ag_input_Wt) * ag_input_Wt;
            uint32_t output_row_off = (ag_tile_id_start / ag_input_Wt) * ag_output_Wt;

            while (tiles_read < ag_tile_id_end) {
                const uint32_t num_pages = std::min(ag_tile_id_end - tiles_read, ag_packet_size_in_pages);

                const uint32_t input_id0 = input_bh_start + input_row_off + pages_in_row;
                const uint32_t gathered_id0 = gathered_bh_start + output_row_off + pages_in_row;
                pages_in_row++;
                if (pages_in_row >= ag_input_Wt) {
                    input_row_off += ag_input_Wt;
                    output_row_off += ag_output_Wt;
                    pages_in_row = 0;
                }

                if constexpr (ag_packet_size_in_pages == 2) {
                    if (num_pages == 2) {
                        const uint32_t input_id1 = input_bh_start + input_row_off + pages_in_row;
                        const uint32_t gathered_id1 = gathered_bh_start + output_row_off + pages_in_row;
                        pages_in_row++;
                        if (pages_in_row >= ag_input_Wt) {
                            input_row_off += ag_input_Wt;
                            output_row_off += ag_output_Wt;
                            pages_in_row = 0;
                        }
                        noc_async_read_tile(input_id0, input_acc, kv_scratch_addr);
                        noc_async_read_tile(input_id1, input_acc, kv_scratch_addr + ag_page_size);
                        noc_async_read_barrier();
                        tt::tt_fabric::linear::to_noc_unicast_scatter_write(
                            ag_page_size, pkt_hdr_write, gathered_id0, gathered_id1, gathered_acc);
                        tt::tt_fabric::fabric_async_write(mux_conn, pkt_hdr_write, kv_scratch_addr, ag_page_size * 2);
                    } else {
                        noc_async_read_tile(input_id0, input_acc, kv_scratch_addr);
                        noc_async_read_barrier();
                        tt::tt_fabric::linear::to_noc_unicast_write(
                            ag_page_size, pkt_hdr_write, gathered_id0, gathered_acc);
                        tt::tt_fabric::fabric_async_write(mux_conn, pkt_hdr_write, kv_scratch_addr, ag_page_size);
                    }
                } else {
                    noc_async_read_tile(input_id0, input_acc, kv_scratch_addr);
                    noc_async_read_barrier();
                    tt::tt_fabric::linear::to_noc_unicast_write(
                        ag_page_size, pkt_hdr_write, gathered_id0, gathered_acc);
                    tt::tt_fabric::fabric_async_write(mux_conn, pkt_hdr_write, kv_scratch_addr, ag_page_size);
                }

                tiles_read += num_pages;
            }
            gathered_bh_start += ag_output_Wt * ag_output_Ht;
            input_bh_start += ag_input_Wt * ag_input_Ht;
        }
    };

    // Lambda: store-and-forward one received slice from gathered DRAM to the next device.
    // slice_tile_id_start: offset of the forwarded chip's slice within the gathered K or V tensor.
    auto forward_slice = [&](const auto& gathered_acc, uint32_t slice_tile_id_start) {
        uint32_t tile_id_start = slice_tile_id_start;
        for (uint32_t bh = 0; bh < ag_batch_head_count; ++bh) {
            uint32_t tiles_read = ag_tile_id_start;
            uint32_t pages_in_row = ag_tile_id_start % ag_input_Wt;
            uint32_t row_offset = (ag_tile_id_start / ag_input_Wt) * ag_output_Wt;

            while (tiles_read < ag_tile_id_end) {
                const uint32_t num_pages = std::min(ag_tile_id_end - tiles_read, ag_packet_size_in_pages);

                const uint32_t gathered_id0 = tile_id_start + row_offset + pages_in_row;

                noc_async_read_tile(gathered_id0, gathered_acc, kv_scratch_addr);
                pages_in_row++;
                if (pages_in_row >= ag_input_Wt) {
                    row_offset += ag_output_Wt;
                    pages_in_row = 0;
                }

                if constexpr (ag_packet_size_in_pages == 2) {
                    if (num_pages == 2) {
                        const uint32_t gathered_id1 = tile_id_start + row_offset + pages_in_row;
                        noc_async_read_tile(gathered_id1, gathered_acc, kv_scratch_addr + ag_page_size);
                        pages_in_row++;
                        if (pages_in_row >= ag_input_Wt) {
                            row_offset += ag_output_Wt;
                            pages_in_row = 0;
                        }
                        noc_async_read_barrier();
                        tt::tt_fabric::linear::to_noc_unicast_scatter_write(
                            ag_page_size, pkt_hdr_write, gathered_id0, gathered_id1, gathered_acc);
                        tt::tt_fabric::fabric_async_write(mux_conn, pkt_hdr_write, kv_scratch_addr, ag_page_size * 2);
                    } else {
                        noc_async_read_barrier();
                        tt::tt_fabric::linear::to_noc_unicast_write(
                            ag_page_size, pkt_hdr_write, gathered_id0, gathered_acc);
                        tt::tt_fabric::fabric_async_write(mux_conn, pkt_hdr_write, kv_scratch_addr, ag_page_size);
                    }
                } else {
                    noc_async_read_barrier();
                    tt::tt_fabric::linear::to_noc_unicast_write(
                        ag_page_size, pkt_hdr_write, gathered_id0, gathered_acc);
                    tt::tt_fabric::fabric_async_write(mux_conn, pkt_hdr_write, kv_scratch_addr, ag_page_size);
                }

                tiles_read += num_pages;
            }
            tile_id_start += ag_output_Wt * ag_output_Ht;
        }
    };
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

        // #ifdef USE_MUX
        //         // Interleaved K/V all-gather step: one phase-1 (ring_iter==0) or phase-2 step per iter.
        //         // This prevents deadlock by ensuring cb_out is drained between AG waits.
        //         if (do_ag) {
        //             if (ring_iter == 0) {
        //                 // Phase 1: send our local K/V slice to the next device
        //                 send_local_slice(
        //                     TensorAccessor(ag_input_k_args, k_addr_ag_rt, ag_page_size),
        //                     TensorAccessor(ag_gathered_k_args, gathered_k_addr_ag_rt, ag_page_size));
        //                 send_local_slice(
        //                     TensorAccessor(ag_input_v_args, v_addr_ag_rt, ag_page_size),
        //                     TensorAccessor(ag_gathered_v_args, gathered_v_addr_ag_rt, ag_page_size));
        //                 noc_async_write_barrier();
        //                 tt::tt_fabric::fabric_atomic_inc(mux_conn, pkt_hdr_sem);
        //                 // Signal SDPA reader that our local slice is available (direction==1 only)
        //                 if (ag_direction == 1) {
        //                     op_signaler.synchronize_workers_and_signal_op(ag_device_index);
        //                 }
        //             } else if (ag_slice_writes < ag_writes_expected) {
        //                 // Phase 2 step: wait for next slice to arrive, signal reader, then forward it
        //                 noc_semaphore_wait_min(out_ready_sem_ptr, ag_slice_writes + 1);

        //                 // Compute which chip's slice just arrived
        //                 int slice_chip_id;
        //                 uint32_t actual_slice_chip_id;
        //                 if (ag_direction == 1) {
        //                     slice_chip_id =
        //                         static_cast<int>(ag_device_index) + static_cast<int>(ag_slice_writes) + 1;
        //                     actual_slice_chip_id = (slice_chip_id >= static_cast<int>(ag_ring_size))
        //                         ? slice_chip_id - ag_ring_size : slice_chip_id;
        //                 } else {
        //                     slice_chip_id =
        //                         static_cast<int>(ag_device_index) - static_cast<int>(ag_slice_writes) - 1;
        //                     actual_slice_chip_id = (slice_chip_id < 0)
        //                         ? ag_ring_size + slice_chip_id : slice_chip_id;
        //                 }

        //                 // Signal SDPA reader that this chip's K/V is now in gathered DRAM
        //                 op_signaler.synchronize_workers_and_signal_op(actual_slice_chip_id);

        //                 const uint32_t slice_tile_id_start = (ag_gather_dim == 3)
        //                     ? actual_slice_chip_id * ag_input_Wt
        //                     : actual_slice_chip_id * ag_input_Ht * ag_input_Wt;

        //                 forward_slice(
        //                     TensorAccessor(ag_gathered_k_args, gathered_k_addr_ag_rt, ag_page_size),
        //                     slice_tile_id_start);
        //                 forward_slice(
        //                     TensorAccessor(ag_gathered_v_args, gathered_v_addr_ag_rt, ag_page_size),
        //                     slice_tile_id_start);
        //                 noc_async_write_barrier();
        //                 tt::tt_fabric::fabric_atomic_inc(mux_conn, pkt_hdr_sem);
        //                 ag_slice_writes++;
        //             }
        //         }
        // #endif

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
        tt::tt_fabric::fabric_client_disconnect(mux_conn);
        if (is_termination_master) {
            auto* termination_sync_ptr = reinterpret_cast<volatile tt_l1_ptr uint32_t*>(termination_sync_sem_addr);
            noc_semaphore_wait(termination_sync_ptr, num_mux_clients - 1);
            tt::tt_fabric::fabric_endpoint_terminate(fabric_mux_x, fabric_mux_y, fabric_mux_termination_signal_address);
        } else {
            uint64_t dest_addr =
                get_noc_addr(termination_master_noc_x, termination_master_noc_y, termination_sync_sem_addr);
            noc_semaphore_inc(dest_addr, 1);
            noc_async_atomic_barrier();
        }
    }
#endif
}
