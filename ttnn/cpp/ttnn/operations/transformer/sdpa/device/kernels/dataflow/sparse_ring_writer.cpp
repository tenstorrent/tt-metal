// SPDX-FileCopyrightText: © 2025 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include <type_traits>

#include "api/dataflow/dataflow_api.h"
#include "api/dataflow/noc.h"
#include "api/dataflow/noc_semaphore.h"
#include "api/dataflow/circular_buffer.h"
#include "api/dataflow/endpoints.h"
#include "api/core_local_mem.h"
#include "ttnn/kernel/dataflow/generate_bcast_scalar.hpp"
#include "ttnn/cpp/ttnn/kernel_lib/reduce_helpers_dataflow.hpp"

// Device-profiler zones around the writer's waits (when built with SPARSE_STEP_PROFILE, under
// TT_METAL_DEVICE_PROFILER=1).
// DEBUG (dev branch): the zones only open inside the profile window (sparse_prof_window.hpp), by pass.
#include "ttnn/operations/transformer/sdpa/device/kernels/sparse_prof_window.hpp"
#define SPARSE_ZONE(name) SPARSE_PROF_ZONE(name, sparse_prof::pass_on)
#include "dataflow_common.hpp"
#include "chunked_prefill_utils.hpp"
#include "ttnn/operations/transformer/sdpa/device/kernels/ring_mla_packing_plan.hpp"
#include "ring_joint_kv_pad_derivation.hpp"
#include "fused_op_receiver.hpp"
#include "ring_utils.hpp"

namespace ring_joint = ttnn::operations::transformer::sdpa::ring_joint;

// Output slice of one local Q chunk (the op has no joint tokens).
inline Slice q_chunk_out_slice(uint32_t q_chunk, uint32_t nb, uint32_t nq, uint32_t Sq_chunk_t, uint32_t DHt) {
    const uint32_t out_row_start_tile = q_chunk * Sq_chunk_t;
    return Slice(nb, nq, out_row_start_tile, out_row_start_tile + Sq_chunk_t, 0, DHt);
}

// Sparse ring attention writer. Compute keeps every resident Q chunk's accumulators in L1 for the whole op, so there
// is no save / restore between ring iterations: this kernel generates the constant tiles (column identity and
// reduce scaler), follows the ring sequence to stay aligned with the reader and the all-gather, and on the
// last active ring iteration of each pass writes that pass's normalized outputs, in Q-chunk order.
void kernel_main() {
    constexpr uint32_t B = get_compile_time_arg_val(0);
    constexpr uint32_t NH = get_compile_time_arg_val(1);
    constexpr uint32_t vDHt = get_compile_time_arg_val(3);
    constexpr uint32_t Sq_chunk_t = get_compile_time_arg_val(4);
    constexpr uint32_t Sk_chunk_t = get_compile_time_arg_val(5);
    constexpr uint32_t q_local_padded_Nt = get_compile_time_arg_val(6);
    constexpr uint32_t kv_local_padded_Nt = get_compile_time_arg_val(7);
    constexpr uint32_t L = get_compile_time_arg_val(10);
    constexpr uint32_t num_q_chunks = get_compile_time_arg_val(15);
    constexpr uint32_t identity_scalar_packed = get_compile_time_arg_val(16);
    constexpr uint32_t ring_size = get_compile_time_arg_val(18);
    constexpr bool use_streaming_compute = get_compile_time_arg_val(21) == 1;
    constexpr bool use_zigzag_balancing = get_compile_time_arg_val(24) == 1;
    constexpr uint32_t out_subblock_h = get_compile_time_arg_val(25);
    constexpr uint32_t chunk_size_t = get_compile_time_arg_val(27);
    constexpr uint32_t kv_stripe_split = get_named_compile_time_arg_val("kv_stripe_split");
    static_assert(kv_stripe_split == 1, "the sparse op does not split KV");
    static_assert(L == 0, "the sparse op has no joint tokens");
    static_assert(use_streaming_compute, "the sparse op runs the streaming compute");
    constexpr uint32_t kv_region_Nt = q_local_padded_Nt / kv_stripe_split;
    static_assert(kv_region_Nt * ring_size == chunk_size_t, "chunk_size_t must span one KV region per ring rank");
    // Slots 32-35: transport-to-tensor rank mapping.
    constexpr bool full_mesh_rank_mapping = get_compile_time_arg_val(32) == 1;
    constexpr auto snake_orientation = static_cast<ttnn::ccl::snake_ring::Orientation>(get_compile_time_arg_val(33));
    constexpr uint32_t mesh_rows = get_compile_time_arg_val(34);
    constexpr uint32_t mesh_cols = get_compile_time_arg_val(35);

    // Slots 28-37 hold scalars; the output accessor starts at slot 38 and the CB block follows it.
    constexpr uint32_t kFirstAccessorArgOffset = 38;
    constexpr auto out_args = TensorAccessorArgs<kFirstAccessorArgOffset>();
    constexpr uint32_t cb_arg_offset = out_args.next_compile_time_args_offset();

    // No Q rotation and no grouped K/V sources.
    static_assert(get_ct_arg<kernel_compile_time_args.size() - 1>() == 0, "no rotated Q split");
    static_assert(GROUPED_KV_SOURCE_COUNT == 1, "no grouped K/V sources");

    uint32_t argidx = 0;
    const uint32_t out_addr = get_arg_val<uint32_t>(argidx++);
    const uint32_t global_q_start = get_arg_val<uint32_t>(argidx++);
    const uint32_t global_q_end = get_arg_val<uint32_t>(argidx++);
    const uint32_t logical_nt = get_arg_val<uint32_t>(argidx++);
    const uint32_t active_ring_iter_mask = get_arg_val<uint32_t>(argidx++);
    argidx++;  // single-valid-K-chunk mask (only the save / restore path needs it)
    RingSDPAOpReceiver fused_op_receiver = RingSDPAOpReceiver(
        false, /* wait_for_op_signal */
        argidx);

    constexpr uint32_t cb_identity_scale_in = get_compile_time_arg_val(cb_arg_offset + 5);
    constexpr uint32_t cb_col_identity = get_compile_time_arg_val(cb_arg_offset + 8);
    constexpr uint32_t cb_out = get_compile_time_arg_val(cb_arg_offset + 13);

    constexpr uint32_t tile_bytes = get_tile_size(cb_out);

    Noc noc;

    const auto out_writer = TensorAccessor(out_args, out_addr);
    constexpr bool output_has_no_padding = q_local_padded_Nt % Sq_chunk_t == 0;
    using StaticOutputTileShape = StaticTensorTileShape<B, NH, q_local_padded_Nt, vDHt>;
    using OutputTileShape = std::conditional_t<output_has_no_padding, StaticOutputTileShape, TensorTileShape>;
    const auto output_tile_logical = OutputTileShape(B, NH, q_local_padded_Nt, vDHt);
    const auto out_generator = PaddedAddrGenerator(out_writer, output_tile_logical);

    {
        // Zero the identity tile first: generate_bcast_col_scalar writes column 0 only, and the compute step
        // multiplies its probabilities by this tile to get row sums, which reads every column of it.
        Noc noc_zero;
        fill_tile_zeros<get_tile_size(cb_col_identity)>(noc_zero, cb_col_identity, 0);
    }
    generate_bcast_col_scalar(CircularBuffer(cb_col_identity), identity_scalar_packed);
    dataflow_kernel_lib::calculate_and_prepare_reduce_scaler<
        cb_identity_scale_in,
        ckernel::PoolType::MAX,
        ckernel::ReduceDim::REDUCE_ROW,
        dataflow_kernel_lib::SUM_AND_MAX_REDUCE_FACTOR>();

    auto arrival_preview = fused_op_receiver.seq;
    const ring_joint::PackedKVSchedule packed_schedule = ring_joint::packed_kv_schedule(
        GROUPED_KV_SOURCE_COUNT,
        ring_size,
        kv_local_padded_Nt,
        logical_nt,
        active_ring_iter_mask,
        kv_region_Nt,
        Sk_chunk_t,
        /*has_sliding_window=*/false,
        [&] {
            return ttnn::ring_attention_all_gather::tensor_rank_from_transport_rank<full_mesh_rank_mapping>(
                arrival_preview.get_next_ring_id([](uint32_t, uint32_t) {}), mesh_rows, mesh_cols, snake_orientation);
        });
    const uint32_t sdpa_ring_iterations = packed_schedule.ring_iterations;
    // Passes split the core's Q blocks evenly (see the reader); each replays the ring order from the start (the
    // writer never waits on the all-gather, so a fresh copy of the receiver is enough).
    constexpr uint32_t num_q_passes = get_named_compile_time_arg_val("num_q_passes");
    const uint32_t q_per_core = global_q_end - global_q_start;
    const RingSDPAOpReceiver receiver_start = fused_op_receiver;
    for (uint32_t q_pass = 0; q_pass < num_q_passes; ++q_pass) {
        const uint32_t pass_q_start = global_q_start + (q_pass * q_per_core) / num_q_passes;
        const uint32_t pass_q_end = global_q_start + ((q_pass + 1) * q_per_core) / num_q_passes;
        RingSDPAOpReceiver pass_receiver = receiver_start;
        for (uint32_t ring_iter = 0; ring_iter < sdpa_ring_iterations; ++ring_iter) {
            sparse_prof::begin_iter(q_pass, ring_iter);  // DEBUG: profile window
            // The ring-id sequence advances on every iteration so the writer stays aligned with the reader, compute
            // and the all-gather; only the pass's last active iteration writes.
            const uint32_t ring_id =
                ttnn::ring_attention_all_gather::tensor_rank_from_transport_rank<full_mesh_rank_mapping>(
                    pass_receiver.get_next_ring_id_and_sync(), mesh_rows, mesh_cols, snake_orientation);
            if (((active_ring_iter_mask >> ring_iter) & 1u) == 0 ||
                !is_last_active_ring_iter(active_ring_iter_mask, ring_iter)) {
                continue;
            }
            SPARSE_ZONE("WR_RES_OUT");  // cb_out waits (compute's final normalization) plus the output writes
            const uint32_t end_seq_tile = q_local_padded_Nt * (ring_id + 1);
            for (uint32_t q_flat = pass_q_start; q_flat < pass_q_end; ++q_flat) {
                const auto decoded_q = decompose_global_q_index(q_flat, num_q_chunks, NH, use_zigzag_balancing);
                write_block_row_grouped_trid<output_has_no_padding>(
                    noc,
                    out_generator,
                    q_chunk_out_slice(decoded_q.q_chunk, decoded_q.nb, decoded_q.nq, Sq_chunk_t, vDHt),
                    end_seq_tile,
                    cb_out,
                    tile_bytes,
                    out_subblock_h,
                    /*flush_trid=*/0);
            }
            noc.async_write_barrier();
        }
    }
}
