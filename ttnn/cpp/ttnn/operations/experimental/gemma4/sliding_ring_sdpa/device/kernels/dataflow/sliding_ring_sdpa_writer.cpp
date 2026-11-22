// SPDX-FileCopyrightText: © 2025 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

// Gemma4 sliding ring SDPA writer (compute: sliding_ring_sdpa.cpp, the precision recipes). It generates the recipe's
// reduce scaler and column identity, then writes each Q chunk's normalized output, in the row groups compute packs it:
// every Q chunk normalizes in its one sliding pass, so there is no state to save or restore.

// Size-optimized: the dataflow kernels leave the kernel config buffer to the recipe compute.
#pragma GCC optimize("Os")

#include <type_traits>

#include "api/dataflow/dataflow_api.h"
#include "api/dataflow/noc.h"
#include "api/dataflow/circular_buffer.h"
#include "ttnn/kernel/dataflow/generate_bcast_scalar.hpp"
#include "ttnn/cpp/ttnn/kernel_lib/reduce_helpers_dataflow.hpp"
#include "ttnn/operations/transformer/sdpa/device/kernels/dataflow/dataflow_common.hpp"
#include "ttnn/operations/transformer/sdpa/device/kernels/recipe_ring_cbs.hpp"
#include "ttnn/operations/experimental/gemma4/sliding_ring_sdpa/device/kernels/dataflow/sliding_ring_sdpa_lane_rows.hpp"
#include "ttnn/operations/transformer/sdpa/device/kernels/dataflow/fused_op_receiver.hpp"
#include "ttnn/operations/experimental/ccl/ring_attention_all_gather_async/device/kernels/ring_attention_rank_mapping.hpp"
#include "ttnn/operations/transformer/sdpa/device/kernels/dataflow/ring_utils.hpp"

void kernel_main() {
    constexpr uint32_t B = get_compile_time_arg_val(0);
    constexpr uint32_t NH = get_compile_time_arg_val(1);
    constexpr uint32_t vDHt = get_compile_time_arg_val(3);
    constexpr uint32_t Sq_chunk_t = get_compile_time_arg_val(4);
    constexpr uint32_t q_local_padded_Nt = get_compile_time_arg_val(6);
    constexpr uint32_t num_joint_q_chunks = get_compile_time_arg_val(12);
    constexpr uint32_t num_q_chunks = get_compile_time_arg_val(15);
    constexpr bool use_zigzag_balancing = get_compile_time_arg_val(24) == 1;
    // The output rows compute packs per row group (two rows, one for an odd Q chunk).
    constexpr uint32_t out_row_group_h = get_compile_time_arg_val(25);
    constexpr uint32_t sliding_window_size = get_compile_time_arg_val(28);
    constexpr uint32_t full_mesh_rank_mapping = get_compile_time_arg_val(32);
    constexpr auto snake_orientation = static_cast<ttnn::ccl::snake_ring::Orientation>(get_compile_time_arg_val(33));
    constexpr uint32_t mesh_rows = get_compile_time_arg_val(34);
    constexpr uint32_t mesh_cols = get_compile_time_arg_val(35);
    constexpr uint32_t kv_stripe_split = get_named_compile_time_arg_val("kv_stripe_split");
    constexpr uint32_t num_lanes = get_named_compile_time_arg_val("num_lanes");
    // The host builds this op only for sliding attention without joint tokens.
    static_assert(sliding_window_size > 0 && num_joint_q_chunks == 0);

    // Output accessor at slot 38 (after the ring geometry slots the factory shares with the other kernels).
    constexpr uint32_t kFirstAccessorArgOffset = 38;
    constexpr auto out_args = TensorAccessorArgs<kFirstAccessorArgOffset>();

    uint32_t argidx = 0;
    const uint32_t out_addr = get_arg_val<uint32_t>(argidx++);
    argidx += 2;  // joint output and stats addresses: no joint tokens, no stats
    const uint32_t global_q_start = get_arg_val<uint32_t>(argidx++);
    const uint32_t global_q_end = get_arg_val<uint32_t>(argidx++);
    argidx += 3;  // logical_nt and the ring work masks: compute and the reader own the sliding geometry
    RingSDPAOpReceiver fused_op_receiver = RingSDPAOpReceiver(
        false, /* wait_for_op_signal */
        argidx);

    constexpr uint32_t cb_out = recipe_ring_cb::out;
    constexpr uint32_t tile_bytes = get_tile_size(cb_out);
    Noc noc;

    // Lanes: the output stacks the lanes' rows like Q, [1, NH, num_lanes * rows, DV].
    constexpr bool output_has_no_padding = q_local_padded_Nt % Sq_chunk_t == 0;
    constexpr uint32_t out_batch = num_lanes > 1 ? 1 : B;
    constexpr uint32_t out_rows_t = num_lanes > 1 ? num_lanes * q_local_padded_Nt : q_local_padded_Nt;
    using StaticOutputTileShape = StaticTensorTileShape<out_batch, NH, out_rows_t, vDHt>;
    using OutputTileShape = std::conditional_t<output_has_no_padding, StaticOutputTileShape, TensorTileShape>;
    const auto out_writer = TensorAccessor(out_args, out_addr);
    const auto out_generator = PaddedAddrGenerator(out_writer, OutputTileShape(out_batch, NH, out_rows_t, vDHt));

    // The recipe's reduce scaler and column identity (as the dense recipe reader, reader_recipe.cpp).
    dataflow_kernel_lib::calculate_and_prepare_reduce_scaler<
        recipe_ring_cb::identity_scale,
        ckernel::PoolType::MAX,
        ckernel::ReduceDim::REDUCE_ROW,
        dataflow_kernel_lib::SUM_AND_MAX_REDUCE_FACTOR>();
    generate_bcast_col_scalar_zeroed(CircularBuffer(recipe_ring_cb::col_identity), 0x3f803f80);

    const uint32_t ring_index =
        ttnn::ring_attention_all_gather::tensor_rank_from_transport_rank<full_mesh_rank_mapping>(
            fused_op_receiver.seq.ring_index, mesh_rows, mesh_cols, snake_orientation);
    for (uint32_t q_flat = global_q_start; q_flat < global_q_end; ++q_flat) {
        const auto decoded = decompose_global_q_index(q_flat, num_q_chunks, NH, use_zigzag_balancing);
        const uint32_t row0 = decoded.q_chunk * Sq_chunk_t;
        Slice out_slice(decoded.nb, decoded.nq, row0, row0 + Sq_chunk_t, 0, vDHt);
        uint32_t end_seq_tile = q_local_padded_Nt * (ring_index / kv_stripe_split + 1);
        if constexpr (num_lanes > 1) {
            const auto lane_rows = gemma4_sliding::lane_q_chunk_rows(
                decoded.nb, decoded.nq, decoded.q_chunk, Sq_chunk_t, q_local_padded_Nt, vDHt);
            out_slice = lane_rows.slice;
            end_seq_tile = lane_rows.end_seq_tile;
        }
        write_block_row_grouped_trid<output_has_no_padding>(
            noc, out_generator, out_slice, end_seq_tile, cb_out, tile_bytes, out_row_group_h, /*flush_trid=*/0);
    }
    noc.async_write_barrier();
}
