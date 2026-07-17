// SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

// Hash-gate compute: activation -> normalize -> scale. Expert selection (cb_out_indices) is produced
// by the reader (tid2eid[input_ids] lookup), and the unbiased-score gather runs in the writer, so the
// compute kernel only needs the shared apply_score_func / normalize_scores / scale blocks.
#include "ttnn/operations/experimental/deepseek_prefill/moe_grouped_topk/device/kernels/compute/moe_gate_common_compute.hpp"

void kernel_main() {
    constexpr uint32_t cb_in_scores = get_named_compile_time_arg_val("cb_in_scores");
    constexpr uint32_t cb_sigmoid_scores = get_named_compile_time_arg_val("cb_sigmoid_scores");
    constexpr uint32_t cb_out_weights = get_named_compile_time_arg_val("cb_out_weights");
    constexpr uint32_t width_tiles = get_named_compile_time_arg_val("width_tiles");
    constexpr uint32_t cb_reduce_intermediate = get_named_compile_time_arg_val("cb_reduce_intermediate");
    constexpr uint32_t cb_reduce_ones_scalar = get_named_compile_time_arg_val("cb_reduce_ones_scalar");
    constexpr uint32_t cb_epsilon_scalar = get_named_compile_time_arg_val("cb_epsilon_scalar");
    constexpr uint32_t cb_route_scale_scalar = get_named_compile_time_arg_val("cb_route_scale_scalar");
    constexpr uint32_t cb_normalized_scores = get_named_compile_time_arg_val("cb_normalized_scores");
    constexpr uint32_t cb_reciprocal_sums = get_named_compile_time_arg_val("cb_reciprocal_sums");
    constexpr uint32_t cb_gathered_sigmoid = get_named_compile_time_arg_val("cb_gathered_sigmoid");
    constexpr uint32_t score_func = get_named_compile_time_arg_val("score_func");

    const uint32_t start_height_tile = get_arg_val<uint32_t>(0);
    const uint32_t end_height_tile = get_arg_val<uint32_t>(1);
    binary_op_init_common(cb_in_scores, cb_reduce_ones_scalar, cb_out_weights);

    for (uint32_t height_tile = start_height_tile; height_tile < end_height_tile; height_tile++) {
        // Activation over all experts. cb_sigmoid_scores is NOT popped here - the writer pops it after gather.
        blocks::apply_score_func<score_func>(cb_in_scores, cb_sigmoid_scores, width_tiles);

        // Weights = scale * normalize(gather(activated_scores, hash_indices)). The gather (writer) fills
        // cb_gathered_sigmoid; normalize_scores waits on it.
        blocks::normalize_scores(
            cb_gathered_sigmoid,
            cb_reduce_ones_scalar,
            cb_reduce_intermediate,
            cb_reciprocal_sums,
            cb_epsilon_scalar,
            cb_normalized_scores);
        blocks::scale(cb_normalized_scores, cb_route_scale_scalar, cb_out_weights);
    }
}
