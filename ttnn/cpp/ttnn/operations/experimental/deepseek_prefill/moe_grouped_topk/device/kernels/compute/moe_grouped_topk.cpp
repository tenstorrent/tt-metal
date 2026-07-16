// SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

// Grouped-topk gate compute: activation -> add_bias -> per-group sort -> group top-k -> expert top-k ->
// normalize -> scale. All stages live in the shared moe_gate_common_compute.hpp blocks, which are also
// used by moe_hash_gate so the activation/normalize/scale logic exists in exactly one place.
#include "ttnn/operations/experimental/deepseek_prefill/moe_grouped_topk/device/kernels/compute/moe_gate_common_compute.hpp"

void kernel_main() {
    // Circular buffer indices
    constexpr uint32_t cb_in_scores = get_named_compile_time_arg_val("cb_in_scores");
    constexpr uint32_t cb_in_bias = get_named_compile_time_arg_val("cb_in_bias");
    constexpr uint32_t cb_sigmoid_scores = get_named_compile_time_arg_val("cb_sigmoid_scores");
    constexpr uint32_t cb_biased_scores = get_named_compile_time_arg_val("cb_biased_scores");
    constexpr uint32_t cb_out_weights = get_named_compile_time_arg_val("cb_out_weights");
    constexpr uint32_t cb_out_indices = get_named_compile_time_arg_val("cb_out_indices");
    constexpr uint32_t width_tiles = get_named_compile_time_arg_val("width_tiles");
    constexpr uint32_t scores_page_size = get_named_compile_time_arg_val("scores_page_size");
    constexpr uint32_t bias_page_size = get_named_compile_time_arg_val("bias_page_size");
    constexpr uint32_t cb_sorted_group_scores = get_named_compile_time_arg_val("cb_sorted_group_scores");
    constexpr uint32_t cb_sorted_expert_indices_temp = get_named_compile_time_arg_val("cb_sorted_expert_indices_temp");
    constexpr uint32_t cb_expert_index_template = get_named_compile_time_arg_val("cb_expert_index_template");
    constexpr uint32_t cb_winning_group_scores = get_named_compile_time_arg_val("cb_winning_group_scores");
    constexpr uint32_t cb_winning_group_indices = get_named_compile_time_arg_val("cb_winning_group_indices");

    constexpr uint32_t log_group_size = get_named_compile_time_arg_val("log_group_size");
    constexpr uint32_t group_size = get_named_compile_time_arg_val("group_size");
    constexpr uint32_t log_topk_groups = get_named_compile_time_arg_val("log_topk_groups");
    constexpr uint32_t topk_groups = get_named_compile_time_arg_val("topk_groups");
    constexpr uint32_t n_activated_experts = get_named_compile_time_arg_val("n_activated_experts");
    constexpr uint32_t n_activated_expert_tiles = get_named_compile_time_arg_val("n_activated_expert_tiles");

    constexpr uint32_t cb_group_index_template = get_named_compile_time_arg_val("cb_group_index_template");
    constexpr uint32_t cb_top_experts_per_group = get_named_compile_time_arg_val("cb_top_experts_per_group");
    constexpr uint32_t cb_group_summed_scores = get_named_compile_time_arg_val("cb_group_summed_scores");
    constexpr uint32_t summed_experts_per_group = get_named_compile_time_arg_val("summed_experts_per_group");
    constexpr uint32_t cb_sorted_group_order = get_named_compile_time_arg_val("cb_sorted_group_order");
    constexpr uint32_t cb_reduce_intermediate = get_named_compile_time_arg_val("cb_reduce_intermediate");
    constexpr uint32_t cb_final_indices_transposed = get_named_compile_time_arg_val("cb_final_indices_transposed");
    constexpr uint32_t cb_reduce_ones_scalar = get_named_compile_time_arg_val("cb_reduce_ones_scalar");
    constexpr uint32_t cb_epsilon_scalar = get_named_compile_time_arg_val("cb_epsilon_scalar");
    constexpr uint32_t cb_route_scale_scalar = get_named_compile_time_arg_val("cb_route_scale_scalar");
    constexpr uint32_t cb_normalized_scores = get_named_compile_time_arg_val("cb_normalized_scores");
    constexpr uint32_t cb_reciprocal_sums = get_named_compile_time_arg_val("cb_reciprocal_sums");
    constexpr uint32_t cb_gathered_sigmoid = get_named_compile_time_arg_val("cb_gathered_sigmoid");

    constexpr uint32_t n_groups = get_named_compile_time_arg_val("n_groups");
    constexpr uint32_t log_n_groups = get_named_compile_time_arg_val("log_n_groups");
    constexpr uint32_t log_width_tiles = get_named_compile_time_arg_val("log_width_tiles");
    constexpr bool stable_sort = get_named_compile_time_arg_val("stable_sort") != 0;
    constexpr uint32_t score_func = get_named_compile_time_arg_val("score_func");

    constexpr uint32_t end_phase = log_group_size - 1;

    const uint32_t start_height_tile = get_arg_val<uint32_t>(0);
    const uint32_t end_height_tile = get_arg_val<uint32_t>(1);
    binary_op_init_common(cb_in_scores, cb_in_bias, cb_biased_scores);

    for (uint32_t height_tile = start_height_tile; height_tile < end_height_tile; height_tile++) {
        blocks::apply_score_func<score_func>(cb_in_scores, cb_sigmoid_scores, width_tiles);

        // Perform add bias on activated scores
        blocks::add_bias(cb_sigmoid_scores, cb_in_bias, cb_biased_scores, width_tiles);
        // Note: cb_sigmoid_scores is NOT popped here - writer will pop it after gather

        if constexpr (n_groups == 1) {
            // Single expert group: grouping is a no-op, so select the top-k directly over the full
            // expert axis. blocks::topk is a general cross-tile top-k; feed it all width_tiles of
            // biased scores together with the identity expert-index template (0..experts-1).
            blocks::topk<stable_sort>(
                cb_biased_scores,
                cb_expert_index_template,
                cb_final_indices_transposed,
                cb_out_indices,
                width_tiles,
                log_width_tiles,
                n_activated_experts);
        } else {
            // Transpose tiles into dest and then perform topk_local_sort
            blocks::process_and_sort_tiles<stable_sort>(
                cb_biased_scores,
                cb_expert_index_template,
                cb_sorted_group_scores,
                cb_sorted_expert_indices_temp,
                width_tiles,
                false,
                false,
                end_phase);
            blocks::sum_top_experts_per_group(
                cb_top_experts_per_group, cb_group_summed_scores, summed_experts_per_group);
            blocks::topk_group_scores<stable_sort>(
                cb_group_summed_scores, cb_group_index_template, cb_sorted_group_order, false, false, log_n_groups - 1);
            blocks::topk<stable_sort>(
                cb_winning_group_scores,
                cb_winning_group_indices,
                cb_final_indices_transposed,
                cb_out_indices,
                topk_groups,
                log_topk_groups,
                n_activated_experts);
        }
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
