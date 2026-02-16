// SPDX-FileCopyrightText: © 2024 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <cstdint>
#include <climits>

#include "ttnn/device_operation.hpp"

#include "sdpa_decode_device_operation_types.hpp"

namespace ttnn::prim {

/******************************************************************************
 *                   Tree Reduction Helpers (Host-side)                       *
 ******************************************************************************/

// Supports up to 2^MAX_TREE_REDUCTION_ROUNDS cores per reduction group.
constexpr uint32_t MAX_TREE_REDUCTION_ROUNDS = 6;

inline uint32_t count_trailing_zeros(uint32_t n) {
    if (n == 0) {
        return 32;
    }
    uint32_t count = 0;
    while ((n & 1) == 0) {
        n >>= 1;
        count++;
    }
    return count;
}

inline uint32_t ceil_log2(uint32_t n) {
    if (n <= 1) {
        return 0;
    }
    uint32_t log = 0;
    n--;
    while (n > 0) {
        n >>= 1;
        log++;
    }
    return log;
}

/**
 * Per-core parameters for binary tree reduction.
 *
 * Example for 8 cores (core 0 is root):
 *   Round 0: 1←0, 3←2, 5←4, 7←6
 *   Round 1: 3←1, 7←5
 *   Round 2: 7←3   (core 0 is root)
 */
struct TreeReductionParams {
    uint32_t num_rounds;                                     // ceil(log2(num_cores))
    uint32_t my_active_rounds;                               // rounds this core receives children
    bool is_root;                                            // final reducer
    uint32_t parent_core_in_group;                           // UINT32_MAX if root
    uint32_t send_at_round;                                  // UINT32_MAX if root
    uint32_t children_per_round[MAX_TREE_REDUCTION_ROUNDS];  // UINT32_MAX if no child at round
    uint32_t num_children;                                   // total children across all rounds
};

inline TreeReductionParams get_tree_reduction_params(uint32_t core_id_in_group, uint32_t num_cores_in_group) {
    TreeReductionParams params{};
    // Virtual Core ID mapping — core 0 of the group is always the root.
    uint32_t vid = (num_cores_in_group - 1) - core_id_in_group;

    params.num_rounds = ceil_log2(num_cores_in_group);
    params.is_root = false;
    params.parent_core_in_group = UINT32_MAX;
    params.send_at_round = UINT32_MAX;
    params.num_children = 0;
    params.my_active_rounds = 0;
    for (uint32_t r = 0; r < MAX_TREE_REDUCTION_ROUNDS; r++) {
        params.children_per_round[r] = UINT32_MAX;
    }

    if (num_cores_in_group <= 1) {
        params.is_root = true;
        return params;
    }

    uint32_t trailing_ones = count_trailing_zeros(~vid);
    uint32_t root_vid = num_cores_in_group - 1;

    // Receiver phase
    for (uint32_t r = 0; r < params.num_rounds; r++) {
        uint32_t step = 1u << r;
        uint32_t mask = (step << 1) - 1;
        if ((vid & mask) == mask) {
            uint32_t child_vid = vid - step;
            if (child_vid < num_cores_in_group) {
                params.children_per_round[r] = (num_cores_in_group - 1) - child_vid;
                params.num_children++;
            }
            params.my_active_rounds = r + 1;
        }
    }

    // Sender phase
    if (vid == root_vid) {
        params.is_root = true;
    } else {
        uint32_t step = 1u << trailing_ones;
        uint32_t parent_vid = vid + step;
        if (parent_vid < num_cores_in_group) {
            params.parent_core_in_group = (num_cores_in_group - 1) - parent_vid;
            params.send_at_round = trailing_ones;
        } else {
            // Non-power-of-two: this core becomes a secondary root.
            params.is_root = true;
        }
    }
    return params;
}

struct SdpaDecodeProgramFactory {
    struct shared_variables_t {
        uint32_t num_active_cores = 0;
        std::vector<CoreCoord> core_group;
        tt::tt_metal::KernelHandle reader_kernels_id{};
        tt::tt_metal::KernelHandle writer_kernels_id{};
        tt::tt_metal::KernelHandle compute_kernels_id{};
        uint32_t num_cores_per_batch = 0;
        uint32_t num_cores_per_head = 0;
        uint32_t num_output_cores = 0;
        tt::tt_metal::CBHandle cb_in8_id{};
        tt::tt_metal::CBHandle cb_in9_id{};
        bool is_output_sharded = false;
        tt::tt_metal::CBHandle cb_out4_id{};
        uint32_t B = 0;
        uint32_t q_heads_parallel_factor = 0;
        bool use_cur_pos_tensor = false;
        bool use_attention_mask = false;
        bool use_attention_sink = false;
        bool is_paged_attention = false;
        bool is_causal = false;
        bool use_mla = false;
    };

    using cached_program_t = ttnn::device_operation::CachedProgram<shared_variables_t>;

    static cached_program_t create(
        const SdpaDecodeParams& operation_attributes, const SdpaDecodeInputs& tensor_args, Tensor& tensor_return_value);

    static void override_runtime_arguments(
        cached_program_t& cached_program,
        const SdpaDecodeParams& operation_attributes,
        const SdpaDecodeInputs& tensor_args,
        Tensor& tensor_return_value);
};

}  // namespace ttnn::prim
