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

constexpr uint32_t MAX_TREE_REDUCTION_ROUNDS = 6;  // Supports up to 2^6 = 64 cores

struct TreeReductionParams {
    uint32_t num_rounds;                                     // ceil(log2(num_cores))
    uint32_t my_active_rounds;                               // rounds this core receives children
    bool is_root;                                            // final reducer (core 0)
    uint32_t parent_core_in_group;                           // UINT32_MAX if root
    uint32_t send_at_round;                                  // UINT32_MAX if root
    uint32_t children_per_round[MAX_TREE_REDUCTION_ROUNDS];  // UINT32_MAX if no child
    uint32_t num_children;
};

// Binary tree reduction: core 0 is root, vid = (N-1) - core_id for tree structure
inline TreeReductionParams get_tree_reduction_params(uint32_t core_id, uint32_t N) {
    TreeReductionParams p{};
    std::fill_n(p.children_per_round, MAX_TREE_REDUCTION_ROUNDS, UINT32_MAX);
    p.parent_core_in_group = p.send_at_round = UINT32_MAX;

    if (N <= 1) {
        p.is_root = true;
        return p;
    }

    uint32_t vid = (N - 1) - core_id;
    p.num_rounds = 32 - __builtin_clz(N - 1);  // ceil(log2(N))
    p.is_root = (core_id == 0);

    // Find children: at round r, vid receives from (vid - 2^r) if vid's low (r+1) bits are all 1
    for (uint32_t r = 0; r < p.num_rounds; r++) {
        uint32_t mask = (2u << r) - 1;
        if ((vid & mask) == mask) {
            uint32_t child_vid = vid - (1u << r);
            if (child_vid < N) {
                p.children_per_round[r] = (N - 1) - child_vid;
                p.num_children++;
                p.my_active_rounds = r + 1;
            }
        }
    }

    // Find parent: send at round = trailing 1s in vid; parent_vid = vid + 2^round
    if (!p.is_root) {
        uint32_t trailing_ones = __builtin_ctz(~vid);
        uint32_t parent_vid = vid + (1u << trailing_ones);
        // If parent_vid >= N (non-power-of-2), orphan sends to root (core 0)
        p.parent_core_in_group = (parent_vid < N) ? (N - 1) - parent_vid : 0;
        p.send_at_round = trailing_ones;
    }

    // Root collects orphans: cores whose natural parent_vid >= N
    if (p.is_root) {
        for (uint32_t c = 1; c < N; c++) {
            uint32_t cv = (N - 1) - c;
            uint32_t t = __builtin_ctz(~cv);
            if (cv + (1u << t) >= N && p.children_per_round[t] == UINT32_MAX) {
                p.children_per_round[t] = c;
                p.num_children++;
                p.my_active_rounds = std::max(p.my_active_rounds, t + 1);
            }
        }
    }
    return p;
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
