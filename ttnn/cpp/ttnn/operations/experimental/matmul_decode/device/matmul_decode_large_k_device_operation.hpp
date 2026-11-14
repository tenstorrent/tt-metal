// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <cstdint>
#include <optional>
#include <variant>
#include <vector>

#include "ttnn/tensor/tensor.hpp"
#include "ttnn/core.hpp"
#include "ttnn/device_operation.hpp"
#include "ttnn/types.hpp"
#include <tt-metalium/program_descriptors.hpp>

namespace ttnn::operations::experimental::matmul_decode {

// Depth of the reduction tree over num_cores leaves with the given fan-in; each level uses one
// semaphore.
inline uint32_t large_k_tree_num_levels(uint32_t num_cores, uint32_t fan_in) {
    uint32_t levels = 0;
    for (uint64_t stride = 1; stride < num_cores; stride *= fan_in) {
        ++levels;
    }
    return levels;
}

// One core's place in the reduction tree. Cores are numbered in A's row-major shard order, so
// core i holds K-slice i. At level l (stride s = fan_in^l) every core whose index is a multiple of
// s * fan_in collects the partials of cores i + j*s (j = 1 .. fan_in-1); everyone else at that
// level sends its running sum to its parent and drops out. Core 0 is the root.
struct LargeKTreeNode {
    int parent = -1;
    uint32_t parent_level = 0;
    // Where this core's partial lands in the parent's receive CB, in partial-sized slots. Slots are
    // numbered level-major, the order in which the parent consumes them.
    uint32_t slot_in_parent = 0;
    std::vector<uint32_t> children_per_level;
    uint32_t num_children = 0;
};

inline std::vector<LargeKTreeNode> build_large_k_tree(uint32_t num_cores, uint32_t fan_in, uint32_t num_levels) {
    std::vector<LargeKTreeNode> nodes(num_cores);
    for (auto& node : nodes) {
        node.children_per_level.assign(num_levels, 0);
    }
    uint64_t stride = 1;
    for (uint32_t level = 0; level < num_levels; ++level, stride *= fan_in) {
        const uint64_t group = stride * fan_in;
        for (uint64_t i = 0; i < num_cores; i += group) {
            for (uint32_t j = 1; j < fan_in; ++j) {
                const uint64_t child = i + j * stride;
                if (child >= num_cores) {
                    break;
                }
                auto& parent = nodes[i];
                nodes[child].parent = static_cast<int>(i);
                nodes[child].parent_level = level;
                nodes[child].slot_in_parent = parent.num_children;
                parent.children_per_level[level]++;
                parent.num_children++;
            }
        }
    }
    return nodes;
}

// Decode matmul for large K / small N. Core i of A's grid holds the i-th K-slice of both
// operands, so every core computes its [M, N] partial locally (no activation gather) and the
// partials are summed up a fan-in tree whose root is the grid's first core.
struct MatmulDecodeLargeKDeviceOperation {
    struct operation_attributes_t {
        int M;
        int N;
        int K;
        DataType output_dtype;
        // Children per tree node. 2 is a binary tree; a wider fan-in trades fewer serial hops
        // for more partials summed per node.
        uint32_t reduce_fan_in = 2;
        // Unweighted RMSNorm of each row of A over the full K, applied before the multiply:
        // C = (A / sqrt(mean(A^2) + eps)) @ B. The per-row scale commutes with the matmul, so
        // each core folds its share of sum(A^2) into the tree reduction and the root applies it.
        bool rms_norm = false;
        float rms_norm_epsilon = 1.0e-6F;
        // Consecutive rows of A that make up one logical row: A [M * fold_rows, K / fold_rows] is
        // consumed as [M, K], core i's K-slice being its whole [fold_rows, Kc / fold_rows] shard.
        uint32_t fold_rows = 1;
    };

    struct tensor_args_t {
        const Tensor& input_tensor_a;
        const Tensor& input_tensor_b;
    };

    using spec_return_value_t = tt::tt_metal::TensorSpec;
    using tensor_return_value_t = Tensor;

    struct TreeReduce {
        static tt::tt_metal::ProgramDescriptor create_descriptor(
            const operation_attributes_t& operation_attributes,
            const tensor_args_t& tensor_args,
            tensor_return_value_t& tensor_return_value);
    };

    using program_factory_t = std::variant<TreeReduce>;

    static program_factory_t select_program_factory(const operation_attributes_t&, const tensor_args_t&);

    static void validate_on_program_cache_miss(const operation_attributes_t&, const tensor_args_t&);

    static spec_return_value_t compute_output_specs(const operation_attributes_t&, const tensor_args_t&);

    static tensor_return_value_t create_output_tensors(const operation_attributes_t&, const tensor_args_t&);
};

}  // namespace ttnn::operations::experimental::matmul_decode

namespace ttnn::prim {
ttnn::operations::experimental::matmul_decode::MatmulDecodeLargeKDeviceOperation::tensor_return_value_t
matmul_decode_large_k(
    const Tensor& input_tensor_a,
    const Tensor& input_tensor_b,
    std::optional<const DataType> dtype = std::nullopt,
    uint32_t reduce_fan_in = 2,
    bool rms_norm = false,
    float rms_norm_epsilon = 1.0e-6F,
    uint32_t fold_rows = 1);
}  // namespace ttnn::prim
