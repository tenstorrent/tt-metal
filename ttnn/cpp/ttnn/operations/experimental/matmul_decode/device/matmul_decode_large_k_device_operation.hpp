// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <cstdint>
#include <optional>
#include <variant>

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
    uint32_t reduce_fan_in = 2);
}  // namespace ttnn::prim
