// SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include "ttnn/decorators.hpp"
#include "device/parallel_device_operation_types.hpp"

namespace ttnn::operations::experimental {

// Import types from ttnn::experimental::prim
using ttnn::experimental::prim::BranchDescriptor;
using ttnn::experimental::prim::make_descriptor;

// Each inner vector is the outputs from one branch
using ParallelReturnType = std::vector<std::vector<Tensor>>;

// =============================================================================
// ExecuteParallel - The registered operation
// =============================================================================

struct ExecuteParallel {
    // Variadic: each branch can be a different operation type
    // Usage:
    //   ttnn::parallel(
    //       ttnn::branch<LayerNormOp>{cores1, op_attrs1, tensor_args1},
    //       ttnn::branch<MatmulOp>{cores2, op_attrs2, tensor_args2}
    //   )
    template <typename... Branches>
    static ParallelReturnType invoke(Branches&&... branches) {
        std::vector<std::shared_ptr<BranchDescriptor>> branch_vec;
        branch_vec.reserve(sizeof...(branches));
        (branch_vec.push_back(make_descriptor(std::forward<Branches>(branches))), ...);
        return invoke_impl(std::move(branch_vec));
    }

    // Vector overload for dynamic number of branches
    static ParallelReturnType invoke(std::vector<std::shared_ptr<BranchDescriptor>> branches);

    // Implementation
    static ParallelReturnType invoke_impl(std::vector<std::shared_ptr<BranchDescriptor>> branches);
};

}  // namespace ttnn::operations::experimental

namespace ttnn {

// Re-export Branch for user convenience
template <typename DeviceOp>
using branch = ttnn::experimental::prim::Branch<DeviceOp>;

// The parallel operation
constexpr auto parallel = ttnn::register_operation<"ttnn::parallel", ttnn::operations::experimental::ExecuteParallel>();

}  // namespace ttnn

// =============================================================================
// Usage Examples
// =============================================================================
//
// Example 1: Two layer_norm operations on different cores
//
// ```cpp
// using LN = ttnn::prim::LayerNormDeviceOperation;
//
// auto results = ttnn::parallel(
//     ttnn::branch<LN>{core_set_1, {.eps = 1e-5, ...}, {.input = t1, .weight = g1}},
//     ttnn::branch<LN>{core_set_2, {.eps = 1e-5, ...}, {.input = t2, .weight = g2}}
// );
//
// auto& output1 = results[0][0];
// auto& output2 = results[1][0];
// ```
//
// Example 2: Different operation types
//
// ```cpp
// using LN = LayerNormDeviceOperation;
// using MM = MatmulDeviceOperation;
//
// auto results = ttnn::parallel(
//     ttnn::branch<LN>{core_set_1, ln_attrs, ln_args},
//     ttnn::branch<MM>{core_set_2, mm_attrs, mm_args}
// );
// ```
//
// Example 3: Dynamic number of branches
//
// ```cpp
// std::vector<std::shared_ptr<BranchDescriptor>> branches;
// for (size_t i = 0; i < num_branches; ++i) {
//     branches.push_back(ttnn::experimental::prim::make_descriptor(
//         ttnn::branch<LN>{core_sets[i], attrs[i], args[i]}));
// }
// auto results = ttnn::parallel(std::move(branches));
// ```
//
