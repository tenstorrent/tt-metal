// SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include "ttnn/decorators.hpp"
#include "device/sequential_device_operation_types.hpp"
#include "device/sequential_branch_descriptor.hpp"

namespace ttnn::operations::experimental::sequential {

// Import types from ttnn::experimental::prim
using ttnn::experimental::prim::BranchDescriptor;
using ttnn::experimental::prim::StepDescriptor;

// =============================================================================
// ExecuteSequential - The registered operation
// =============================================================================

struct ExecuteSequential {
    // Execute a sequence of operations
    // Each step carries its own core range specification
    // Usage:
    //   ttnn::sequential([step1, step2, ...])
    static std::vector<Tensor> invoke(std::vector<std::shared_ptr<StepDescriptor>> steps);

    // Create a branch descriptor for use with ttnn::parallel
    // The branch's core range is taken from the first step
    // Usage:
    //   auto seq_branch = ttnn::sequential.branch([step1, step2]);
    //   ttnn::parallel([seq_branch, other_branch]);
    static std::shared_ptr<BranchDescriptor> branch(std::vector<std::shared_ptr<StepDescriptor>> steps);
};

}  // namespace ttnn::operations::experimental::sequential

namespace ttnn {

// Re-export Step for user convenience
template <typename DeviceOp>
using step = ttnn::experimental::prim::Step<DeviceOp>;

// The sequential operation
constexpr auto sequential =
    ttnn::register_operation<"ttnn::sequential", ttnn::operations::experimental::sequential::ExecuteSequential>();

}  // namespace ttnn

// =============================================================================
// Usage Examples
// =============================================================================
//
// Example 1: Sequential execution of RMS norm then LayerNorm
//
// ```cpp
// using RMS = ttnn::prim::RMSNormDeviceOperation;
// using LN = ttnn::prim::LayerNormDeviceOperation;
//
// std::vector<std::shared_ptr<StepDescriptor>> steps;
// steps.push_back(create_step<RMS>(cores, {.eps = 1e-5}, {.input = t1, .weight = g1}));
// steps.push_back(create_step<LN>(cores, {.eps = 1e-6}, {.input = t2, .weight = g2}));
//
// auto results = ttnn::sequential(steps);
// auto& output = results[0];  // Output from last step
// ```
//
// Example 2: Sequential as a branch in parallel
//
// ```cpp
// auto seq_branch = ttnn::sequential.branch(steps);
// auto other_branch = ttnn::rms_norm.branch(input, cores_b, epsilon=1e-5);
//
// // Both branches execute in parallel, but seq_branch runs its steps sequentially
// auto results = ttnn::parallel([seq_branch, other_branch]);
// ```
//
