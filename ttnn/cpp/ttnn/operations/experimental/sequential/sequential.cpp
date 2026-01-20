// SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#include "sequential.hpp"
#include "device/sequential_device_operation.hpp"

namespace ttnn::operations::experimental::sequential {

tensor_return_value_t ExecuteSequential::invoke(std::vector<std::shared_ptr<StepDescriptor>> steps) {
    TT_FATAL(!steps.empty(), "Sequential operation requires at least one step");

    // Create operation attributes - each step carries its own cores
    operation_attributes_t op_attrs{.steps = std::move(steps), .mesh_device = nullptr};

    // Launch the device operation
    return ttnn::prim::sequential(op_attrs);
}

std::shared_ptr<parallel::BranchDescriptor> ExecuteSequential::branch(
    std::vector<std::shared_ptr<StepDescriptor>> steps) {
    TT_FATAL(!steps.empty(), "Sequential branch requires at least one step");

    // Use the first step's cores as the branch's core range
    const auto& cores = steps.front()->get_cores();
    return create_sequential_branch(cores, std::move(steps));
}

}  // namespace ttnn::operations::experimental::sequential
