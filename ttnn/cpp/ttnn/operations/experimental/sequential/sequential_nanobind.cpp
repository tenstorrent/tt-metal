// SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#include "sequential_nanobind.hpp"

#include <nanobind/nanobind.h>
#include <nanobind/stl/shared_ptr.h>
#include <nanobind/stl/vector.h>

#include "sequential.hpp"
#include "device/sequential_device_operation_types.hpp"
#include "device/sequential_branch_descriptor.hpp"
#include "device/sequential_device_operation.hpp"

namespace nb = nanobind;

namespace ttnn::operations::experimental::detail {

// Import types from ttnn::experimental::prim
using ttnn::experimental::prim::create_sequential_branch;
using ttnn::experimental::prim::StepDescriptor;

//=============================================================================
// Binding implementation
//=============================================================================

void bind_sequential_operation(nb::module_& mod) {
    // Bind StepDescriptor as an opaque type
    nb::class_<StepDescriptor>(
        mod,
        "StepDescriptor",
        R"doc(
            A descriptor for a sequential step operation.

            StepDescriptor instances are created via operation-specific step() methods.
            Each operation that supports sequential execution provides a step() method.
            Each step carries its own core range specification.

            Example:
                >>> cores = ttnn.CoreRangeSet([ttnn.CoreRange((0, 0), (3, 3))])
                >>> step1 = ttnn.rms_norm.step(input1, cores, epsilon=1e-5, weight=w1)
                >>> step2 = ttnn.layer_norm.step(input2, cores, epsilon=1e-6, weight=w2)
                >>> results = ttnn.sequential([step1, step2])
        )doc");

    // Bind the sequential execution function
    mod.def(
        "sequential",
        [](const std::vector<std::shared_ptr<StepDescriptor>>& steps) {
            return ExecuteSequential::invoke(std::vector<std::shared_ptr<StepDescriptor>>(steps));
        },
        nb::arg("steps"),
        R"doc(
            Execute multiple operations in sequence as a single fused program.

            Each step uses its own core range (specified when the step was created).
            Only the outputs from the last step are returned.

            Args:
                steps (list[StepDescriptor]): List of step descriptors created via
                    operation-specific step() methods (e.g., ttnn.rms_norm.step()).
                    Each step carries its own core range.

            Returns:
                list[Tensor]: Output tensors from the last step in the sequence.

            Example:
                >>> cores = ttnn.CoreRangeSet([ttnn.CoreRange((0, 0), (3, 3))])
                >>> step1 = ttnn.rms_norm.step(input1, cores, epsilon=1e-5, weight=w1)
                >>> step2 = ttnn.layer_norm.step(input2, cores, epsilon=1e-6, weight=w2)
                >>> results = ttnn.sequential([step1, step2])
                >>> output = results[0]  # Output from layer_norm (last step)

            Notes:
                - Each step executes on its own specified core range
                - Intermediate outputs are discarded (future: CB chaining)
                - Only the last step's outputs are returned
        )doc");

    // Bind the sequential branch function (for use with parallel)
    mod.def(
        "sequential_branch",
        [](const std::vector<std::shared_ptr<StepDescriptor>>& steps) {
            // For sequential branch, we need a shared core range
            // We use the first step's cores as the branch's cores
            TT_FATAL(!steps.empty(), "Sequential branch requires at least one step");
            const auto& cores = steps.front()->get_cores();
            return create_sequential_branch(cores, std::vector<std::shared_ptr<StepDescriptor>>(steps));
        },
        nb::arg("steps"),
        R"doc(
            Create a sequential branch descriptor for use with ttnn.parallel.

            This allows running a sequence of operations as a single branch in a parallel
            execution context. All steps should specify the same core range (the branch
            uses the first step's cores).

            Args:
                steps (list[StepDescriptor]): List of step descriptors created via
                    operation-specific step() methods. Each step carries its core range.
                    All steps should use the same cores for proper sequential execution.

            Returns:
                BranchDescriptor: A branch descriptor that can be used with ttnn.parallel.

            Example:
                >>> # Create a sequential branch - all steps on the same cores
                >>> cores = ttnn.CoreRangeSet([ttnn.CoreRange((0, 0), (3, 3))])
                >>> step1 = ttnn.rms_norm.step(input, cores, epsilon=1e-5)
                >>> step2 = ttnn.layer_norm.step(input2, cores, epsilon=1e-6)
                >>> seq_branch = ttnn.sequential.branch([step1, step2])
                >>>
                >>> # Use with parallel - dispatched as single fused program
                >>> other_cores = ttnn.CoreRangeSet([ttnn.CoreRange((4, 0), (7, 3))])
                >>> other_branch = ttnn.rms_norm.branch(input_b, other_cores, epsilon=1e-5)
                >>> results = ttnn.parallel([seq_branch, other_branch])
        )doc");
}

}  // namespace ttnn::operations::experimental::detail
