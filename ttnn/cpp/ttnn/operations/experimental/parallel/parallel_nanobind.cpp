// SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#include "parallel_nanobind.hpp"

#include <nanobind/nanobind.h>
#include <nanobind/stl/shared_ptr.h>
#include <nanobind/stl/vector.h>

#include "parallel.hpp"
#include "device/parallel_device_operation_types.hpp"

namespace nb = nanobind;

namespace ttnn::operations::experimental::parallel::detail {

//=============================================================================
// Binding implementation
//=============================================================================

void bind_parallel_operation(nb::module_& mod) {
    // Bind BranchDescriptor as an opaque type
    // BranchDescriptor instances are created via operation-specific branch() methods
    // e.g., ttnn.rms_norm.branch(input, cores, epsilon=1e-5, weight=w)
    nb::class_<BranchDescriptor>(
        mod,
        "BranchDescriptor",
        R"doc(
            A descriptor for a parallel branch operation.

            BranchDescriptor instances are created via operation-specific branch() methods.
            Each operation that supports parallel execution provides a branch() method that
            returns a BranchDescriptor.

            Example:
                >>> branch = ttnn.rms_norm.branch(input, cores, epsilon=1e-5, weight=w)
                >>> results = ttnn.parallel([branch1, branch2])
        )doc");

    // Bind the parallel execution function
    mod.def(
        "parallel",
        [](const std::vector<std::shared_ptr<BranchDescriptor>>& branches) {
            return ExecuteParallel::invoke(std::vector<std::shared_ptr<BranchDescriptor>>(branches));
        },
        nb::arg("branches"),
        R"doc(
            Execute multiple operations in parallel as a single fused program.

            Each branch runs on disjoint core ranges within a single program dispatch.
            This enables operation fusion without kernel boundaries, maximizing hardware utilization.

            Branches are created using the operation-specific branch() methods. Any operation
            that supports parallel execution will have a branch() method with the same signature
            as its regular invoke, plus a `cores` parameter.

            Args:
                branches (list[BranchDescriptor]): List of branch descriptors created via
                    operation-specific branch() methods (e.g., ttnn.rms_norm.branch()).

            Returns:
                list[list[Tensor]]: Nested list where results[i] contains the output tensors
                from the i-th branch.

            Example:
                >>> # Create branch descriptors using operation-specific branch() methods
                >>> cores_a = ttnn.CoreRangeSet([ttnn.CoreRange((0, 0), (3, 3))])
                >>> cores_b = ttnn.CoreRangeSet([ttnn.CoreRange((4, 0), (7, 3))])
                >>>
                >>> branch_a = ttnn.rms_norm.branch(input_a, cores_a, epsilon=1e-5, weight=w_a)
                >>> branch_b = ttnn.rms_norm.branch(input_b, cores_b, epsilon=1e-5, weight=w_b)
                >>>
                >>> results = ttnn.experimental.parallel([branch_a, branch_b])
                >>> output_a = results[0][0]
                >>> output_b = results[1][0]

            Supported Operations:
                - ttnn.rms_norm.branch() - RMS normalization
                - ttnn.layer_norm.branch() - Layer normalization (when implemented)
                - Additional operations can be added by implementing a branch() method

            Notes:
                - Core ranges must be disjoint between branches
                - All branches execute within a single program dispatch
                - Each branch can be a different operation type
                - The operation must implement the add_to() factory method for parallel support
        )doc");
}

}  // namespace ttnn::operations::experimental::parallel::detail
