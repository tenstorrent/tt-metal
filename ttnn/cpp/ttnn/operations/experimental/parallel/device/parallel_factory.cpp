// SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#include "parallel_factory.hpp"

#include <tt-metalium/host_api.hpp>

namespace ttnn::experimental::prim {

ParallelProgramFactory::cached_program_t ParallelProgramFactory::create(
    const ParallelParams& operation_attributes,
    const ParallelInputs& /*tensor_args*/,
    tensor_return_value_t& tensor_return_value) {
    // Note: We need non-const access to branches because add_to_program modifies internal state
    // This is safe because ParallelParams is passed by value to the top-level prim::parallel
    auto& branches = const_cast<std::vector<BranchDescriptor>&>(operation_attributes.branches);
    TT_FATAL(!branches.empty(), "ParallelProgramFactory requires at least one branch");

    // Create a single shared program - no merging needed!
    tt::tt_metal::Program program = tt::tt_metal::CreateProgram();

    // Each branch adds its kernels, CBs, and semaphores directly to the program
    for (size_t i = 0; i < branches.size(); ++i) {
        auto& branch_outputs = tensor_return_value[i];

        // Branch adds its resources directly to the shared program
        // using its core_range to restrict execution to specific cores
        branches[i].add_to_program(program, branch_outputs);
    }

    return cached_program_t{std::move(program), ParallelSharedVariables{.num_branches = branches.size()}};
}

void ParallelProgramFactory::override_runtime_arguments(
    cached_program_t& cached_program,
    const ParallelParams& operation_attributes,
    const ParallelInputs& /*tensor_args*/,
    tensor_return_value_t& tensor_return_value) {
    // Note: We need non-const access to branches because update_runtime_args modifies internal state
    auto& branches = const_cast<std::vector<BranchDescriptor>&>(operation_attributes.branches);

    TT_FATAL(
        branches.size() == cached_program.shared_variables.num_branches,
        "Branch count mismatch: expected {} but got {}",
        cached_program.shared_variables.num_branches,
        branches.size());

    // Each branch manages its own shared variables internally
    for (size_t i = 0; i < branches.size(); ++i) {
        auto& branch_outputs = tensor_return_value[i];
        branches[i].update_runtime_args(cached_program.program, branch_outputs);
    }
}

}  // namespace ttnn::experimental::prim
