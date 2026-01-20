// SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#include "sequential_factory.hpp"

#include <tt-metalium/host_api.hpp>

namespace ttnn::operations::experimental::sequential {

SequentialProgramFactory::cached_program_t SequentialProgramFactory::create(
    const operation_attributes_t& operation_attributes,
    const tensor_args_t& /*tensor_args*/,
    tensor_return_value_t& tensor_return_value) {
    const auto& steps = operation_attributes.steps;

    TT_FATAL(!steps.empty(), "SequentialProgramFactory requires at least one step");

    // Create a single shared program
    tt::tt_metal::Program program = tt::tt_metal::CreateProgram();

    // Each step adds its kernels, CBs, and semaphores directly to the program
    // Steps are added in order - they will execute sequentially on the device
    // Each step uses its own stored core range
    for (size_t i = 0; i < steps.size(); ++i) {
        // Create output tensors for this step
        auto step_outputs = steps[i]->make_output_tensors();

        // Add this step's resources to the program
        // Each step uses its own stored core range
        steps[i]->add_to_program(program, step_outputs);

        // For sequential execution, only the LAST step's outputs are returned
        if (i == steps.size() - 1) {
            tensor_return_value = std::move(step_outputs);
        }
        // Intermediate outputs are discarded (future: could be connected via CBs)
    }

    return cached_program_t{std::move(program), SequentialSharedVariables{.num_steps = steps.size()}};
}

void SequentialProgramFactory::override_runtime_arguments(
    cached_program_t& cached_program,
    const operation_attributes_t& operation_attributes,
    const tensor_args_t& /*tensor_args*/,
    tensor_return_value_t& tensor_return_value) {
    const auto& steps = operation_attributes.steps;

    TT_FATAL(
        steps.size() == cached_program.shared_variables.num_steps,
        "Step count mismatch: expected {} but got {}",
        cached_program.shared_variables.num_steps,
        steps.size());

    // Each step manages its own shared variables internally
    // Only update the last step's outputs since that's what we return
    if (!steps.empty()) {
        steps.back()->update_runtime_args(cached_program.program, tensor_return_value);
    }
}

}  // namespace ttnn::operations::experimental::sequential
