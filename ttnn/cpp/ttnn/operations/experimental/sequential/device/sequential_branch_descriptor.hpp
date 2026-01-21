// SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include "sequential_device_operation_types.hpp"
#include "ttnn/operations/experimental/parallel/device/parallel_device_operation_types.hpp"

namespace ttnn::experimental::prim {

// =============================================================================
// SequentialBranchDescriptor - Allows sequential to be used as a branch in parallel
// =============================================================================
//
// This implements the BranchDescriptor interface from parallel, enabling patterns like:
//
//   ttnn::parallel([
//       branch_a,  // Single op on cores [0,0]-[3,3]
//       ttnn::sequential::branch(...)  // Multiple ops in sequence on cores [4,0]-[7,3]
//   ])
//
// When add_to_program is called, all steps are added to the shared program
// in sequence, running on the branch's core_range.
//

struct SequentialBranchDescriptor : BranchDescriptor {
    std::vector<std::shared_ptr<StepDescriptor>> steps;

    SequentialBranchDescriptor(const CoreRangeSet& cores, std::vector<std::shared_ptr<StepDescriptor>> steps_) :
        BranchDescriptor{cores}, steps(std::move(steps_)) {}

    std::vector<const Tensor*> get_input_tensors() const override {
        // Gather input tensors from all steps
        std::vector<const Tensor*> all_inputs;
        for (const auto& step : steps) {
            auto step_inputs = step->get_input_tensors();
            all_inputs.insert(all_inputs.end(), step_inputs.begin(), step_inputs.end());
        }
        return all_inputs;
    }

    std::vector<TensorSpec> get_output_specs() const override {
        // Sequential returns outputs from the LAST step
        if (steps.empty()) {
            return {};
        }
        return steps.back()->get_output_specs();
    }

    std::vector<Tensor> make_output_tensors() const override {
        // Sequential returns outputs from the LAST step
        if (steps.empty()) {
            return {};
        }
        return steps.back()->make_output_tensors();
    }

    void check_on_cache_hit() const override {
        for (const auto& step : steps) {
            step->check_on_cache_hit();
        }
    }

    void check_on_cache_miss() const override {
        for (const auto& step : steps) {
            step->check_on_cache_miss();
        }
    }

    void add_to_program(tt::tt_metal::Program& program, std::vector<Tensor>& outputs) override {
        // Add all steps to the program in sequence
        // Each step uses its own stored core_range
        for (size_t i = 0; i < steps.size(); ++i) {
            if (i < steps.size() - 1) {
                // Intermediate step - create its own outputs
                auto step_outputs = steps[i]->make_output_tensors();
                steps[i]->add_to_program(program, step_outputs);
                // Intermediate outputs are discarded (future: CB chaining)
            } else {
                // Last step - use the provided outputs vector
                steps[i]->add_to_program(program, outputs);
            }
        }
    }

    void update_runtime_args(tt::tt_metal::Program& program, std::vector<Tensor>& outputs) override {
        // Update runtime args for the last step (which produces outputs)
        if (!steps.empty() && steps.back()->has_shared_variables()) {
            steps.back()->update_runtime_args(program, outputs);
        }
    }

    bool has_shared_variables() const override {
        // Check if any step has shared variables
        for (const auto& step : steps) {
            if (step->has_shared_variables()) {
                return true;
            }
        }
        return false;
    }

    const std::type_info& type_info() const override { return typeid(SequentialBranchDescriptor); }

    std::string operation_name() const override {
        std::string name = "Sequential[";
        for (size_t i = 0; i < steps.size(); ++i) {
            if (i > 0) {
                name += " -> ";
            }
            name += steps[i]->operation_name();
        }
        name += "]";
        return name;
    }
};

// =============================================================================
// Factory function to create a sequential branch
// =============================================================================

inline std::shared_ptr<BranchDescriptor> create_sequential_branch(
    const CoreRangeSet& cores, std::vector<std::shared_ptr<StepDescriptor>> steps) {
    return std::make_shared<SequentialBranchDescriptor>(cores, std::move(steps));
}

}  // namespace ttnn::experimental::prim

// Backward compatibility alias
namespace ttnn::operations::experimental::sequential {
using ttnn::experimental::prim::create_sequential_branch;
using ttnn::experimental::prim::SequentialBranchDescriptor;
}  // namespace ttnn::operations::experimental::sequential
