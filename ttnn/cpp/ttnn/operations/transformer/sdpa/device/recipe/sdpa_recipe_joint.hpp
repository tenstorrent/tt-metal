// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
// SPDX-License-Identifier: Apache-2.0
#pragma once

#include <tt-metalium/program_descriptors.hpp>

#include "sdpa_recipe_program.hpp"
#include "ttnn/operations/transformer/sdpa/device/joint_sdpa_device_operation_types.hpp"

// The JointSDPA device operation on the recipe host library: the joint Q/K/V are a second key-stream segment of
// the dense recipe program.
namespace ttnn::operations::transformer::sdpa::detail {

PrecisionPolicy joint_recipe_policy(const ttnn::prim::JointSDPAParams& attrs, const ttnn::prim::JointSDPAInputs& tensors);

// Runs before the program-cache lookup: chunk sizes left to the op and the L1 budget (both hashed).
void resolve_joint_recipe(ttnn::prim::JointSDPAParams& attrs, const ttnn::prim::JointSDPAInputs& tensors);

std::vector<tt::tt_metal::TensorSpec> joint_recipe_output_specs(
    const ttnn::prim::JointSDPAParams& attrs, const ttnn::prim::JointSDPAInputs& tensors);

tt::tt_metal::ProgramDescriptor joint_recipe_descriptor(
    const ttnn::prim::JointSDPAParams& attrs,
    const ttnn::prim::JointSDPAInputs& tensors,
    std::vector<Tensor>& outputs);

}  // namespace ttnn::operations::transformer::sdpa::detail
