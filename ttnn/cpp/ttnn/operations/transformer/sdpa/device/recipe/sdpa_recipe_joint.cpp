// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
// SPDX-License-Identifier: Apache-2.0

#include "sdpa_recipe_joint.hpp"

#include "sdpa_recipe_blocking.hpp"

namespace ttnn::operations::transformer::sdpa::detail {
using namespace tt::tt_metal;
using ttnn::prim::JointSDPAInputs;
using ttnn::prim::JointSDPAParams;

namespace {
std::vector<std::array<Tensor, 3>> joint_segments(const JointSDPAInputs& t) {
    return {{t.input_q, t.input_k, t.input_v}, {t.joint_q, t.joint_k, t.joint_v}};
}
}  // namespace

PrecisionPolicy joint_recipe_policy(const JointSDPAParams& attrs, const JointSDPAInputs& tensors) {
    return resolve_recipe_policy(
        tensors.input_q,
        tensors.input_k,
        attrs.precision.value_or(SDPAPrecision::STANDARD),
        attrs.scale,
        std::nullopt,
        attrs.program_config);
}

void resolve_joint_recipe(JointSDPAParams& attrs, const JointSDPAInputs& tensors) {
    const auto policy = joint_recipe_policy(attrs, tensors);
    attrs.program_config = resolve_dense_recipe_blocking(
        policy,
        tensors.input_q,
        tensors.input_k,
        &tensors.joint_q,
        &tensors.joint_k,
        attrs.program_config,
        nullptr,
        0,
        nullptr,
        nullptr,
        false,
        recipe_kv_storage(tensors.input_v.dtype()));
    attrs.recipe_l1_budget = recipe_l1_budget(
        joint_segments(tensors), policy, attrs.program_config, std::nullopt, attrs.output_memory_config, {}, {});
}

std::vector<TensorSpec> joint_recipe_output_specs(const JointSDPAParams& attrs, const JointSDPAInputs& tensors) {
    return {
        recipe_output_spec(tensors.input_q, 0, 0, false, attrs.output_memory_config),
        recipe_output_spec(tensors.joint_q, 0, 0, false, attrs.output_memory_config)};
}

ProgramDescriptor joint_recipe_descriptor(
    const JointSDPAParams& attrs, const JointSDPAInputs& tensors, std::vector<Tensor>& outputs) {
    return recipe_program(
        joint_segments(tensors),
        joint_recipe_policy(attrs, tensors),
        attrs.program_config,
        std::nullopt,
        attrs.scale,
        attrs.output_memory_config,
        {},
        {},
        outputs,
        attrs.recipe_l1_budget);
}

}  // namespace ttnn::operations::transformer::sdpa::detail
