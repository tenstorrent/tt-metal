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
    validate_recipe_grid(attrs.program_config);
    {
        const uint32_t nqh = tensors.input_q.logical_shape()[1], nkv = tensors.input_k.logical_shape()[1];
        const uint32_t joint_nqh = tensors.joint_q.logical_shape()[1], joint_nkv = tensors.joint_k.logical_shape()[1];
        TT_FATAL(
            nkv > 0 && nqh == nkv,
            "Q num_heads must be equal to K num_heads, and greater than 0. Got Q: {}, K: {}",
            nqh,
            nkv);
        TT_FATAL(
            joint_nqh == joint_nkv,
            "Joint Q num_heads must be equal to Joint K num_heads. Got Q: {}, K: {}",
            joint_nqh,
            joint_nkv);
        TT_FATAL(
            joint_nkv == nkv, "Joint K num_heads must be equal to K num_heads. Got Joint K: {}, K: {}", joint_nkv, nkv);
    }
    const auto resolution = memoized_recipe_resolution(recipe_resolution_key(tensors.input_q, attrs, tensors), [&] {
        const auto policy = joint_recipe_policy(attrs, tensors);
        const auto segments = joint_segments(tensors);
        auto choose = [&](const std::optional<SDPAProgramConfig>& config) {
            return resolve_dense_recipe_blocking(
                policy,
                tensors.input_q,
                tensors.input_k,
                &tensors.joint_q,
                &tensors.joint_k,
                config,
                nullptr,
                0,
                nullptr,
                nullptr,
                false,
                recipe_kv_storage(tensors.input_v.dtype()));
        };
        RecipeResolution r;
        r.program_config = choose(attrs.program_config);
        if (r.program_config &&
            !recipe_chunks_fit(segments, policy, r.program_config, std::nullopt, attrs.output_memory_config, {}, {})) {
            // The caller's chunks do not fit the free L1 in any recipe layout: the op chooses them on its grid.
            auto config = *r.program_config;
            config.q_chunk_size = 0;
            config.k_chunk_size = 0;
            r.program_config = choose(config);
        }
        r.l1_budget =
            recipe_l1_budget(segments, policy, r.program_config, std::nullopt, attrs.output_memory_config, {}, {});
        return r;
    });
    attrs.program_config = resolution.program_config;
    attrs.recipe_l1_budget = resolution.l1_budget;
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
