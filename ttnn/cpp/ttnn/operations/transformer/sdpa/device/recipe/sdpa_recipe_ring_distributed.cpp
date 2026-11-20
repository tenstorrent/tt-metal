// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
// SPDX-License-Identifier: Apache-2.0

#include "sdpa_recipe_ring_distributed.hpp"

#include "sdpa_recipe_blocking.hpp"

namespace ttnn::operations::transformer::sdpa::detail {
using namespace tt::tt_metal;
using ttnn::prim::RingDistributedSDPAInputs;
using ttnn::prim::RingDistributedSDPAParams;

namespace {
PrecisionPolicy ring_distributed_policy(const RingDistributedSDPAParams& attrs, const RingDistributedSDPAInputs& t) {
    return resolve_recipe_policy(
        t.q, t.k, attrs.precision.value_or(SDPAPrecision::STANDARD), attrs.scale, std::nullopt, attrs.program_config);
}

uint32_t slab_rows(const RingDistributedSDPAParams& attrs, const RingDistributedSDPAInputs& t) {
    return t.q.logical_shape()[2] / (2 * attrs.ring_size);
}

RecipeKeyRange ring_distributed_key_range(const RingDistributedSDPAParams& attrs, const RingDistributedSDPAInputs& t) {
    return RecipeKeyRange{
        .causal = true,
        .q_offset = static_cast<uint32_t>(attrs.chunk_start_idx.value_or(0)),
        .page_table = t.page_table,
        .q_slab_rows = slab_rows(attrs, t)};
}

void validate_ring_distributed(const RingDistributedSDPAParams& attrs, const RingDistributedSDPAInputs& t) {
    const uint32_t sq = t.q.logical_shape()[2];
    const uint32_t ring_size = attrs.ring_size;
    validate_recipe_grid(attrs.program_config);
    TT_FATAL(
        ring_size > 0 && ring_size % 2 == 0 && ring_size <= 32,
        "ring_size must be positive, even and at most 32, got {}",
        ring_size);
    TT_FATAL(
        !attrs.ring_id || *attrs.ring_id < ring_size,
        "ring_id must be less than ring_size, got {}",
        attrs.ring_id.value_or(0));
    TT_FATAL(
        sq % (64 * ring_size) == 0,
        "Ring-distributed SDPA splits the sequence into 2 * ring_size tile-aligned slabs; got sequence length {} for "
        "ring size {}",
        sq,
        ring_size);
    TT_FATAL(
        attrs.chunk_start_idx.has_value() == t.page_table.has_value(),
        "Ring-distributed SDPA takes chunk_start_idx and page_table together (prefix caching)");
    TT_FATAL(
        t.page_table || t.k.logical_shape()[2] == sq,
        "Ring-distributed SDPA is causal and requires Q and K to have the same sequence length when not using prefix "
        "caching. Got Q: {}, K: {}",
        sq,
        t.k.logical_shape()[2]);
}
}  // namespace

namespace {
RecipeResolution resolve_ring_distributed_recipe_uncached(
    const RingDistributedSDPAParams& attrs, const RingDistributedSDPAInputs& tensors) {
    validate_ring_distributed(attrs, tensors);
    const auto policy = ring_distributed_policy(attrs, tensors);
    const auto key_range = ring_distributed_key_range(attrs, tensors);
    const RecipeDenseOptions options{};
    RecipeResolution resolution;
    auto choose = [&](const std::optional<SDPAProgramConfig>& config) {
        resolution.program_config = resolve_dense_recipe_blocking(
            policy,
            tensors.q,
            tensors.k,
            nullptr,
            nullptr,
            config,
            nullptr,
            0,
            &key_range,
            &options,
            false,
            recipe_kv_storage(tensors.v.dtype()));
    };
    choose(attrs.program_config);
    if (resolution.program_config && !recipe_chunks_fit(
                                         {{tensors.q, tensors.k, tensors.v}},
                                         policy,
                                         resolution.program_config,
                                         std::nullopt,
                                         attrs.output_mem_config,
                                         key_range,
                                         options)) {
        // The caller's chunks do not fit the free L1 in any recipe layout: the op chooses them on the caller's grid.
        auto config = *resolution.program_config;
        config.q_chunk_size = 0;
        config.k_chunk_size = 0;
        choose(config);
    }
    resolution.l1_budget = recipe_l1_budget(
        {{tensors.q, tensors.k, tensors.v}},
        policy,
        resolution.program_config,
        std::nullopt,
        attrs.output_mem_config,
        key_range,
        options);
    return resolution;
}
}  // namespace

void resolve_ring_distributed_recipe(RingDistributedSDPAParams& attrs, const RingDistributedSDPAInputs& tensors) {
    const auto resolution = memoized_recipe_resolution(recipe_resolution_key(tensors.q, attrs, tensors), [&] {
        return resolve_ring_distributed_recipe_uncached(attrs, tensors);
    });
    attrs.program_config = resolution.program_config;
    attrs.recipe_l1_budget = resolution.l1_budget;
}

TensorSpec ring_distributed_recipe_output_spec(
    const RingDistributedSDPAParams& attrs, const RingDistributedSDPAInputs& tensors) {
    return recipe_output_spec(tensors.q, 0, slab_rows(attrs, tensors), false, attrs.output_mem_config);
}

ProgramDescriptor ring_distributed_recipe_descriptor(
    const RingDistributedSDPAParams& attrs, const RingDistributedSDPAInputs& tensors, Tensor& output, uint32_t ring_id) {
    const uint32_t rows = slab_rows(attrs, tensors);
    auto program = recipe_program(
        {{tensors.q, tensors.k, tensors.v}},
        ring_distributed_policy(attrs, tensors),
        attrs.program_config,
        std::nullopt,
        attrs.scale,
        attrs.output_mem_config,
        ring_distributed_key_range(attrs, tensors),
        {},
        {output},
        attrs.recipe_l1_budget);
    set_recipe_q_slabs(
        program,
        recipe_dense_q_tiles(attrs.program_config) * 32,
        {ring_id * rows, (2 * attrs.ring_size - 1 - ring_id) * rows});
    return program;
}

}  // namespace ttnn::operations::transformer::sdpa::detail
