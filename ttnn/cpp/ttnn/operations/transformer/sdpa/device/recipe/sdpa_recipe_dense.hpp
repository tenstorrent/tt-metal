// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
// SPDX-License-Identifier: Apache-2.0
#pragma once

#include <array>
#include <vector>

#include <tt-metalium/program_descriptors.hpp>

#include "sdpa_recipe_program.hpp"
#include "ttnn/operations/transformer/sdpa/device/sdpa_device_operation_types.hpp"

// The SDPA device operation (dense, chunked, MLA, windowed prefill) on the recipe host library.
namespace ttnn::operations::transformer::sdpa::detail {

// A dense call in the recipe library's terms: its Q/K/V segment (V is K for MLA without a V tensor), key range and
// dense options.
struct DenseRecipeCall {
    std::vector<std::array<Tensor, 3>> segments;
    RecipeKeyRange key_range;
    RecipeDenseOptions options;
};

DenseRecipeCall dense_recipe_call(const ttnn::prim::SDPAParams& attrs, const ttnn::prim::SDPAInputs& tensors);

// The precision policy of a call (attrs.precision, STANDARD when unset).
PrecisionPolicy dense_recipe_policy(const ttnn::prim::SDPAParams& attrs, const ttnn::prim::SDPAInputs& tensors);

// The attn_mask checks of a dense call, run by scaled_dot_product_attention before it pre-scales the mask (so a
// rejected call dispatches nothing).
void validate_dense_recipe_mask(
    const Tensor& q,
    const Tensor& k,
    const Tensor& mask,
    std::optional<SDPAPrecision> precision,
    bool is_causal,
    std::optional<uint32_t> sliding_window_size,
    bool windowed);

// Runs before the program-cache lookup: fills the chunk sizes the caller left to the op (attrs.program_config) and
// the L1 budget (attrs.recipe_l1_budget), both part of the program hash.
void resolve_dense_recipe(ttnn::prim::SDPAParams& attrs, const ttnn::prim::SDPAInputs& tensors);

tt::tt_metal::TensorSpec dense_recipe_output_spec(
    const ttnn::prim::SDPAParams& attrs, const ttnn::prim::SDPAInputs& tensors);

tt::tt_metal::ProgramDescriptor dense_recipe_descriptor(
    const ttnn::prim::SDPAParams& attrs, const ttnn::prim::SDPAInputs& tensors, Tensor& output);

}  // namespace ttnn::operations::transformer::sdpa::detail
