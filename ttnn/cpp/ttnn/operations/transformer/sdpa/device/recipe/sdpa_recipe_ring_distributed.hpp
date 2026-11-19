// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
// SPDX-License-Identifier: Apache-2.0
#pragma once

#include <tt-metalium/program_descriptors.hpp>

#include "sdpa_recipe_program.hpp"
#include "ttnn/operations/transformer/sdpa/device/ring_distributed_sdpa_device_operation_types.hpp"

// The RingDistributedSDPA device operation on the recipe host library: each device computes the causal attention of
// two slabs of the whole Q (sequence chunks ring_id and 2 * ring_size - 1 - ring_id), RecipeKeyRange::q_slab_rows.
namespace ttnn::operations::transformer::sdpa::detail {

// Runs before the program-cache lookup: chunk sizes left to the op and the L1 budget (both hashed).
void resolve_ring_distributed_recipe(
    ttnn::prim::RingDistributedSDPAParams& attrs, const ttnn::prim::RingDistributedSDPAInputs& tensors);

tt::tt_metal::TensorSpec ring_distributed_recipe_output_spec(
    const ttnn::prim::RingDistributedSDPAParams& attrs, const ttnn::prim::RingDistributedSDPAInputs& tensors);

// The program of the device at ring position `ring_id`.
tt::tt_metal::ProgramDescriptor ring_distributed_recipe_descriptor(
    const ttnn::prim::RingDistributedSDPAParams& attrs,
    const ttnn::prim::RingDistributedSDPAInputs& tensors,
    Tensor& output,
    uint32_t ring_id);

}  // namespace ttnn::operations::transformer::sdpa::detail
