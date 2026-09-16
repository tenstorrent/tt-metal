// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <cstdint>

#include "unified_routed_expert_ffn_types.hpp"

#include <tt-metalium/program_descriptors.hpp>
#include "ttnn/tensor/tensor.hpp"

namespace ttnn::operations::experimental::deepseek_prefill::hybrid_routed_expert_ffn::unified {

// Appends this implementation's circular buffers, semaphores and kernels to a descriptor the
// caller owns, so the merged op can carry both implementations in ONE program.
//
// `next_semaphore_id` is the caller's allocator: descriptor semaphores carry explicit ids, and a
// caller appending more than one implementation needs to know which ids are spoken for. It is
// advanced past every id this call takes. (The merged op does not thread one counter through both
// halves -- it lets each number from zero and folds the two sets by id afterwards, which is sound
// only because the passes are strictly ordered and the shared ids are re-zeroed at the barrier.)
//
// Buffer addresses go in as Buffer* rather than raw values, which registers them as buffer
// bindings -- the framework then patches them in place on a program-cache hit instead of rebuilding
// the descriptor.
void append_to_descriptor(
    tt::tt_metal::ProgramDescriptor& descriptor,
    uint32_t& next_semaphore_id,
    const UnifiedRoutedExpertFfnParams& operation_attributes,
    const UnifiedRoutedExpertFfnInputs& tensor_args,
    Tensor& tensor_return_value);

// The standalone form, for a program that carries this implementation alone.
tt::tt_metal::ProgramDescriptor create_unified_routed_expert_ffn_program_descriptor(
    const UnifiedRoutedExpertFfnParams& operation_attributes,
    const UnifiedRoutedExpertFfnInputs& tensor_args,
    Tensor& tensor_return_value);

}  // namespace ttnn::operations::experimental::deepseek_prefill::hybrid_routed_expert_ffn::unified
