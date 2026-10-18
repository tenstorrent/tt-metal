// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <tt-metalium/program_descriptors.hpp>

#include "hybrid_routed_expert_ffn_types.hpp"
#include "ttnn/tensor/tensor.hpp"

namespace ttnn::operations::experimental::deepseek_prefill::hybrid_routed_expert_ffn {

// Everything the merged op rejects, checked before anything is placed. Each half also validates
// what it will actually be handed, so a merged dispatch cannot pass a configuration either
// implementation would reject on its own.
void validate_arguments(const HybridRoutedExpertFfnParams& op, const HybridRoutedExpertFfnInputs& t);

// The merged op's whole program, from one call.
//
// Which implementations are carried depends on the threshold: with no expert below it the fused
// half is not built at all, and the result is a plain unified program. Otherwise both are built
// and folded into one binary per RISC-V that runs them as ordered passes.
tt::tt_metal::ProgramDescriptor create_hybrid_program_descriptor(
    const HybridRoutedExpertFfnParams& op, const HybridRoutedExpertFfnInputs& t, ttnn::Tensor& output);

}  // namespace ttnn::operations::experimental::deepseek_prefill::hybrid_routed_expert_ffn
