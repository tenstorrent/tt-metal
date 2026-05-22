// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include "unified_routed_expert_ffn_program_factory.hpp"

#include <cstdint>
#include <utility>

#include <tt-metalium/constants.hpp>
#include <tt-metalium/core_coord.hpp>
#include <tt-metalium/host_api.hpp>
#include <tt-metalium/math.hpp>
#include <tt-metalium/tensor_accessor_args.hpp>

#include "ttnn/operation.hpp"

namespace ttnn::operations::experimental::deepseek_prefill::unified_routed_expert_ffn {

// Placeholder program factory for the unified routed expert. The real fused
// SwiGLU compute kernel is being written separately; this stub creates an
// empty program so the device op machinery (validation, output spec, Python
// binding) can be wired up and tested end-to-end ahead of the kernel landing.
//
// Once the compute / reader / writer kernels are in place this factory will:
// - Allocate CBs for x (tiled L1, mcast in0), weights (DRAM-streamed in1 per
//   K-block), gate_intermediate, up_intermediate, activated (block-sharded
//   L1 per-chunk), and the output.
// - Create reader kernels on the data-movement risc(s) to stream tiles from
//   DRAM into the in0/in1 CBs and the counts/idx-table scratch CBs.
// - Create one compute kernel (`fused_swiglu.cpp`) per core that performs
//   gate matmul + silu + up matmul + multiply + down matmul for each active
//   M-chunk; the kernel reads counts[idx_table[local_expert_id]] at runtime
//   and skips chunks whose M-row start exceeds the count.
// - Create writer kernels that drain the down-matmul output CB to the DRAM-
//   interleaved output tensor.
UnifiedRoutedExpertFfnProgramFactory::cached_program_t UnifiedRoutedExpertFfnProgramFactory::create(
    const UnifiedRoutedExpertFfnParams& operation_attributes,
    const UnifiedRoutedExpertFfnInputs& tensor_args,
    Tensor& tensor_return_value) {
    (void)operation_attributes;
    (void)tensor_args;
    (void)tensor_return_value;
    tt::tt_metal::Program program = tt::tt_metal::CreateProgram();
    UnifiedRoutedExpertFfnSharedVariables shared;
    return cached_program_t{std::move(program), std::move(shared)};
}

void UnifiedRoutedExpertFfnProgramFactory::override_runtime_arguments(
    cached_program_t& /*cached_program*/,
    const UnifiedRoutedExpertFfnParams& /*operation_attributes*/,
    const UnifiedRoutedExpertFfnInputs& /*tensor_args*/,
    Tensor& /*tensor_return_value*/) {
    // No-op while the compute kernel is being written.
}

}  // namespace ttnn::operations::experimental::deepseek_prefill::unified_routed_expert_ffn
