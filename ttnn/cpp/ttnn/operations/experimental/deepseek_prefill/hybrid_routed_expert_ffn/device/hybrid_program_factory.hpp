// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <tt-metalium/program_descriptors.hpp>
#include <tt-metalium/workload_descriptor.hpp>

#include "hybrid_routed_expert_ffn_types.hpp"
#include "ttnn/distributed/types.hpp"
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

// The op dispatches as a workload rather than one replicated program because the combine half it
// can carry is coord-dependent -- each chip's compile-time args name its ring neighbour -- and
// because only a WorkloadDescriptor has somewhere to park combine's GlobalSemaphores so they
// outlive the cached workload. With the combine half off this still emits one program per
// coordinate RANGE, so a mesh-wide op is a single entry exactly as before.
struct HybridRoutedExpertFfnProgramFactory {
    static tt::tt_metal::WorkloadDescriptor create_workload_descriptor(
        const HybridRoutedExpertFfnParams& operation_attributes,
        const HybridRoutedExpertFfnInputs& tensor_args,
        ttnn::Tensor& tensor_return_value,
        const ttnn::MeshCoordinateRangeSet& tensor_coords);
};

}  // namespace ttnn::operations::experimental::deepseek_prefill::hybrid_routed_expert_ffn
