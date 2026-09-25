// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <tt-metalium/program_descriptors.hpp>
#include <tt-metalium/workload_descriptor.hpp>

#include "hybrid_routed_expert_ffn_types.hpp"
#include "combine/combine_fabric2d_types.hpp"
#include "ttnn/distributed/types.hpp"
#include "ttnn/tensor/tensor.hpp"

namespace ttnn::operations::experimental::deepseek_prefill::hybrid_routed_expert_ffn {

// The routed expert alone: one program, the same on every chip.
struct HybridSoloProgramFactory {
    static tt::tt_metal::ProgramDescriptor create_descriptor(
        const HybridRoutedExpertFfnParams& op, const HybridRoutedExpertFfnInputs& t, ttnn::Tensor& output);
};

// The routed expert on rows 2-9 and combine on rows 0-1 of ONE program per chip. Every routed-expert writer
// reports each expert it finishes to that chip's combine collector, which releases combine to read it.
//
// A workload rather than one program because combine's placement follows each chip's ethernet cores, and
// because it owns what has to outlive the program: fwd_arrived and, when the fused pass runs, the L1 arena, allocated
// in that order at cache miss. The arena spans every worker core and both ops lay their L1 over it: the
// routed expert's static circular buffers would otherwise have to clear combine's, which the device-wide
// lowest-allocation check does not allow on disjoint cores.
struct HybridOverlapProgramFactory {
    static tt::tt_metal::WorkloadDescriptor create_workload_descriptor(
        const HybridRoutedExpertFfnParams& op,
        const HybridRoutedExpertFfnInputs& t,
        ttnn::Tensor& output,
        const ttnn::MeshCoordinateRangeSet& tensor_coords);
};

// What combine is handed when it overlaps the routed expert; `output` there is the routed expert's.
combine::CombineFabric2dParams combine_attributes(
    const HybridRoutedExpertFfnParams& op, const HybridRoutedExpertFfnInputs& t);
combine::CombineFabric2dInputs combine_inputs(const HybridRoutedExpertFfnInputs& t);

}  // namespace ttnn::operations::experimental::deepseek_prefill::hybrid_routed_expert_ffn
