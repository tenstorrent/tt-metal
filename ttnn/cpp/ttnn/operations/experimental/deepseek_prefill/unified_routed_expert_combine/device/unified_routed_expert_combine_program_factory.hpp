// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <tt-metalium/program_descriptors.hpp>
#include <tt-metalium/workload_descriptor.hpp>

#include "unified_routed_expert_combine_types.hpp"
#include "ttnn/distributed/types.hpp"
#include "ttnn/operations/experimental/deepseek_prefill/unified_routed_expert_combine/device/combine_overlap_program_factory.hpp"
#include "ttnn/operations/experimental/deepseek_prefill/unified_routed_expert_ffn/device/unified_routed_expert_ffn_types.hpp"
#include "ttnn/tensor/tensor.hpp"

namespace ttnn::operations::experimental::deepseek_prefill::unified_routed_expert_combine {

// The routed expert alone: the wave-planned program on the whole 11x8 grid, the same on every chip.
struct UnifiedRoutedExpertSoloProgramFactory {
    static tt::tt_metal::ProgramDescriptor create_descriptor(
        const UnifiedRoutedExpertCombineParams& op, const UnifiedRoutedExpertCombineInputs& t, ttnn::Tensor& output);
};

// The routed expert on rows 2-9 and combine_fabric2d on rows 0-1 of ONE program per chip. The routed expert's
// writers report each expert to that chip's combine collector as soon as they are done with it, at its place in the
// order the program is predicted to finish them (adaptive_chunk::predicted_order); combine releases them in the order
// they actually finish, walking that prediction for every ring chip.
//
// A workload rather than one program because combine's placement follows each chip's ethernet cores.
struct UnifiedRoutedExpertCombineProgramFactory {
    static tt::tt_metal::WorkloadDescriptor create_workload_descriptor(
        const UnifiedRoutedExpertCombineParams& op,
        const UnifiedRoutedExpertCombineInputs& t,
        ttnn::Tensor& output,
        const ttnn::MeshCoordinateRangeSet& tensor_coords);
};

// What each of the two programs is handed. The routed expert's are unified_routed_expert_moe's with no active-token
// band; combine's `dispatched_buffer` is the routed expert's output.
::ttnn::operations::experimental::deepseek_prefill::unified_routed_expert_ffn::UnifiedRoutedExpertFfnParams
routed_expert_attributes(const UnifiedRoutedExpertCombineParams& op);
::ttnn::operations::experimental::deepseek_prefill::unified_routed_expert_ffn::UnifiedRoutedExpertFfnInputs
routed_expert_inputs(const UnifiedRoutedExpertCombineInputs& t);
overlap::CombineFabric2dParams combine_attributes(
    const UnifiedRoutedExpertCombineParams& op, const UnifiedRoutedExpertCombineInputs& t);
overlap::CombineFabric2dInputs combine_inputs(const UnifiedRoutedExpertCombineInputs& t);

}  // namespace ttnn::operations::experimental::deepseek_prefill::unified_routed_expert_combine
