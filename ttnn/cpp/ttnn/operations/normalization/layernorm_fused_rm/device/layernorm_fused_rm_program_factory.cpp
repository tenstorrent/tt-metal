// SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#include "layernorm_fused_rm_program_factory.hpp"

#include <tt-metalium/host_api.hpp>

namespace ttnn::operations::normalization::layernorm_fused_rm::program {

using namespace tt;
using namespace tt::tt_metal;

// Stub implementation - to be implemented by ttnn-factory-builder agent in Stages 4-6
LayernormFusedRmProgramFactory::cached_program_t LayernormFusedRmProgramFactory::create(
    [[maybe_unused]] const LayernormFusedRmParams& operation_attributes,
    [[maybe_unused]] const LayernormFusedRmInputs& tensor_args,
    [[maybe_unused]] Tensor& tensor_return_value) {
    TT_THROW(
        "LayernormFusedRmProgramFactory::create is not yet implemented. "
        "This stub awaits Stage 4-6 implementation by the ttnn-factory-builder agent.");
}

void LayernormFusedRmProgramFactory::override_runtime_arguments(
    [[maybe_unused]] cached_program_t& cached_program,
    [[maybe_unused]] const LayernormFusedRmParams& operation_attributes,
    [[maybe_unused]] const LayernormFusedRmInputs& tensor_args,
    [[maybe_unused]] Tensor& tensor_return_value) {
    // Stub - update runtime arguments for cached program
    // This will update tensor buffer addresses when program is reused
}

}  // namespace ttnn::operations::normalization::layernorm_fused_rm::program
