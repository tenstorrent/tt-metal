// SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#include "layer_norm_w_rm_program_factory.hpp"

#include <tt-metalium/host_api.hpp>

namespace ttnn::operations::normalization::layer_norm_w_rm::program {

using namespace tt;
using namespace tt::tt_metal;

// Stub implementation - to be implemented by ttnn-factory-builder agent in Stages 4-6
LayerNormWRmProgramFactory::cached_program_t LayerNormWRmProgramFactory::create(
    const LayerNormWRmParams& operation_attributes,
    const LayerNormWRmInputs& tensor_args,
    Tensor& tensor_return_value) {
    (void)operation_attributes;
    (void)tensor_args;
    (void)tensor_return_value;
    TT_THROW(
        "LayerNormWRmProgramFactory::create is not yet implemented. "
        "This stub awaits Stage 4-6 implementation by the ttnn-factory-builder agent.");
}

void LayerNormWRmProgramFactory::override_runtime_arguments(
    cached_program_t& cached_program,
    const LayerNormWRmParams& operation_attributes,
    const LayerNormWRmInputs& tensor_args,
    Tensor& tensor_return_value) {
    (void)cached_program;
    (void)operation_attributes;
    (void)tensor_args;
    (void)tensor_return_value;
    // Stub - update runtime arguments for cached program
    // This will update tensor buffer addresses when program is reused
}

}  // namespace ttnn::operations::normalization::layer_norm_w_rm::program