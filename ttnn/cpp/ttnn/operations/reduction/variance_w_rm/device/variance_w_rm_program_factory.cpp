// SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#include "variance_w_rm_program_factory.hpp"

#include <tt-metalium/host_api.hpp>

namespace ttnn::operations::reduction::variance_w_rm::program {

using namespace tt;
using namespace tt::tt_metal;

// Stub implementation - to be implemented by ttnn-factory-builder agent in Stages 4-6
VarianceWRmProgramFactory::cached_program_t VarianceWRmProgramFactory::create(
    [[maybe_unused]] const VarianceWRmParams& operation_attributes,
    [[maybe_unused]] const VarianceWRmInputs& tensor_args,
    [[maybe_unused]] Tensor& tensor_return_value) {
    TT_THROW(
        "VarianceWRmProgramFactory::create is not yet implemented. "
        "This stub awaits Stage 4-6 implementation by the ttnn-factory-builder agent.");
}

void VarianceWRmProgramFactory::override_runtime_arguments(
    [[maybe_unused]] cached_program_t& cached_program,
    [[maybe_unused]] const VarianceWRmParams& operation_attributes,
    [[maybe_unused]] const VarianceWRmInputs& tensor_args,
    [[maybe_unused]] Tensor& tensor_return_value) {
    // Stub - update runtime arguments for cached program
    // This will update tensor buffer addresses when program is reused
}

}  // namespace ttnn::operations::reduction::variance_w_rm::program
