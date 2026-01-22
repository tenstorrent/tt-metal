// SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#include "standardize_w_rm_program_factory.hpp"

#include <tt-metalium/host_api.hpp>

namespace ttnn::operations::reduction::standardize_w_rm::program {

using namespace tt;
using namespace tt::tt_metal;

// Stub implementation - to be implemented by ttnn-factory-builder agent in Stages 4-6
StandardizeWRmProgramFactory::cached_program_t StandardizeWRmProgramFactory::create(
    [[maybe_unused]] const StandardizeWRmParams& operation_attributes,
    [[maybe_unused]] const StandardizeWRmInputs& tensor_args,
    [[maybe_unused]] Tensor& tensor_return_value) {
    TT_THROW(
        "StandardizeWRmProgramFactory::create is not yet implemented. "
        "This stub awaits Stage 4-6 implementation by the ttnn-factory-builder agent.");
}

void StandardizeWRmProgramFactory::override_runtime_arguments(
    [[maybe_unused]] cached_program_t& cached_program,
    [[maybe_unused]] const StandardizeWRmParams& operation_attributes,
    [[maybe_unused]] const StandardizeWRmInputs& tensor_args,
    [[maybe_unused]] Tensor& tensor_return_value) {
    // Stub - update runtime arguments for cached program
    // This will update tensor buffer addresses when program is reused
}

}  // namespace ttnn::operations::reduction::standardize_w_rm::program
