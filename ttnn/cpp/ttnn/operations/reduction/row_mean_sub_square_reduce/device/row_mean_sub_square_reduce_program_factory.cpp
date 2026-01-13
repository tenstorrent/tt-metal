// SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#include "row_mean_sub_square_reduce_program_factory.hpp"

#include <tt-metalium/host_api.hpp>

namespace ttnn::operations::reduction::row_mean_sub_square_reduce::program {

using namespace tt;
using namespace tt::tt_metal;

// Stub implementation - to be implemented by ttnn-factory-builder agent in Stages 4-6
RowMeanSubSquareReduceProgramFactory::cached_program_t RowMeanSubSquareReduceProgramFactory::create(
    const operation_attributes_t& operation_attributes,
    const tensor_args_t& tensor_args,
    tensor_return_value_t& tensor_return_value) {
    TT_THROW(
        "RowMeanSubSquareReduceProgramFactory::create is not yet implemented. "
        "This stub awaits Stage 4-6 implementation by the ttnn-factory-builder agent.");
}

void RowMeanSubSquareReduceProgramFactory::override_runtime_arguments(
    cached_program_t& cached_program,
    const operation_attributes_t& operation_attributes,
    const tensor_args_t& tensor_args,
    tensor_return_value_t& tensor_return_value) {
    // Stub - update runtime arguments for cached program
    // This will update tensor buffer addresses when program is reused
}

}  // namespace ttnn::operations::reduction::row_mean_sub_square_reduce::program
