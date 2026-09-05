// SPDX-FileCopyrightText: © 2026 Tenstorrent Inc.
// SPDX-License-Identifier: Apache-2.0

#include "single_tile_stockham_factory.hpp"

#include "batched_stockham_factory.hpp"

namespace ttnn::experimental::prim {

ttnn::device_operation::ProgramArtifacts SingleTileStockhamFactory::create_program_artifacts(
    const FFTParams& operation_attributes,
    const FFTTensorArgs& tensor_args,
    std::tuple<ttnn::Tensor, ttnn::Tensor>& tensor_return_value) {
    TT_FATAL(
        !operation_attributes.input_imag_provided,
        "SingleTileStockhamFactory is reserved for real-only input");
    return BatchedStockhamFactory::create_program_artifacts(operation_attributes, tensor_args, tensor_return_value);
}

}  // namespace ttnn::experimental::prim
