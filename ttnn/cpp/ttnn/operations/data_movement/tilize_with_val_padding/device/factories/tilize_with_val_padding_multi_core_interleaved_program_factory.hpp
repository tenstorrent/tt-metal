// SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC.
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include "ttnn/device_operation.hpp"
#include "ttnn/operations/data_movement/tilize_with_val_padding/device/tilize_with_val_padding_device_operation_types.hpp"
#include "ttnn/operations/data_movement/tilize_with_val_padding/device/factories/tilize_with_val_padding_shared_variables.hpp"

namespace ttnn::prim {

struct TilizeWithValPaddingMultiCoreInterleavedFactory {
    using shared_variables_t = shared_variables_interleaved;
    using cached_program_t = ttnn::device_operation::CachedProgram<shared_variables_t>;

    using operation_attributes_t = ttnn::prim::TilizeWithValPaddingParams;
    using tensor_args_t = Tensor;
    using tensor_return_value_t = Tensor;

    static cached_program_t create(
        const operation_attributes_t& operation_attributes, const Tensor& input_tensor, const Tensor& output_tensor);

    static void override_runtime_arguments(
        cached_program_t& cached_program,
        const operation_attributes_t& operation_attributes,
        const Tensor& input_tensor,
        const Tensor& output_tensor);
};

}  // namespace ttnn::prim
