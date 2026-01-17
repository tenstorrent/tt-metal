// SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC.
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include "ttnn/device_operation.hpp"
#include "ttnn/operations/data_movement/tilize_with_val_padding/device/tilize_with_val_padding_device_operation_types.hpp"

namespace ttnn::prim {

struct TilizeWithValPaddingMultiCoreShardedFactory {
    struct shared_variables_t {
        tt::tt_metal::KernelHandle reader_kernel_id{};
        tt::tt_metal::KernelHandle writer_kernel_id{};
        tt::tt_metal::CBHandle cb_src0{};
        tt::tt_metal::CBHandle cb_output{};
    };

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
