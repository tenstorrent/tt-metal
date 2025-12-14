// SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC.
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include "ttnn/operation.hpp"
#include "ttnn/device_operation.hpp"
#include "ttnn/operations/data_movement/tilize_with_val_padding/device/tilize_with_val_padding_device_operation_types.hpp"

namespace ttnn::operations::data_movement::tilize_with_val_padding::program {

struct TilizeWithValPaddingMultiCoreShardedFactory {
    struct shared_variables_t {
        tt::tt_metal::KernelHandle reader_kernel_id{};
        tt::tt_metal::KernelHandle writer_kernel_id{};
        tt::tt_metal::CBHandle cb_src0{};
        tt::tt_metal::CBHandle cb_output{};
    };

    using cached_program_t = ttnn::device_operation::CachedProgram<shared_variables_t>;

    using operation_attributes_t = tilize_with_val_padding::operation_attributes_t;
    using tensor_args_t = tilize_with_val_padding::tensor_args_t;
    using tensor_return_value_t = tilize_with_val_padding::tensor_return_value_t;

    static cached_program_t create(
        const operation_attributes_t& operation_attributes,
        const tensor_args_t& tensor_args,
        const tensor_return_value_t& tensor_return_value);

    static void override_runtime_arguments(
        cached_program_t& cached_program,
        const operation_attributes_t& operation_attributes,
        const tensor_args_t& tensor_args,
        const tensor_return_value_t& output);
};

}  // namespace ttnn::operations::data_movement::tilize_with_val_padding::program
