// SPDX-FileCopyrightText: © 2025 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0
#pragma once

#include "ttnn/device_operation.hpp"
#include "ttnn/operations/experimental/where/device/where_device_operation_types.hpp"

namespace ttnn::operations::experimental::ternary {

struct ElementWiseMultiCoreWhereProgram {
    struct shared_variables_t {
        tt::tt_metal::KernelHandle reader_kernel_id;
        tt::tt_metal::KernelHandle writer_kernel_id;
        tt::tt_metal::KernelHandle eltwise_kernel_id;
        tt::tt_metal::CBHandle condition_cb;
        tt::tt_metal::CBHandle true_values_cb;
        tt::tt_metal::CBHandle false_values_cb;
        tt::tt_metal::CBHandle output_cb;
        CoreRangeSet all_device_cores;
        uint32_t condition_tensor_tile_size;
        uint32_t true_values_tensor_tile_size;
        uint32_t false_values_tensor_tile_size;
        uint32_t dst_tile_size;
    };
    using cached_program_t = ttnn::device_operation::CachedProgram<shared_variables_t>;

    static cached_program_t create(
        const where_ttt_args::operation_attributes_type& operation_attributes,
        const where_ttt_args::tensor_args_type& tensor_args,
        where_ttt_args::tensor_return_value_type& tensor_return_value);

    static void override_runtime_arguments(
        cached_program_t& cached_program,
        const where_ttt_args::operation_attributes_type& operation_attributes,
        const where_ttt_args::tensor_args_type& tensor_args,
        where_ttt_args::tensor_return_value_type& tensor_return_value);
};
}  // namespace ttnn::operations::experimental::ternary
