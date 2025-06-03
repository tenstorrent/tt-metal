// SPDX-FileCopyrightText: © 2025 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0
#pragma once

#include "ttnn/device_operation.hpp"
#include "ttnn/operations/experimental/where/device/where_device_operation_types.hpp"

namespace ttnn::operations::ternary::experimental {

struct ElementWiseMultiCoreWhereProgram {
    struct shared_variables_t {
        tt::tt_metal::KernelHandle reader_kernel_id;
        tt::tt_metal::KernelHandle writer_kernel_id;
        tt::tt_metal::KernelHandle eltwise_kernel_id;
        tt::tt_metal::CBHandle cb_src0;
        tt::tt_metal::CBHandle cb_src1;
        tt::tt_metal::CBHandle cb_src2;
        tt::tt_metal::CBHandle cb_output;
        CoreRangeSet all_device_cores;
        uint32_t src0_single_tile_size;
        uint32_t src1_single_tile_size;
        uint32_t src2_single_tile_size;
        uint32_t dst_single_tile_size;
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
}  // namespace ttnn::operations::ternary::experimental
