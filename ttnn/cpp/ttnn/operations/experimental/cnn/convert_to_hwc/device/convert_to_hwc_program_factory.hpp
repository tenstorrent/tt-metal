// SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include "ttnn/operation.hpp"
#include "ttnn/device_operation.hpp"
#include "convert_to_hwc_device_operation_types.hpp"
#include "ttnn/operations/data_movement/sharded/sharded_common.hpp"

namespace ttnn::operations::experimental::cnn::program {

struct ConvertToHWCSharedVariables {
    tt::tt_metal::CBHandle cb_in{};
    tt::tt_metal::CBHandle cb_out{};
    bool is_input_in_dram = false;
    std::vector<tt::tt_metal::CoreCoord> l1_input_cores;
    std::vector<std::vector<ttnn::operations::data_movement::detail::WidthShardingReshardSegment>>
        runtime_args_for_each_core;
    tt::tt_metal::KernelHandle writer_kernel_id0{};
    tt::tt_metal::KernelHandle writer_kernel_id1{};
    uint32_t total_num_sticks_kernel_0 = 0;
    uint32_t total_num_sticks_kernel_1 = 0;
    uint32_t dram_read_stride_bytes = 0;
    uint32_t dram_write_stride_bytes = 0;
    uint32_t remote_address = 0;
};

struct ConvertToHWCProgramFactory {
    using shared_variables_t = ConvertToHWCSharedVariables;
    using cached_program_t = ttnn::device_operation::CachedProgram<shared_variables_t>;

    static cached_program_t create(
        const operation_attributes_t& operation_attributes,
        const tensor_args_t& tensor_args,
        tensor_return_value_t& tensor_return_value);

    static void override_runtime_arguments(
        cached_program_t& cached_program,
        const operation_attributes_t& operation_attributes,
        const tensor_args_t& tensor_args,
        tensor_return_value_t& tensor_return_value);
};

}  // namespace ttnn::operations::experimental::cnn::program

namespace ttnn::operations::experimental::cnn::detail {

uint32_t compute_alignment_requirement_in_elements(const Tensor& input_tensor);

}  // namespace ttnn::operations::experimental::cnn::detail
