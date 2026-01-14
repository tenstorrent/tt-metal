// SPDX-FileCopyrightText: © 2024 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include "hc_sum_reduce_device_operation_types.hpp"
#include "ttnn/device_operation.hpp"

namespace ttnn::operations::experimental::ssm::hc_sum_reduce::program {

struct HCSumReduceSharedVariables {
    tt::tt_metal::KernelHandle reader_kernel_id = 0;
    tt::tt_metal::KernelHandle writer_kernel_id = 0;
    tt::tt_metal::KernelHandle compute_kernel_id = 0;
    uint32_t num_cores = 0;
    CoreRangeSet all_cores;
    std::vector<CoreCoord> cores;
    uint32_t g1_numcores = 0;
    uint32_t g2_numcores = 0;
    uint32_t num_blocks_per_core_group_1 = 0;
    uint32_t num_blocks_per_core_group_2 = 0;
};

struct HCSumReduceProgramFactory {
    using shared_variables_t = HCSumReduceSharedVariables;
    using cached_program_t = ttnn::device_operation::CachedProgram<shared_variables_t>;

    static cached_program_t create(
        const operation_attributes_t& operation_attributes, const tensor_args_t& tensor_args, Tensor& output);

    static void override_runtime_arguments(
        cached_program_t& cached_program,
        const operation_attributes_t& operation_attributes,
        const tensor_args_t& tensor_args,
        Tensor& tensor_return_value);
};

}  // namespace ttnn::operations::experimental::ssm::hc_sum_reduce::program
