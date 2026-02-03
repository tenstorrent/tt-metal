// SPDX-FileCopyrightText: © 2024 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include "repeat_and_interleave_eltwise_mul_device_operation_types.hpp"
#include "ttnn/device_operation.hpp"

namespace ttnn::experimental::prim {

struct RepeatAndInterleaveEltwiseMulSharedVariables {
    tt::tt_metal::KernelHandle reader_kernel_id = 0;
    tt::tt_metal::KernelHandle writer_kernel_id = 0;
    tt::tt_metal::KernelHandle compute_kernel_id = 0;
    CoreCoord compute_with_storage_grid_size;
    CoreRangeSet all_cores;
    std::vector<CoreCoord> cores;
    uint32_t num_cores = 0;
    uint32_t g1_numcores = 0;
    uint32_t g2_numcores = 0;
    uint32_t num_blocks_per_core_group_1 = 0;
    uint32_t num_blocks_per_core_group_2 = 0;
    Shape ashape;
    Shape bshape;
    uint32_t hidden_size = 0;
};

struct RepeatAndInterleaveEltwiseMulProgramFactory {
    using shared_variables_t = RepeatAndInterleaveEltwiseMulSharedVariables;
    using cached_program_t = ttnn::device_operation::CachedProgram<shared_variables_t>;

    static cached_program_t create(
        const RepeatMulParams& operation_attributes, const RepeatMulInputs& tensor_args, Tensor& tensor_return_value);

    static void override_runtime_arguments(
        cached_program_t& cached_program,
        const RepeatMulParams& operation_attributes,
        const RepeatMulInputs& tensor_args,
        Tensor& tensor_return_value);
};

}  // namespace ttnn::experimental::prim
