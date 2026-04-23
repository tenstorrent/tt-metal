// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include "sp_eq_mul_mask_device_operation_types.hpp"
#include "ttnn/device_operation.hpp"

namespace ttnn::experimental::prim {

struct SpEqMulMaskSharedVariables {
    tt::tt_metal::KernelHandle reader_kernel_id = 0;
    tt::tt_metal::KernelHandle writer_kernel_id = 0;
    tt::tt_metal::KernelHandle compute_kernel_id = 0;
    uint32_t num_cores = 0;
    CoreRangeSet all_cores;
    std::vector<CoreCoord> cores;
    uint32_t g1_numcores = 0;
    uint32_t g2_numcores = 0;
    uint32_t num_tiles_per_core_g1 = 0;
    uint32_t num_tiles_per_core_g2 = 0;
};

struct SpEqMulMaskProgramFactory {
    using shared_variables_t = SpEqMulMaskSharedVariables;
    using cached_program_t = ttnn::device_operation::CachedProgram<shared_variables_t>;

    static cached_program_t create(
        const SpEqMulMaskParams& operation_attributes, const SpEqMulMaskInputs& tensor_args, Tensor& output);

    static void override_runtime_arguments(
        cached_program_t& cached_program,
        const SpEqMulMaskParams& operation_attributes,
        const SpEqMulMaskInputs& tensor_args,
        Tensor& tensor_return_value);
};

}  // namespace ttnn::experimental::prim
