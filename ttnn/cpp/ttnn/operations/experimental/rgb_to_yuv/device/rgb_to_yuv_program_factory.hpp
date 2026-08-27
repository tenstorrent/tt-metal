// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include "rgb_to_yuv_device_op_types.hpp"
#include "ttnn/device_operation.hpp"

namespace ttnn::experimental::prim {

struct RgbToYuvProgramFactory {
    struct shared_variables_t {
        tt::tt_metal::KernelHandle reader_kernel_id;
        tt::tt_metal::KernelHandle writer_kernel_id;
        tt::tt_metal::KernelHandle compute_kernel_id;
        uint32_t num_cores;
        tt::tt_metal::CoreRangeSet core_group_1;
        tt::tt_metal::CoreRangeSet core_group_2;
        uint32_t units_per_core_g1;
        uint32_t units_per_core_g2;
    };

    using cached_program_t = ttnn::device_operation::CachedProgram<shared_variables_t>;

    static cached_program_t create(
        const RgbToYuvParams& op_attrs, const RgbToYuvInputs& tensor_args, std::tuple<Tensor, Tensor, Tensor>& output);

    static void override_runtime_arguments(
        cached_program_t& cached_program,
        const RgbToYuvParams& op_attrs,
        const RgbToYuvInputs& tensor_args,
        std::tuple<Tensor, Tensor, Tensor>& output);
};

}  // namespace ttnn::experimental::prim
