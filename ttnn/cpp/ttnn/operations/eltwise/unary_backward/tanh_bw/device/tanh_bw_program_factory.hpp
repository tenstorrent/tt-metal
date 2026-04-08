// SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include "tanh_bw_device_operation_types.hpp"
#include "ttnn/device_operation.hpp"

namespace ttnn::operations::unary_backward::tanh_bw {

struct TanhBwProgramFactory {
    struct shared_variables_t {
        tt::tt_metal::KernelHandle reader_kernel_id{};
        tt::tt_metal::KernelHandle compute_kernel_id{};
        tt::tt_metal::KernelHandle writer_kernel_id{};
        uint32_t num_cores = 0;
        uint32_t num_cores_y = 0;
    };
    using cached_program_t = ttnn::device_operation::CachedProgram<shared_variables_t>;

    static cached_program_t create(const TanhBwParams& args, const TanhBwInputs& tensor_args, Tensor& output);

    static void override_runtime_arguments(
        cached_program_t& cached_program,
        const TanhBwParams& operation_attributes,
        const TanhBwInputs& tensor_args,
        Tensor& output);
};

}  // namespace ttnn::operations::unary_backward::tanh_bw
