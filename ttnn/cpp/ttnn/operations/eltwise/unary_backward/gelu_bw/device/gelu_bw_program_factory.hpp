// SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include "gelu_bw_device_operation_types.hpp"
#include "ttnn/device_operation.hpp"

namespace ttnn::operations::unary_backward::gelu_bw {

struct GeluBwProgramFactory {
    struct shared_variables_t {
        tt::tt_metal::KernelHandle reader_kernel_id{};
        tt::tt_metal::KernelHandle compute_kernel_id{};
        tt::tt_metal::KernelHandle writer_kernel_id{};
        uint32_t num_cores = 0;
        uint32_t num_cores_y = 0;
    };
    using cached_program_t = ttnn::device_operation::CachedProgram<shared_variables_t>;

    static cached_program_t create(const GeluBwParams& args, const GeluBwInputs& tensor_args, Tensor& output);

    static void override_runtime_arguments(
        cached_program_t& cached_program,
        const GeluBwParams& operation_attributes,
        const GeluBwInputs& tensor_args,
        Tensor& output);
};

}  // namespace ttnn::operations::unary_backward::gelu_bw
