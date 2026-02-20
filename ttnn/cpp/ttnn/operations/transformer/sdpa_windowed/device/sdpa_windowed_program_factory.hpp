// SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include "ttnn/device_operation.hpp"
#include "ttnn/operations/core/compute_kernel/compute_kernel_config.hpp"
#include "ttnn/operations/transformer/sdpa_windowed/device/sdpa_windowed_device_operation_types.hpp"
#include "ttnn/operation.hpp"
#include "ttnn/operations/transformer/sdpa_config.hpp"

namespace ttnn::prim {

struct WindowedSDPASharedVariables {
    tt::tt_metal::KernelHandle reader_kernels_id{};
    tt::tt_metal::KernelHandle writer_kernels_id{};
    tt::tt_metal::KernelHandle compute_kernels_id{};
    tt::tt_metal::CoreCoord grid_size;
    uint32_t num_cores = 0;
};

struct WindowedSDPAProgramFactory {
    using shared_variables_t = WindowedSDPASharedVariables;
    using cached_program_t = ttnn::device_operation::CachedProgram<shared_variables_t>;

    static cached_program_t create(
        const SdpaWindowedParams& operation_attributes,
        const SdpaWindowedInputs& tensor_args,
        Tensor& tensor_return_value);

    static void override_runtime_arguments(
        cached_program_t& cached_program,
        const SdpaWindowedParams& operation_attributes,
        const SdpaWindowedInputs& tensor_args,
        Tensor& tensor_return_value);
};

}  // namespace ttnn::prim
