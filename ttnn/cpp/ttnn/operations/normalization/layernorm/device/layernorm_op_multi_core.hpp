// SPDX-FileCopyrightText: © 2023 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <optional>

#include "ttnn/operations/core/compute_kernel/compute_kernel_config.hpp"
#include "ttnn/run_operation.hpp"
#include "ttnn/tensor/tensor.hpp"
#include "ttnn/operations/core/core.hpp"

#include "ttnn/operations/normalization/layernorm/device/layernorm_types.hpp"

namespace ttnn::operations::normalization {

struct LayerNormMultiCoreOverrideRuntimeArgsCapture {
    tt::tt_metal::KernelHandle reader_kernel_id{};
    tt::tt_metal::KernelHandle writer_kernel_id{};
    uint32_t num_cores = 0;
    CoreCoord grid_size;
};

tt::tt_metal::operation::ProgramWithCallbacks layernorm_multi_core(
    const Tensor& a,
    const std::optional<const Tensor>& b,
    const std::optional<const Tensor>& gamma,
    const std::optional<const Tensor>& beta,
    Tensor& output,
    LayerNormType norm_type,
    float eps,
    bool legacy_reduction,
    bool legacy_rsqrt,
    bool use_welford,
    DeviceComputeKernelConfig compute_kernel_config);

void update_layernorm_multi_core_args(
    const LayerNormMultiCoreOverrideRuntimeArgsCapture& capture,
    const void* operation,
    const Program& program,
    const std::vector<Tensor>& input_tensors,
    const std::vector<std::optional<const Tensor>>& optional_input_tensors,
    const std::vector<Tensor>& output_tensors);

}  // namespace ttnn::operations::normalization
