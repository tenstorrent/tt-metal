// SPDX-FileCopyrightText: © 2023 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <functional>

#include "ttnn/tensor/tensor.hpp"
#include "ttnn/run_operation.hpp"
#include "ttnn/operations/core/compute_kernel/compute_kernel_config.hpp"

namespace tt {
namespace tt_metal {

operation::ProgramWithCallbacks rotary_embedding_llama_multi_core(
    const Tensor &input,
    const Tensor &cos,
    const Tensor &sin,
    const Tensor &trans_mat,
    Tensor &output,
    ttnn::DeviceComputeKernelConfig compute_kernel_config);

}  // namespace tt_metal
}  // namespace tt
