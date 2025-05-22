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

tt::tt_metal::operation::ProgramWithCallbacks rotary_embedding_llama_fused_qk_multi_core_sharded(
    const Tensor& q_input,
    const Tensor& k_input,
    const Tensor& cos,
    const Tensor& sin,
    const Tensor& trans_mat,
    Tensor& q_output,
    Tensor& k_output,
    ttnn::DeviceComputeKernelConfig compute_kernel_config,
    const bool row_major_QK);

}  // namespace tt_metal
}  // namespace tt
