// SPDX-FileCopyrightText: © 2023 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <stdint.h>
#include <functional>
#include <optional>

#include "ttnn/decorators.hpp"
#include "ttnn/operation.hpp"
#include "ttnn/operations/core/compute_kernel/compute_kernel_config.hpp"
#include "ttnn/run_operation.hpp"
#include "ttnn/tensor/tensor.hpp"

namespace tt {
namespace tt_metal {
class Tensor;

tt::tt_metal::operation::ProgramWithCallbacks rotary_embedding_multi_core(
    const Tensor& input,
    const Tensor& cos,
    const Tensor& sin,
    Tensor& output,
    std::optional<uint32_t> token_idx,
    ttnn::DeviceComputeKernelConfig compute_kernel_config);

}  // namespace tt_metal
}  // namespace tt
