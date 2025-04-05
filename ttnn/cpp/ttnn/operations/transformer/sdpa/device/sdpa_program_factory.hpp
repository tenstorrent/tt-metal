// SPDX-FileCopyrightText: © 2024 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <stdint.h>
#include <cstddef>
#include <optional>

#include "ttnn/operation.hpp"
#include "ttnn/operations/core/compute_kernel/compute_kernel_config.hpp"
#include "ttnn/operations/transformer/sdpa_config.hpp"
#include "ttnn/tensor/tensor.hpp"

namespace ttnn {
namespace operations {
namespace transformer {
struct SDPAProgramConfig;
}  // namespace transformer
}  // namespace operations
}  // namespace ttnn

namespace ttnn::operations::transformer::detail {

tt::tt_metal::operation::ProgramWithCallbacks sdpa_multi_core(
    const Tensor& input_tensor_q,
    const Tensor& input_tensor_k,
    const Tensor& input_tensor_v,
    const Tensor& output_tensor,
    const std::optional<const Tensor>& attn_mask,
    const std::optional<const Tensor>& page_table,
    const std::optional<int64_t>& chunk_start_idx,
    std::optional<float> scale,
    bool is_causal,
    std::size_t q_chunk_size,
    std::size_t k_chunk_size,
    DeviceComputeKernelConfig compute_kernel_config,
    std::optional<SDPAProgramConfig> program_config);

}  // namespace ttnn::operations::transformer::detail
