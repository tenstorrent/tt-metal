// SPDX-FileCopyrightText: 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <optional>

#include "ttnn/operations/core/compute_kernel/compute_kernel_config.hpp"
#include "ttnn/operations/experimental/minimal_matmul/device/minimal_matmul_device_operation_types.hpp"
#include "ttnn/tensor/tensor.hpp"

namespace ttnn::experimental::nomic_embed_text_v2_moe {

// input_tensor @ weight + bias with M > N: minimal_matmul's compute kernel and split, its operands multicast down
// the core columns and along the core rows. See the binding for the contract.
Tensor pirotski_matmul(
    const Tensor& input_tensor,
    const Tensor& weight,
    const ttnn::experimental::prim::MinimalMatmulConfig& config,
    tt::tt_metal::DataType dtype,
    const std::optional<Tensor>& bias = std::nullopt,
    const std::optional<tt::tt_metal::MemoryConfig>& memory_config = std::nullopt,
    const std::optional<DeviceComputeKernelConfig>& compute_kernel_config = std::nullopt,
    bool resident_weights = false,
    uint32_t strip_m_tiles = 0,
    bool strip_read_ahead = false,
    bool read_ahead = false);

}  // namespace ttnn::experimental::nomic_embed_text_v2_moe
