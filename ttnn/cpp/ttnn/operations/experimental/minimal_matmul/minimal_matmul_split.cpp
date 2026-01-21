// SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#include "minimal_matmul_split.hpp"
#include "device/minimal_matmul_device_operation.hpp"
#include <tt-metalium/math.hpp>
#include <tt-metalium/tt_metal.hpp>
#include <tt-metalium/constants.hpp>
#include "ttnn/operations/core/compute_kernel/compute_kernel_config.hpp"

using namespace tt::tt_metal;
using namespace tt::constants;

namespace ttnn::operations::experimental::minimal_matmul_split {

std::vector<ttnn::Tensor> ExecuteMinimalMatmulSplit::invoke(
    const ttnn::Tensor& input_tensor,
    const ttnn::Tensor& weight_tensor,
    int32_t chunks,
    int32_t dim,
    const std::optional<ttnn::Tensor>& bias_tensor,
    std::optional<unary::UnaryWithParam> fused_activation,
    const std::optional<const MinimalMatmulSplitConfig>& config,
    const std::optional<MemoryConfig>& memory_config,
    std::optional<const DataType> dtype,
    std::optional<DeviceComputeKernelConfig> compute_kernel_config) {
    // Validate chunks
    TT_FATAL(chunks >= 1, "minimal_matmul_split requires chunks >= 1, got chunks={}", chunks);

    // Validate dim
    TT_FATAL(dim == -1, "minimal_matmul_split currently only supports dim=-1, got dim={}", dim);

    // Validate that N is divisible by chunks
    const auto& weight_shape = weight_tensor.logical_shape();
    const uint32_t N = weight_shape[-1];
    TT_FATAL(N % chunks == 0, "Output width N={} must be divisible by chunks={}", N, chunks);

    // Validate that each chunk is tile-aligned (N/chunks must be multiple of TILE_WIDTH=32)
    const uint32_t N_per_chunk = N / chunks;
    TT_FATAL(
        N_per_chunk % tt::constants::TILE_WIDTH == 0,
        "Each chunk size N/chunks={} must be a multiple of TILE_WIDTH={}",
        N_per_chunk,
        tt::constants::TILE_WIDTH);

    // Call the unified minimal_matmul device operation with chunks and dim parameters
    return ttnn::prim::minimal_matmul(
        input_tensor,
        weight_tensor,
        bias_tensor,
        std::move(fused_activation),
        config,
        memory_config,
        dtype,
        compute_kernel_config,
        chunks,
        dim);
}

}  // namespace ttnn::operations::experimental::minimal_matmul_split
