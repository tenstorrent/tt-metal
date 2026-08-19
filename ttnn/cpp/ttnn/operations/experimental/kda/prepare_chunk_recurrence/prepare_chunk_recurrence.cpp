// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
// SPDX-License-Identifier: Apache-2.0

#include "prepare_chunk_recurrence.hpp"

#include "device/prepare_chunk_recurrence_device_operation.hpp"

namespace ttnn::experimental::kda {
namespace {

void validate_allocated_device_tensor(const ttnn::Tensor& tensor, const char* name) {
    TT_FATAL(
        tensor.storage_type() == StorageType::DEVICE && tensor.buffer() != nullptr,
        "prepare_chunk_recurrence: {} must be an allocated device tensor",
        name);
}

}  // namespace

std::vector<ttnn::Tensor> prepare_chunk_recurrence(
    const ttnn::Tensor& q,
    const ttnn::Tensor& k,
    const ttnn::Tensor& v,
    const ttnn::Tensor& g,
    const ttnn::Tensor& beta,
    const ttnn::Tensor& eye,
    const ttnn::Tensor& tril,
    const ttnn::Tensor& ones,
    uint32_t num_heads,
    const std::optional<ttnn::MemoryConfig>& memory_config,
    const std::optional<ttnn::DeviceComputeKernelConfig>& compute_kernel_config,
    uint32_t output_bf16_mask) {
    validate_allocated_device_tensor(q, "q");
    validate_allocated_device_tensor(k, "k");
    validate_allocated_device_tensor(v, "v");
    validate_allocated_device_tensor(g, "g");
    validate_allocated_device_tensor(beta, "beta");
    validate_allocated_device_tensor(eye, "eye");
    validate_allocated_device_tensor(tril, "tril");
    validate_allocated_device_tensor(ones, "ones");
    const auto output_memory_config = memory_config.value_or(ttnn::DRAM_MEMORY_CONFIG);
    TT_FATAL(num_heads > 0, "prepare_chunk_recurrence: num_heads must be positive");
    constexpr uint32_t allowed_bf16_mask = 0x37;
    TT_FATAL(
        (output_bf16_mask & ~allowed_bf16_mask) == 0,
        "prepare_chunk_recurrence: unsupported KDA prep BF16 mask 0x{:x}",
        output_bf16_mask);
    TT_FATAL(!output_memory_config.is_sharded(), "prepare_chunk_recurrence: output memory must be interleaved");
    const auto kernel_config = init_device_compute_kernel_config(
        q.device()->arch(),
        compute_kernel_config,
        MathFidelity::HiFi4,
        /*default_approx_mode=*/true,
        /*default_fp32_acc=*/true,
        /*default_l1_acc=*/false);
    return ttnn::experimental::prim::prepare_chunk_recurrence(
        q, k, v, g, beta, eye, tril, ones, num_heads, output_memory_config, kernel_config, output_bf16_mask);
}

}  // namespace ttnn::experimental::kda
