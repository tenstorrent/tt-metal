// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
// SPDX-License-Identifier: Apache-2.0

#include "prepare_chunk_recurrence.hpp"

#include "device/prepare_chunk_recurrence_device_operation.hpp"
#include "ttnn/operations/experimental/kda/factory/kda_factory_utils.hpp"

namespace ttnn::experimental::kda {

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
    using namespace ttnn::experimental::prim::kda_factory_detail;
    constexpr std::string_view operation_name = "prepare_chunk_recurrence";
    check_allocated_device_tensor(q, operation_name, "q");
    check_allocated_device_tensor(k, operation_name, "k");
    check_allocated_device_tensor(v, operation_name, "v");
    check_allocated_device_tensor(g, operation_name, "g");
    check_allocated_device_tensor(beta, operation_name, "beta");
    check_allocated_device_tensor(eye, operation_name, "eye");
    check_allocated_device_tensor(tril, operation_name, "tril");
    check_allocated_device_tensor(ones, operation_name, "ones");
    const auto output_memory_config = memory_config.value_or(ttnn::DRAM_MEMORY_CONFIG);
    TT_FATAL(num_heads > 0, "prepare_chunk_recurrence: num_heads must be positive");
    constexpr uint32_t allowed_bf16_mask = 0x37;
    TT_FATAL(
        (output_bf16_mask & ~allowed_bf16_mask) == 0,
        "prepare_chunk_recurrence: unsupported KDA prep BF16 mask 0x{:x}",
        output_bf16_mask);
    check_output_interleaved(output_memory_config, operation_name);
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
