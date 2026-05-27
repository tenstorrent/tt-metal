// SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#include "rotary_embedding_hf.hpp"

#include "device/rotary_embedding_hf_device_operation.hpp"

namespace ttnn::prim {

tt::tt_metal::Tensor rotary_embedding_hf(
    const tt::tt_metal::Tensor& input,
    const tt::tt_metal::Tensor& cos,
    const tt::tt_metal::Tensor& sin,
    bool is_decode_mode,
    const tt::tt_metal::MemoryConfig& output_mem_config,
    ttnn::DeviceComputeKernelConfig compute_kernel_config);

}  // namespace ttnn::prim

namespace ttnn::experimental {

Tensor rotary_embedding_hf(
    const Tensor& input_tensor,
    const Tensor& cos_cache,
    const Tensor& sin_cache,
    const bool is_decode_mode,
    const std::optional<MemoryConfig>& memory_config,
    std::optional<const ttnn::DeviceComputeKernelConfig> compute_kernel_config) {
    // Device-only op: kernel config needs arch() and the primitive enqueues on device.
    TT_FATAL(
        input_tensor.storage_type() == StorageType::DEVICE,
        "rotary_embedding_hf requires input_tensor on device; host tensors are not supported.");
    TT_FATAL(
        cos_cache.storage_type() == StorageType::DEVICE,
        "rotary_embedding_hf requires cos_cache on device; host tensors are not supported.");
    TT_FATAL(
        sin_cache.storage_type() == StorageType::DEVICE,
        "rotary_embedding_hf requires sin_cache on device; host tensors are not supported.");

    auto output_mem_config = memory_config.value_or(input_tensor.memory_config());

    auto* mesh_device = input_tensor.device();
    TT_FATAL(mesh_device != nullptr, "rotary_embedding_hf requires a non-null device on input_tensor.");
    auto arch = mesh_device->arch();
    auto kernel_config = init_device_compute_kernel_config(
        arch, compute_kernel_config, tt::tt_metal::MathFidelity::HiFi4, true, false, false);

    return ttnn::prim::rotary_embedding_hf(
        input_tensor, cos_cache, sin_cache, is_decode_mode, output_mem_config, kernel_config);
}

}  // namespace ttnn::experimental
