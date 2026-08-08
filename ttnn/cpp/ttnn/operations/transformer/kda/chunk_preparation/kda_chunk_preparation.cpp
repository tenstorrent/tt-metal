// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
// SPDX-License-Identifier: Apache-2.0

#include "kda_chunk_preparation.hpp"

#include <tt-metalium/allocator.hpp>

#include "device/kda_chunk_preparation_device_operation.hpp"
#include "device/kda_chunk_preparation_program_factory.hpp"
#include "ttnn/device.hpp"

namespace ttnn::transformer {
namespace {

size_t output_l1_bytes_per_bank(
    uint32_t batch_heads,
    uint32_t num_chunks,
    uint32_t chunk_size,
    uint32_t key_dim,
    uint32_t value_dim,
    uint32_t output_bf16_mask,
    MeshDevice* device) {
    const auto spec = [&](const ttnn::Shape& shape, uint32_t output_index) {
        const auto dtype = (output_bf16_mask & (1U << output_index)) ? DataType::BFLOAT16 : DataType::FLOAT32;
        return tt::tt_metal::TensorSpec(shape, TensorLayout(dtype, PageConfig(Layout::TILE), ttnn::L1_MEMORY_CONFIG));
    };
    const std::vector<tt::tt_metal::TensorSpec> specs = {
        spec(ttnn::Shape({batch_heads, num_chunks, chunk_size, value_dim}), 0),
        spec(ttnn::Shape({batch_heads, num_chunks, chunk_size, key_dim}), 1),
        spec(ttnn::Shape({batch_heads, num_chunks, chunk_size, key_dim}), 2),
        spec(ttnn::Shape({batch_heads, num_chunks, chunk_size, chunk_size}), 3),
        spec(ttnn::Shape({batch_heads, num_chunks, key_dim, chunk_size}), 4),
        spec(ttnn::Shape({batch_heads, num_chunks, key_dim, 1}), 5),
        spec(ttnn::Shape({batch_heads, num_chunks, chunk_size, chunk_size}), 6),
    };
    const auto num_banks = device->allocator()->get_num_banks(tt::tt_metal::BufferType::L1);
    const auto alignment = device->allocator()->get_alignment(tt::tt_metal::BufferType::L1);
    size_t bytes_per_bank = 0;
    for (const auto& output_spec : specs) {
        bytes_per_bank += tt::tt_metal::detail::calculate_bank_size_spread(
            output_spec.compute_packed_buffer_size_bytes(),
            output_spec.compute_page_size_bytes(),
            num_banks,
            alignment);
    }
    return bytes_per_bank;
}

tt::tt_metal::MemoryConfig select_output_memory_config(
    const ttnn::Tensor& q,
    const ttnn::Tensor& v,
    const ttnn::Tensor& g,
    uint32_t chunk_size,
    bool v_flat,
    uint32_t value_heads,
    bool qk_flat,
    uint32_t key_heads,
    uint32_t output_bf16_mask) {
    const auto& q_shape = q.logical_shape();
    const auto& v_shape = v.logical_shape();
    const uint32_t batch_heads = qk_flat ? q_shape[0] * value_heads : q_shape[0];
    const uint32_t num_chunks = qk_flat ? q_shape[1] / chunk_size : q_shape[1];
    const uint32_t key_dim = qk_flat ? q_shape[2] / key_heads : q_shape[3];
    const uint32_t value_dim = v_flat ? v_shape[2] / value_heads : v_shape[3];
    auto* device = q.device();
    const auto cb_bytes =
        ttnn::prim::kda_chunk_preparation_cb_size_bytes(chunk_size, key_dim, value_dim, g.dtype(), output_bf16_mask);
    const auto output_bytes =
        output_l1_bytes_per_bank(batch_heads, num_chunks, chunk_size, key_dim, value_dim, output_bf16_mask, device);
    const auto l1_budget =
        device->l1_size_per_core() - device->allocator()->get_base_allocator_addr(tt::tt_metal::HalMemType::L1);
    return cb_bytes + output_bytes <= l1_budget ? ttnn::L1_MEMORY_CONFIG : ttnn::DRAM_MEMORY_CONFIG;
}

}  // namespace

std::vector<ttnn::Tensor> kda_chunk_preparation(
    const ttnn::Tensor& q,
    const ttnn::Tensor& k,
    const ttnn::Tensor& v,
    const ttnn::Tensor& g,
    const ttnn::Tensor& beta,
    const ttnn::Tensor& eye,
    const ttnn::Tensor& tril,
    const ttnn::Tensor& ones,
    const ttnn::Tensor& masks,
    uint32_t chunk_size,
    const std::optional<ttnn::MemoryConfig>& memory_config,
    const std::optional<ttnn::DeviceComputeKernelConfig>& compute_kernel_config,
    bool v_flat,
    uint32_t value_heads,
    bool normalize_qk,
    float scale,
    bool qk_flat,
    uint32_t key_heads,
    bool gate_flat,
    uint32_t output_bf16_mask) {
    const auto output_memory_config = memory_config.value_or(
        select_output_memory_config(q, v, g, chunk_size, v_flat, value_heads, qk_flat, key_heads, output_bf16_mask));
    const auto kernel_config = init_device_compute_kernel_config(
        q.device()->arch(),
        compute_kernel_config,
        MathFidelity::HiFi4,
        /*default_approx_mode=*/false,
        /*default_fp32_acc=*/true,
        /*default_l1_acc=*/false);
    return ttnn::prim::kda_chunk_preparation(
        q,
        k,
        v,
        g,
        beta,
        eye,
        tril,
        ones,
        masks,
        chunk_size,
        output_memory_config,
        kernel_config,
        v_flat,
        value_heads,
        normalize_qk,
        scale,
        qk_flat,
        key_heads,
        gate_flat,
        output_bf16_mask);
}

}  // namespace ttnn::transformer
