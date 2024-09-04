// SPDX-FileCopyrightText: © 2024 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include "sdpa_decode_gqa.hpp"

#include "device/sdpa_decode_gqa_op.hpp"
#include "ttnn/common/constants.hpp"
#include "ttnn/operations/core/core.hpp"
#include "ttnn/operations/data_movement/transpose/transpose.hpp"
#include "ttnn/run_operation.hpp"

namespace {
uint32_t get_chunk_size(uint32_t s) {
    if (s <= 128) {
        return 32;
    }
    if (s <= 256) {
        return 256;
    }
    return 512;
}
}  // namespace

namespace ttnn::operations::transformer {

ttnn::Tensor ExecuteScaledDotProductAttentionGQADecode::invoke(
    uint8_t queue_id,
    const ttnn::Tensor &input_tensor_q,
    const ttnn::Tensor &input_tensor_k,
    const ttnn::Tensor &input_tensor_v,
    const std::vector<uint32_t> cur_pos,
    std::optional<float> scale,
    const std::optional<MemoryConfig> &memory_config,
    std::optional<SDPAProgramConfig> program_config,
    std::optional<DeviceComputeKernelConfig> compute_kernel_config) {
    auto arch = input_tensor_q.storage_type() == StorageType::DEVICE ? input_tensor_q.device()->arch()
                                                                     : ttnn::operations::experimental::auto_format::AutoFormat::GetDefaultDevice()->arch();
    // formatting input tensors
    auto q_shape = input_tensor_q.get_legacy_shape();
    auto k_shape = input_tensor_k.get_legacy_shape();
    uint32_t B = k_shape[0];
    uint32_t NQH = q_shape[1];
    uint32_t NKH = k_shape[1];
    uint32_t D = k_shape[3];
    uint32_t NG = NQH / NKH;

    auto input_tensor_q_gqa =
        ttnn::to_layout(input_tensor_q, ttnn::ROW_MAJOR_LAYOUT, std::nullopt, std::nullopt, (Device *)nullptr);
    input_tensor_q_gqa = ttnn::transpose(input_tensor_q_gqa, 1, 2);
    input_tensor_q_gqa = ttnn::reshape(input_tensor_q_gqa, ttnn::Shape{std::array<uint32_t, 4>{1, B * NKH, NG, D}});
    input_tensor_q_gqa =
        ttnn::to_layout(input_tensor_q_gqa, ttnn::TILE_LAYOUT, std::nullopt, std::nullopt, (Device *)nullptr);

    auto input_tensor_k_gqa =
        ttnn::reshape(input_tensor_k, ttnn::Shape{std::array<uint32_t, 4>{1, B * NKH, k_shape[2], D}});
    auto input_tensor_v_gqa =
        ttnn::reshape(input_tensor_v, ttnn::Shape{std::array<uint32_t, 4>{1, B * NKH, k_shape[2], D}});

    uint32_t max_cur_pos = *std::max_element(cur_pos.begin(), cur_pos.end());
    uint32_t k_chunk_size = get_chunk_size(max_cur_pos + 1);
    // get chunk size and then pass to sdpa decode as an attribute for prgm cache
    auto kernel_config_val = init_device_compute_kernel_config(
        input_tensor_q.device()->arch(), compute_kernel_config, MathFidelity::HiFi2, true, false, false);

    auto output_tensors = operation::run(
        ScaledDotProductAttentionGQADecode{
            .cur_pos = cur_pos,
            .scale = scale,
            .output_mem_config = memory_config.value_or(operation::DEFAULT_OUTPUT_MEMORY_CONFIG),
            .program_config = program_config,
            .compute_kernel_config = kernel_config_val,
            .k_chunk_size = k_chunk_size},
        {input_tensor_q_gqa, input_tensor_k_gqa, input_tensor_v_gqa},
        {},
        {},
        queue_id);

    // formatting output tensor
    auto output_tensor = output_tensors.at(0);
    output_tensor =
        ttnn::to_layout(output_tensor, ttnn::ROW_MAJOR_LAYOUT, std::nullopt, std::nullopt, (Device *)nullptr);
    output_tensor = ttnn::reshape(output_tensor, ttnn::Shape{std::array<uint32_t, 4>{1, B, NQH, D}});
    output_tensor = ttnn::to_layout(output_tensor, ttnn::TILE_LAYOUT, std::nullopt, std::nullopt, (Device *)nullptr);
    return output_tensor;
}

ttnn::Tensor ExecuteScaledDotProductAttentionGQADecode::invoke(
    const ttnn::Tensor &input_tensor_q,
    const ttnn::Tensor &input_tensor_k,
    const ttnn::Tensor &input_tensor_v,
    const std::vector<uint32_t> cur_pos,
    std::optional<float> scale,
    const std::optional<MemoryConfig> &memory_config,
    std::optional<SDPAProgramConfig> program_config,
    std::optional<DeviceComputeKernelConfig> compute_kernel_config) {
    return invoke(
        DefaultQueueId,
        input_tensor_q,
        input_tensor_k,
        input_tensor_v,
        cur_pos,
        scale,
        memory_config,
        program_config,
        compute_kernel_config);
}

}  // namespace ttnn::operations::transformer
