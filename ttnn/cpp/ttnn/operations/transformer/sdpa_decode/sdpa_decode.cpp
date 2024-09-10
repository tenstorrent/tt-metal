// SPDX-FileCopyrightText: © 2024 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include "sdpa_decode.hpp"

#include "device/sdpa_decode_op.hpp"
#include "ttnn/common/constants.hpp"
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

ttnn::Tensor ExecuteScaledDotProductAttentionDecode::invoke(
    uint8_t queue_id,
    const ttnn::Tensor &input_tensor_q,
    const ttnn::Tensor &input_tensor_k,
    const ttnn::Tensor &input_tensor_v,
    const std::vector<uint32_t> cur_pos,
    const std::optional<const Tensor> cur_pos_tensor,
    std::optional<float> scale,
    const std::optional<MemoryConfig> &memory_config,
    std::optional<SDPAProgramConfig> program_config,
    std::optional<DeviceComputeKernelConfig> compute_kernel_config) {
    auto arch = input_tensor_q.storage_type() == StorageType::DEVICE ? input_tensor_q.device()->arch()
                                                                     : ttnn::operations::experimental::auto_format::AutoFormat::GetDefaultDevice()->arch();
    //uint32_t max_cur_pos = *std::max_element(cur_pos.begin(), cur_pos.end());
    uint32_t k_chunk_size = 512; //get_chunk_size(max_cur_pos + 1);

    // get chunk size and then pass to sdpa decode as an attribute for prgm cache
    auto kernel_config_val = init_device_compute_kernel_config(
        input_tensor_q.device()->arch(), compute_kernel_config, MathFidelity::HiFi2, true, false, false);

    return operation::run(
               ScaledDotProductAttentionDecode{
                   .cur_pos = cur_pos,
                   .scale = scale,
                   .output_mem_config = memory_config.value_or(operation::DEFAULT_OUTPUT_MEMORY_CONFIG),
                   .program_config = program_config,
                   .compute_kernel_config = kernel_config_val,
                   .k_chunk_size = k_chunk_size,
                   .paged_attention = false},
               {input_tensor_q, input_tensor_k, input_tensor_v},
               {cur_pos_tensor, std::nullopt},
               {},
               queue_id)
        .at(0);
}

ttnn::Tensor ExecuteScaledDotProductAttentionDecode::invoke(
    const ttnn::Tensor &input_tensor_q,
    const ttnn::Tensor &input_tensor_k,
    const ttnn::Tensor &input_tensor_v,
    const std::vector<uint32_t> cur_pos,
    const std::optional<const Tensor> cur_pos_tensor,
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
        cur_pos_tensor,
        scale,
        memory_config,
        program_config,
        compute_kernel_config);
}


ttnn::Tensor ExecutePagedScaledDotProductAttentionDecode::invoke(
    uint8_t queue_id,
    const ttnn::Tensor &input_tensor_q,
    const ttnn::Tensor &input_tensor_k,
    const ttnn::Tensor &input_tensor_v,
    const ttnn::Tensor &cur_pos_tensor,
    const ttnn::Tensor &page_table_tensor,
    std::optional<float> scale,
    const std::optional<MemoryConfig> &memory_config,
    std::optional<SDPAProgramConfig> program_config,
    std::optional<DeviceComputeKernelConfig> compute_kernel_config) {
    auto arch = input_tensor_q.storage_type() == StorageType::DEVICE ? input_tensor_q.device()->arch()
                                                                     : ttnn::operations::experimental::auto_format::AutoFormat::GetDefaultDevice()->arch();
    //uint32_t max_cur_pos = *std::max_element(cur_pos.begin(), cur_pos.end());
    uint32_t k_chunk_size = 512; //get_chunk_size(max_cur_pos + 1);

    // get chunk size and then pass to sdpa decode as an attribute for prgm cache
    auto kernel_config_val = init_device_compute_kernel_config(
        input_tensor_q.device()->arch(), compute_kernel_config, MathFidelity::HiFi2, true, false, false);

    return operation::run(
               ScaledDotProductAttentionDecode{
                   .cur_pos = std::vector<uint32_t>(),
                   .scale = scale,
                   .output_mem_config = memory_config.value_or(operation::DEFAULT_OUTPUT_MEMORY_CONFIG),
                   .program_config = program_config,
                   .compute_kernel_config = kernel_config_val,
                   .k_chunk_size = k_chunk_size,
                   .paged_attention = true},
               {input_tensor_q, input_tensor_k, input_tensor_v},
               {cur_pos_tensor, page_table_tensor},
               {},
               queue_id)
        .at(0);
}

ttnn::Tensor ExecutePagedScaledDotProductAttentionDecode::invoke(
    const ttnn::Tensor &input_tensor_q,
    const ttnn::Tensor &input_tensor_k,
    const ttnn::Tensor &input_tensor_v,
    const ttnn::Tensor &cur_pos_tensor,
    const ttnn::Tensor &page_table_tensor,
    std::optional<float> scale,
    const std::optional<MemoryConfig> &memory_config,
    std::optional<SDPAProgramConfig> program_config,
    std::optional<DeviceComputeKernelConfig> compute_kernel_config) {
    return invoke(
        DefaultQueueId,
        input_tensor_q,
        input_tensor_k,
        input_tensor_v,
        cur_pos_tensor,
        page_table_tensor,
        scale,
        memory_config,
        program_config,
        compute_kernel_config);
}

}  // namespace ttnn::operations::transformer
