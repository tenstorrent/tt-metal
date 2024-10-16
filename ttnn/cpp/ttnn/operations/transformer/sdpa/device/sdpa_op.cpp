// SPDX-FileCopyrightText: © 2024 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include "sdpa_op.hpp"

#include "sdpa_program_factory.hpp"
#include "ttnn/run_operation.hpp"

namespace ttnn::operations::transformer {

void ScaledDotProductAttention::validate(
    const std::vector<Tensor>& input_tensors,
    const std::vector<std::optional<const Tensor>>& optional_input_tensors) const {
    TT_FATAL(input_tensors.size() == 3 and optional_input_tensors.size() == 1, "Must have 3 input tensors and optional mask");

    for (auto& input_tensor : input_tensors) {
        TT_FATAL(input_tensor.storage_type() == StorageType::DEVICE, "Operands to SDPA need to be on device");
        TT_FATAL(input_tensor.buffer() != nullptr, "Operands to SDPA need to be allocated in buffers on device");
        TT_FATAL((input_tensor.get_layout() == Layout::TILE), "Inputs to SDPA must be tilized");
        TT_FATAL(input_tensor.get_dtype() == DataType::BFLOAT16 || input_tensor.get_dtype() == DataType::BFLOAT8_B, "Error");
        TT_FATAL(input_tensor.buffer()->buffer_type() == tt::tt_metal::BufferType::DRAM, "Operands to SDPA need to be in DRAM");
    }

    TT_FATAL(!(this->is_causal && optional_input_tensors.at(0).has_value()), "is_causal and attn_mask cannot both be present. Got is_causal: {}, attn_mask: {}", this->is_causal, optional_input_tensors.at(0).has_value());

    const auto& mask_option = optional_input_tensors.at(0);
    if (mask_option.has_value()){
        auto mask = optional_input_tensors.at(0).value();
        TT_FATAL(mask.storage_type() == StorageType::DEVICE, "When mask is provided to SDPA, the tensor must be on device");
        TT_FATAL(input_tensors.at(0).device() == mask.device(), "When mask is provided to SDPA, it must be on the same device as the input tensors");
        TT_FATAL(mask.get_layout() == Layout::TILE, "When mask is provided to SDPA, it must be tilized");
        TT_FATAL(mask.get_dtype() == DataType::BFLOAT16 || mask.get_dtype() == DataType::BFLOAT8_B || mask.get_dtype() == DataType::BFLOAT4_B, "When mask is provided to SDPA, it must be in BF16, BFP8, or BFP4 dataformat");

        TT_FATAL(mask.buffer()->buffer_type() == tt::tt_metal::BufferType::DRAM, "When mask is provided to SDPA, it must be in DRAM");
    }

    // assert all dataformats are the same
    TT_FATAL(
        input_tensors.at(0).get_dtype() == input_tensors.at(1).get_dtype() &&
        input_tensors.at(0).get_dtype() == input_tensors.at(2).get_dtype(), "All inputs to SDPA must have the same dataformat");

    TT_FATAL(this->output_mem_config.buffer_type == tt::tt_metal::BufferType::DRAM, "Output must be in DRAM");

    // Check shapes
    const auto q_shape = input_tensors.at(0).get_legacy_shape();
    const auto k_shape = input_tensors.at(1).get_legacy_shape();
    const auto v_shape = input_tensors.at(2).get_legacy_shape();
    const auto B = q_shape[0];
    const auto nqh = q_shape[1];
    const auto nkv = k_shape[1];
    const auto S = q_shape[2];
    const auto DH = q_shape[3];

    TT_FATAL(k_shape[0] == B && v_shape[0] == B, "K and V batch must match. Got K: {}, V: {}", k_shape[0], v_shape[0]);
    TT_FATAL(v_shape[1] == nkv, "K and V num_heads must match. Got K: {}, V: {}", k_shape[1], v_shape[1]);
    TT_FATAL(k_shape[2] == S && v_shape[2] == S, "K and V sequence length must match. Got K: {}, V: {}", k_shape[2], v_shape[2]);
    TT_FATAL(k_shape[3] == DH && v_shape[3] == DH, "K and V hidden dim must match. Got K: {}, V: {}", k_shape[3], v_shape[3]);
    TT_FATAL(nqh >= nkv && nqh % nkv == 0, "Q num_heads must be >= K num_heads and divisible by K num_heads. Got Q: {}, K: {}", nqh, nkv);

    if (mask_option.has_value()) {
        const auto mask_shape = mask_option.value().get_legacy_shape();

        TT_FATAL(mask_shape[0] == B, "Mask batch dim must match Q batch dim");
        TT_FATAL(mask_shape[1] == 1, "Mask num_heads must be 1 to be broadcasted across all heads");
        TT_FATAL(mask_shape[2] == S, "Mask sequence length must match Q sequence length");
        TT_FATAL(mask_shape[3] == S, "Mask sequence length must match Q sequence length");
    }

    if (this->program_config.has_value()) {
        auto q_chunk_size = program_config->q_chunk_size;
        auto k_chunk_size = program_config->k_chunk_size;

        TT_FATAL(q_shape[-2] % q_chunk_size == 0, "q_chunk_size must divide q_shape[-2]. Got q_chunk_size: {}, q_shape[-2]: {}", q_chunk_size, q_shape[-2]);
        TT_FATAL(k_shape[-2] % k_chunk_size == 0, "k_chunk_size must divide k_shape[-2]. Got k_chunk_size: {}, k_shape[-2]: {}", k_chunk_size, k_shape[-2]);

    }
}

std::vector<tt::tt_metal::LegacyShape> ScaledDotProductAttention::compute_output_shapes(
    const std::vector<Tensor>& input_tensors) const {
    return {input_tensors.at(0).get_legacy_shape()};
}

std::vector<Tensor> ScaledDotProductAttention::create_output_tensors(const std::vector<Tensor>& input_tensors) const {
    return operation::generic_create_output_tensors(
        *this, input_tensors, input_tensors.at(0).get_dtype(), Layout::TILE, this->output_mem_config);
}

operation::ProgramWithCallbacks ScaledDotProductAttention::create_program(
    const std::vector<Tensor>& input_tensors,
    const std::vector<std::optional<const Tensor>>& optional_input_tensors,
    std::vector<Tensor>& output_tensors) const {
    auto& input_tensor_q = input_tensors.at(0);
    auto& input_tensor_k = input_tensors.at(1);
    auto& input_tensor_v = input_tensors.at(2);
    auto& output_tensor = output_tensors.at(0);
    const auto& attn_mask = optional_input_tensors.at(0);

    auto scale = this->scale;
    if (not scale.has_value()) {
        scale = 1.0f / std::sqrt(static_cast<float>(input_tensor_q.get_legacy_shape()[-1]));
    }

    std::size_t q_chunk_size = this->program_config ? this->program_config->q_chunk_size : 32;
    std::size_t k_chunk_size = this->program_config ? this->program_config->k_chunk_size : 32;

    return detail::sdpa_multi_core(
        input_tensor_q,
        input_tensor_k,
        input_tensor_v,
        output_tensor,
        attn_mask,
        scale,
        this->is_causal,
        q_chunk_size,
        k_chunk_size,
        this->compute_kernel_config,
        this->program_config);
}

}  // namespace ttnn::operations::transformer
