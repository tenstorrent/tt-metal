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
        TT_FATAL(input_tensor.get_dtype() == DataType::BFLOAT16 || input_tensor.get_dtype() == DataType::BFLOAT8_B);
    }

    const auto& mask_option = optional_input_tensors.at(0);
    if (mask_option.has_value()){
        TT_FATAL(!this->is_causal, "Causal SDPA does not take mask as input");
        auto mask = optional_input_tensors.at(0).value();
        TT_FATAL(mask.storage_type() == StorageType::DEVICE, "When mask is provided to SDPA, the tensor must be on device");
        TT_FATAL(input_tensors.at(0).device() == mask.device(), "When mask is provided to SDPA, it must be on the same device as the input tensors");
        TT_FATAL(mask.get_layout() == Layout::TILE, "When mask is provided to SDPA, it must be tilized");
        TT_FATAL(mask.get_dtype() == DataType::BFLOAT16 || mask.get_dtype() == DataType::BFLOAT8_B, "When mask is provided to SDPA, it must be in BF16 or BFP8 dataformat");
        TT_FATAL(input_tensors.at(0).get_dtype() == mask_option.value().get_dtype(), "When mask is provided to SDPA, it must have the same dataformat as the input tensors");

        TT_FATAL(mask.buffer()->buffer_type() == tt::tt_metal::BufferType::DRAM, "When mask is provided to SDPA, it must be in DRAM");
    }

    const auto q_shape = input_tensors.at(0).get_legacy_shape();
    const auto k_shape = input_tensors.at(1).get_legacy_shape();
    const auto v_shape = input_tensors.at(2).get_legacy_shape();

    // assert all dataformats are the same
    TT_FATAL(
        input_tensors.at(0).get_dtype() == input_tensors.at(1).get_dtype() &&
        input_tensors.at(0).get_dtype() == input_tensors.at(2).get_dtype(), "All inputs to SDPA must have the same dataformat");

    if (this->is_causal) {
        // All inputs must be in DRAM
        for (auto& input_tensor : input_tensors) {
            TT_FATAL(input_tensor.buffer()->buffer_type() == tt::tt_metal::BufferType::DRAM, "All inputs to causal SDPA must be in DRAM");
        }
        // Check sequence lengths
        TT_FATAL(q_shape[-2] == k_shape[-2] && q_shape[-2] == v_shape[-2], "Q, K, V sequence dim must match. Got Q: {}, K: {}, V: {}", q_shape[-2], k_shape[-2], v_shape[-2]);

        // Check batch size
        TT_FATAL(q_shape[-4] == k_shape[-4] && q_shape[-4] == v_shape[-4], "Q, K, V batch dim must match. Got Q: {}, K: {}, V: {}", q_shape[-4], k_shape[-4], v_shape[-4]);

        // Check hidden size
        TT_FATAL(q_shape[-1] == k_shape[-1] && q_shape[-1] == v_shape[-1], "Q, K, V hidden dim must match. Got Q: {}, K: {}, V: {}", q_shape[-1], k_shape[-1], v_shape[-1]);

        // Check kv heads
        TT_FATAL(k_shape[-3] == v_shape[-3], "K, V heads dim must match. Got K: {}, V: {}", k_shape[-3], v_shape[-3]);

        // Check qkv heads
        TT_FATAL(q_shape[-3] >= k_shape[-3], "Q heads must be >= K heads. Got Q: {}, K: {}", q_shape[-3], k_shape[-3]);

        TT_FATAL(this->output_mem_config.buffer_type == tt::tt_metal::BufferType::DRAM, "Output must be in DRAM");

    } else {
        const auto mask_shape = mask_option.value().get_legacy_shape();

        // Input 0 must be sharded by height. All other inputs must be in DRAM.
        const auto Q_memcfg = input_tensors.at(0).memory_config();
        TT_FATAL(input_tensors.at(0).is_sharded() == true);
        TT_FATAL(Q_memcfg.memory_layout == TensorMemoryLayout::HEIGHT_SHARDED);

        for (std::size_t i = 1; i < input_tensors.size(); i++) {
            TT_FATAL(input_tensors.at(i).buffer()->buffer_type() == tt::tt_metal::BufferType::DRAM);
        }
        // Output memconfig must be height sharded, same as input
        TT_FATAL(this->output_mem_config.is_sharded());
        TT_FATAL(this->output_mem_config.memory_layout == TensorMemoryLayout::HEIGHT_SHARDED);

        // Assert we are in decode mode if not causal
        // Q: [1, B, NH, D]
        // K: [1, B, S, D]
        // V: [1, B, S, D]
        // mask: [1, B, NH, VS]

        // Batch must match
        const auto B = q_shape[1];
        TT_FATAL(k_shape[1] == B);
        TT_FATAL(v_shape[1] == B);
        TT_FATAL(mask_shape[1] == B);
        TT_FATAL(Q_memcfg.shard_spec.value().grid.num_cores() == B, "Q must be height sharded by batch ");

        // NKV must be 1 if we are running in this decode mode
        TT_FATAL(q_shape[0] == 1);
        TT_FATAL(k_shape[0] == 1);
        TT_FATAL(v_shape[0] == 1);
        TT_FATAL(mask_shape[0] == 1);

        // Check sequence lengths
        TT_FATAL(k_shape[-2] == v_shape[-2]);

        // Check hidden size
        const auto D = q_shape[-1];
        TT_FATAL(k_shape[-1] == D);
        TT_FATAL(v_shape[-1] == D);

        // Check NH
        TT_FATAL(q_shape[2] == mask_shape[2]);

        // Check valid seqlen
        TT_FATAL(valid_seq_len.has_value(), "Non-causal SDPA must set valid_seq_len");
        TT_FATAL(valid_seq_len.value() == mask_shape[-1], "Mask sequence dim must match valid_seq_len");
        TT_FATAL(valid_seq_len.value() <= k_shape[-2], "valid_seq_len must be <= K sequence dim");
    }

    if (this->program_config.has_value()) {
        auto q_chunk_size = program_config->q_chunk_size;
        auto k_chunk_size = program_config->k_chunk_size;

        TT_FATAL(q_shape[-2] % q_chunk_size == 0);
        TT_FATAL(k_shape[-2] % k_chunk_size == 0);

        if (!this->is_causal) {
            TT_FATAL(q_chunk_size == q_shape[-2], "Non-causal SDPA must have q_chunk_size == q_shape[-2]");
            TT_FATAL(
                this->valid_seq_len.value() % k_chunk_size == 0, "valid_seq_len must be divisible by k_chunk_size");
        }
    } else {
        if (!this->is_causal) {
            TT_FATAL(false, "Non-causal SDPA must use multi-core program config");
        }
    }
}

std::vector<tt::tt_metal::Shape> ScaledDotProductAttention::compute_output_shapes(
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
        this->program_config,
        this->valid_seq_len);
}

}  // namespace ttnn::operations::transformer
