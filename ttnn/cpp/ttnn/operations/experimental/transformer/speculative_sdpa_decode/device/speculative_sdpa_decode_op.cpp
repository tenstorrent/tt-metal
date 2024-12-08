// SPDX-FileCopyrightText: © 2024 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include "speculative_sdpa_decode_op.hpp"

#include "speculative_sdpa_decode_program_factory.hpp"
#include "ttnn/run_operation.hpp"

using namespace tt::tt_metal;

namespace ttnn::operations::experimental::transformer {

void SpeculativeScaledDotProductAttentionDecode::validate(
    const std::vector<Tensor>& input_tensors,
    const std::vector<std::optional<const Tensor>>& optional_input_tensors) const {
    TT_FATAL(input_tensors.size() == 3, "Must have 3 input tensors and mask");

    // non-causal mode and paged attention are not supported yet
    TT_FATAL(this->is_causal, "Non-causal mode is not supported yet");
    TT_FATAL(!this->paged_attention, "Paged attention is not supported yet");

    for (auto& input_tensor : input_tensors) {
        TT_FATAL(input_tensor.storage_type() == StorageType::DEVICE, "Operands to SDPA need to be on device!");
        TT_FATAL(input_tensor.buffer() != nullptr, "Operands to SDPA need to be allocated in buffers on device!");
        TT_FATAL((input_tensor.get_layout() == Layout::TILE), "Inputs to SDPA must be tilized");
        TT_FATAL(
            input_tensor.get_dtype() == DataType::BFLOAT16 || input_tensor.get_dtype() == DataType::BFLOAT8_B ||
                input_tensor.get_dtype() == DataType::BFLOAT4_B,
            "Unsupported data type {}.",
            input_tensor.get_dtype());
    }

    const auto q_shape = input_tensors.at(0).get_padded_shape();
    const auto q_shape_unpadded = input_tensors.at(0).get_logical_shape();
    const auto k_shape = input_tensors.at(1).get_padded_shape();
    const auto v_shape = input_tensors.at(2).get_padded_shape();

    // Input 0 must be sharded by height or DRAM interleaved. All other inputs must be in DRAM.
    const auto Q_memcfg = input_tensors.at(0).memory_config();
    if (input_tensors.at(0).is_sharded()) {
        TT_FATAL(Q_memcfg.memory_layout == TensorMemoryLayout::HEIGHT_SHARDED, "Error");
    } else {
        TT_FATAL(Q_memcfg.buffer_type == tt::tt_metal::BufferType::DRAM, "Error");
    }

    for (std::size_t i = 1; i < input_tensors.size(); i++) {
        TT_FATAL(input_tensors.at(i).buffer()->buffer_type() == tt::tt_metal::BufferType::DRAM, "Error");
    }
    // Output memconfig must be height sharded or DRAM
    if (this->output_mem_config.is_sharded()) {
        TT_FATAL(this->output_mem_config.memory_layout == TensorMemoryLayout::HEIGHT_SHARDED, "Error");
    }

    if (!this->is_causal) {
        if (optional_input_tensors.at(2).has_value()) {
            // Causal attention verification
            const auto& mask_tensor = optional_input_tensors.at(2).value();
            const auto mask_shape = mask_tensor.get_padded_shape();
            const auto mask_shape_unpadded = mask_tensor.get_logical_shape();

            TT_FATAL(
                mask_shape[2] == q_shape[2],
                "Expect same number of padded heads in mask as in Q, got {} and {}",
                mask_shape[2],
                q_shape[2]);
            TT_FATAL(
                mask_shape_unpadded[2] == q_shape_unpadded[2],
                "Expect same number of heads in mask as in Q, got {} and {}",
                mask_shape_unpadded[2],
                q_shape_unpadded[2]);
            if (!this->paged_attention) {
                TT_FATAL(
                    mask_shape[3] == k_shape[2],
                    "Expect same sequence length in mask as in K, got {} and {}",
                    mask_shape[3],
                    k_shape[2]);
            }
            TT_FATAL(
                mask_shape[3] % k_chunk_size == 0,
                "Mask sequence length must be multiple of chunk size, got: {} and {}",
                mask_shape[3],
                k_chunk_size);

            TT_FATAL(
                mask_tensor.get_dtype() == DataType::BFLOAT16 || mask_tensor.get_dtype() == DataType::BFLOAT8_B ||
                    mask_tensor.get_dtype() == DataType::BFLOAT4_B,
                "Unsupported data type for mask tensor: {}.",
                mask_tensor.get_dtype());
        }
    } else {
        // Uncausal attention verification
        TT_FATAL(
            not optional_input_tensors.at(2).has_value(), "Must not have attn_mask tensor for non-causal attention");
    }

    if (this->paged_attention) {
        // Paged attention verification
        TT_FATAL(!this->share_cache.value_or(false), "Share cache feature not supported for paged attention");
        const auto B = q_shape[1];

        if (this->is_causal) {
            // Check cur pos tensor for causal mode
            TT_FATAL(
                optional_input_tensors.at(0).has_value(),
                "Must have cur_pos tensor for paged attention in causal mode");
            const auto& cur_pos_tensor = optional_input_tensors.at(0).value();
            TT_FATAL(
                cur_pos_tensor.get_dtype() == DataType::INT32,
                "Expect cur_pos to be INT32, got {}",
                cur_pos_tensor.get_dtype());
            TT_FATAL(
                cur_pos_tensor.get_layout() == Layout::ROW_MAJOR,
                "Expect cur_pos to be ROW_MAJOR, got {}",
                cur_pos_tensor.get_layout());
            const auto cur_pos_shape = cur_pos_tensor.get_padded_shape();
            TT_FATAL(
                cur_pos_shape[0] == B, "cur_pos must have batch size equal to Q, got {} and {}", cur_pos_shape[0], B);
        }

        TT_FATAL(optional_input_tensors.at(1).has_value(), "Must have page_table tensor for paged attention");
        const auto& page_table_tensor = optional_input_tensors.at(1).value();

        TT_FATAL(page_table_tensor.get_dtype() == DataType::INT32, "Error");
        TT_FATAL(page_table_tensor.get_layout() == Layout::ROW_MAJOR, "Error");

        const auto page_table_shape = page_table_tensor.get_padded_shape();

        TT_FATAL(page_table_shape[0] == B, "page_table must have hidden size equal to Q");

        TT_FATAL(k_shape[2] == v_shape[2], "K and V must have same block size");
        TT_FATAL(k_shape[3] == v_shape[3] && k_shape[3] == q_shape[3], "Q, K, V must have same hidden size");

        // Validate chunk size for paged version
        TT_FATAL(k_chunk_size % 32 == 0, "Chunk size must be multiple of 32, got: {}", k_chunk_size);
        if (!this->is_causal) {
            TT_FATAL(
                (page_table_shape[1] * k_shape[2]) % k_chunk_size == 0,
                "K sequence length must be multiple of chunk size, got: {} and {}",
                page_table_shape[1] * k_shape[2],
                k_chunk_size);
        }
    } else {
        // Unpaged attention verification
        TT_FATAL(not optional_input_tensors.at(1).has_value(), "Must not have page_table tensor for unpaged attention");
        // Assert we are in decode mode if not causal
        // Q: [1, B, NH, D]
        // K: [1, B, S, D]
        // V: [1, B, S, D]

        // Batch must match
        const auto B = q_shape[1];
        if (this->share_cache.value_or(false)) {
            TT_FATAL(k_shape[0] == 1, "Share cache expects K to have batch size of 1, but got {}", k_shape[0]);
            TT_FATAL(v_shape[0] == 1, "Share cache expects V to have batch size of 1, but got {}", v_shape[0]);
        } else {
            TT_FATAL(k_shape[0] == B, "Error");
            TT_FATAL(v_shape[0] == B, "Error");
        }
        // TT_FATAL(Q_memcfg.shard_spec.value().grid.num_cores() == B, "Q must be height sharded by batch ");

        // Q seqlen must be 1 if we are running decode mode
        TT_FATAL(q_shape[0] == 1, "Error");

        // Check sequence lengths
        TT_FATAL(k_shape[-2] == v_shape[-2], "Error");

        // Validate chunk size for unpaged version
        TT_FATAL(k_chunk_size % 32 == 0, "Chunk size must be multiple of 32, got: {}", k_chunk_size);
        TT_FATAL(
            k_shape[2] % k_chunk_size == 0,
            "K sequence length must be multiple of chunk size, got: {} and {}",
            k_shape[2],
            k_chunk_size);

        // Check hidden size
        const auto D = q_shape[-1];
        TT_FATAL(k_shape[-1] == D, "Error");
        TT_FATAL(v_shape[-1] == D, "Error");

        // Check valid seqlen
        for (int i = 0; i < this->cur_pos.size(); i++) {
            TT_FATAL(this->cur_pos[i] < k_shape[-2], "cur_pos must be <= K sequence dim");
        }
    }

    // Check gqa specific validation
    TT_FATAL(
        k_shape[1] == v_shape[1],
        "Flash decode expects K and V to have same number of heads, but got {} and {}",
        k_shape[1],
        v_shape[1]);
    bool is_gqa = (k_shape[1] > 1);
    if (is_gqa) {
        TT_FATAL(!output_mem_config.is_sharded(), "Sharded output not supported for GQA");
        TT_FATAL(
            input_tensors.at(0).get_dtype() == DataType::BFLOAT16,
            "GQA expects BFLOAT16 input tensor, but got {}",
            input_tensors.at(0).get_dtype());
        uint32_t num_heads_per_kv = q_shape_unpadded[2] / k_shape[1];
        TT_FATAL(
            q_shape_unpadded[2] % k_shape[1] == 0,
            "GQA expects Q to have a multiple of K heads, but got {} and {}",
            q_shape_unpadded[2],
            k_shape[1]);
    }
}

std::vector<TensorSpec> SpeculativeScaledDotProductAttentionDecode::compute_output_specs(
    const std::vector<Tensor>& input_tensors) const {
    auto& input = input_tensors.at(0);
    auto full_output = TensorSpec(
        input.get_logical_shape(), TensorLayout(input.get_dtype(), PageConfig(Layout::TILE), output_mem_config));
    auto speculated_output = TensorSpec(
        input.get_logical_shape(), TensorLayout(input.get_dtype(), PageConfig(Layout::TILE), output_mem_config));

    auto batch_size = input.get_logical_shape()[1];
    ttnn::SimpleShape spec_stat_shape{1, 1, 1, batch_size};
    MemoryConfig stat_mem_cfg = MemoryConfig{
        .memory_layout = tt::tt_metal::TensorMemoryLayout::INTERLEAVED,
        .buffer_type = BufferType::DRAM,
        .shard_spec = std::nullopt};
    auto l2_dist_tensor =
        TensorSpec(spec_stat_shape, TensorLayout(DataType::FLOAT32, PageConfig(Layout::ROW_MAJOR), stat_mem_cfg));
    auto l2_norm_tensor =
        TensorSpec(spec_stat_shape, TensorLayout(DataType::FLOAT32, PageConfig(Layout::ROW_MAJOR), stat_mem_cfg));
    return {full_output, speculated_output, l2_dist_tensor, l2_norm_tensor};
}

operation::ProgramWithCallbacks SpeculativeScaledDotProductAttentionDecode::create_program(
    const std::vector<Tensor>& input_tensors,
    const std::vector<std::optional<const Tensor>>& optional_input_tensors,
    std::vector<Tensor>& output_tensors) const {
    auto& input_tensor_q = input_tensors.at(0);
    auto& input_tensor_k = input_tensors.at(1);
    auto& input_tensor_v = input_tensors.at(2);

    auto& cur_pos_tensor = optional_input_tensors.at(0);
    auto& page_table_tensor = optional_input_tensors.at(1);
    auto& attn_mask = optional_input_tensors.at(2);

    auto& full_output_tensor = output_tensors.at(0);
    auto& speculated_output_tensor = output_tensors.at(1);
    auto& l2_dist_tensor = output_tensors.at(2);
    auto& l2_norm_tensor = output_tensors.at(3);

    // set default values for scale, lambda, and speculative_chunk_size if not provided
    auto scale = this->scale;
    if (not scale.has_value()) {
        scale = 1.0f / std::sqrt(static_cast<float>(input_tensor_q.get_padded_shape()[-1]));
    }

    auto lambda = this->lambda_;
    if (not lambda.has_value()) {
        lambda = 0.2f;
    }

    auto speculative_chunk_size = this->speculative_chunk_size;
    if (not speculative_chunk_size.has_value()) {
        speculative_chunk_size = 128;
    }

    return detail::speculative_sdpa_decode_multi_core(
        input_tensor_q,
        input_tensor_k,
        input_tensor_v,
        cur_pos_tensor,
        page_table_tensor,
        attn_mask,
        full_output_tensor,
        speculated_output_tensor,
        l2_dist_tensor,
        l2_norm_tensor,
        this->is_causal,
        this->cur_pos,
        scale,
        this->compute_kernel_config,
        this->program_config,
        this->k_chunk_size,
        this->share_cache);
}

operation::Hash SpeculativeScaledDotProductAttentionDecode::compute_program_hash(
    const std::vector<Tensor>& input_tensors,
    const std::vector<std::optional<const Tensor>>& optional_input_tensors) const {
    bool has_cur_pos = optional_input_tensors.at(0).has_value();
    bool has_attn_mask = optional_input_tensors.at(2).has_value();
    return operation::hash_operation<SpeculativeScaledDotProductAttentionDecode>(
        this->lambda_,
        this->speculative_chunk_size,
        this->scale,
        this->output_mem_config,
        this->program_config,
        this->compute_kernel_config,
        this->k_chunk_size,
        this->paged_attention,
        this->is_causal,
        has_attn_mask,
        has_cur_pos,
        input_tensors);
}

}  // namespace ttnn::operations::experimental::transformer
