// SPDX-FileCopyrightText: © 2024 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include "sdpa_decode_op.hpp"

#include "sdpa_decode_program_factory.hpp"
#include "ttnn/run_operation.hpp"

using namespace tt::tt_metal;

namespace ttnn::operations::transformer {

void ScaledDotProductAttentionDecode::validate(
    const std::vector<Tensor>& input_tensors,
    const std::vector<std::optional<const Tensor>>& optional_input_tensors) const {
    bool use_mla = this->use_mla.value_or(false);

    if (use_mla) {
        TT_FATAL(input_tensors.size() == 2, "Must have 2 input tensors and mask");
        TT_FATAL(this->head_dim_v.has_value(), "Must provide head_dim_v for multi-latent attention decode");
        TT_FATAL(!this->paged_attention, "Paged attention is untested for multi-latent attention decode!");
        TT_FATAL(this->is_causal, "Multi-latent attention decode only tested for causal!");
    } else {
        TT_FATAL(input_tensors.size() == 3, "Must have 3 input tensors and mask");
    }

    for (auto& input_tensor : input_tensors) {
        TT_FATAL(input_tensor.storage_type() == StorageType::DEVICE, "Operands to SDPA need to be on device!");
        TT_FATAL(input_tensor.buffer() != nullptr, "Operands to SDPA need to be allocated in buffers on device!");
        TT_FATAL(
            input_tensor.dtype() == DataType::BFLOAT16 || input_tensor.dtype() == DataType::BFLOAT8_B ||
                input_tensor.dtype() == DataType::BFLOAT4_B,
            "Unsupported data type {}.",
            input_tensor.dtype());
    }

    for (size_t i = 1; i < input_tensors.size(); i++) {
        TT_FATAL(input_tensors.at(i).layout() == Layout::TILE, "Inputs to SDPA must be tilized except for Q");
    }

    const auto q_shape = input_tensors.at(0).padded_shape();
    const auto q_shape_unpadded = input_tensors.at(0).logical_shape();
    const auto k_shape = input_tensors.at(1).padded_shape();

    // When using multi-latent attention, the V tensor is the same as K tensor, but of smaller head dimension.
    // For the sake validation, we will use the K tensor shape for V tensor, and validate head_dim_v separately.
    const auto v_shape = use_mla ? input_tensors.at(1).padded_shape() : input_tensors.at(2).padded_shape();

    if (use_mla) {
        // Head dim v validation
        TT_FATAL(
            q_shape[3] == k_shape[3],
            "Head dimension of Q must be equal to head dim of K, got {} and {}",
            q_shape[3],
            k_shape[3]);
        TT_FATAL(
            this->head_dim_v.value() <= q_shape[3],
            "Head dimension of V must be less than or equal to head dim of Q, got {} and {}",
            head_dim_v,
            q_shape[3]);
    }

    // Input 0 must be sharded by height or DRAM interleaved. All other inputs must be in DRAM.
    const auto Q_memcfg = input_tensors.at(0).memory_config();
    if (input_tensors.at(0).is_sharded()) {
        TT_FATAL(Q_memcfg.memory_layout() == TensorMemoryLayout::HEIGHT_SHARDED, "Error");
    } else {
        TT_FATAL(Q_memcfg.buffer_type() == tt::tt_metal::BufferType::DRAM, "Error");
    }

    for (std::size_t i = 1; i < input_tensors.size(); i++) {
        TT_FATAL(input_tensors.at(i).buffer()->buffer_type() == tt::tt_metal::BufferType::DRAM, "Error");
    }
    // Output memconfig must be height sharded or DRAM
    if (this->output_mem_config.is_sharded()) {
        TT_FATAL(this->output_mem_config.memory_layout() == TensorMemoryLayout::HEIGHT_SHARDED, "Error");
    }

    if (!this->is_causal) {
        if (optional_input_tensors.at(2).has_value()) {
            // Causal attention verification
            const auto& mask_tensor = optional_input_tensors.at(2).value();
            const auto mask_shape = mask_tensor.padded_shape();
            const auto mask_shape_unpadded = mask_tensor.logical_shape();

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

            TT_FATAL(k_chunk_size > 0, "Must provide k_chunk_size if using attention mask!");
            TT_FATAL(
                mask_shape[3] % k_chunk_size == 0,
                "Mask sequence length must be multiple of chunk size, got: {} and {}",
                mask_shape[3],
                k_chunk_size);

            TT_FATAL(
                mask_tensor.dtype() == DataType::BFLOAT16 || mask_tensor.dtype() == DataType::BFLOAT8_B ||
                    mask_tensor.dtype() == DataType::BFLOAT4_B,
                "Unsupported data type for mask tensor: {}.",
                mask_tensor.dtype());
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
                cur_pos_tensor.dtype() == DataType::INT32,
                "Expect cur_pos to be INT32, got {}",
                cur_pos_tensor.dtype());
            TT_FATAL(
                cur_pos_tensor.layout() == Layout::ROW_MAJOR,
                "Expect cur_pos to be ROW_MAJOR, got {}",
                cur_pos_tensor.layout());
            const auto cur_pos_shape = cur_pos_tensor.padded_shape();
            TT_FATAL(
                cur_pos_shape[0] == B, "cur_pos must have batch size equal to Q, got {} and {}", cur_pos_shape[0], B);
        }

        TT_FATAL(optional_input_tensors.at(1).has_value(), "Must have page_table tensor for paged attention");
        const auto& page_table_tensor = optional_input_tensors.at(1).value();

        TT_FATAL(page_table_tensor.dtype() == DataType::INT32, "Error");
        TT_FATAL(page_table_tensor.layout() == Layout::ROW_MAJOR, "Error");

        const auto page_table_shape = page_table_tensor.padded_shape();

        TT_FATAL(page_table_shape[0] == B, "page_table must have hidden size equal to Q");

        TT_FATAL(k_shape[2] == v_shape[2], "K and V must have same block size");
        TT_FATAL(k_shape[3] == v_shape[3] && k_shape[3] == q_shape[3], "Q, K, V must have same hidden size");

        // Validate chunk size for paged version
        // k_chunk_size can also be zero; if k_chunk_size = 0, figure it out in kernels
        TT_FATAL(k_chunk_size % 32 == 0, "Chunk size must be multiple of 32, got: {}", k_chunk_size);
        if (!this->is_causal) {
            TT_FATAL(k_chunk_size > 0, "Must provide k_chunk_size if paged and non-causal!");
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
        TT_FATAL(k_chunk_size > 0, "Must provide k_chunk_size if non-causal!");
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
            input_tensors.at(0).dtype() == DataType::BFLOAT16,
            "GQA expects BFLOAT16 input tensor, but got {}",
            input_tensors.at(0).dtype());
        uint32_t num_heads_per_kv = q_shape_unpadded[2] / k_shape[1];
        TT_FATAL(
            q_shape_unpadded[2] % k_shape[1] == 0,
            "GQA expects Q to have a multiple of K heads, but got {} and {}",
            q_shape_unpadded[2],
            k_shape[1]);
    }
}

std::vector<TensorSpec> ScaledDotProductAttentionDecode::compute_output_specs(
    const std::vector<Tensor>& input_tensors) const {
    auto& input = input_tensors.at(0);
    Layout output_layout = Layout::TILE;
    ttnn::Shape output_shape = input.logical_shape();
    if (input.layout() == Layout::ROW_MAJOR) {
        output_shape[2] = round_up_to_tile(output_shape[2], tt::constants::TILE_HEIGHT);
        output_layout = Layout::ROW_MAJOR;
    }
    if (this->use_mla.value_or(false)) {
        // Multi Latent Attention
        output_shape[3] = this->head_dim_v.value();
    }
    return {TensorSpec(output_shape, TensorLayout(input.dtype(), PageConfig(output_layout), output_mem_config))};
}

operation::ProgramWithCallbacks ScaledDotProductAttentionDecode::create_program(
    const std::vector<Tensor>& input_tensors,
    const std::vector<std::optional<const Tensor>>& optional_input_tensors,
    std::vector<Tensor>& output_tensors) const {
    auto& input_tensor_q = input_tensors.at(0);
    auto& input_tensor_k = input_tensors.at(1);
    auto& input_tensor_v = this->use_mla.value_or(false) ? input_tensors.at(1) : input_tensors.at(2);

    auto& cur_pos_tensor = optional_input_tensors.at(0);
    auto& page_table_tensor = optional_input_tensors.at(1);
    auto& attn_mask = optional_input_tensors.at(2);

    auto& output_tensor = output_tensors.at(0);

    auto scale = this->scale;
    if (not scale.has_value()) {
        scale = 1.0f / std::sqrt(static_cast<float>(input_tensor_q.padded_shape()[-1]));
    }

    return detail::sdpa_decode_multi_core(
        input_tensor_q,
        input_tensor_k,
        input_tensor_v,
        cur_pos_tensor,
        page_table_tensor,
        attn_mask,
        output_tensor,
        this->is_causal,
        this->cur_pos,
        scale,
        this->compute_kernel_config,
        this->program_config,
        this->k_chunk_size,
        this->share_cache,
        this->use_mla.value_or(false),
        this->head_dim_v.value_or(0));
}

operation::Hash ScaledDotProductAttentionDecode::compute_program_hash(
    const std::vector<Tensor>& input_tensors,
    const std::vector<std::optional<const Tensor>>& optional_input_tensors) const {
    bool has_cur_pos = optional_input_tensors.at(0).has_value();
    bool has_attn_mask = optional_input_tensors.at(2).has_value();
    return operation::hash_operation<ScaledDotProductAttentionDecode>(
        this->scale,
        this->output_mem_config,
        this->program_config,
        this->compute_kernel_config,
        this->k_chunk_size,
        this->paged_attention,
        this->is_causal,
        this->use_mla,
        this->head_dim_v,
        has_attn_mask,
        has_cur_pos,
        input_tensors,
        // Hash on page_table_tensor to properly size page table CB
        optional_input_tensors.at(1));
}

}  // namespace ttnn::operations::transformer
