// SPDX-FileCopyrightText: © 2024 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include "sdpa_decode_op.hpp"

#include "sdpa_decode_program_factory.hpp"
#include "ttnn/run_operation.hpp"

namespace ttnn::operations::transformer {

void ScaledDotProductAttentionDecode::validate(const std::vector<Tensor>& input_tensors, const std::vector<std::optional<const Tensor>>& optional_input_tensors) const {
    TT_FATAL(input_tensors.size() == 3, "Must have 3 input tensors and mask");

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

    const auto q_shape = input_tensors.at(0).get_legacy_shape();
    const auto q_shape_unpadded = input_tensors.at(0).get_shape();
    const auto k_shape = input_tensors.at(1).get_legacy_shape();
    const auto v_shape = input_tensors.at(2).get_legacy_shape();

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

    if (this->paged_attention) {
        // Paged attention verification
        TT_FATAL(! this->share_cache.value_or(false), "Share cache feature not supported for paged attention");
        TT_FATAL(optional_input_tensors.at(0).has_value(), "Must have cur_pos tensor for paged attention");
        TT_FATAL(optional_input_tensors.at(1).has_value(), "Must have page_table tensor for paged attention");

        const auto& cur_pos_tensor = optional_input_tensors.at(0).value();
        const auto& page_table_tensor = optional_input_tensors.at(1).value();

        TT_FATAL(cur_pos_tensor.get_dtype() == DataType::INT32, "Error");
        TT_FATAL(cur_pos_tensor.get_layout() == Layout::ROW_MAJOR, "Error");

        TT_FATAL(page_table_tensor.get_dtype() == DataType::INT32, "Error");
        TT_FATAL(page_table_tensor.get_layout() == Layout::ROW_MAJOR, "Error");

        const auto cur_pos_shape = cur_pos_tensor.get_legacy_shape();
        const auto page_table_shape = page_table_tensor.get_legacy_shape();

        const auto B = q_shape[1];
        TT_FATAL(cur_pos_shape[0] == B, "cur_pos must have batch size equal to Q");
        TT_FATAL(page_table_shape[0] == B, "page_table must have hidden size equal to Q");

        TT_FATAL(k_shape[2] == v_shape[2], "K and V must have same block size");
        TT_FATAL(k_shape[3] == v_shape[3] && k_shape[3] == q_shape[3], "Q, K, V must have same hidden size");
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
    TT_FATAL(k_shape[1] == v_shape[1], "Flash decode expects K and V to have same number of heads, but got {} and {}", k_shape[1], v_shape[1]);
    bool is_gqa = (k_shape[1] > 1);
    if (is_gqa) {
        TT_FATAL(! output_mem_config.is_sharded(), "Sharded output not supported for GQA");
        TT_FATAL(input_tensors.at(0).get_dtype() == DataType::BFLOAT16, "GQA expects BFLOAT16 input tensor, but got {}", input_tensors.at(0).get_dtype());
        uint32_t num_heads_per_kv = q_shape_unpadded[2]/k_shape[1];
        TT_FATAL(q_shape_unpadded[2]%k_shape[1] == 0, "GQA expects Q to have a multiple of K heads, but got {} and {}", q_shape_unpadded[2], k_shape[1]);
    }

    // Check compute kernel config
    std::visit(
        [&](auto&& compute_kernel_config) {
            using T = std::decay_t<decltype(compute_kernel_config)>;
            if constexpr (std::is_same_v<T, WormholeComputeKernelConfig>) {
                TT_FATAL(
                    compute_kernel_config.fp32_dest_acc_en == false,
                    "FP32 dest acc disabled due to nd pcc and unpacker hang issue.");
            }
        },
        this->compute_kernel_config);
}

std::vector<tt::tt_metal::LegacyShape> ScaledDotProductAttentionDecode::compute_output_shapes(
    const std::vector<Tensor>& input_tensors) const {
    return {input_tensors.at(0).get_legacy_shape()};
}

std::vector<Tensor> ScaledDotProductAttentionDecode::create_output_tensors(
    const std::vector<Tensor>& input_tensors) const {
    return operation::generic_create_output_tensors(
        *this, input_tensors, input_tensors.at(0).get_dtype(), Layout::TILE, this->output_mem_config);
}

operation::ProgramWithCallbacks ScaledDotProductAttentionDecode::create_program(
    const std::vector<Tensor>& input_tensors, const std::vector<std::optional<const Tensor>>& optional_input_tensors, std::vector<Tensor>& output_tensors) const {
    auto& input_tensor_q = input_tensors.at(0);
    auto& input_tensor_k = input_tensors.at(1);
    auto& input_tensor_v = input_tensors.at(2);

    auto& cur_pos_tensor = optional_input_tensors.at(0);
    auto& page_table_tensor = optional_input_tensors.at(1);

    auto& output_tensor = output_tensors.at(0);

    auto scale = this->scale;
    if (not scale.has_value()) {
        scale = 1.0f / std::sqrt(static_cast<float>(input_tensor_q.get_legacy_shape()[-1]));
    }

    return detail::sdpa_decode_multi_core(
        input_tensor_q,
        input_tensor_k,
        input_tensor_v,
        cur_pos_tensor,
        page_table_tensor,
        output_tensor,
        this->cur_pos,
        scale,
        this->compute_kernel_config,
        this->program_config,
        this->k_chunk_size,
        this->share_cache);
}

operation::Hash ScaledDotProductAttentionDecode::compute_program_hash(const std::vector<Tensor>& input_tensors, const std::vector<std::optional<const Tensor>>& optional_input_tensors) const {
    return operation::hash_operation<ScaledDotProductAttentionDecode>(
        this->scale,
        this->output_mem_config,
        this->program_config,
        this->compute_kernel_config,
        this->k_chunk_size,
        this->paged_attention,
        input_tensors);
}

}  // namespace ttnn::operations::transformer
