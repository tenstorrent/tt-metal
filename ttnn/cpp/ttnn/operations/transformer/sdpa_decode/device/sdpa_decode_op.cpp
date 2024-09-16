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
            input_tensor.get_dtype() == DataType::BFLOAT4_B);
    }

    const auto q_shape = input_tensors.at(0).get_legacy_shape();
    const auto k_shape = input_tensors.at(1).get_legacy_shape();
    const auto v_shape = input_tensors.at(2).get_legacy_shape();

    // Input 0 must be sharded by height or DRAM interleaved. All other inputs must be in DRAM.
    const auto Q_memcfg = input_tensors.at(0).memory_config();
    if (input_tensors.at(0).is_sharded()) {
        TT_FATAL(Q_memcfg.memory_layout == TensorMemoryLayout::HEIGHT_SHARDED);
    } else {
        TT_FATAL(Q_memcfg.buffer_type == tt::tt_metal::BufferType::DRAM);
    }

    for (std::size_t i = 1; i < input_tensors.size(); i++) {
        TT_FATAL(input_tensors.at(i).buffer()->buffer_type() == tt::tt_metal::BufferType::DRAM);
    }
    // Output memconfig must be height sharded or DRAM
    if (this->output_mem_config.is_sharded()) {
        TT_FATAL(this->output_mem_config.memory_layout == TensorMemoryLayout::HEIGHT_SHARDED);
    }

    // Assert we are in decode mode if not causal
    // Q: [1, B, NH, D]
    // K: [1, B, S, D]
    // V: [1, B, S, D]

    // Batch must match
    const auto B = q_shape[1];
    TT_FATAL(k_shape[1] == B);
    TT_FATAL(v_shape[1] == B);
    // TT_FATAL(Q_memcfg.shard_spec.value().grid.num_cores() == B, "Q must be height sharded by batch ");

    // NKV must be 1 if we are running in this decode mode
    TT_FATAL(q_shape[0] == 1);
    TT_FATAL(k_shape[0] == 1);
    TT_FATAL(v_shape[0] == 1);

    // Check sequence lengths
    TT_FATAL(k_shape[-2] == v_shape[-2]);

    // Check hidden size
    const auto D = q_shape[-1];
    TT_FATAL(k_shape[-1] == D);
    TT_FATAL(v_shape[-1] == D);

    // Check valid seqlen
    for (int i = 0; i < this->cur_pos.size(); i++) {
        TT_FATAL(this->cur_pos[i] < k_shape[-2], "cur_pos must be <= K sequence dim");
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

    auto& output_tensor = output_tensors.at(0);

    auto scale = this->scale;
    if (not scale.has_value()) {
        scale = 1.0f / std::sqrt(static_cast<float>(input_tensor_q.get_legacy_shape()[-1]));
    }

    // TODO: get this from program_config
    std::size_t q_chunk_size = program_config.has_value() ? program_config->q_chunk_size : 32;
    std::size_t k_chunk_size = program_config.has_value() ? program_config->k_chunk_size : 32;

    return detail::sdpa_decode_multi_core(
        input_tensor_q,
        input_tensor_k,
        input_tensor_v,
        cur_pos_tensor,
        output_tensor,
        this->cur_pos,
        scale,
        this->compute_kernel_config,
        this->program_config,
        this->k_chunk_size);
}

operation::Hash ScaledDotProductAttentionDecode::compute_program_hash(const std::vector<Tensor>& input_tensors, const std::vector<std::optional<const Tensor>>& optional_input_tensors) const {
    return operation::hash_operation<ScaledDotProductAttentionDecode>(
        this->scale,
        this->output_mem_config,
        this->program_config,
        this->compute_kernel_config,
        this->k_chunk_size,
        input_tensors,
        optional_input_tensors);
}

}  // namespace ttnn::operations::transformer
