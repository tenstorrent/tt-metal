// SPDX-FileCopyrightText: © 2025 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include "joint_sdpa_op.hpp"

#include "joint_sdpa_program_factory.hpp"
#include "ttnn/run_operation.hpp"
#include <tt-metalium/constants.hpp>

using namespace tt::tt_metal;

namespace ttnn::operations::transformer {

void JointScaledDotProductAttention::validate(const std::vector<Tensor>& input_tensors) const {
    TT_FATAL(input_tensors.size() == 6, "Must have 6 input tensors (Q, K, V, joint_Q, joint_K, joint_V)");

    const auto& input_tensor_q = input_tensors.at(0);
    const auto& input_tensor_k = input_tensors.at(1);
    const auto& input_tensor_v = input_tensors.at(2);
    const auto& joint_tensor_q = input_tensors.at(3);
    const auto& joint_tensor_k = input_tensors.at(4);
    const auto& joint_tensor_v = input_tensors.at(5);

    // Validate joint strategy is 'rear'
    TT_FATAL(this->joint_strategy == "rear", "Joint strategy must be 'rear'. Got: {}", this->joint_strategy);

    // Validate all tensors have the same dtype
    const auto dtype = input_tensor_q.dtype();
    for (const auto& tensor : input_tensors) {
        TT_FATAL(
            tensor.dtype() == dtype,
            "All tensors must have the same dtype. Expected {}, got {}",
            dtype,
            tensor.dtype());
    }

    // Get shapes
    const auto q_shape = input_tensor_q.logical_shape();
    const auto k_shape = input_tensor_k.logical_shape();
    const auto v_shape = input_tensor_v.logical_shape();
    const auto joint_q_shape = joint_tensor_q.logical_shape();
    const auto joint_k_shape = joint_tensor_k.logical_shape();
    const auto joint_v_shape = joint_tensor_v.logical_shape();

    // Validate storage types and buffers
    for (auto& tensor : input_tensors) {
        TT_FATAL(tensor.storage_type() == StorageType::DEVICE, "Operands to Joint SDPA need to be on device");
        TT_FATAL(tensor.buffer() != nullptr, "Operands to Joint SDPA need to be allocated in buffers on device");
        TT_FATAL(tensor.layout() == Layout::TILE, "Inputs to Joint SDPA must be tilized");
        TT_FATAL(
            tensor.dtype() == DataType::BFLOAT16 || tensor.dtype() == DataType::BFLOAT8_B,
            "Inputs to Joint SDPA must be BF16 or BF8");
        TT_FATAL(
            tensor.buffer()->buffer_type() == tt::tt_metal::BufferType::DRAM,
            "Operands to Joint SDPA need to be in DRAM");
    }

    // Validate input shapes match
    TT_FATAL(
        k_shape[0] == q_shape[0] && v_shape[0] == q_shape[0],
        "Batch sizes must match. Got Q: {}, K: {}, V: {}",
        q_shape[0],
        k_shape[0],
        v_shape[0]);

    // Validate joint input shapes match
    TT_FATAL(
        joint_k_shape[0] == joint_q_shape[0] && joint_v_shape[0] == joint_q_shape[0],
        "Joint batch sizes must match. Got Q: {}, K: {}, V: {}",
        joint_q_shape[0],
        joint_k_shape[0],
        joint_v_shape[0]);

    // Validate Q and joint Q have same batch size and num heads
    TT_FATAL(
        q_shape[0] == joint_q_shape[0],
        "Q and joint Q must have same batch size. Got Q: {}, joint Q: {}",
        q_shape[0],
        joint_q_shape[0]);

    // Validate head dimensions match
    TT_FATAL(
        k_shape[3] == q_shape[3] && v_shape[3] == q_shape[3],
        "Head dimensions must match. Got Q: {}, K: {}, V: {}",
        q_shape[3],
        k_shape[3],
        v_shape[3]);

    TT_FATAL(
        joint_k_shape[3] == joint_q_shape[3] && joint_v_shape[3] == joint_q_shape[3],
        "Joint head dimensions must match. Got Q: {}, K: {}, V: {}",
        joint_q_shape[3],
        joint_k_shape[3],
        joint_v_shape[3]);

    TT_FATAL(
        q_shape[3] == joint_q_shape[3],
        "Q and joint Q must have same head dimension. Got Q: {}, joint Q: {}",
        q_shape[3],
        joint_q_shape[3]);

    // Validate num_heads relationship
    const auto nqh = q_shape[1];
    const auto nkv = k_shape[1];
    const auto joint_nqh = joint_q_shape[1];
    const auto joint_nkv = joint_k_shape[1];

    TT_FATAL(nqh == nkv, "Q num_heads must be equal to K num_heads. Got Q: {}, K: {}", nqh, nkv);

    TT_FATAL(
        joint_nqh == joint_nkv,
        "Joint Q num_heads must be equal to Joint K num_heads. Got Q: {}, K: {}",
        joint_nqh,
        joint_nkv);
    TT_FATAL(
        joint_nkv == nkv, "Joint K num_heads must be equal to K num_heads. Got Joint K: {}, K: {}", joint_nkv, nkv);

    // Validate chunk sizes if program config is provided
    auto q_chunk_size = this->get_q_chunk_size();
    auto k_chunk_size = this->get_k_chunk_size();

    TT_FATAL(
        q_chunk_size % tt::constants::TILE_WIDTH == 0,
        "q_chunk_size must be divisible by TILE_SIZE. Got q_chunk_size: {}, TILE_SIZE: {}",
        q_chunk_size,
        tt::constants::TILE_WIDTH);
    TT_FATAL(
        k_chunk_size % tt::constants::TILE_WIDTH == 0,
        "k_chunk_size must be divisible by TILE_SIZE. Got k_chunk_size: {}, TILE_SIZE: {}",
        k_chunk_size,
        tt::constants::TILE_WIDTH);

    // Validate padding: Only the sequence dimension may be padded
    auto validate_padding = [](const Tensor& tensor) {
        auto logical_shape = tensor.logical_shape();
        auto padded_shape = tensor.padded_shape();
        TT_FATAL(logical_shape[0] == padded_shape[0], "Padding is not supported on the batch dimension");
        TT_FATAL(logical_shape[1] == padded_shape[1], "Padding is not supported on the num_heads dimension");
        TT_FATAL(logical_shape[3] == padded_shape[3], "Padding is not supported on the head_dim dimension");
    };

    for (const auto& tensor : input_tensors) {
        validate_padding(tensor);
    }
}

std::uint32_t JointScaledDotProductAttention::get_q_chunk_size() const {
    return this->program_config ? this->program_config->q_chunk_size : 32;
}

std::uint32_t JointScaledDotProductAttention::get_k_chunk_size() const {
    return this->program_config ? this->program_config->k_chunk_size : 32;
}

std::vector<TensorSpec> JointScaledDotProductAttention::compute_output_specs(
    const std::vector<Tensor>& input_tensors) const {
    auto& input = input_tensors.at(0);
    auto& joint_input = input_tensors.at(3);
    return {
        TensorSpec(input.logical_shape(), TensorLayout(input.dtype(), PageConfig(Layout::TILE), output_mem_config)),
        TensorSpec(
            joint_input.logical_shape(),
            TensorLayout(joint_input.dtype(), PageConfig(Layout::TILE), output_mem_config))};
}

operation::ProgramWithCallbacks JointScaledDotProductAttention::create_program(
    const std::vector<Tensor>& input_tensors, std::vector<Tensor>& output_tensors) const {
    auto& input_tensor_q = input_tensors.at(0);
    auto& input_tensor_k = input_tensors.at(1);
    auto& input_tensor_v = input_tensors.at(2);
    auto& joint_tensor_q = input_tensors.at(3);
    auto& joint_tensor_k = input_tensors.at(4);
    auto& joint_tensor_v = input_tensors.at(5);
    auto& output_tensor = output_tensors.at(0);
    auto& joint_output_tensor = output_tensors.at(1);

    auto scale = this->scale;
    if (not scale.has_value()) {
        scale = 1.0f / std::sqrt(static_cast<float>(input_tensor_q.logical_shape()[-1]));
    }

    std::size_t q_chunk_size = this->get_q_chunk_size();
    std::size_t k_chunk_size = this->get_k_chunk_size();

    return detail::joint_sdpa(
        input_tensor_q,
        input_tensor_k,
        input_tensor_v,
        joint_tensor_q,
        joint_tensor_k,
        joint_tensor_v,
        output_tensor,
        joint_output_tensor,
        scale,
        q_chunk_size,
        k_chunk_size,
        this->compute_kernel_config,
        this->program_config);
}

}  // namespace ttnn::operations::transformer
