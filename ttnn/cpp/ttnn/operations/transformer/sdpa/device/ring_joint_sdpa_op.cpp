// SPDX-FileCopyrightText: © 2025 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include "ring_joint_sdpa_op.hpp"

#include "ring_joint_sdpa_program_factory.hpp"
#include "ttnn/run_operation.hpp"
#include <tt-metalium/constants.hpp>

using namespace tt::tt_metal;

namespace ttnn::operations::transformer {

void RingJointScaledDotProductAttention::validate(const std::vector<Tensor>& input_tensors) const {
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
    const auto dtype = input_tensor_q.get_dtype();
    for (const auto& tensor : input_tensors) {
        TT_FATAL(
            tensor.get_dtype() == dtype,
            "All tensors must have the same dtype. Expected {}, got {}",
            dtype,
            tensor.get_dtype());
    }

    // Get shapes
    const auto q_shape = input_tensor_q.get_logical_shape();
    const auto k_shape = input_tensor_k.get_logical_shape();
    const auto v_shape = input_tensor_v.get_logical_shape();
    const auto joint_q_shape = joint_tensor_q.get_logical_shape();
    const auto joint_k_shape = joint_tensor_k.get_logical_shape();
    const auto joint_v_shape = joint_tensor_v.get_logical_shape();

    // Validate storage types and buffers
    for (auto& tensor : input_tensors) {
        TT_FATAL(tensor.storage_type() == StorageType::DEVICE, "Operands to Joint SDPA need to be on device");
        TT_FATAL(tensor.buffer() != nullptr, "Operands to Joint SDPA need to be allocated in buffers on device");
        TT_FATAL(tensor.get_layout() == Layout::TILE, "Inputs to Joint SDPA must be tilized");
        TT_FATAL(
            tensor.get_dtype() == DataType::BFLOAT16 || tensor.get_dtype() == DataType::BFLOAT8_B,
            "Inputs to Joint SDPA must be BF16 or BF8");
        TT_FATAL(
            tensor.buffer()->buffer_type() == tt::tt_metal::BufferType::DRAM,
            "Operands to Joint SDPA need to be in DRAM");
    }

    // Validate input shapes match
    const auto B = q_shape[0];
    const auto NQH = q_shape[1];
    const auto NKH = k_shape[1];
    const auto N_local = q_shape[2];
    const auto N_global = k_shape[2];
    const auto L = joint_q_shape[2];
    const auto DH = q_shape[3];

    TT_FATAL(
        k_shape[0] == B && v_shape[0] == B && joint_q_shape[0] == B && joint_k_shape[0] == B && joint_v_shape[0] == B,
        "Batch sizes must match. Got Q: {}, K: {}, V: {}, joint_Q: {}, joint_K: {}, joint_V: {}",
        B,
        k_shape[0],
        v_shape[0],
        joint_q_shape[0],
        joint_k_shape[0],
        joint_v_shape[0]);

    // Validate head dimensions match
    TT_FATAL(
        k_shape[3] == DH && v_shape[3] == DH && joint_q_shape[3] == DH && joint_k_shape[3] == DH &&
            joint_v_shape[3] == DH,
        "Head dimensions must match. Got Q: {}, K: {}, V: {}, joint_Q: {}, joint_K: {}, joint_V: {}",
        DH,
        k_shape[3],
        v_shape[3],
        joint_q_shape[3],
        joint_k_shape[3],
        joint_v_shape[3]);

    TT_FATAL(
        v_shape[1] == NKH && joint_q_shape[1] == NQH && joint_k_shape[1] == NKH && joint_v_shape[1] == NKH,
        "Num heads must match. Got Q: {}, K: {}, V: {}, joint_Q: {}, joint_K: {}, joint_V: {}",
        NQH,
        NKH,
        v_shape[1],
        joint_q_shape[1],
        joint_k_shape[1],
        joint_v_shape[1]);

    TT_FATAL(
        v_shape[2] == N_global,
        "V sequence length must be equal to global sequence length. Got V: {}, global sequence length: {}",
        v_shape[2],
        N_global);

    TT_FATAL(
        N_global == N_local * this->ring_size,
        "Global sequence length must be equal to local sequence length times ring size. Got global sequence length: "
        "{}, local sequence length: {}, ring size: {}",
        N_global,
        N_local,
        this->ring_size);

    TT_FATAL(
        this->logical_n <= N_global,
        "Logical sequence length must be less than or equal to global sequence length. Got logical sequence length: "
        "{}, global sequence length: {}",
        this->logical_n,
        N_global);

    TT_FATAL(
        joint_k_shape[2] == L && joint_v_shape[2] == L,
        "Joint sequence length must match. Got joint_K: {}, joint_V: {}",
        joint_k_shape[2],
        joint_v_shape[2]);

    // Check shapes based on ring
    TT_FATAL(
        q_shape[2] * this->ring_size == k_shape[2],
        "Q sequence length times ring size must be equal to K sequence length. Got Q: {}, K: {}, ring_size: {}",
        q_shape[2],
        k_shape[2],
        this->ring_size);
    TT_FATAL(
        k_shape[2] == v_shape[2],
        "K sequence length must be equal to V sequence length. Got K: {}, V: {}",
        k_shape[2],
        v_shape[2]);

    TT_FATAL(NQH == NKH, "Q num_heads must be equal to K num_heads. Got Q: {}, K: {}", NQH, NKH);

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

    TT_FATAL(
        N_local % q_chunk_size == 0,
        "Local sequence length must be divisible by q_chunk_size. Got N_local: {}, q_chunk_size: {}",
        N_local,
        q_chunk_size);
    TT_FATAL(
        N_local % k_chunk_size == 0,
        "Local sequence length must be divisible by k_chunk_size. Got N_local: {}, k_chunk_size: {}",
        N_local,
        k_chunk_size);

    // Validate padding: Only the sequence dimension may be padded
    auto validate_padding = [](const Tensor& tensor) {
        auto logical_shape = tensor.get_logical_shape();
        auto padded_shape = tensor.get_padded_shape();
        TT_FATAL(logical_shape[0] == padded_shape[0], "Padding is not supported on the batch dimension");
        TT_FATAL(logical_shape[1] == padded_shape[1], "Padding is not supported on the num_heads dimension");
        TT_FATAL(logical_shape[3] == padded_shape[3], "Padding is not supported on the head_dim dimension");
    };

    for (const auto& tensor : input_tensors) {
        validate_padding(tensor);
    }
}

std::uint32_t RingJointScaledDotProductAttention::get_q_chunk_size() const {
    return this->program_config ? this->program_config->q_chunk_size : 32;
}

std::uint32_t RingJointScaledDotProductAttention::get_k_chunk_size() const {
    return this->program_config ? this->program_config->k_chunk_size : 32;
}

std::vector<TensorSpec> RingJointScaledDotProductAttention::compute_output_specs(
    const std::vector<Tensor>& input_tensors) const {
    auto& input = input_tensors.at(0);
    auto& joint_input = input_tensors.at(3);
    auto lse_shape = input.get_logical_shape();
    lse_shape[3] = 1;
    lse_shape[2] = input.get_padded_shape()[2] + joint_input.get_padded_shape()[2];
    lse_shape[0] = this->ring_size;  // DEBUG: Returning intermediate LSEs for testing

    return {
        TensorSpec(
            input.get_logical_shape(), TensorLayout(input.get_dtype(), PageConfig(Layout::TILE), output_mem_config)),
        TensorSpec(
            joint_input.get_logical_shape(),
            TensorLayout(joint_input.get_dtype(), PageConfig(Layout::TILE), output_mem_config)),
        TensorSpec(lse_shape, TensorLayout(input.get_dtype(), PageConfig(Layout::TILE), output_mem_config))};
}

operation::ProgramWithCallbacks RingJointScaledDotProductAttention::create_program(
    const std::vector<Tensor>& input_tensors, std::vector<Tensor>& output_tensors) const {
    auto& input_tensor_q = input_tensors.at(0);
    auto& input_tensor_k = input_tensors.at(1);
    auto& input_tensor_v = input_tensors.at(2);
    auto& joint_tensor_q = input_tensors.at(3);
    auto& joint_tensor_k = input_tensors.at(4);
    auto& joint_tensor_v = input_tensors.at(5);
    auto& output_tensor = output_tensors.at(0);
    auto& joint_output_tensor = output_tensors.at(1);
    auto& lse_output_tensor = output_tensors.at(2);
    auto scale = this->scale;
    if (not scale.has_value()) {
        scale = 1.0f / std::sqrt(static_cast<float>(input_tensor_q.get_logical_shape()[-1]));
    }

    std::size_t q_chunk_size = this->get_q_chunk_size();
    std::size_t k_chunk_size = this->get_k_chunk_size();

    return detail::ring_joint_sdpa(
        input_tensor_q,
        input_tensor_k,
        input_tensor_v,
        joint_tensor_q,
        joint_tensor_k,
        joint_tensor_v,
        output_tensor,
        joint_output_tensor,
        lse_output_tensor,
        this->logical_n,
        scale,
        q_chunk_size,
        k_chunk_size,
        this->ring_size,
        this->compute_kernel_config,
        this->program_config);
}

}  // namespace ttnn::operations::transformer
