// SPDX-FileCopyrightText: © 2023 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include "rotary_embedding_llama_fused_qk_device_operation.hpp"
#include "rotary_embedding_llama_fused_qk_program_factory.hpp"

#include "tt_metal/common/constants.hpp"
#include "tt_metal/host_api.hpp"

namespace tt {

namespace tt_metal {

void RotaryEmbeddingLlamaFusedQK::validate(const std::vector<Tensor>& input_tensors) const {
    using namespace tt::constants;
    TT_FATAL(input_tensors.size() == 5, "Error");
    const auto& q_input_tensor = input_tensors.at(0);
    const auto& k_input_tensor = input_tensors.at(1);
    const auto& cos = input_tensors.at(2);
    const auto& sin = input_tensors.at(3);
    const auto& trans_mat = input_tensors.at(4);

    auto ref_device = q_input_tensor.device();
    for (const auto& input : input_tensors) {
        TT_FATAL(input.storage_type() == StorageType::DEVICE, "Operands to rotary embedding need to be on device!");
        TT_FATAL(input.buffer() != nullptr, "Operands to rotary embedding need to be allocated in buffers on device!");
        TT_FATAL(input.device() == ref_device, "Operands to rotary embedding need to be on same device!");
        TT_FATAL((input.get_layout() == Layout::TILE), "Inputs to rotary embedding must be tilized");
    }

    uint32_t head_dim = q_input_tensor.get_logical_shape()[-1];
    TT_FATAL(head_dim <= 128 || std::get<ttnn::WormholeComputeKernelConfig>(this->compute_kernel_config).fp32_dest_acc_en == false, "If head_dim is > 128, fp32_dest_acc_en must be False");
    // Check that head_dim is less than 256
    TT_FATAL(head_dim <= 256, "Head dim must be less than 256");
    // Check that head_dim is a multiple of 32
    TT_FATAL(head_dim % TILE_WIDTH == 0, "Head dim must be a multiple of TILE_WIDTH");

    TT_FATAL(q_input_tensor.get_dtype() == cos.get_dtype()  && cos.get_dtype() == sin.get_dtype()
        && sin.get_dtype() == trans_mat.get_dtype() && trans_mat.get_dtype() == DataType::BFLOAT16, "All input tensors must have dtype = bfloat16");
    TT_FATAL(q_input_tensor.memory_config().memory_layout == this->q_output_mem_config.memory_layout, "Input tensor and output tensor must have same memory layout");

    // Check that cos and sin have same dims
    TT_FATAL(cos.get_logical_shape() == sin.get_logical_shape(), "Cos and Sin dims must match");


    uint32_t seq_len = q_input_tensor.get_logical_shape()[0];
    TT_FATAL(seq_len == 1, "rotary_embedding_llama_fused_qk currently only supports sharded inputs in decode mode, and therefore, seq_len (in dim 0) must be 1.");

    for (const auto& input : input_tensors) {
        TT_FATAL((input.memory_config().memory_layout == TensorMemoryLayout::HEIGHT_SHARDED), "Sharded inputs for RoPE must be HEIGHT_SHARDED.");
    }

    uint32_t num_cores = q_input_tensor.device()->compute_with_storage_grid_size().x * q_input_tensor.device()->compute_with_storage_grid_size().y;
    uint32_t batch = q_input_tensor.get_logical_shape()[1];
    TT_FATAL(batch <= num_cores, "In decode mode, RoPE is parallelized over batch dimension, and therefore, batch size must be less than or equal to the number of cores");

    // Checks for cos and sin
    // TT_FATAL(batch == cos.get_logical_shape()[1], "Cos and Sin must have the same batch size as the input");
    TT_FATAL(cos.shard_spec()->shape[0] == TILE_HEIGHT, "In decode mode, RoPE only supports n_heads (shard_shape[0]) less than equal to TILE_HEIGHT"); // TODO: might be supported by kernel currently, but need to check with pytest

    // Checks for transformation matrix
    TT_FATAL(trans_mat.get_logical_shape()[0] == 1 && trans_mat.get_logical_shape()[1] == 1, "Transformation matrix must have 1st & 2nd dim equal to 1");
    TT_FATAL(trans_mat.shard_spec()->shape[0] == TILE_HEIGHT, "Transformation matrix must have 3rd dim equal to TILE_HEIGHT");
    TT_FATAL(trans_mat.shard_spec()->shape[1] == TILE_WIDTH, "Transformation matrix must have 4rd dim equal to TILE_WIDTH");

}

std::vector<ttnn::SimpleShape> RotaryEmbeddingLlamaFusedQK::compute_output_shapes(const std::vector<Tensor>& input_tensors) const {
    const auto& q_input_tensor = input_tensors.at(0);
    const auto& k_input_tensor = input_tensors.at(1);
    auto q_shape = q_input_tensor.get_logical_shape();
    auto k_shape = k_input_tensor.get_logical_shape();
    return {q_shape, k_shape};
}

std::vector<Tensor> RotaryEmbeddingLlamaFusedQK::create_output_tensors(const std::vector<Tensor>& input_tensors) const {
    const auto& q_input_tensor = input_tensors.at(0);
    const auto& k_input_tensor = input_tensors.at(1);
    auto output_shapes = this->compute_output_shapes(input_tensors);
    return {create_device_tensor(
        output_shapes[0], q_input_tensor.get_dtype(), q_input_tensor.get_layout(), q_input_tensor.device(), this->q_output_mem_config),
        create_device_tensor(
        output_shapes[1], k_input_tensor.get_dtype(), k_input_tensor.get_layout(), k_input_tensor.device(), this->k_output_mem_config)
        };
}

operation::ProgramWithCallbacks RotaryEmbeddingLlamaFusedQK::create_program(
    const std::vector<Tensor>& input_tensors, std::vector<Tensor>& output_tensors) const {
    const auto& q_input_tensor = input_tensors.at(0);
    const auto& k_input_tensor = input_tensors.at(1);
    const auto& cos = input_tensors.at(2);
    const auto& sin = input_tensors.at(3);
    const auto& trans_mat = input_tensors.at(4);
    auto& q_output_tensor = output_tensors.at(0);
    auto& k_output_tensor = output_tensors.at(1);

    return rotary_embedding_llama_fused_qk_multi_core_sharded(q_input_tensor, k_input_tensor, cos, sin, trans_mat, q_output_tensor, k_output_tensor, this->compute_kernel_config);

}

}  // namespace tt_metal

}  // namespace tt
