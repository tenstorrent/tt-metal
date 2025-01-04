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
        TT_FATAL(
            input.storage_type() == StorageType::DEVICE || input.storage_type() == StorageType::MULTI_DEVICE,
            "Operands to rotary embedding need to be on device!");
        TT_FATAL(input.buffer() != nullptr, "Operands to rotary embedding need to be allocated in buffers on device!");
        TT_FATAL(input.device() == ref_device, "Operands to rotary embedding need to be on same device!");
        TT_FATAL((input.get_layout() == Layout::TILE), "Inputs to rotary embedding must be tilized");
        TT_FATAL(
            (input.memory_config().memory_layout == TensorMemoryLayout::HEIGHT_SHARDED),
            "inputs for RoPE must be HEIGHT_SHARDED.");
        TT_FATAL((input.get_dtype() == DataType::BFLOAT16), "Inputs to rotary embedding must be bfloat16");
    }

    // Check for decode mode
    TT_FATAL(
        q_input_tensor.get_logical_shape()[0] == 1 && k_input_tensor.get_logical_shape()[0] == 1,
        "rotary_embedding_llama_fused_qk currently only supports deocde mode qith seq_len=1.");

    TT_FATAL(
        q_input_tensor.get_logical_shape()[-1] == k_input_tensor.get_logical_shape()[-1],
        "Q input tensor and K input tensor must have same head dimensions");
    uint32_t head_dim = q_input_tensor.get_logical_shape()[-1];
    TT_FATAL(
        head_dim <= 128 ||
            std::get<ttnn::WormholeComputeKernelConfig>(this->compute_kernel_config).fp32_dest_acc_en == false,
        "If head_dim is > 128, fp32_dest_acc_en must be False");

    // Check that head_dim is a multiple of 32
    TT_FATAL(head_dim % TILE_WIDTH == 0, "Head dim must be a multiple of TILE_WIDTH");

    TT_FATAL(
        q_input_tensor.memory_config().memory_layout == this->q_output_mem_config.memory_layout,
        "Q Input tensor and Q output tensor must have same memory layout");
    TT_FATAL(
        k_input_tensor.memory_config().memory_layout == this->k_output_mem_config.memory_layout,
        "K Input tensor and K output tensor must have same memory layout");

    // check that q and k have same batch size and lesser that equal to 32
    uint32_t q_batch_size = q_input_tensor.get_logical_shape()[1];
    uint32_t k_batch_size = k_input_tensor.get_logical_shape()[1];
    TT_FATAL(q_batch_size == k_batch_size, "Q and K must have the equal batch size");
    TT_FATAL(
        q_batch_size <= 32,
        "Q and K must have batch size less than or equal to 32, due to parallelization over core-grid of 64");
    uint32_t q_num_cores = q_input_tensor.shard_spec()->grid.num_cores();
    uint32_t k_num_cores = k_input_tensor.shard_spec()->grid.num_cores();
    TT_FATAL(q_num_cores + k_num_cores <= 64, "Q and K must not exceed max core grid size of 64");

    bool is_overlap = q_input_tensor.shard_spec()->grid.intersects(k_input_tensor.shard_spec()->grid);
    TT_FATAL(!is_overlap, "Q and K must not overlap");

    // Check that cos and sin have same dims
    TT_FATAL(cos.get_logical_shape() == sin.get_logical_shape(), "Cos and Sin dims must match");
    uint32_t cos_sin_batch_size = cos.get_logical_shape()[1];
    TT_FATAL(
        cos_sin_batch_size == (q_batch_size + k_batch_size),
        "Cos and Sin are repeated for Q and K, so they must have the same batch size as the sum of Q and K batch "
        "sizes");

    // Checks for transformation matrix
    uint32_t trans_mat_num_cores = trans_mat.shard_spec()->grid.num_cores();
    TT_FATAL(
        trans_mat_num_cores >= (q_num_cores + k_num_cores),
        "Transformation matrix is repeated for Q and K must be sharded over core grid of Q and K");
    TT_FATAL(
        trans_mat.shard_spec()->shape[0] == TILE_HEIGHT && trans_mat.shard_spec()->shape[1] == TILE_WIDTH,
        "Transformation matrix must be sharded to single tile of shape (32, 32)");
}

std::vector<ttnn::SimpleShape> RotaryEmbeddingLlamaFusedQK::compute_output_shapes(
    const std::vector<Tensor>& input_tensors) const {
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
    return {
        create_device_tensor(
            output_shapes[0],
            q_input_tensor.get_dtype(),
            q_input_tensor.get_layout(),
            q_input_tensor.device(),
            this->q_output_mem_config),
        create_device_tensor(
            output_shapes[1],
            k_input_tensor.get_dtype(),
            k_input_tensor.get_layout(),
            k_input_tensor.device(),
            this->k_output_mem_config)};
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

    return rotary_embedding_llama_fused_qk_multi_core_sharded(
        q_input_tensor,
        k_input_tensor,
        cos,
        sin,
        trans_mat,
        q_output_tensor,
        k_output_tensor,
        this->compute_kernel_config);
}

}  // namespace tt_metal

}  // namespace tt
