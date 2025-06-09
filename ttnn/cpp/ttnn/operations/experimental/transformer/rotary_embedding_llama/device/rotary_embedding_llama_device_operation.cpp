// SPDX-FileCopyrightText: © 2023 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include "rotary_embedding_llama_device_operation.hpp"
#include "rotary_embedding_llama_program_factory.hpp"

#include <tt-metalium/constants.hpp>

namespace tt {

namespace tt_metal {

void RotaryEmbeddingLlama::validate(const std::vector<Tensor>& input_tensors) const {
    using namespace tt::constants;
    const auto& input_tensor = input_tensors.at(0);
    const auto& cos = input_tensors.at(1);
    const auto& sin = input_tensors.at(2);
    const auto& trans_mat = input_tensors.at(3);
    TT_FATAL(input_tensors.size() == 4, "Error");

    auto ref_device = input_tensor.device();
    for (const auto& input : input_tensors) {
        TT_FATAL(input.storage_type() == StorageType::DEVICE, "Operands to rotary embedding need to be on device!");
        TT_FATAL(input.buffer() != nullptr, "Operands to rotary embedding need to be allocated in buffers on device!");
        TT_FATAL(input.device() == ref_device, "Operands to rotary embedding need to be on same device!");
        TT_FATAL((input.layout() == Layout::TILE), "Inputs to rotary embedding must be tilized");
    }

    uint32_t head_dim = input_tensor.logical_shape()[-1];
    TT_FATAL(
        head_dim <= 128 ||
            std::get<ttnn::WormholeComputeKernelConfig>(this->compute_kernel_config).fp32_dest_acc_en == false,
        "If head_dim is > 128, fp32_dest_acc_en must be False");
    // Check that head_dim is less than 256
    TT_FATAL(head_dim <= 256, "Head dim must be less than 256");
    // Check that head_dim is a multiple of 32
    TT_FATAL(head_dim % TILE_WIDTH == 0, "Head dim must be a multiple of TILE_WIDTH");

    TT_FATAL(
        input_tensor.dtype() == cos.dtype() && cos.dtype() == sin.dtype() && sin.dtype() == trans_mat.dtype() &&
            trans_mat.dtype() == DataType::BFLOAT16,
        "All input tensors must have dtype = bfloat16");
    TT_FATAL(
        input_tensor.memory_config().memory_layout() == this->output_mem_config.memory_layout(),
        "Input tensor and output tensor must have same memory layout");

    // Check that cos and sin have same dims
    TT_FATAL(cos.logical_shape() == sin.logical_shape(), "Cos and Sin dims must match");

    if (this->is_decode_mode) {  // Decode mode validation
        uint32_t seq_len = input_tensor.logical_shape()[0];
        TT_FATAL(
            seq_len == 1,
            "rotary_embedding_llama currently only supports sharded inputs in decode mode, and therefore, seq_len (in "
            "dim 0) must be 1.");

        for (const auto& input : input_tensors) {
            TT_FATAL(
                (input.memory_config().memory_layout() == TensorMemoryLayout::HEIGHT_SHARDED),
                "Sharded inputs for RoPE must be HEIGHT_SHARDED.");
        }

        uint32_t num_cores = input_tensor.device()->compute_with_storage_grid_size().x *
                             input_tensor.device()->compute_with_storage_grid_size().y;
        uint32_t batch = input_tensor.logical_shape()[1];
        TT_FATAL(
            batch <= num_cores,
            "In decode mode, RoPE is parallelized over batch dimension, and therefore, batch size must be less than or "
            "equal to the number of cores");

        // Checks for cos and sin
        TT_FATAL(batch == cos.logical_shape()[1], "Cos and Sin must have the same batch size as the input");
        TT_FATAL(
            cos.shard_spec()->shape[0] == TILE_HEIGHT,
            "In decode mode, RoPE only supports n_heads (shard_shape[0]) less than equal to TILE_HEIGHT");  // TODO:
                                                                                                            // might be
                                                                                                            // supported
                                                                                                            // by kernel
                                                                                                            // currently,
                                                                                                            // but need
                                                                                                            // to check
                                                                                                            // with
                                                                                                            // pytest

        // Checks for transformation matrix
        TT_FATAL(
            trans_mat.logical_shape()[0] == 1 && trans_mat.logical_shape()[1] == 1,
            "Transformation matrix must have 1st & 2nd dim equal to 1");
        TT_FATAL(
            trans_mat.shard_spec()->shape[0] == TILE_HEIGHT,
            "Transformation matrix must have 3rd dim equal to TILE_HEIGHT");
        TT_FATAL(
            trans_mat.shard_spec()->shape[1] == TILE_WIDTH,
            "Transformation matrix must have 4rd dim equal to TILE_WIDTH");
    } else {  // Prefill mode validation
        TT_FATAL(
            input_tensor.memory_config().memory_layout() == TensorMemoryLayout::INTERLEAVED,
            "Input tensor must be interleaved in prefill mode");

        // Checks for cos and sin
        TT_FATAL(cos.logical_shape()[0] == 1 && cos.logical_shape()[-1] == head_dim, "Cos dims must match input dims");
        // Check num_heads in cos/sin
        TT_FATAL(
            cos.logical_shape()[1] == input_tensor.logical_shape()[1] || cos.logical_shape()[1] == 1,
            "Num heads in cos/sin must match input tensor num heads or be 1. Expected {}, got {}",
            input_tensor.logical_shape()[1],
            cos.logical_shape()[1]);
        TT_FATAL(
            input_tensor.memory_config().memory_layout() == sin.memory_config().memory_layout(),
            "Input tensor and sin tensor must have same memory layout");
        TT_FATAL(
            input_tensor.memory_config().memory_layout() == cos.memory_config().memory_layout(),
            "Input tensor and cos tensor must have same memory layout");

        // Checks for transformation matrix
        TT_FATAL(
            trans_mat.logical_shape()[0] == 1 && trans_mat.logical_shape()[1] == 1,
            "Transformation matrix must have 1st & 2nd dim equal to 1");
        TT_FATAL(
            trans_mat.logical_shape()[-2] == TILE_HEIGHT,
            "Transformation matrix must have 3rd dim equal to TILE_HEIGHT");
        TT_FATAL(
            trans_mat.logical_shape()[-1] == TILE_WIDTH, "Transformation matrix must have 4rd dim equal to TILE_WIDTH");
    }
}

std::vector<ttnn::TensorSpec> RotaryEmbeddingLlama::compute_output_specs(
    const std::vector<Tensor>& input_tensors) const {
    const auto& input_tensor = input_tensors.at(0);
    auto shape = input_tensor.logical_shape();
    return {
        TensorSpec(shape, TensorLayout(input_tensor.dtype(), PageConfig(input_tensor.layout()), output_mem_config))};
}

operation::ProgramWithCallbacks RotaryEmbeddingLlama::create_program(
    const std::vector<Tensor>& input_tensors, std::vector<Tensor>& output_tensors) const {
    const auto& input_tensor = input_tensors.at(0);
    const auto& cos = input_tensors.at(1);
    const auto& sin = input_tensors.at(2);
    const auto& trans_mat = input_tensors.at(3);
    auto& output_tensor = output_tensors.at(0);

    // Works on single core as well
    if (this->is_decode_mode) {
        return rotary_embedding_llama_multi_core_sharded(
            input_tensor, cos, sin, trans_mat, output_tensor, this->compute_kernel_config);
    } else {
        return rotary_embedding_llama_multi_core(
            input_tensor, cos, sin, trans_mat, output_tensor, this->compute_kernel_config);
    }
}

}  // namespace tt_metal

}  // namespace tt
