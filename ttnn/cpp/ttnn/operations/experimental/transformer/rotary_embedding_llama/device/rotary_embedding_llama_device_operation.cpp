// SPDX-FileCopyrightText: © 2023 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include "rotary_embedding_llama_device_operation.hpp"
#include "rotary_embedding_llama_program_factory.hpp"

#include "tt_metal/common/constants.hpp"
#include "tt_metal/host_api.hpp"

using namespace tt::constants;

namespace tt {

namespace tt_metal {

void RotaryEmbeddingLlama::validate(const std::vector<Tensor>& input_tensors) const {
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
        TT_FATAL((input.get_layout() == Layout::TILE), "Inputs to rotary embedding must be tilized");
    }

    TT_FATAL(input_tensor.get_legacy_shape()[-1] % TILE_WIDTH  == 0, "Input X dim must be divisible into tiles");
    uint32_t seq_len = input_tensor.get_legacy_shape()[-2];
    uint32_t B = input_tensor.get_legacy_shape()[0];
    uint32_t head_dim = input_tensor.get_legacy_shape()[-1];

    TT_FATAL(head_dim <= 128 || std::get<ttnn::WormholeComputeKernelConfig>(this->compute_kernel_config).fp32_dest_acc_en == false, "If head_dim is > 128, fp32_dest_acc_en must be False");
    // Check that head_dim is less than 256
    TT_FATAL(head_dim <= 256, "Head dim must be less than 256");
    // Check that head_dim is a multiple of 32
    TT_FATAL(head_dim % 32 == 0, "Head dim must be a multiple of 32");
    // Check datatypes
    TT_FATAL(input_tensor.get_dtype() == cos.get_dtype()  && cos.get_dtype() == sin.get_dtype()
        && sin.get_dtype() == trans_mat.get_dtype() && trans_mat.get_dtype() == DataType::BFLOAT16, "All input tensors must have dtype = bfloat16");
    TT_FATAL(cos.get_dtype() == sin.get_dtype(), "Cos and Sin dtypes must match");
    TT_FATAL(cos.get_legacy_shape() == sin.get_legacy_shape(), "Cos and Sin dims must match");
    TT_FATAL(cos.get_legacy_shape()[0] == 1 && cos.get_legacy_shape()[1] == 1 && cos.get_legacy_shape()[-1] == head_dim, "Cos dims must match input dims");

    TT_FATAL(trans_mat.get_legacy_shape()[0] == 1 && trans_mat.get_legacy_shape()[1] == 1, "Transformation matrix must have 1st & 2nd dim equal to 1");
    TT_FATAL(trans_mat.get_legacy_shape()[-2] == TILE_HEIGHT, "Transformation matrix must have 3rd dim equal to TILE_HEIGHT");
    TT_FATAL(trans_mat.get_legacy_shape()[-1] == TILE_WIDTH, "Transformation matrix must have 4rd dim equal to TILE_WIDTH");


    TT_FATAL(input_tensor.memory_config().memory_layout == TensorMemoryLayout::INTERLEAVED, "Error");
    TT_FATAL(this->output_mem_config.memory_layout == TensorMemoryLayout::INTERLEAVED, "Error");

}

std::vector<tt::tt_metal::LegacyShape> RotaryEmbeddingLlama::compute_output_shapes(const std::vector<Tensor>& input_tensors) const {
    const auto& input_tensor = input_tensors.at(0);
    auto shape = input_tensor.get_legacy_shape();
    return {shape};
}

std::vector<Tensor> RotaryEmbeddingLlama::create_output_tensors(const std::vector<Tensor>& input_tensors) const {
    const auto& input_tensor = input_tensors.at(0);
    auto output_shape = this->compute_output_shapes(input_tensors)[0];
    return {create_device_tensor(
        output_shape, input_tensor.get_dtype(), input_tensor.get_layout(), input_tensor.device(), this->output_mem_config)};
}

operation::ProgramWithCallbacks RotaryEmbeddingLlama::create_program(
    const std::vector<Tensor>& input_tensors, std::vector<Tensor>& output_tensors) const {
    const auto& input_tensor = input_tensors.at(0);
    const auto& cos = input_tensors.at(1);
    const auto& sin = input_tensors.at(2);
    const auto& trans_mat = input_tensors.at(3);
    auto& output_tensor = output_tensors.at(0);

    // Works on single core as well
    return rotary_embedding_llama_multi_core(input_tensor, cos, sin, trans_mat, output_tensor, this->compute_kernel_config);
}

}  // namespace tt_metal

}  // namespace tt
