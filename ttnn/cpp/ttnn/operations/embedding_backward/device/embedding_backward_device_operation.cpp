// SPDX-FileCopyrightText: © 2024 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include "ttnn/operations/embedding_backward/device/embedding_backward_device_operation.hpp"

#include "tt_metal/common/constants.hpp"
#include "ttnn/cpp/ttnn/run_operation.hpp"

using namespace tt::constants;
using namespace std;
using namespace tt::tt_metal;

namespace ttnn::operations::embedding_backward {

void EmbeddingBackward::validate(const std::vector<Tensor> &input_tensors) const {
    TT_FATAL(input_tensors.size() == 2, "Must have between 2 input tensors");

    const auto &index_tensor = input_tensors.at(0);
    const auto &grad_tensor = input_tensors.at(1);
    const auto &index_tensor_shape = index_tensor.get_legacy_shape();
    const auto &grad_tensor_shape = grad_tensor.get_legacy_shape();

    TT_FATAL(
        index_tensor.device()->arch() == tt::ARCH::WORMHOLE_B0,
        "Embedding backwards is only implemented for Wormhole!");

    TT_FATAL(index_tensor.get_layout() == Layout::ROW_MAJOR);
    TT_FATAL(
        index_tensor.get_dtype() == DataType::UINT32 or index_tensor.get_dtype() == DataType::BFLOAT16,
        "Index tensor must be UINT32 or BFLOAT16");

    TT_FATAL(
        index_tensor_shape[1] == 1 && index_tensor_shape[2] == 1,
        fmt::format(
            "Only dim 0 && 3 for the index tensor can be non 1, but found {} && {}.",
            index_tensor_shape[1],
            index_tensor_shape[2]));

    TT_FATAL(
        index_tensor_shape[-1] % TILE_WIDTH == 0,
        "Number of columns in the index tensor must be divisible by tile width");

    TT_FATAL(grad_tensor.get_layout() == Layout::TILE);
    TT_FATAL(
        grad_tensor.get_dtype() == DataType::BFLOAT16 or grad_tensor.get_dtype() == DataType::BFLOAT8_B,
        "Output gradient tensor must be BFLOAT16 or BFLOAT8_B");
    TT_FATAL(
        grad_tensor.get_dtype() == this->output_dtype, "Output and input gradient tensors must have the same dtype");

    TT_FATAL(
        grad_tensor.memory_config().memory_layout == TensorMemoryLayout::INTERLEAVED or
            index_tensor.memory_config().memory_layout == TensorMemoryLayout::INTERLEAVED or
            this->output_mem_config.memory_layout == TensorMemoryLayout::INTERLEAVED,
        "Embedding b/w does not currently support sharding");

    TT_FATAL(
        grad_tensor_shape[0] == 1 && grad_tensor_shape[1] == 1,
        fmt::format(
            "First two dimensions for the gradient tensor must be 1, but found {} && {}.",
            grad_tensor_shape[0],
            grad_tensor_shape[1]));

    TT_FATAL(
        grad_tensor_shape[-1] % TILE_WIDTH == 0,
        "Number of columns in the gradient tensor must be divisible by tile width");

    TT_FATAL(
        grad_tensor_shape[2] == index_tensor_shape[0] * index_tensor_shape[-1],
        "Number of rows in gradient tensor must be equal to number of indices in index tensor");
}

std::vector<tt::tt_metal::LegacyShape> EmbeddingBackward::compute_output_shapes(
    const std::vector<Tensor> &input_tensors) const {
    const auto &grad_tensor = input_tensors.at(1);
    auto embedding_dim = grad_tensor.get_legacy_shape()[-1];

    tt::tt_metal::LegacyShape output_shape({1, 1, this->num_embeddings, embedding_dim});
    return {output_shape};
}

std::vector<Tensor> EmbeddingBackward::create_output_tensors(const std::vector<Tensor> &input_tensors) const {
    return operation::generic_create_output_tensors(
        *this, input_tensors, this->output_dtype, Layout::TILE, this->output_mem_config);
}

operation::ProgramWithCallbacks EmbeddingBackward::create_program(
    const std::vector<Tensor> &input_tensors, std::vector<Tensor> &output_tensors) const {
    const auto &index_tensor = input_tensors.at(0);
    const auto &grad_tensor = input_tensors.at(1);
    auto &output_tensor = output_tensors.at(0);
    return detail::embedding_backward_multi_core(index_tensor, grad_tensor, output_tensor, this->num_embeddings);
}

}  // namespace ttnn::operations::embedding_backward
