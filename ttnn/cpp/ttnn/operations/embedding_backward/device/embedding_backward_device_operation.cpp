// SPDX-FileCopyrightText: © 2023 Tenstorrent Inc.
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
    // WH only
    // Output dype (input gradient) same as output gradient
    // Seq from BS must be // 32
    // BS must be same for index and grad height
    TT_FATAL(input_tensors.size() == 2, "Must have between 2 input tensors");

    const auto &index_tensor = input_tensors.at(0);
    const auto &grad_tensor = input_tensors.at(1);

    TT_FATAL(grad_tensor.get_layout() == Layout::TILE);
    TT_FATAL(grad_tensor.get_dtype() == DataType::BFLOAT16);

    TT_FATAL(index_tensor.get_layout() == Layout::ROW_MAJOR);
    TT_FATAL(
        index_tensor.get_dtype() == DataType::UINT32 or index_tensor.get_dtype() == DataType::BFLOAT16,
        "Index tensor must be UINT32 or BFLOAT16");

    TT_FATAL(
        grad_tensor.memory_config().memory_layout == TensorMemoryLayout::INTERLEAVED ||
            index_tensor.memory_config().memory_layout == TensorMemoryLayout::INTERLEAVED ||
            this->output_mem_config.memory_layout == TensorMemoryLayout::INTERLEAVED,
        "Embedding b/w does not currently support sharding");

    TT_FATAL(
        grad_tensor.get_legacy_shape()[0] == 1 && grad_tensor.get_legacy_shape()[1] == 1,
        "First two dimensions for the gradient tensor must be 1");

    TT_FATAL(
        grad_tensor.get_legacy_shape()[-1] % TILE_WIDTH == 0,
        "Number of columns in the gradient tensor must be divisible by tile width");

    TT_FATAL(
        index_tensor.get_legacy_shape()[1] == 1 && index_tensor.get_legacy_shape()[2] == 1,
        "Only dim 0 && 3 for the index tensor can be non 1");

    TT_FATAL(
        index_tensor.get_legacy_shape()[-1] % TILE_WIDTH == 0,
        "Number of columns in the index tensor must be divisible by tile width");
}

std::vector<tt::tt_metal::Shape> EmbeddingBackward::compute_output_shapes(const std::vector<Tensor> &input_tensors) const {
    const auto &grad_tensor = input_tensors.at(1);
    auto embedding_dim = grad_tensor.get_legacy_shape()[-1];

    tt::tt_metal::Shape output_shape({1, 1, this->num_embeddings, embedding_dim});
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
