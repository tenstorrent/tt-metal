// SPDX-FileCopyrightText: © 2023 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include "ttnn/operations/embedding/device/embedding_device_operation.hpp"
#include "ttnn/deprecated/tt_dnn/op_library/math.hpp"
#include "tt_metal/common/work_split.hpp"
#include "tt_metal/common/constants.hpp"
#include "tt_metal/detail/util.hpp"
#include "tt_metal/host_api.hpp"
#include "ttnn/operations/embedding/device/embedding_program_factory.hpp"

using namespace tt::constants;
using namespace std;
using namespace tt::tt_metal;

#define RISC_CORES_PER_TENSIX 2

namespace ttnn::operations::embedding {

void Embeddings::validate(const std::vector<Tensor> &input_tensors) const {
    TT_FATAL(input_tensors.size() == 2, "Must have between 2 input tensors");
    auto &a = input_tensors.at(0);
    const auto &weights = input_tensors.at(1);
    TT_FATAL(a.get_layout() == Layout::ROW_MAJOR, "Error");
    TT_FATAL(weights.get_layout() == Layout::ROW_MAJOR, "Error");
    TT_FATAL(a.get_dtype() == DataType::UINT32 or a.get_dtype() == DataType::BFLOAT16, "Input must be UINT32 or BFLOAT16");
    TT_FATAL(weights.get_dtype() == DataType::BFLOAT16, "Error");
    TT_FATAL(a.memory_config().memory_layout == TensorMemoryLayout::INTERLEAVED, "Embedding does not currently support sharding");
    TT_FATAL(weights.memory_config().memory_layout == TensorMemoryLayout::INTERLEAVED, "Embedding does not currently support sharding");
    TT_FATAL(this->output_mem_config.memory_layout == TensorMemoryLayout::INTERLEAVED, "Embedding does not currently support sharding");

    TT_FATAL(weights.get_legacy_shape()[0] == 1 && weights.get_legacy_shape()[1] == 1, "First two dimensions for the weights must be 1");
    if (this->tilized) {
        TT_FATAL(a.get_legacy_shape()[-1] % TILE_HEIGHT == 0, "Error");
        TT_FATAL(weights.get_legacy_shape()[-1] % TILE_WIDTH == 0, "Number of columns in table must be factor of tile width");
    } else {
        TT_FATAL(this->output_dtype != DataType::BFLOAT8_B, "Error");
    }
    TT_FATAL(a.get_legacy_shape()[1] == 1 && a.get_legacy_shape()[2] == 1, "Only dim 0 && 3 for the input can be non 1");
    switch (this->embeddings_type) {
        case EmbeddingsType::PADDED: TT_FATAL(this->pad_token.has_value(), "Error"); break;
        case EmbeddingsType::BINARY: TT_FATAL(weights.get_legacy_shape()[-2] == 2, "Error");
        default: TT_FATAL(!this->pad_token.has_value(), "Error");
    }
}

std::vector<tt::tt_metal::Shape> Embeddings::compute_output_shapes(const std::vector<Tensor> &input_tensors) const {
    const auto &input_tensor = input_tensors.at(0);
    const auto &weight_tensor = input_tensors.at(1);
    auto num_output_embeddings = input_tensor.get_legacy_shape()[3];
    auto batch_num = input_tensor.get_legacy_shape()[0];
    auto num_embedding_dims = weight_tensor.get_legacy_shape()[3];

    tt::tt_metal::Shape output_shape({batch_num, 1, num_output_embeddings, num_embedding_dims});
    return {output_shape};
}

std::vector<Tensor> Embeddings::create_output_tensors(const std::vector<Tensor> &input_tensors) const {
    const auto &weight_tensor = input_tensors.at(1);
    if (!tilized) {
        return operation::generic_create_output_tensors(
            *this, input_tensors, this->output_dtype, Layout::ROW_MAJOR, this->output_mem_config);
    } else {
        return operation::generic_create_output_tensors(
            *this, input_tensors, this->output_dtype, Layout::TILE, this->output_mem_config);
    }
}

operation::ProgramWithCallbacks Embeddings::create_program(
    const std::vector<Tensor> &input_tensors, std::vector<Tensor> &output_tensors) const {
    const auto &a = input_tensors.at(0);
    const auto &weights = input_tensors.at(1);
    auto &output_tensor = output_tensors.at(0);
    return detail::embeddings_(a, weights, output_tensor, this->tilized, this->embeddings_type, this->pad_token);
}

}  // namespace ttnn::operations::embedding
