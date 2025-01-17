// SPDX-FileCopyrightText: © 2023 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include "ttnn/operations/embedding/device/embedding_device_operation.hpp"
#include "ttnn/operations/math.hpp"
#include <tt-metalium/work_split.hpp>
#include <tt-metalium/constants.hpp>
#include <tt-metalium/util.hpp>
#include <tt-metalium/host_api.hpp>
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
    TT_FATAL(weights.get_layout() == Layout::ROW_MAJOR, "Error");
    TT_FATAL(a.get_dtype() == DataType::UINT32 or a.get_dtype() == DataType::BFLOAT16, "Input must be UINT32 or BFLOAT16");
    TT_FATAL(weights.get_dtype() == DataType::BFLOAT16, "Weights tensor must have BFLOAT16 dtype");
    TT_FATAL(a.memory_config().memory_layout == TensorMemoryLayout::INTERLEAVED, "Embedding does not currently support sharded inputs");
    TT_FATAL(weights.memory_config().memory_layout == TensorMemoryLayout::INTERLEAVED, "Embedding does not currently support sharded weights");

    TT_FATAL(weights.get_legacy_shape()[0] == 1 && weights.get_legacy_shape()[1] == 1, "First two dimensions for the weights must be 1");
    if (this->tilized) {
        TT_FATAL(a.get_legacy_shape()[-1] % TILE_HEIGHT == 0, "Input tensor width {} must be a multiple of tile height {} to have the output tensor tilized", a.get_legacy_shape()[-1], TILE_HEIGHT);
        TT_FATAL(weights.get_legacy_shape()[-1] % TILE_WIDTH == 0, "Number of columns in table {} must be factor of tile width {}", weights.get_legacy_shape()[-1], TILE_WIDTH);
        if (is_sharded(this->output_mem_config.memory_layout)) {
            const auto& shard_spec = this->output_mem_config.shard_spec;
            TT_FATAL(shard_spec.has_value(), "Sharded memory config must have a shard spec");
            TT_FATAL(shard_spec->shape[0] % TILE_HEIGHT == 0, "Shard height {} must be a multiple of tile height {} to have the output tensor tilized", shard_spec->shape[0], TILE_HEIGHT);
            TT_FATAL(shard_spec->shape[1] % TILE_WIDTH == 0, "Shard width {} must be a multiple of tile width {} to have the output tensor tilized", shard_spec->shape[1], TILE_WIDTH);
            TT_FATAL(a.volume() % shard_spec->shape[0] == 0, "Input tensor volume {} must be a multiple of shard height {}", a.volume(), shard_spec->shape[0]);
            TT_FATAL(weights.get_legacy_shape()[-1] % shard_spec->shape[1] == 0, "Number of columns in table {} must be factor of shard width {}", weights.get_legacy_shape()[-1], shard_spec->shape[1]);
        }
    } else {
        TT_FATAL(this->output_mem_config.memory_layout == TensorMemoryLayout::INTERLEAVED, "Embedding only supports interleaved RM outputs");
        TT_FATAL(!is_block_float(this->output_dtype), "Output cannot be a block float dtype when not tilized");
    }
    if(a.get_layout() == Layout::ROW_MAJOR) {
        TT_FATAL(a.get_legacy_shape()[1] == 1 && a.get_legacy_shape()[2] == 1, "Only dim 0 && 3 for the input can be non 1");
    }
    switch (this->embeddings_type) {
        case EmbeddingsType::PADDED: TT_FATAL(this->pad_token.has_value(), "Pad token must be specified when PADDED Embeddings Type is specified"); break;
        case EmbeddingsType::BINARY: TT_FATAL(weights.get_legacy_shape()[-2] == 2, "Weight tensor must have 2 embeddings for BINARY Embeddings Type"); break;
        default: TT_FATAL(!this->pad_token.has_value(), "Pad token must not be specified when PADDED Embeddings Type is not specified");
    }
}

std::vector<TensorSpec> Embeddings::compute_output_specs(const std::vector<Tensor> &input_tensors) const {
    const auto &input_tensor = input_tensors.at(0);
    const auto &weight_tensor = input_tensors.at(1);
    auto num_output_embeddings = input_tensor.logical_shape()[-1];
    auto batch_num = input_tensor.logical_shape()[0];
    auto num_embedding_dims = weight_tensor.logical_shape()[-1];

    ttnn::SimpleShape output_shape({batch_num, 1, num_output_embeddings, num_embedding_dims});
    auto output_layout = tilized ? Layout::TILE : Layout::ROW_MAJOR;
    return {TensorSpec(output_shape, TensorLayout(output_dtype, PageConfig(output_layout), output_mem_config))};
}

operation::ProgramWithCallbacks Embeddings::create_program(
    const std::vector<Tensor> &input_tensors, std::vector<Tensor> &output_tensors) const {
    const auto &a = input_tensors.at(0);
    const auto &weights = input_tensors.at(1);
    auto &output_tensor = output_tensors.at(0);
    return detail::embeddings_(a, weights, output_tensor, this->tilized, this->embeddings_type, this->pad_token);
}

}  // namespace ttnn::operations::embedding
