// SPDX-FileCopyrightText: © 2024 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include "ttnn/operations/moreh_embedding/device/moreh_embedding_device_operation.hpp"

#include "tt_metal/common/constants.hpp"
#include "tt_metal/detail/util.hpp"
#include "tt_metal/host_api.hpp"
#include "ttnn/deprecated/tt_dnn/op_library/math.hpp"
#include "ttnn/deprecated/tt_dnn/op_library/work_split.hpp"
#include "ttnn/operations/moreh_embedding/device/moreh_embedding_program_factory.hpp"

using namespace tt::constants;
using namespace std;
using namespace tt::tt_metal;

namespace ttnn::operations::moreh_embedding {

void MorehEmbeddings::validate_with_output_tensors(const std::vector<Tensor> &input_tensors, const std::vector<std::optional<Tensor>> &output_tensors) const {
    TT_FATAL(input_tensors.size() == 2, "Must have between 2 input tensors");
    auto &input = input_tensors[0];
    const auto &weight = input_tensors[1];

    TT_FATAL(this->max_norm.has_value() == false, "The max_norm feature is currently not available.");

    TT_FATAL(input.get_layout() == Layout::TILE);
    TT_FATAL(input.get_dtype() == DataType::INT32, "Input must be INT32");
    TT_FATAL(
        input.memory_config().memory_layout == TensorMemoryLayout::INTERLEAVED,
        "Moreh Embedding does not currently support sharding");

    TT_FATAL(weight.get_layout() == Layout::TILE);
    TT_FATAL(weight.get_dtype() == DataType::BFLOAT16);
    TT_FATAL(
        weight.memory_config().memory_layout == TensorMemoryLayout::INTERLEAVED,
        "Moreh Embedding does not currently support sharding");
    TT_FATAL(weight.get_legacy_shape().rank() == 2, "weight rank must be 2");

    TT_FATAL(
        this->output_mem_config.memory_layout == TensorMemoryLayout::INTERLEAVED,
        "Moreh Embedding does not currently support sharding");

    if (output_tensors[0].has_value()) {
        auto &output = output_tensors[0].value();
        TT_FATAL(output.get_layout() == Layout::TILE);
        TT_FATAL(output.get_dtype() == DataType::BFLOAT16);
        TT_FATAL(
            output.memory_config().memory_layout == TensorMemoryLayout::INTERLEAVED,
            "Moreh Embedding does not currently support sharding");
    }

}

std::vector<tt::tt_metal::Shape> MorehEmbeddings::compute_output_shapes(
    const std::vector<Tensor> &input_tensors) const {
    using namespace tt;
    using namespace tt::tt_metal;

    const auto &input = input_tensors.at(0);
    auto input_shape = input.get_legacy_shape();
    auto input_shape_without_padding = input_shape.without_padding();
    auto input_rank = input_shape.rank();

    const auto &weight = input_tensors.at(1);
    auto weight_shape = weight.get_legacy_shape().without_padding();

    auto embedding_dim = weight_shape[-1];

    auto dimensions_pads = std::vector<Padding::PadDimension>();
    std::vector<uint32_t> output_shape_vec;

    for (uint32_t dim = 0; dim < input_rank; dim++) {
        // padding for h dim
        if (dim == input_rank - 1) {
            uint32_t up32_shape = round_up(input_shape[dim], 32);
            uint32_t padding_back = up32_shape - input_shape_without_padding[dim];
            output_shape_vec.push_back(up32_shape);
            dimensions_pads.push_back(Padding::PadDimension{.front = 0, .back = padding_back});
        } else {
            output_shape_vec.push_back(input_shape_without_padding[dim]);
            dimensions_pads.push_back(Padding::PadDimension{.front = 0, .back = 0});
        }
    }

    // padding for w dim
    uint32_t up32_shape = round_up(embedding_dim, 32);
    uint32_t padding_back = up32_shape - embedding_dim;
    output_shape_vec.push_back(up32_shape);
    dimensions_pads.push_back(Padding::PadDimension{.front = 0, .back = padding_back});

    const auto padding = Padding(dimensions_pads, Padding::PadValue::Any);
    auto output_shape = tt::tt_metal::Shape(output_shape_vec, padding);

    return {output_shape};
}

std::vector<Tensor> MorehEmbeddings::create_output_tensors(const std::vector<Tensor> &input_tensors, const std::vector<std::optional<Tensor>> &output_tensors) const {
    if (output_tensors[0].has_value()) {
        return {output_tensors[0].value()};
    }

    const auto &weight_tensor = input_tensors.at(1);
    return operation::generic_create_output_tensors(
        *this, input_tensors, this->output_dtype, Layout::TILE, this->output_mem_config);
}

operation::ProgramWithCallbacks MorehEmbeddings::create_program(
    const std::vector<Tensor> &input_tensors, std::vector<Tensor> &output_tensors) const {
    const auto &input = input_tensors.at(0);
    const auto &weight = input_tensors.at(1);
    auto &output_tensor = output_tensors.at(0);

    auto device = input.device();
    auto grid_coord = device->compute_with_storage_grid_size();
    const CoreRange all_cores({0, 0}, {grid_coord.x - 1, grid_coord.y - 1});

    return detail::moreh_embeddings_(
        input,
        weight,
        this->max_norm,
        this->norm_type,
        output_tensor,
        core_range.value_or(all_cores),
        this->compute_kernel_config);
}

}  // namespace ttnn::operations::moreh_embedding
