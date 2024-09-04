// SPDX-FileCopyrightText: © 2024 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include "ttnn/run_operation.hpp"
#include "ttnn/deprecated/tt_dnn/op_library/moreh_getitem/moreh_getitem_op.hpp"
#include "ttnn/deprecated/tt_dnn/op_library/moreh_helper_functions.hpp"
#include "ttnn/deprecated/tt_dnn/op_library/work_split.hpp"
#include "tt_metal/common/constants.hpp"
#include "tt_metal/host_api.hpp"

using namespace tt::constants;
using namespace std;
using namespace tt::tt_metal;

namespace tt {
namespace operations {
namespace primary {

void MorehGetitem::validate_with_output_tensors(
    const std::vector<Tensor>& input_tensors, const std::vector<std::optional<Tensor>>& output_tensors) const {
    // validate input tensor
    auto& input_tensor = input_tensors.at(0);
    auto input_layout = input_tensor.get_layout();

    TT_FATAL(input_tensor.storage_type() == StorageType::DEVICE, "Operands to getitem need to be on device!");
    TT_FATAL(input_tensor.buffer() != nullptr, "Operands to getitem need to be allocated in buffers on device!");
    auto dtype = input_tensor.get_dtype();
    TT_FATAL(dtype == DataType::INT32 || dtype == DataType::BFLOAT16);

    // validate index tensors
    uint32_t index_size = input_tensors.at(1).get_legacy_shape()[-1];
    for (uint32_t i = 1; i < input_tensors.size(); i++) {
        auto& index_tensor = input_tensors.at(i);
        TT_FATAL(index_tensor.storage_type() == StorageType::DEVICE, "Operands to getitem need to be on device!");
        TT_FATAL(index_tensor.buffer() != nullptr, "Operands to getitem need to be allocated in buffers on device!");
        TT_FATAL(index_tensor.get_dtype() == DataType::INT32);

        auto index_shape = index_tensor.get_legacy_shape();
        auto index_layout = index_tensor.get_layout();
        if (index_layout == Layout::ROW_MAJOR) {
            TT_FATAL(index_shape.rank() == 1);
        } else if (index_layout == Layout::TILE) {
            TT_FATAL(index_shape.rank() == 5);
        }
        TT_FATAL(
            !(input_layout == Layout::ROW_MAJOR && index_layout == Layout::TILE),
            "input layout ROW_MAJOR and index layout TILE not supported");
        TT_FATAL(index_size == index_shape[-1], "The shapes of all index tensors must be identical!");
    }

    if (input_layout == Layout::ROW_MAJOR) {
        for (auto dim : this->index_dims) {
            TT_FATAL(dim != 4, "getitem for ROW_MAJOR layout not support W index tensor!");
        }
    }

    uint32_t dim_start = this->index_dims.front();
    uint32_t i = 0;
    for (auto dim : this->index_dims) {
        TT_FATAL(
            dim_start + i == dim,
            fmt::format("The value of index_dims={} must be consecutive integers.", this->index_dims));
        i++;
    }

    if (output_tensors.empty() || !output_tensors.at(0).has_value()) {
        // If the user decided to not use any optional output tensors, then this would be empty or would be a nullptr.
        return;
    }
    TT_FATAL(output_tensors.size() == 1, "Must have 1 output tensor");
    TT_FATAL(dtype == output_tensors.front().value().get_dtype());
}

std::vector<Shape> MorehGetitem::compute_output_shapes(const std::vector<Tensor>& input_tensors) const {
    auto input_shape = input_tensors.at(0).get_legacy_shape();
    auto output_shape = input_shape;
    auto layout = input_tensors.at(0).get_layout();

    if (layout == Layout::TILE) {
        // compute output shape
        // ex)
        // input: (10, 20, 30, 40)
        // index_tensor: [(100), (100)]
        // index_dims = 1,2
        // output: (10, 1, 100, 40)

        auto dim_offset = 5 - input_shape.rank();
        auto dimensions_pads = std::vector<Padding::PadDimension>();
        std::vector<uint32_t> output_size_vec;

        for (int dim = 0; dim < output_shape.size(); dim++) {
            dimensions_pads.push_back(output_shape.padding()[dim]);
            output_size_vec.push_back(output_shape[dim]);
        }

        auto index = input_tensors.at(1);
        uint32_t index_size = index.get_legacy_shape()[-1];
        uint32_t index_size_without_padding = index.get_legacy_shape().without_padding()[-1];

        uint32_t last_dim = this->index_dims.back() + dim_offset;

        for (uint32_t i = 0; i < this->index_dims.size(); i++) {
            uint32_t out_put_dim = this->index_dims.at(i);
            uint32_t dim = out_put_dim + dim_offset;
            auto index = input_tensors.at(i + 1);

            if (dim == 3 || dim == 4) {
                dimensions_pads[out_put_dim] = Padding::PadDimension{.front = 0, .back = 31};
                output_size_vec[out_put_dim] = 32;
            } else {
                output_size_vec[out_put_dim] = 1;
            }
        }

        if (last_dim == 3 || last_dim == 4) {
            output_size_vec[this->index_dims.back()] = round_up_to_mul32(index_size);
            uint32_t padding_back = round_up_to_mul32(index_size_without_padding) - index_size_without_padding;
            dimensions_pads[this->index_dims.back()] = Padding::PadDimension{.front = 0, .back = padding_back};
        } else {
            output_size_vec[this->index_dims.back()] = index_size_without_padding;
        }

        const auto padding = Padding(dimensions_pads, Padding::PadValue::Any);
        output_shape = Shape(output_size_vec, padding);
    } else {
        // compute output shape
        // ex)
        // input: (10, 20, 30, 40)
        // index_tensor: [(100), (100)]
        // index_dims = 1,2
        // output: (10, 100, 40)
        std::vector<uint32_t> output_size_vec;

        auto input_shape = input_tensors.at(0).get_legacy_shape();
        uint32_t input_rank = input_shape.rank();

        auto index = input_tensors.at(1);
        uint32_t index_size = index.get_legacy_shape()[0];

        uint32_t start_dim = this->index_dims.front();
        uint32_t last_dim = this->index_dims.back();
        for (uint32_t input_dim = 0; input_dim < input_rank; input_dim++) {
            if (input_dim < start_dim) {
                output_size_vec.push_back(input_shape[input_dim]);
            } else if (start_dim == input_dim) {
                output_size_vec.push_back(index_size);
            } else if (last_dim < input_dim) {
                output_size_vec.push_back(input_shape[input_dim]);
            }
        }

        output_shape = Shape(output_size_vec);
    }

    return {output_shape};
}

std::vector<Tensor> MorehGetitem::create_output_tensors(
    const std::vector<Tensor>& input_tensors, const std::vector<std::optional<Tensor>>& output_tensors) const {
    if (!output_tensors.empty() && output_tensors.at(0).has_value()) {
        return {output_tensors.at(0).value()};
    }

    auto dtype = input_tensors.at(0).get_dtype();
    auto layout = input_tensors.at(0).get_layout();
    Tensor output =
        operation::generic_create_output_tensors(*this, input_tensors, dtype, layout, this->output_mem_config).at(0);

    return {output};
}

operation::ProgramWithCallbacks MorehGetitem::create_program(
    const std::vector<Tensor>& input_tensors, std::vector<Tensor>& output_tensors) const {
    auto& input = input_tensors.at(0);
    auto& output = output_tensors.at(0);

    std::vector<Tensor> index_tensors;
    for (uint32_t i = 1; i < input_tensors.size(); i++) {
        index_tensors.push_back(input_tensors.at(i));
    }

    if (input.get_layout() == Layout::ROW_MAJOR) {
        return {moreh_getitem_rm(input, index_tensors, this->index_dims, output, this->core_range)};
    }

    return {moreh_getitem_tilized(input, index_tensors, this->index_dims, output, this->core_range)};
}

Tensor moreh_getitem(
    const Tensor& input_tensor,
    const std::vector<Tensor>& index_tensors,
    const std::vector<uint32_t>& index_dims,
    std::optional<Tensor> output_tensor,
    const MemoryConfig& output_mem_config) {
    auto device = input_tensor.device();
    auto grid_coord = device->compute_with_storage_grid_size();
    const CoreRange all_cores({0, 0}, {grid_coord.x - 1, grid_coord.y - 1});

    std::vector<Tensor> new_input_tensors;
    new_input_tensors.push_back(input_tensor);
    new_input_tensors.insert(new_input_tensors.end(), index_tensors.begin(), index_tensors.end());

    std::vector<Tensor> output_tensors = {Tensor(operation::get_workers_for_op_output(new_input_tensors))};

    operation::launch_op(
        [index_dims, all_cores, output_mem_config](
            const std::vector<Tensor>& input_tensors,
            const std::vector<std::optional<const Tensor>>& optional_input_tensors,
            const std::vector<std::optional<Tensor>>& optional_output_tensors) mutable -> std::vector<Tensor> {
            return operation::run(
                MorehGetitem{.index_dims = index_dims, .core_range = all_cores, .output_mem_config = output_mem_config},
                input_tensors,
                optional_input_tensors,
                optional_output_tensors);
        },
        new_input_tensors,
        output_tensors,
        {},
        {output_tensor});

    return output_tensors.at(0);
}

}  // namespace primary
}  // namespace operations
}  // namespace tt
