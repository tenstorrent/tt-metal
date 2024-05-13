// SPDX-FileCopyrightText: © 2023 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

#pragma once


#include "ttnn/cpp/ttnn/operations/core.hpp"
#include "tt_eager/tt_dnn/op_library/permute/permute_op.hpp"
#include "tt_eager/tt_dnn/op_library/concat/concat_op.hpp"

namespace ttnn {
namespace operations {
namespace data_movement {

inline bool is_on_device(const Tensor& t) {
    return t.storage_type() == tt::tt_metal::StorageType::DEVICE or t.storage_type() == tt::tt_metal::StorageType::MULTI_DEVICE;
}

inline bool has_tile_padding(const Tensor& t) {
    if(t.get_shape().rank() > 1) {
        auto the_shape = t.get_shape();
        auto the_shape_with_padding = t.get_shape().with_tile_padding();
        return the_shape[-1] != the_shape_with_padding[-1] or the_shape[-2] != the_shape_with_padding[-2];
    }
    return false;
}

inline bool has_tile_padding(const Tensor& t, int dim) {
    int rank = t.get_shape().rank();
    // Wrap dim
    dim = dim < 0 ? rank + dim : dim;
    TT_FATAL(dim >= 0 and dim < rank, "ttnn: Dimension out of range: dim {} cannot be used for tensors of rank {}", dim, rank);

    if(dim < rank) {
        auto the_shape = t.get_shape();
        auto the_shape_with_padding = t.get_shape().with_tile_padding();
        return the_shape[dim] != the_shape_with_padding[dim];
    }
    return false;
}

inline ttnn::Tensor permute(
    const ttnn::Tensor& input_tensor,
    const std::vector<int>&  order
) {
    const bool initial_input_tensor_on_device = is_on_device(input_tensor);
    const auto input_layout = input_tensor.get_layout();
    const auto input_rank = input_tensor.get_shape().rank();

    TT_FATAL(input_rank <= 4);
    TT_FATAL(input_rank == order.size(), "The number of dimensions in the tensor input does not match the length of the desired ordering");

    auto adjust_order = [](const std::vector<int>&  order) {
        std::vector<std::int64_t>  new_order;
        TT_FATAL(order.size() <= 4);
        int additional_ranks = 4 - order.size();
        for (int i = 0; i < additional_ranks; i++) {
            new_order.push_back(i);
        }
        for (int i = 0; i < order.size(); i++) {
            new_order.push_back(order.at(i)+additional_ranks);
        }
        return new_order;
    };
    auto itensor = (input_tensor.get_shape().rank() < 4) ? ttnn::unsqueeze_to_4D(input_tensor) : input_tensor;
    auto iorder  = adjust_order(order);

    if(has_tile_padding(itensor)) {
        itensor = ttnn::to_layout(itensor, ttnn::ROW_MAJOR_LAYOUT, std::nullopt, std::nullopt, (Device*)nullptr);
    }

    TT_FATAL(is_on_device(itensor) and itensor.get_shape().rank() == 4);
    auto output_tensor = tt::tt_metal::permute(itensor, iorder, ttnn::DRAM_MEMORY_CONFIG);
    output_tensor = ttnn::to_layout(output_tensor, input_layout, std::nullopt, std::nullopt, (Device*)nullptr);

    if(input_rank < 4){
        const auto shape = output_tensor.get_shape();
        const auto full_shape = output_tensor.get_shape().with_tile_padding();
        std::vector<uint32_t> shape_vec{};
        std::vector<uint32_t> full_shape_vec{};
        int i = 0;
        while(i < 3 and shape[i] == 1) i++;
        for(i; i < shape.rank(); i++) {
            shape_vec.push_back(shape[i]);
            full_shape_vec.push_back(full_shape[i]);
        }
        auto metal_shape = tt::tt_metal::Shape(shape_vec, full_shape_vec);
        output_tensor = ttnn::reshape(output_tensor, ttnn::Shape(metal_shape));
    }

    if(initial_input_tensor_on_device and not is_on_device(output_tensor)) {
        output_tensor = ttnn::to_device(output_tensor, input_tensor.device(), ttnn::DRAM_MEMORY_CONFIG);
    }

    return output_tensor;
}

inline ttnn::Tensor concat(
    const std::vector<ttnn::Tensor> & input_tensors,
    int dim,
    const std::optional<MemoryConfig>& memory_config_arg
) {
    TT_FATAL(input_tensors.size() > 0, "ttnn.concat: expected a non-empty list of Tensors!");

    const auto memory_config = memory_config_arg.value_or(ttnn::DRAM_MEMORY_CONFIG);

    if(input_tensors.size() == 1) {
        return ttnn::to_memory_config(input_tensors.at(0), memory_config, std::nullopt);
    }

    // TODO: Issue #8426: Add validation for ttnn.concat for sharded inputs
    //const bool all_tensors_are_tile_layout_without_padding = std::all_of(input_tensors.begin(), input_tensors.end(), [dim](const ttnn::Tensor& input_tensor){
    //    return input_tensor.get_layout() == ttnn::TILE_LAYOUT and not has_tile_padding(input_tensor, dim);
    //});
    //TT_FATAL(all_tensors_are_tile_layout_without_padding, "Not Implemented");

    const ttnn::Tensor& first_tensor = input_tensors.front();
    const int rank = first_tensor.get_shape().rank();

    // Wrap dim
    dim = dim < 0 ? rank + dim : dim;
    TT_FATAL(dim >= 0 and dim < rank, "ttnn: Dimension out of range: dim {} cannot be used for tensors of rank {}", dim, rank);

    const bool shapes_match = std::all_of(input_tensors.begin(), input_tensors.end(), [first_tensor, dim](const ttnn::Tensor& t){

        const auto& ft_shape = first_tensor.get_shape();
        const auto& t_shape = t.get_shape();

        const bool ranks_match = ft_shape.rank() == t_shape.rank();
        bool non_concat_dims_match = true;
        for(int i = 0; i < ft_shape.rank(); i++) {
            non_concat_dims_match &= dim == i or t_shape[i] == ft_shape[i];
        }
        //bool non_concat_padded_dims_match = true;
        //for(int i = 0; i < ft_shape.rank(); i++) {
        //    non_concat_padded_dims_match &= dim == i or t_shape.with_tile_padding()[i] == ft_shape.with_tile_padding()[i];
        //}
        return ranks_match and non_concat_dims_match; // and non_concat_padded_dims_match;
    });

    TT_FATAL(shapes_match, "All dimensions must be the same size except for the dimension along which the contenation is taking place.");

    std::vector<ttnn::Tensor> itensor;
    std::transform(input_tensors.begin(), input_tensors.end(), std::back_inserter(itensor),
        [rank](const ttnn::Tensor& input_tensor) -> ttnn::Tensor {
            auto output = (rank < 4) ? ttnn::unsqueeze_to_4D(input_tensor) : input_tensor;
            return output;
        }
    );
    // Convert dim after unsqueeze
    dim = dim + 4 - rank;
    auto output_tensor = tt::tt_metal::concat(itensor, dim, memory_config);
    while(output_tensor.get_shape().rank() > rank) {
        const auto shape = output_tensor.get_shape();
        const auto full_shape = output_tensor.get_shape().with_tile_padding();
        std::vector<uint32_t> shape_vec{};
        std::vector<uint32_t> full_shape_vec{};
        //int i = 0;
        //while(i < 3 and shape[i] == 1) i++;
        for(int i = 1; i < shape.rank(); i++) {
            shape_vec.push_back(shape[i]);
            full_shape_vec.push_back(full_shape[i]);
        }
        auto metal_shape = tt::tt_metal::Shape(shape_vec, full_shape_vec);
        output_tensor = ttnn::reshape(output_tensor, ttnn::Shape(metal_shape));
    }

    return output_tensor;
}

} // namespace data_movement
} // namespace operations
} // namespace ttnn
