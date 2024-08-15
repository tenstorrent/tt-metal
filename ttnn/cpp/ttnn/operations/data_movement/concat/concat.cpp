// SPDX-FileCopyrightText: © 2024 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include "ttnn/common/constants.hpp"
#include "ttnn/tensor/types.hpp"
#include "ttnn/operations/core/core.hpp"

#include "ttnn/cpp/ttnn/operations/data_movement/concat/device/concat_device_operation.hpp"
#include "ttnn/cpp/ttnn/operations/data_movement/concat/concat.hpp"

#include <ranges>


namespace ttnn {
namespace operations {
namespace data_movement {

    // Wrapper for TTDNN
    ttnn::Tensor ConcatOperation::invoke(
        uint8_t queue_id,
        const std::vector<ttnn::Tensor>& input_tensors,
        int dim,
        const std::optional<MemoryConfig>& memory_config,
        std::optional<ttnn::Tensor> optional_output_tensor) {
        TT_FATAL(input_tensors.size() > 0, "ttnn.concat: expected a non-empty list of Tensors!");
        TT_FATAL(!optional_output_tensor.has_value(), "optional output tensor currently unsupported!");
        const auto mem_config = memory_config.value_or(ttnn::DRAM_MEMORY_CONFIG); // should match input tensor memory config when unpopulated but causes CI errors for now

        if (input_tensors.size() == 1) {
            return ttnn::to_memory_config(input_tensors.at(0), mem_config, std::nullopt);
        }

        // TODO: Issue #8426: Add validation for ttnn.concat for sharded inputs
        // const bool all_tensors_are_tile_layout_without_padding = std::all_of(input_tensors.begin(), input_tensors.end(),
        // [dim](const ttnn::Tensor& input_tensor){
        //    return input_tensor.get_layout() == ttnn::TILE_LAYOUT and not has_tile_padding(input_tensor, dim);
        //});
        // TT_FATAL(all_tensors_are_tile_layout_without_padding, "Not Implemented");

        const ttnn::Tensor& first_tensor = input_tensors.front();
        const int rank = first_tensor.get_shape().rank();

        dim = first_tensor.get_legacy_shape().get_normalized_index(dim);

        TT_FATAL(
            dim >= 0 and dim < rank,
            "ttnn: Dimension out of range: dim {} cannot be used for tensors of rank {}",
            dim,
            rank);

        const bool shapes_match =
            std::all_of(input_tensors.begin(), input_tensors.end(), [first_tensor, dim](const ttnn::Tensor& t) {
                const auto& ft_shape = first_tensor.get_shape();
                const auto& t_shape = t.get_shape();

                const bool ranks_match = ft_shape.rank() == t_shape.rank();
                bool non_concat_dims_match = true;
                for (int i = 0; i < ft_shape.rank(); i++) {
                    non_concat_dims_match &= dim == i or t_shape[i] == ft_shape[i];
                }
                // bool non_concat_padded_dims_match = true;
                // for(int i = 0; i < ft_shape.rank(); i++) {
                //     non_concat_padded_dims_match &= dim == i or t_shape.with_tile_padding()[i] ==
                //     ft_shape.with_tile_padding()[i];
                // }
                return ranks_match and non_concat_dims_match;  // and non_concat_padded_dims_match;
            });

        TT_FATAL(
            shapes_match,
            "All dimensions must be the same size except for the dimension along which the contenation is taking place.");

        std::vector<ttnn::Tensor> itensor;
        std::transform(
            input_tensors.begin(),
            input_tensors.end(),
            std::back_inserter(itensor),
            [rank](const ttnn::Tensor& input_tensor) -> ttnn::Tensor {
                auto output = (rank < 4) ? ttnn::unsqueeze_to_4D(input_tensor) : input_tensor;
                return output;
            });
        // Convert dim after unsqueeze
        dim = dim + 4 - rank;
        auto output_tensor = concat_impl(itensor, dim, mem_config);
        while (output_tensor.get_shape().rank() > rank) {
            const auto shape = output_tensor.get_shape();
            const auto full_shape = output_tensor.get_shape().with_tile_padding();
            std::vector<uint32_t> shape_vec{};
            std::vector<uint32_t> full_shape_vec{};
            // int i = 0;
            // while(i < 3 and shape[i] == 1) i++;
            for (int i = 1; i < shape.rank(); i++) {
                shape_vec.push_back(shape[i]);
                full_shape_vec.push_back(full_shape[i]);
            }
            output_tensor = ttnn::reshape(output_tensor, ttnn::Shape(shape_vec, full_shape_vec));
        }

        return output_tensor;
    }

    ttnn::Tensor ConcatOperation::invoke (
        const std::vector<ttnn::Tensor>& input_tensors,
        int dim,
        const std::optional<MemoryConfig>& memory_config,
        std::optional<ttnn::Tensor> optional_output_tensor) {
        return invoke(DefaultQueueId, input_tensors, dim, memory_config, optional_output_tensor);
    }
};

}
}
