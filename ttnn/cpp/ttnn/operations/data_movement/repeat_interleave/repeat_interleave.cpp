// SPDX-FileCopyrightText: © 2023 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0


#include "repeat_interleave.hpp"

#include "ttnn/cpp/ttnn/operations/data_movement/reshape_on_device/reshape.hpp"
#include "ttnn/cpp/ttnn/operations/data_movement/unsqueeze/unsqueeze.hpp"
#include "ttnn/cpp/ttnn/operations/data_movement/transpose/transpose.hpp"
#include "ttnn/cpp/ttnn/operations/copy.hpp"

namespace ttnn {
namespace operations {
namespace data_movement {

// repeat interleave supports repeats as 1 to inf, dim between 0 to 2
ttnn::Tensor ExecuteRepeatInterleave::invoke(const ttnn::Tensor& input_a, uint32_t repeat, int32_t dim, std::optional<MemoryConfig> output_mem_config) {
    std::vector<Tensor> combined_tensors;
    combined_tensors.reserve(repeat);
    MemoryConfig mem_config = output_mem_config.value_or(input_a.memory_config());
    if (repeat == 1) {
        return ttnn::to_memory_config(input_a, mem_config);
    }
    uint32_t input_rank = input_a.get_shape().rank();
    uint32_t normalized_dim = input_a.get_shape().get_normalized_index(dim);
    if (normalized_dim == input_rank - 1) {
        auto transposed_input = ttnn::transpose(input_a, -1, -2, mem_config);
        auto repeated_input = ExecuteRepeatInterleave::invoke(transposed_input, repeat, -2, mem_config);
        return ttnn::transpose(repeated_input, -1, -2, mem_config);
    }

    ttnn::Tensor rm_input = input_a;
    bool typecast = input_a.get_dtype() != DataType::BFLOAT16;
    if (typecast) {
        rm_input = ttnn::typecast(rm_input, DataType::BFLOAT16, mem_config);
    }

    rm_input = ttnn::to_layout(rm_input, Layout::ROW_MAJOR, std::nullopt, std::nullopt, (Device*)nullptr);
    std::vector<uint32_t> final_shape;
    final_shape.reserve(input_rank);
    for (uint32_t i = 0; i < rm_input.get_shape().rank(); i++) {
        final_shape.push_back(rm_input.get_shape()[i]);
    }
<<<<<<< HEAD

    final_shape[normalized_dim] *= repeat;

    auto unsqueezed_tensor = ttnn::unsqueeze(rm_input, normalized_dim + 1);
    for (uint32_t i = 0; i < repeat; i++) {
        combined_tensors.push_back(unsqueezed_tensor);
=======
    if (normalized_dim <= 1) {
        for (int i = 0; i < repeat; i++) {
            combined_tensors.push_back(input_a);
        }
        // TODO: For dim = 1 facing issue with concat_op
        if (normalized_dim) {
            Tensor concat_out = ttnn::concat(combined_tensors, 2);
            return ttnn::reshape_on_device(concat_out, ttnn::SimpleShape{shape_wh[0], shape_wh[1] * repeat, shape_wh[2], shape_wh[3]});
        } else {
            Tensor concat_out = ttnn::concat(combined_tensors, 1);
            return ttnn::reshape_on_device(concat_out, ttnn::SimpleShape{shape_wh[0] * repeat, shape_wh[1], shape_wh[2], shape_wh[3]});
        }
    } else {
        Tensor reshape_out = ttnn::reshape_on_device(input_a, ttnn::SimpleShape{1, 1, shape_wh[0] * shape_wh[1] * shape_wh[2], shape_wh[3]});
        for (int i = 0; i < repeat; i++) {
            combined_tensors.push_back(reshape_out);
        }
        Tensor concat_out = ttnn::concat(combined_tensors, 1);
        std::vector<int64_t> permute_dims = {0, 2, 1, 3};
        Tensor permute_out = ttnn::permute(concat_out, permute_dims);
        return ttnn::reshape_on_device(permute_out, ttnn::SimpleShape{shape_wh[0], shape_wh[1], shape_wh[2] * repeat, shape_wh[3]});
>>>>>>> #13707: Rework
    }

    auto concatenated_tensor = ttnn::concat(combined_tensors, normalized_dim + 1);
    auto reshaped_tensor = ttnn::reshape(concatenated_tensor, ttnn::Shape(final_shape));
    auto original_layout = ttnn::to_layout(reshaped_tensor, input_a.get_layout(), std::nullopt, std::nullopt, (Device*)nullptr);
    return typecast ? ttnn::typecast(original_layout, input_a.get_dtype(), mem_config) : original_layout;

}

}  // namespace data_movement
}  // namespace operations
}  // namespace ttnn
