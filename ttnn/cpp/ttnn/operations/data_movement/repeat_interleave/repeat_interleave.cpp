// SPDX-FileCopyrightText: © 2023 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0


#include "repeat_interleave.hpp"

#include "ttnn/cpp/ttnn/operations/data_movement/reshape/reshape.hpp"

namespace ttnn {
namespace operations {
namespace data_movement {


// repeat interleave supports repeats as 1 to inf, dim between 0 to 2
ttnn::Tensor ExecuteRepeatInterleave::invoke(const ttnn::Tensor& input_a, uint32_t repeat, int32_t dim, std::optional<MemoryConfig> output_mem_config) {
    std::vector<Tensor> combined_tensors;
    combined_tensors.reserve(repeat);
    auto shape_wh = input_a.get_legacy_shape();
    MemoryConfig mem_config = output_mem_config.value_or(input_a.memory_config());
    // normalizing the negative dim
    uint32_t normalized_dim = input_a.get_legacy_shape().get_normalized_index(dim);
    // check if dim is 1 or 3
    if (normalized_dim & 1) {
        constexpr uint32_t tmp_dim = 2;
        std::vector<int64_t> dims = {0, 1, 2, 3};
        std::swap(dims[dim], dims[tmp_dim]);
        Tensor transpose_input = ttnn::permute(input_a, dims);
        Tensor ril_result = ExecuteRepeatInterleave::invoke(transpose_input, repeat, tmp_dim, mem_config);
        return ttnn::permute(ril_result, dims);
    }
    if (normalized_dim <= 1) {
        for (int i = 0; i < repeat; i++) {
            combined_tensors.push_back(input_a);
        }
        // TODO: For dim = 1 facing issue with concat_op
        if (normalized_dim) {
            Tensor concat_out = ttnn::concat(combined_tensors, 2);
            return ttnn::reshape_on_device(concat_out, shape_wh[0], shape_wh[1] * repeat, shape_wh[2], shape_wh[3]);
        } else {
            Tensor concat_out = ttnn::concat(combined_tensors, 1);
            return ttnn::reshape_on_device(concat_out, shape_wh[0] * repeat, shape_wh[1], shape_wh[2], shape_wh[3]);
        }
    } else {
        Tensor reshape_out = ttnn::reshape_on_device(input_a, 1, 1, shape_wh[0] * shape_wh[1] * shape_wh[2], shape_wh[3]);
        for (int i = 0; i < repeat; i++) {
            combined_tensors.push_back(reshape_out);
        }
        Tensor concat_out = ttnn::concat(combined_tensors, 1);
        std::vector<int64_t> permute_dims = {0, 2, 1, 3};
        Tensor permute_out = ttnn::permute(concat_out, permute_dims);
        return ttnn::reshape_on_device(permute_out, shape_wh[0], shape_wh[1], shape_wh[2] * repeat, shape_wh[3]);
    }
}

}  // namespace data_movement
}  // namespace operations
}  // namespace ttnn
