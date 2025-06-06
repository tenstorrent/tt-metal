// SPDX-FileCopyrightText: © 2024 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include "ttnn/operations/data_movement/permute/permute.hpp"
#include "ttnn/operations/data_movement/concat/concat.hpp"
#include "ttnn/operations/eltwise/binary/binary.hpp"
#include "ttnn/operations/eltwise/unary/unary.hpp"
#include "ttnn/operations/experimental/reduction/argmax/argmax.hpp"
#include "ttnn/operations/creation.hpp"
#include "cpp/ttnn/operations/eltwise/ternary/where.hpp"
#include "ttnn/operations/reduction/generic/generic_reductions.hpp"
#include "ttnn/operations/core/core.hpp"

namespace ttnn::operations::experimental::reduction {

Tensor create_mask(const Tensor& input_a, const std::optional<MemoryConfig>& output_mem_config) {
    auto& padded_shape = input_a.padded_shape();
    auto& unpadded_shape = input_a.logical_shape();
    if (padded_shape == unpadded_shape) {
        return input_a;
    }
    float t_inf = -std::numeric_limits<float>::infinity();
    Tensor masked_input = ttnn::mask_padded_input<::bfloat16>(padded_shape, unpadded_shape, DataType::BFLOAT16);
    masked_input = ttnn::where(masked_input, input_a, t_inf, output_mem_config.value());
    return masked_input;
}
// Argmax returns the index of maximum element in the tensor
Tensor ArgmaxOperation::invoke(
    const Tensor& input, int64_t _dim, bool all, const std::optional<MemoryConfig>& output_mem_config) {
    auto output_memory_config = output_mem_config.value_or(input.memory_config());
    auto input_shape = input.padded_shape();
    TT_FATAL(input_shape.rank() == 4, "supported for rank-4 tensors at this time");

    Tensor input_a = create_mask(input, output_memory_config);

    uint32_t dim = input_shape.get_normalized_index(_dim);
    int size = input_a.physical_volume();

    if (!all) {
        if ((dim == (input_shape.rank() - 1)) || (dim == (input_shape.rank() - 2))) {
            bool is_width = (dim == (input_shape.rank() - 1));
            Tensor max_val = ttnn::max(input_a, (int)dim, true, output_memory_config);
            Tensor max_tensor = ttnn::zeros_like(input_a);
            Tensor tindex = ttnn::index_width<::bfloat16>(
                input.logical_shape(),
                input.padded_shape(),
                DataType::BFLOAT16,
                Layout::TILE,
                input_a.device(),
                output_memory_config);
            if (is_width) {
                max_tensor = ttnn::add(max_tensor, max_val, std::nullopt, output_memory_config);
            } else {
                tindex = ttnn::index_height<::bfloat16>(
                    input.logical_shape(),
                    input.padded_shape(),
                    DataType::BFLOAT16,
                    Layout::TILE,
                    input_a.device(),
                    output_memory_config);
                max_tensor = ttnn::add(max_tensor, max_val, std::nullopt, output_memory_config);
            }
            tindex = tindex.to_device(input_a.device());
            max_val.deallocate();
            Tensor cmp_results = ttnn::eq(input_a, max_tensor, std::nullopt, output_memory_config);
            Tensor max_indices = ttnn::multiply(cmp_results, tindex, std::nullopt, output_memory_config);
            cmp_results.deallocate();
            Tensor result = ttnn::where(ttnn::eqz(max_indices), size, max_indices, output_memory_config);
            max_indices.deallocate();
            result = ttnn::min(result, (int)dim, true, output_memory_config);
            Tensor res_index = ttnn::zeros_like(result);
            result = ttnn::where(ttnn::eq(result, size), res_index, result, output_memory_config);
            ttnn::SmallVector<int64_t> permute_dims = {3, 0, 1, 2};
            if (is_width) {
                res_index = ttnn::add(res_index, result, std::nullopt, output_memory_config);
            } else {
                res_index = ttnn::add(res_index, result, std::nullopt, output_memory_config);
                permute_dims[0] = 2;
                permute_dims[3] = 3;
            }
            result.deallocate();
            Tensor transpose_res = ttnn::permute(res_index, permute_dims, output_memory_config);
            return transpose_res;
        } else if ((dim == (input_shape.rank() - 3)) || (dim == (input_shape.rank() - 4))) {
            bool is_channel = (dim == (input_shape.rank() - 3));
            Tensor max_val = ttnn::max(input_a, (int)dim, true, output_memory_config);
            int repeat = input.logical_shape()[dim];
            std::vector<Tensor> combined_tensors;
            for (int cid = 0; cid < repeat; cid++) {
                combined_tensors.emplace_back(max_val);
            }
            max_val.deallocate();
            Tensor concat_out = ttnn::concat(combined_tensors, dim, output_memory_config);
            // Needed till `max` stops autoformatting output
            concat_out = ttnn::reshape(concat_out, input_a.logical_shape());
            Tensor cmp_results = ttnn::eq(input_a, concat_out, std::nullopt, output_memory_config);
            concat_out.deallocate();
            Tensor tindex = ttnn::index_channel<::bfloat16>(
                input.logical_shape(),
                input.padded_shape(),
                DataType::BFLOAT16,
                Layout::TILE,
                input_a.device(),
                output_memory_config);
            if (!is_channel) {
                tindex = ttnn::index_batch<::bfloat16>(
                    input.logical_shape(),
                    input.padded_shape(),
                    DataType::BFLOAT16,
                    Layout::TILE,
                    input_a.device(),
                    output_memory_config);
            }
            tindex = tindex.to_device(input_a.device());
            Tensor max_indices = ttnn::multiply(cmp_results, tindex, std::nullopt, output_memory_config);
            cmp_results.deallocate();
            Tensor midx = full_like(max_indices, size);
            Tensor result = ttnn::where(ttnn::eqz(max_indices), midx, max_indices, output_memory_config);
            result = ttnn::min(result, (int)dim, true, output_memory_config);
            Tensor res_index = ttnn::zeros_like(result);
            result = ttnn::where(ttnn::eq(result, full_like(result, size)), res_index, result, output_memory_config);
            if (is_channel) {
                ttnn::SmallVector<int64_t> permute_dims = {1, 0, 2, 3};
                Tensor transpose_res = ttnn::permute(result, permute_dims, output_memory_config);
                return transpose_res;
            } else {
                return result;
            }
        }
    }
    // TODO: Fix the index generation code. With the fix the code will work for argmax that return entire
    // maximum value index
    Tensor tindex = ttnn::index_all<::bfloat16>(
        input.logical_shape(),
        input.padded_shape(),
        DataType::BFLOAT16,
        Layout::TILE,
        input_a.device(),
        output_memory_config);
    Tensor max_val = ttnn::max(input_a, std::nullopt, true, output_memory_config);
    Tensor max_tensor = ttnn::zeros_like(input_a);
    max_tensor = ttnn::add(max_tensor, max_val, std::nullopt, output_memory_config);
    max_val.deallocate();
    Tensor cmp_results = ttnn::eq(input_a, max_tensor, std::nullopt, output_memory_config);
    Tensor max_indices = ttnn::multiply(cmp_results, tindex, std::nullopt, output_memory_config);
    cmp_results.deallocate();
    Tensor result = ttnn::where(ttnn::eqz(max_indices), size, max_indices, output_memory_config);
    max_indices.deallocate();
    result = ttnn::min(result, std::nullopt, true, output_memory_config);
    return result;
}

Tensor ArgminOperation::invoke(
    const Tensor& input_a, int64_t _dim, bool all, const std::optional<MemoryConfig>& output_mem_config) {
    auto output_memory_config = output_mem_config.value_or(input_a.memory_config());
    Tensor neg_input = ttnn::neg(input_a, output_memory_config);
    return ttnn::experimental::argmax(neg_input, _dim, all, output_memory_config);
}

}  // namespace ttnn::operations::experimental::reduction
