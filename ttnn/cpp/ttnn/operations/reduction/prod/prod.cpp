// SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC.
// SPDX-License-Identifier: Apache-2.0

#include "prod.hpp"
#include "device/prod_nc_op.hpp"
#include "device/prod_op_all.hpp"

#include "ttnn/operations/core/core.hpp"
#include "ttnn/operations/creation.hpp"
#include "ttnn/operations/data_movement/common/common.hpp"
#include "ttnn/operations/data_movement/permute/permute.hpp"
#include "ttnn/operations/data_movement/slice/slice.hpp"
#include "ttnn/operations/data_movement/squeeze/squeeze.hpp"
#include "ttnn/operations/data_movement/tilize_with_val_padding/tilize_with_val_padding.hpp"
#include "ttnn/operations/functions.hpp"

namespace ttnn::operations::reduction {

inline Tensor compute_prod_all(const Tensor& input_a, const MemoryConfig& output_mem_config) {
    auto formatted_input_tensor = input_a;
    if (formatted_input_tensor.layout() != Layout::TILE) {
        auto a_pad_shape = ttnn::operations::data_movement::pad_to_tile_shape(input_a.padded_shape());

        auto need_format = input_a.layout() != Layout::TILE || input_a.padded_shape() != a_pad_shape;
        if (need_format) {
            formatted_input_tensor =
                ttnn::tilize_with_val_padding(input_a, a_pad_shape, PadValue(1.0f), input_a.memory_config());
        }
    }

    return tt::operations::primary::prod_all(formatted_input_tensor, output_mem_config);
}

inline Tensor compute_prod_nc(const Tensor& temp, int64_t dim, const MemoryConfig& output_mem_config) {
    // layout conversion
    auto formatted_input_tensor = temp;
    if (formatted_input_tensor.layout() == Layout::ROW_MAJOR) {
        auto a_pad_shape = ttnn::operations::data_movement::pad_to_tile_shape(temp.padded_shape());
        auto out_shape = temp.padded_shape();
        out_shape = ttnn::Shape({out_shape[0], out_shape[1], out_shape[2], out_shape[3]});
        auto need_format = temp.layout() != Layout::TILE || temp.padded_shape() != a_pad_shape;
        if (need_format) {
            formatted_input_tensor =
                ttnn::tilize_with_val_padding(temp, a_pad_shape, PadValue(1.0f), temp.memory_config());
        }
    }
    // Apply prod
    ttnn::SmallVector<int64_t> dimension = {(dim == 1 || dim == -3) ? 1 : 0};
    const auto& input_shape = formatted_input_tensor.logical_shape();
    std::array<uint32_t, 4> required = {
        ((dim == 1 || dim == -3) ? input_shape[0] : 1),
        ((dim == 1 || dim == -3) ? 1 : input_shape[1]),
        input_shape[2],
        input_shape[3]};

    auto ttnn_shape = ttnn::Shape(required);
    auto* ttnn_device = formatted_input_tensor.device();

    return tt::operations::primary::prod_nc(
        formatted_input_tensor,
        ttnn::zeros(
            ttnn_shape,
            formatted_input_tensor.dtype(),
            formatted_input_tensor.layout(),
            *ttnn_device,
            output_mem_config),
        dimension,
        output_mem_config);
}

Tensor ProdOperation::invoke(
    const Tensor& input_a,
    const std::optional<int64_t> dim,
    const bool keepdim,
    const std::optional<MemoryConfig>& memory_config) {
    auto output_mem_config = memory_config.value_or(input_a.memory_config());
    const int size = static_cast<int>(input_a.logical_shape().rank());

    const auto old_rank = input_a.logical_shape().rank();
    const ttnn::Shape& input_shape = input_a.logical_shape();

    // If no dim is provided, compute the prod across all dimensions
    if (!dim.has_value()) {
        Tensor result = compute_prod_all(input_a, output_mem_config);
        if (keepdim) {
            // Reshape to have all dimensions (as many as the input rank) set to 1.
            ttnn::SmallVector<uint32_t> output_shape(old_rank, 1);
            result = ttnn::reshape(result, ttnn::Shape{output_shape});
        }
        return result;
    }

    TT_FATAL(size > 0, "Tensor has no dimensions");
    TT_FATAL(
        *dim >= -size && *dim <= size - 1,
        "Dimension for prod is out of range (expected to be in range of [{}, {}]",
        -size,
        size - 1);

    // For higher dimension Tensors, we need to squeeze to 4D to perform the reduction
    if (old_rank > 4) {
        // Bring dim into range [0, size - 1]
        const int64_t positive_dim = *dim < 0 ? *dim + size : *dim;

        // Prod only can do reduction on dim0 or dim1.
        // After squeezing the ND tensor to 4D, the third last dimension will become the second (i.e, [1]) dimension
        // We will permute the target reduction dim to this eventual position in a 4D tensor, and reduce it there
        // Then unsqueeze back to ND, and move the reduction dim back to its original position

        // First, permute the target reduction dim to the third last position
        const int third_last_dim_idx = input_a.logical_shape().rank() - 3;
        const bool permute_required = third_last_dim_idx != positive_dim;

        ttnn::SmallVector<int64_t> post_permute_dims(input_a.logical_shape().rank());
        std::iota(post_permute_dims.begin(), post_permute_dims.end(), 0);
        std::swap(post_permute_dims[third_last_dim_idx], post_permute_dims[positive_dim]);

        // Tensor with target reduction dim at third last position
        ttnn::Tensor permuted =
            permute_required ? ttnn::permute(input_a, post_permute_dims, output_mem_config) : input_a;

        // Now squeeze to 4D and do the 4D prod.
        // Dim0 grows to include the rest of the dimensions, and our "third last" dim moves into dim1, which is our 4D
        // reduction dim
        auto input_tensor_4d = data_movement::squeeze_from_ND_to_4D(permuted);
        Tensor result = compute_prod_nc(input_tensor_4d, /*dim=*/1, output_mem_config);

        // Unsqueeze dim0 to restore the original tensor rank
        ttnn::Shape output_shape = ttnn::Shape(input_shape);
        output_shape[positive_dim] = output_shape[third_last_dim_idx];
        output_shape[third_last_dim_idx] = 1;

        result = ttnn::reshape(result, output_shape);

        // Can now permute the reduced dim to the correct position
        if (permute_required) {
            result = ttnn::permute(result, post_permute_dims, output_mem_config);
        }

        if (!keepdim) {
            result = ttnn::squeeze(result, positive_dim);
        }
        return result;
    }
    // 4D or lower dimension Tensors
    // For lower dimension Tensors, we need to unsqueeze to 4D
    auto input_tensor_4d = ttnn::unsqueeze_to_4D(input_a);

    // Update the dim because we unsqueezed input to 4d
    // If dim is negative, counting from the back is not impacted by the unsqueeze
    const int64_t dim_4d = (*dim >= 0) ? (4 - old_rank + *dim) : *dim;

    Tensor temp = input_tensor_4d;
    // Permute for dim 2,3
    if (dim_4d == 2 || dim_4d == -2) {
        ttnn::SmallVector<int64_t> permute_dims = {2, 0, 1, 3};
        temp = ttnn::permute(input_tensor_4d, permute_dims, output_mem_config);
    } else if (dim_4d == 3 || dim_4d == -1) {
        ttnn::SmallVector<int64_t> permute_dims = {3, 0, 1, 2};
        temp = ttnn::permute(input_tensor_4d, permute_dims, output_mem_config);
    }
    Tensor result = compute_prod_nc(temp, dim_4d, output_mem_config);
    // Permute and unpad result for dim 2,3. Don't need to process dim 0,1.
    auto step = ttnn::SmallVector<uint32_t>({1, 1, 1, 1});
    if (dim_4d == 0 || dim_4d == 1 || dim_4d == -4 || dim_4d == -3) {
        result = ttnn::squeeze_from_4D(result, old_rank);
    } else if (dim_4d == 2 || dim_4d == -2) {
        ttnn::SmallVector<int64_t> after_permute_dims = {1, 2, 0, 3};
        Tensor required = ttnn::permute(result, after_permute_dims, output_mem_config);
        const auto& input_shape = input_tensor_4d.logical_shape();
        ttnn::SmallVector<uint32_t> start_index = {0, 0, 0, 0};
        ttnn::SmallVector<uint32_t> end_index = {input_shape[0], input_shape[1], 1, input_shape[3]};
        result = ttnn::squeeze_from_4D(ttnn::slice(required, start_index, end_index, step, std::nullopt), old_rank);
    } else {  // dim 3
        // permute
        ttnn::SmallVector<int64_t> after_permute_dims = {1, 2, 0, 3};
        Tensor required = ttnn::permute(result, after_permute_dims, output_mem_config);
        // unpad
        const auto& input_shape = input_tensor_4d.logical_shape();
        ttnn::SmallVector<uint32_t> start_index = {0, 0, 0, 0};
        ttnn::SmallVector<uint32_t> end_index = {input_shape[0], input_shape[1], 1, input_shape[2]};
        Tensor new_unpad_tensor = ttnn::slice(required, start_index, end_index, step, std::nullopt);
        // permute back
        after_permute_dims = {0, 1, 3, 2};
        Tensor res_host = ttnn::permute(new_unpad_tensor, after_permute_dims, output_mem_config);
        result = ttnn::squeeze_from_4D(res_host, old_rank);
    }
    return keepdim ? result : ttnn::squeeze(result, *dim);
}

Tensor ProdOperation::invoke(
    const Tensor& input,
    const Tensor& output,
    ttnn::SmallVector<int64_t>& dims,
    const std::optional<MemoryConfig>& memory_config) {
    auto mem_cfg = memory_config.value_or(input.memory_config());

    if (dims.empty()) {
        return compute_prod_all(input, mem_cfg);
    }
    return tt::operations::primary::prod_nc(input, output, dims, mem_cfg);
}

}  // namespace ttnn::operations::reduction
