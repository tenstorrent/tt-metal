// SPDX-FileCopyrightText: © 2024 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include "prod.hpp"
#include "device/prod_nc_op.hpp"
#include "device/prod_op_all.hpp"
#include "ttnn/operations/experimental/auto_format/auto_format.hpp"
#include "cpp/ttnn/operations/creation.hpp"
#include "ttnn/operations/data_movement/slice/slice.hpp"
#include "ttnn/operations/data_movement/permute/permute.hpp"
#include "ttnn/operations/functions.hpp"
#include "ttnn/types.hpp"
#include "ttnn/common/queue_id.hpp"
#include "cpp/ttnn/operations/data_movement/squeeze/squeeze.hpp"
#include "ttnn/operations/core/core.hpp"

namespace ttnn::operations::reduction {

// Autoformat support
inline Tensor change_layout_to_tile(const Tensor& temp, const MemoryConfig& output_mem_config) {
    using ttnn::operations::experimental::auto_format::AutoFormat;
    auto formatted_input_tensor = temp;
    if (formatted_input_tensor.get_layout() == Layout::ROW_MAJOR) {
        auto a_pad_shape = AutoFormat::pad_to_tile_shape(temp.get_padded_shape(), false, false, true, true);
        if (!AutoFormat::check_input_tensor_format(temp, a_pad_shape)) {
            formatted_input_tensor =
                AutoFormat::format_input_tensor(temp, temp.device(), a_pad_shape, 1.0, Layout::TILE);
        }
    }
    return formatted_input_tensor;
}

inline Tensor prod_all(const Tensor& input_a, const MemoryConfig& output_mem_config) {
    using ttnn::operations::experimental::auto_format::AutoFormat;
    auto formatted_input_tensor = input_a;
    if (formatted_input_tensor.get_layout() == Layout::ROW_MAJOR) {
        auto a_pad_shape = AutoFormat::pad_to_tile_shape(input_a.get_padded_shape(), false, false, true, true);
        auto out_shape = input_a.get_padded_shape();
        out_shape = ttnn::Shape({out_shape[0], out_shape[1], out_shape[2], out_shape[3]});
        if (!AutoFormat::check_input_tensor_format(input_a, a_pad_shape)) {
            formatted_input_tensor =
                AutoFormat::format_input_tensor(input_a, input_a.device(), a_pad_shape, 1.0, Layout::TILE);
        }
    }
    return tt::operations::primary::prod_all(formatted_input_tensor, output_mem_config);
}

inline Tensor prod_nc(const Tensor& temp, int64_t dim, const MemoryConfig& output_mem_config) {
    using ttnn::operations::experimental::auto_format::AutoFormat;
    // layout conversion
    auto formatted_input_tensor = temp;
    if (formatted_input_tensor.get_layout() == Layout::ROW_MAJOR) {
        auto a_pad_shape = AutoFormat::pad_to_tile_shape(temp.get_padded_shape(), false, false, true, true);
        auto out_shape = temp.get_padded_shape();
        out_shape = ttnn::Shape({out_shape[0], out_shape[1], out_shape[2], out_shape[3]});
        if (!AutoFormat::check_input_tensor_format(temp, a_pad_shape)) {
            formatted_input_tensor =
                AutoFormat::format_input_tensor(temp, temp.device(), a_pad_shape, 1.0, Layout::TILE);
        }
    }
    // Apply prod
    ttnn::SmallVector<int64_t> dimension = {(dim == 1 || dim == -3) ? 1 : 0};
    const auto& input_shape = formatted_input_tensor.get_logical_shape();
    std::array<uint32_t, 4> required = {
        ((dim == 1 || dim == -3) ? input_shape[0] : 1),
        ((dim == 1 || dim == -3) ? 1 : input_shape[1]),
        input_shape[2],
        input_shape[3]};

    auto ttnn_shape = ttnn::Shape(required);
    auto ttnn_device = formatted_input_tensor.device();

    return tt::operations::primary::prod_nc(
        formatted_input_tensor,
        ttnn::zeros(
            ttnn_shape,
            formatted_input_tensor.get_dtype(),
            formatted_input_tensor.get_layout(),
            std::optional<std::reference_wrapper<tt::tt_metal::IDevice>>(*ttnn_device),
            output_mem_config),
        dimension,
        output_mem_config);
}

Tensor ProdOperation::invoke(
    const Tensor& input_a,
    bool all_dimensions,
    int64_t dim,
    const bool keepdim,
    const std::optional<MemoryConfig>& memory_config) {
    auto output_mem_config = memory_config.value_or(input_a.memory_config());
    const int size = static_cast<int>(input_a.get_logical_shape().rank());
    TT_FATAL(
        size && dim >= -size && dim <= size - 1,
        "Dimension out of range (expected to be in range of [-{}, {}]",
        size,
        size - 1);

    if (all_dimensions) {
        return prod_all(input_a, output_mem_config);
    }

    // FIXME: all the prod code is based on 4D tensors, so we need to convert the input tensor to 4D.
    // TODO: We need to handle the case where the input tensor is not 4D.
    const auto old_rank = input_a.get_logical_shape().rank();
    auto input_tensor_4d = ttnn::unsqueeze_to_4D(input_a);

    // update the dim because we unsqueezed input to 4d
    const int64_t old_dim = dim;
    if (dim >= 0) {
        dim = (4 - old_rank + dim);
    }

    Tensor temp = input_tensor_4d;
    // Permute for dim 2,3
    if (dim == 2 || dim == -2) {
        ttnn::SmallVector<int64_t> permute_dims = {2, 0, 1, 3};
        temp = ttnn::permute(input_tensor_4d, permute_dims, output_mem_config);
    } else if (dim == 3 || dim == -1) {
        ttnn::SmallVector<int64_t> permute_dims = {3, 0, 1, 2};
        temp = ttnn::permute(input_tensor_4d, permute_dims, output_mem_config);
    }
    Tensor result = prod_nc(temp, dim, output_mem_config);
    // Permute and unpad result for dim 2,3. Don't need to process dim 0,1.
    auto step = ttnn::SmallVector<uint32_t>({1, 1, 1, 1});
    if (dim == 0 || dim == 1 || dim == -4 || dim == -3) {
        result = ttnn::squeeze_from_4D(result, old_rank);
    } else if (dim == 2 || dim == -2) {
        ttnn::SmallVector<int64_t> after_permute_dims = {1, 2, 0, 3};
        Tensor required = ttnn::permute(result, after_permute_dims, output_mem_config);
        const auto& input_shape = input_tensor_4d.get_logical_shape();
        ttnn::SmallVector<uint32_t> start_index = {0, 0, 0, 0};
        ttnn::SmallVector<uint32_t> end_index = {input_shape[0], input_shape[1], 1, input_shape[3]};
        result = ttnn::squeeze_from_4D(ttnn::slice(required, start_index, end_index, step, std::nullopt), old_rank);
    } else {  // dim 3
        // permute
        ttnn::SmallVector<int64_t> after_permute_dims = {1, 2, 0, 3};
        Tensor required = ttnn::permute(result, after_permute_dims, output_mem_config);
        // unpad
        const auto& input_shape = input_tensor_4d.get_logical_shape();
        ttnn::SmallVector<uint32_t> start_index = {0, 0, 0, 0};
        ttnn::SmallVector<uint32_t> end_index = {input_shape[0], input_shape[1], 1, input_shape[2]};
        Tensor new_unpad_tensor = ttnn::slice(required, start_index, end_index, step, std::nullopt);
        // permute back
        after_permute_dims = {0, 1, 3, 2};
        Tensor res_host = ttnn::permute(new_unpad_tensor, after_permute_dims, output_mem_config);
        result = ttnn::squeeze_from_4D(res_host, old_rank);
    }
    return keepdim ? result : ttnn::squeeze(result, old_dim);
}

Tensor ProdOperation::invoke(
    const Tensor& input,
    const Tensor& output,
    ttnn::SmallVector<int64_t>& dims,
    const std::optional<MemoryConfig>& memory_config) {
    auto mem_cfg = memory_config.value_or(input.memory_config());
    return tt::operations::primary::prod_nc(input, output, dims, mem_cfg);
}

}  // namespace ttnn::operations::reduction
