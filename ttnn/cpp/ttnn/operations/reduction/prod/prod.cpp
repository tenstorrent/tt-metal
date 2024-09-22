// SPDX-FileCopyrightText: © 2024 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0


#include "prod.hpp"
#include "device/prod_nc_op.hpp"
#include "device/prod_op_all.hpp"
#include "ttnn/operations/experimental/auto_format/auto_format.hpp"
#include "ttnn/cpp/ttnn/operations/creation.hpp"
#include "ttnn/operations/data_movement/slice/slice.hpp"
#include "ttnn/operations/data_movement/permute/permute.hpp"
#include "ttnn/operations/numpy/functions.hpp"
#include "ttnn/types.hpp"
#include "ttnn/common/constants.hpp"

namespace ttnn::operations::reduction {

// Autoformat support
inline Tensor change_layout_to_tile(const Tensor& temp, const MemoryConfig& output_mem_config) {
    using ttnn::operations::experimental::auto_format::AutoFormat;
    auto formatted_input_tensor = temp;
    if(formatted_input_tensor.get_layout()==Layout::ROW_MAJOR){
        auto a_pad_shape = AutoFormat::pad_to_tile_shape(temp.get_legacy_shape(), false, false, true, true);
        if (!AutoFormat::check_input_tensor_format(temp, a_pad_shape)) {
            formatted_input_tensor = AutoFormat::format_input_tensor(temp, temp.device(), a_pad_shape, 1.0, Layout::TILE);
        }
    }
    return formatted_input_tensor;
}

inline Tensor prod_all(const Tensor& input_a, const MemoryConfig& output_mem_config) {
    using ttnn::operations::experimental::auto_format::AutoFormat;
    auto formatted_input_tensor = input_a;
    if (formatted_input_tensor.get_layout() == Layout::ROW_MAJOR) {
        auto a_pad_shape = AutoFormat::pad_to_tile_shape(input_a.get_legacy_shape(), false, false, true, true);
        auto out_shape = input_a.get_legacy_shape();
        out_shape = {out_shape[0], out_shape[1], out_shape[2], out_shape[3]};
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
    if(formatted_input_tensor.get_layout() == Layout::ROW_MAJOR) {
        auto a_pad_shape = AutoFormat::pad_to_tile_shape(temp.get_legacy_shape(), false, false, true, true);
        auto out_shape = temp.get_legacy_shape();
        out_shape = {out_shape[0], out_shape[1], out_shape[2], out_shape[3]};
        if (!AutoFormat::check_input_tensor_format(temp, a_pad_shape)) {
            formatted_input_tensor =
                AutoFormat::format_input_tensor(temp, temp.device(), a_pad_shape, 1.0, Layout::TILE);
        }
    }
    // Apply prod
    std::vector<int64_t> dimension = {(dim == 1 || dim == -3) ? 1 : 0};
    tt::tt_metal::LegacyShape input_shape = formatted_input_tensor.get_legacy_shape();
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
            std::optional<std::reference_wrapper<tt::tt_metal::Device>>(*ttnn_device),
            output_mem_config),
        dimension,
        output_mem_config);
}


Tensor ProdOperation::invoke(const Tensor& input_a, bool all_dimensions, int64_t dim, const std::optional<MemoryConfig>& memory_config) {
    auto output_mem_config = memory_config.value_or(input_a.memory_config());
    if (all_dimensions) {
        return prod_all(input_a, output_mem_config);
    }
    TT_FATAL(dim >= -4 && dim <= 3, "Dimension out of range (expected to be in range of [-4, 3]");
    Tensor temp = input_a;
    // Permute for dim 2,3
    if (dim == 2 || dim == -2) {
        std::vector<int64_t> permute_dims = {2, 0, 1, 3};
        temp = ttnn::permute(input_a, permute_dims, output_mem_config);
    } else if (dim == 3 || dim == -1) {
        std::vector<int64_t> permute_dims = {3, 0, 1, 2};
        temp = ttnn::permute(input_a, permute_dims, output_mem_config);
    }
    Tensor result = prod_nc(temp, dim, output_mem_config);
    // Permute and unpad result for dim 2,3
    auto step = std::vector<uint32_t>({1, 1, 1, 1});
    if (dim == 0 || dim == 1 || dim == -4 || dim == -3) {
        return result;
    } else if (dim == 2 || dim == -2) {
        std::vector<int64_t> after_permute_dims = {1, 2, 0, 3};
        Tensor required = ttnn::permute(result, after_permute_dims, output_mem_config);
        tt::tt_metal::LegacyShape input_shape = input_a.get_legacy_shape();
        std::vector<uint32_t> start_index = {0, 0, 0, 0};
        std::vector<uint32_t> end_index = {input_shape[0], input_shape[1], 1, input_shape[3]};
        return ttnn::slice(DefaultQueueId, required, start_index, end_index, step, std::nullopt);
    } else {  // dim 3
        // permute
        std::vector<int64_t> after_permute_dims = {1, 2, 0, 3};
        Tensor required = ttnn::permute(result, after_permute_dims, output_mem_config);
        // unpad
        tt::tt_metal::LegacyShape input_shape = input_a.get_legacy_shape();
        std::vector<uint32_t> start_index = {0, 0, 0, 0};
        std::vector<uint32_t> end_index = {input_shape[0], input_shape[1], 1, input_shape[2]};
        Tensor new_unpad_tensor = ttnn::slice(DefaultQueueId, required, start_index, end_index, step, std::nullopt);
        // permute back
        after_permute_dims = {0, 1, 3, 2};
        Tensor res_host = ttnn::permute(new_unpad_tensor, after_permute_dims, output_mem_config);
        if(res_host.storage_type() != StorageType::DEVICE or res_host.storage_type() != StorageType::MULTI_DEVICE) {
            res_host = res_host.pad_to_tile(0.0f);
            res_host = res_host.to(Layout::TILE);
            res_host = res_host.to(input_a.device());
        }
        return res_host;
    }
}

Tensor ProdOperation::invoke(const Tensor &input, const Tensor &output, std::vector<int64_t> &dims, const std::optional<MemoryConfig>& memory_config) {
        auto mem_cfg = memory_config.value_or(input.memory_config());
        return tt::operations::primary::prod_nc(input, output, dims, mem_cfg);
}

} // namespace ttnn::operations::reduction
