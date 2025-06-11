// SPDX-FileCopyrightText: © 2025 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include "sort.hpp"
#include "device/sort_device_operation.hpp"

#include "ttnn/run_operation.hpp"
#include "ttnn/common/queue_id.hpp"
#include "ttnn/operations/core/core.hpp"
#include "ttnn/operations/creation.hpp"
#include "ttnn/operations/data_movement/fill_pad/fill_pad.hpp"
#include "ttnn/operations/data_movement/pad/pad.hpp"
#include "ttnn/operations/reduction/reduction_common/reduction_common.hpp"
#include "ttnn/operations/data_movement/slice/slice.hpp"
#include "ttnn/tensor/shape/shape.hpp"
#include "ttnn/operations/data_movement/transpose/transpose.hpp"

namespace ttnn::operations::experimental::reduction::sort {
namespace {
namespace CMAKE_UNIQUE_NAMESPACE {

uint32_t next_power_of_two(uint32_t n) {
    if (n <= 1) {
        return 1;
    }

    // If n is already a power of two, return it
    if ((n & (n - 1)) == 0) {
        return n;
    }

    // Otherwise, compute the next power of two
    uint32_t power = 1;
    while (power < n) {
        power <<= 1;
    }
    return power;
}

Tensor pre_sort_transform_tensor(
    const Tensor& input_tensor,
    const int8_t dim,
    const bool is_dim_last_idx,
    const bool is_rank_le_4d,
    const bool descending) {
    if (input_tensor.logical_shape() == ttnn::Shape{1}) {
        // Early exit for scalar tensors, return the same tensor
        // Scalar tensors do not require sorting.
        return input_tensor;
    }
    // If dim is not last dimension transpose it
    const Tensor transposed_tensor = reduction_common::perform_transpose(input_tensor, is_dim_last_idx, dim, -1);
    // If input is not rank 4 transorm it to 4D
    const Tensor transformed_tensor = reduction_common::transform_to_4d_tensor(transposed_tensor, is_rank_le_4d);
    // Fill implicit tile padding with the appropriate value
    Tensor padded_tensor = ttnn::fill_implicit_tile_padding(
        transformed_tensor,
        descending ? -std::numeric_limits<float>::infinity() : std::numeric_limits<float>::infinity());

    // Check for need of manual padding - Bitonic sort works on dataset that are the size of power of two - add manual
    // padding if needed
    const auto current_padded_shape = padded_tensor.padded_shape();
    const auto last_dim = current_padded_shape[-1];
    auto padded_last_dim = next_power_of_two(last_dim);
    if ((padded_last_dim == last_dim) && (last_dim > tt::constants::TILE_WIDTH)) {
        // If the last dimension is already a power of two and is multiple of 64, no padding is needed
        return padded_tensor;
    }
    if (padded_last_dim == tt::constants::TILE_WIDTH) {
        // Bitonic sort works on tiles that are the size of power of two - need at least 2 tiles
        padded_last_dim = tt::constants::TILE_WIDTH * 2;
    }
    const Tensor padded_output_tensor = ttnn::pad(
        padded_tensor,
        tt::tt_metal::Array4D(
            {current_padded_shape[0], current_padded_shape[1], current_padded_shape[2], padded_last_dim}),
        tt::tt_metal::Array4D({0, 0, 0, 0}),
        descending ? -std::numeric_limits<float>::infinity() : std::numeric_limits<float>::infinity());

    return padded_output_tensor;
}

std::vector<Tensor> post_sort_transform_tensor(
    const Tensor& input_tensor,
    std::vector<Tensor>& result,
    const int8_t dim,
    const bool is_dim_last_idx,
    const Shape& original_lshape,
    const MemoryConfig& input_memory_config) {
    auto input_shape = input_tensor.padded_shape();
    const auto orig_rank = input_shape.rank();

    // Check if manual padding was applied for the last dimension
    const auto output_logical_shape = result[0].logical_shape();
    if (output_logical_shape[-1] != original_lshape[-1]) {
        const ttnn::SmallVector<uint32_t> step = {1, 1, 1, 1};
        const ttnn::SmallVector<uint32_t> start_index = {0, 0, 0, 0};
        const ttnn::SmallVector<uint32_t> end_index = {
            original_lshape[-4], original_lshape[-3], original_lshape[-2], original_lshape[-1]};
        result[0] = ttnn::slice(result[0], start_index, end_index, step, input_memory_config);
        result[1] = ttnn::slice(result[1], start_index, end_index, step, input_memory_config);
    }

    if (orig_rank < 4) {
        result[0] = ttnn::squeeze_from_4D(result[0], orig_rank);
        result[1] = ttnn::squeeze_from_4D(result[1], orig_rank);
    } else if (orig_rank > 4) {
        ttnn::SmallVector<uint32_t> result_shape(input_shape.cbegin(), input_shape.cend());
        result[0] = ttnn::reshape(result[0], ttnn::Shape{result_shape});
        result[1] = ttnn::reshape(result[1], ttnn::Shape{result_shape});
    }

    if (!is_dim_last_idx) {
        result[0] = ttnn::transpose(result[0], dim, -1, input_tensor.memory_config());
        result[1] = ttnn::transpose(result[1], dim, -1, input_tensor.memory_config());
    }

    TT_FATAL(
        result[0].logical_shape() == original_lshape,
        "Output tensor transformation did not create correct output shape! Got: {}, expected: {}",
        result[0].logical_shape(),
        original_lshape);

    return result;
}

bool validate_optional_output_tensors_for_early_exit(
    const std::optional<std::tuple<Tensor, Tensor>>& optional_output_tensors, const Shape& original_lshape) {
    if (!optional_output_tensors.has_value()) {
        return false;
    }

    auto output_tensor_0 = std::get<0>(optional_output_tensors.value());
    auto output_tensor_1 = std::get<1>(optional_output_tensors.value());

    return output_tensor_0.logical_shape() == original_lshape && output_tensor_1.logical_shape() == original_lshape;
}

void convert_tensor_dtype(Tensor& tensor, const DataType& target_dtype, IDevice* device) {
    if (tensor.dtype() == target_dtype) {
        // No need to change the dtype
        return;
    }
    // Convert the tensor to the target dtype
    // ttnn::to_dtype does not convert the tensor on Device, need to move it to CPU first
    tensor = tensor.cpu();  // blocking
    tensor = ttnn::to_dtype(tensor, target_dtype);
    tensor = tensor.to_device(device);
}

}  // namespace CMAKE_UNIQUE_NAMESPACE
}  // namespace

std::vector<Tensor> ExecuteSort::invoke(
    QueueId queue_id,
    const Tensor& input_tensor,
    const int8_t dim,
    const bool descending,
    const bool stable,
    const std::optional<MemoryConfig>& memory_config,
    std::optional<std::tuple<Tensor&, Tensor&>> optional_output_tensors) {
    ttnn::Shape original_lshape = input_tensor.logical_shape();
    auto rank = input_tensor.padded_shape().rank();

    // Check for early exit for scalar or empty tensors tensors
    if ((original_lshape == ttnn::Shape{}) || (original_lshape == ttnn::Shape{1})) {
        if (CMAKE_UNIQUE_NAMESPACE::validate_optional_output_tensors_for_early_exit(
                optional_output_tensors, original_lshape)) {
            std::get<0>(*optional_output_tensors).tensor_attributes->get_storage() =
                input_tensor.tensor_attributes->get_storage();
            return {std::get<0>(optional_output_tensors.value()), std::get<1>(optional_output_tensors.value())};
        } else {
            return {input_tensor, ttnn::zeros_like(input_tensor)};
        }
    }

    const bool is_dim_last_idx = (dim == -1 || dim == rank - 1);
    const bool is_rank_le_4d = rank <= 4;

    const auto memory_config_value = memory_config.has_value() ? memory_config.value() : input_tensor.memory_config();

    Tensor padded_input_tensor = CMAKE_UNIQUE_NAMESPACE::pre_sort_transform_tensor(
        input_tensor, dim, is_dim_last_idx, is_rank_le_4d, descending);

    std::vector<std::optional<Tensor>> output_tensors;
    if (optional_output_tensors.has_value()) {
        output_tensors = reduction_common::tuple_to_vector_optional(*optional_output_tensors);
        output_tensors[0] = CMAKE_UNIQUE_NAMESPACE::pre_sort_transform_tensor(
            output_tensors[0].value(), dim, is_dim_last_idx, is_rank_le_4d, descending);

        output_tensors[1] = CMAKE_UNIQUE_NAMESPACE::pre_sort_transform_tensor(
            output_tensors[1].value(), dim, is_dim_last_idx, is_rank_le_4d, descending);

        const auto target_index_dtype = DataType::UINT16;
        CMAKE_UNIQUE_NAMESPACE::convert_tensor_dtype(
            output_tensors[1].value(), target_index_dtype, input_tensor.device());

    } else {
        output_tensors = std::vector<std::optional<Tensor>>{
            std::nullopt,  // Placeholder for values tensor
            std::nullopt   // Placeholder for indices tensor
        };
    }

    auto sorted_tensors =
        ttnn::prim::sort(queue_id, padded_input_tensor, dim, descending, stable, memory_config_value, output_tensors);

    auto post_transform_output_tensors = CMAKE_UNIQUE_NAMESPACE::post_sort_transform_tensor(
        input_tensor, sorted_tensors, dim, is_dim_last_idx, original_lshape, memory_config_value);

    // Check if padding or dtype conversion changed buffer address
    if (optional_output_tensors.has_value()) {
        if (std::get<0>(optional_output_tensors.value()).buffer() != output_tensors.at(0)->buffer()) {
            std::get<0>(optional_output_tensors.value()) = post_transform_output_tensors.at(0);
        }
        if (std::get<1>(optional_output_tensors.value()).buffer() != output_tensors.at(1)->buffer()) {
            std::get<1>(optional_output_tensors.value()) = post_transform_output_tensors.at(1);
        }
    }

    return post_transform_output_tensors;
}

}  // namespace ttnn::operations::experimental::reduction::sort
