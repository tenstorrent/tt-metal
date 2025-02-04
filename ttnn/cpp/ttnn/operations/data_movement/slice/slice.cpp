// SPDX-FileCopyrightText: © 2023 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include "ttnn/common/constants.hpp"
#include "slice.hpp"
#include "device/slice_op.hpp"
#include "ttnn/run_operation.hpp"
#include "ttnn/operations/core/core.hpp"
#include "cpp/ttnn/operations/creation.hpp"
#include "ttnn/common/constants.hpp"
#include "cpp/ttnn/operations/data_movement/copy/copy.hpp"
#include "cpp/ttnn/operations/data_movement/unsqueeze/unsqueeze.hpp"
#include "cpp/ttnn/operations/data_movement/common/common.hpp"

namespace ttnn::operations::data_movement {

template <typename T>
ttnn::Tensor SliceOperation::invoke(
    uint8_t queue_id,
    const ttnn::Tensor& input_tensor,
    tt::stl::Span<const T> begins,
    tt::stl::Span<const T> ends,
    tt::stl::Span<const T> step,
    const std::optional<MemoryConfig>& memory_config_arg,
    const std::optional<Tensor>& optional_output_tensor) {
    // Ensure start and end vectors have matching sizes and correct tensor rank

    const auto& input_shape = input_tensor.get_logical_shape();
    uint32_t input_rank = input_shape.rank();

    auto input_layout = input_tensor.get_layout();
    if (input_rank == 0) {
        return input_tensor;
    }
    TT_FATAL(
        input_rank == begins.size(), "Input rank {} and begins {} must have the same size", input_rank, begins.size());
    TT_FATAL(begins.size() == ends.size(), "Start {} and end {} must have the same size", begins.size(), ends.size());
    TT_FATAL(
        step.size() == begins.size(),
        "Step {} must have the same size as start {} and end",
        step.size(),
        begins.size());

    bool no_step = std::ranges::all_of(step, [](uint32_t s) { return s == 1; });
    bool starts_zero = std::ranges::all_of(begins, [](uint32_t s) { return s == 0; });
    bool ends_max = true;
    for (size_t i = 0; i < ends.size(); ++i) {
        ends_max &= ends[i] == input_shape[i];
        if (!ends_max) {
            break;
        }
    }

    const auto& tile_shape = input_tensor.get_tensor_spec().tile().get_tile_shape();

    auto memory_config = optional_output_tensor.has_value() ? optional_output_tensor.value().memory_config()
                                                            : memory_config_arg.value_or(input_tensor.memory_config());

    auto ret_adjustment([&](const ttnn::Tensor& input_tensor) {
        if (input_tensor.storage_type() == StorageType::DEVICE) {
            auto tensor = ttnn::to_memory_config(input_tensor, memory_config, std::nullopt);
            tensor = ttnn::to_layout(tensor, input_layout, std::nullopt, std::nullopt, (IDevice*)nullptr);
            return tensor;
        }
        return input_tensor;
    });

    // No-op check
    if (no_step && starts_zero && ends_max) {
        return ret_adjustment(input_tensor);
    }

    // Create modified vectors with wrapped indices and adjust them to match the tensor's rank
    ttnn::SmallVector<uint32_t> modified_begins(input_rank, 0);
    ttnn::SmallVector<uint32_t> modified_ends(input_rank, 0);
    ttnn::SmallVector<uint32_t> modified_step(input_rank, 1);

    // Wrap indices and adjust begins, ends, and step
    for (size_t i = 0; i < begins.size(); ++i) {
        if constexpr (std::is_signed_v<T>) {
            modified_begins[i] = wrap_index(begins[i], input_shape[i]);
            modified_ends[i] = wrap_index(ends[i], input_shape[i]);
            modified_step[i] = static_cast<uint32_t>(step[i]);
        } else {
            modified_begins[i] = begins[i];
            modified_ends[i] = ends[i];
            modified_step[i] = step[i];
        }
    }

    bool aligned_begins = true;
    bool aligned_ends = true;
    bool rm_only = false;
    bool one_dimensional = input_rank == 1;

    Tensor input = input_tensor;
    if (input_tensor.get_layout() == Layout::TILE) {
        if (!one_dimensional) {
            auto slice_aligned_to_tile = [&tile_shape, &input_rank](const auto& v) -> bool {
                return (v[input_rank - 2] % tile_shape[0] == 0) && (v[input_rank - 1] % tile_shape[1] == 0);
            };

            aligned_begins &= slice_aligned_to_tile(modified_begins);
            // TODO: if only the ends are unaligned, we can use fill_pad to pad the tensor
            aligned_ends &= slice_aligned_to_tile(modified_ends) || (modified_ends[input_rank - 1] == input_shape[-1] &&
                                                                     modified_ends[input_rank - 2] == input_shape[-2]);
        }
        rm_only = !no_step || !aligned_begins || !aligned_ends || one_dimensional;
        if (rm_only) {
            TT_FATAL(input.get_dtype() == DataType::BFLOAT16, "Strided slice is not supported for BFLOAT8 tensors");
            input = ttnn::to_layout(input, Layout::ROW_MAJOR, std::nullopt, memory_config, (IDevice*)nullptr);
        }
    }

    auto output_dim_i = [&modified_begins, &modified_step](size_t i, const ttnn::SmallVector<uint32_t>& modified_ends) {
        return (modified_ends[i] - modified_begins[i] + modified_step[i] - 1) / modified_step[i];
    };

    ttnn::SmallVector<uint32_t> actual_shape_vec;
    actual_shape_vec.reserve(input_rank);
    bool empty = false;

    // Compute actual and padded shapes for the original input rank
    for (size_t i = 0; i < input_rank; ++i) {
        TT_FATAL(
            modified_begins[i] <= modified_ends[i],
            "Invalid slice operation: begin[{}] must be less than or equal to end[{}], but got {} > {}",
            i,
            i,
            modified_begins[i],
            modified_ends[i]);
        auto val = output_dim_i(i, modified_ends);
        if (val == 0) {
            empty = true;
        }
        actual_shape_vec.push_back(val);
    }
    ttnn::Shape actual_shape(actual_shape_vec);

    if (empty) {
        TT_FATAL(
            input.storage_type() == StorageType::DEVICE, "Host tensor slice cannot return a scalar or empty tensor");
        return ttnn::empty(
            actual_shape,
            input_tensor.dtype(),
            input_tensor.layout(),
            input_tensor.device(),
            memory_config_arg.value_or(input_tensor.memory_config()));
    }

    auto res = operation::run(
                   SliceDeviceOperation{
                       ttnn::Shape(modified_begins),
                       ttnn::Shape(modified_ends),
                       ttnn::Shape(modified_step),
                       memory_config},
                   {input},
                   {},
                   {optional_output_tensor},
                   queue_id)
                   .at(0);
    return ret_adjustment(res);
}

template <typename T>
ttnn::Tensor SliceOperation::invoke(
    const ttnn::Tensor& input_tensor,
    tt::stl::Span<const T> begins,
    tt::stl::Span<const T> ends,
    tt::stl::Span<const T> step,
    const std::optional<MemoryConfig>& memory_config_arg,
    const std::optional<Tensor>& optional_output_tensor) {
    return SliceOperation::invoke<T>(ttnn::DefaultQueueId, input_tensor, begins, ends, step, memory_config_arg, optional_output_tensor);
}

template <typename T, std::size_t N>
ttnn::Tensor SliceOperation::invoke(
    uint8_t queue_id,
    const ttnn::Tensor& input_tensor,
    const std::array<T, N>& output_tensor_start,
    const std::array<T, N>& output_tensor_end,
    const std::array<T, N>& step,
    const std::optional<MemoryConfig>& memory_config_arg,
    const std::optional<Tensor>& optional_output_tensor) {
    tt::stl::Span<const T> start(output_tensor_start.begin(), output_tensor_start.end());
    tt::stl::Span<const T> end(output_tensor_end.begin(), output_tensor_end.end());
    tt::stl::Span<const T> step_vec(step.begin(), step.end());
    return SliceOperation::invoke<T>(
        queue_id, input_tensor, start, end, step_vec, memory_config_arg, optional_output_tensor);
}

template <typename T, std::size_t N>
ttnn::Tensor SliceOperation::invoke(
    const ttnn::Tensor& input_tensor,
    const std::array<T, N>& output_tensor_start,
    const std::array<T, N>& output_tensor_end,
    const std::array<T, N>& step,
    const std::optional<MemoryConfig>& memory_config_arg,
    const std::optional<Tensor>& optional_output_tensor) {
    return SliceOperation::invoke<T, N>(
        ttnn::DefaultQueueId,
        input_tensor,
        output_tensor_start,
        output_tensor_end,
        step,
        memory_config_arg,
        optional_output_tensor);
}

template ttnn::Tensor SliceOperation::invoke<int>(
    uint8_t queue_id,
    const ttnn::Tensor& input_tensor,
    tt::stl::Span<const int> begins,
    tt::stl::Span<const int> ends,
    tt::stl::Span<const int> step,
    const std::optional<MemoryConfig>& memory_config_arg,
    const std::optional<Tensor>& optional_output_tensor);

template ttnn::Tensor SliceOperation::invoke<int>(
    const ttnn::Tensor& input_tensor,
    tt::stl::Span<const int> begins,
    tt::stl::Span<const int> ends,
    tt::stl::Span<const int> step,
    const std::optional<MemoryConfig>& memory_config_arg,
    const std::optional<Tensor>& optional_output_tensor);

template ttnn::Tensor SliceOperation::invoke<uint32_t>(
    uint8_t queue_id,
    const ttnn::Tensor& input_tensor,
    tt::stl::Span<const uint32_t> begins,
    tt::stl::Span<const uint32_t> ends,
    tt::stl::Span<const uint32_t> step,
    const std::optional<MemoryConfig>& memory_config_arg,
    const std::optional<Tensor>& optional_output_tensor);

template ttnn::Tensor SliceOperation::invoke<uint32_t>(
    const ttnn::Tensor& input_tensor,
    tt::stl::Span<const uint32_t> begins,
    tt::stl::Span<const uint32_t> ends,
    tt::stl::Span<const uint32_t> step,
    const std::optional<MemoryConfig>& memory_config_arg,
    const std::optional<Tensor>& optional_output_tensor);

template ttnn::Tensor SliceOperation::invoke<uint32_t, 4>(
    uint8_t queue_id,
    const ttnn::Tensor& input_tensor,
    const std::array<uint32_t, 4>& output_tensor_start,
    const std::array<uint32_t, 4>& output_tensor_end,
    const std::array<uint32_t, 4>& step,
    const std::optional<MemoryConfig>& memory_config_arg,
    const std::optional<Tensor>& optional_output_tensor);

template ttnn::Tensor SliceOperation::invoke<uint32_t, 4>(
    const ttnn::Tensor& input_tensor,
    const std::array<uint32_t, 4>& output_tensor_start,
    const std::array<uint32_t, 4>& output_tensor_end,
    const std::array<uint32_t, 4>& step,
    const std::optional<MemoryConfig>& memory_config_arg,
    const std::optional<Tensor>& optional_output_tensor);

template ttnn::Tensor SliceOperation::invoke<uint32_t, 3>(
    uint8_t queue_id,
    const ttnn::Tensor& input_tensor,
    const std::array<uint32_t, 3>& output_tensor_start,
    const std::array<uint32_t, 3>& output_tensor_end,
    const std::array<uint32_t, 3>& step,
    const std::optional<MemoryConfig>& memory_config_arg,
    const std::optional<Tensor>& optional_output_tensor);

template ttnn::Tensor SliceOperation::invoke<uint32_t, 3>(
    const ttnn::Tensor& input_tensor,
    const std::array<uint32_t, 3>& output_tensor_start,
    const std::array<uint32_t, 3>& output_tensor_end,
    const std::array<uint32_t, 3>& step,
    const std::optional<MemoryConfig>& memory_config_arg,
    const std::optional<Tensor>& optional_output_tensor);

template ttnn::Tensor SliceOperation::invoke<uint32_t, 1>(
    uint8_t queue_id,
    const ttnn::Tensor& input_tensor,
    const std::array<uint32_t, 1>& output_tensor_start,
    const std::array<uint32_t, 1>& output_tensor_end,
    const std::array<uint32_t, 1>& step,
    const std::optional<MemoryConfig>& memory_config_arg,
    const std::optional<Tensor>& optional_output_tensor);

template ttnn::Tensor SliceOperation::invoke<uint32_t, 1>(
    const ttnn::Tensor& input_tensor,
    const std::array<uint32_t, 1>& output_tensor_start,
    const std::array<uint32_t, 1>& output_tensor_end,
    const std::array<uint32_t, 1>& step,
    const std::optional<MemoryConfig>& memory_config_arg,
    const std::optional<Tensor>& optional_output_tensor);

}  // namespace ttnn::operations::data_movement
