// SPDX-FileCopyrightText: © 2023 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include "device/slice_op.hpp"
#include "ttnn/run_operation.hpp"
#include "ttnn/common/constants.hpp"
#include "ttnn/common/queue_id.hpp"
#include "ttnn/operations/creation.hpp"
#include "ttnn/operations/core/core.hpp"
#include "ttnn/operations/data_movement/common/common.hpp"
#include "ttnn/operations/data_movement/fill_pad/fill_pad.hpp"
#include "ttnn/operations/experimental/reshape/view.hpp"
#include "slice.hpp"

namespace ttnn::operations::data_movement {

template <typename T>
ttnn::Tensor SliceOperation::invoke(
    QueueId queue_id,
    const ttnn::Tensor& input_tensor,
    tt::stl::Span<const T> begins,
    tt::stl::Span<const T> ends,
    tt::stl::Span<const T> step,
    const std::optional<MemoryConfig>& memory_config_arg,
    const std::optional<Tensor>& optional_output_tensor,
    const std::optional<float>& pad_value) {
    // Ensure start and end vectors have matching sizes and correct tensor rank

    const auto& input_shape = input_tensor.logical_shape();
    uint32_t input_rank = input_shape.rank();
    auto input_layout = input_tensor.layout();

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

    const auto& tile_shape = input_tensor.tensor_spec().tile().get_tile_shape();

    auto memory_config = optional_output_tensor.has_value() ? optional_output_tensor.value().memory_config()
                                                            : memory_config_arg.value_or(input_tensor.memory_config());

    auto ret_adjustment([&](const ttnn::Tensor& input_tensor) {
        if (input_tensor.storage_type() == StorageType::DEVICE) {
            auto tensor = ttnn::to_memory_config(input_tensor, memory_config, std::nullopt);
            tensor = ttnn::to_layout(tensor, input_layout);
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

    auto output_dim_i = [&modified_begins, &modified_step](size_t i, const ttnn::SmallVector<uint32_t>& modified_ends) {
        return (modified_ends[i] - modified_begins[i] + modified_step[i] - 1) / modified_step[i];
    };

    auto check_handled_tile_alignment = [&modified_begins, &input_rank, &tile_shape]() -> bool {
        return (
            modified_begins[input_rank - 1] % tile_shape[1] == 0 &&
            modified_begins[input_rank - 2] % tile_shape[0] == 0);
    };

    bool rm_only = false;
    bool one_dimensional = input_rank == 1;
    bool handled_tile_alignment = one_dimensional ? true : check_handled_tile_alignment();

    Tensor input = input_tensor;
    rm_only =
        (input_tensor.layout() == Layout::TILE &&
         (!no_step || one_dimensional || input_tensor.is_sharded() || !handled_tile_alignment));
    if (rm_only) {
        if (!no_step) {
            TT_FATAL(input.dtype() != DataType::BFLOAT8_B, "Strided slice is not supported for BFLOAT8 tensors");
        }
        TT_FATAL(
            input.dtype() != DataType::UINT16,
            "This slice requires an implicit Tile->RM conversion and that is not currently supported for uint16");
        input = ttnn::to_layout(input, Layout::ROW_MAJOR, std::nullopt, memory_config);
    }

    ttnn::SmallVector<uint32_t> padded_ends = modified_ends;
    if (input.layout() == Layout::TILE) {
        padded_ends[input_rank - 2] = std::max(tt::round_up(padded_ends[input_rank - 2], tile_shape[0]), tile_shape[0]);
        padded_ends[input_rank - 1] = std::max(tt::round_up(padded_ends[input_rank - 1], tile_shape[1]), tile_shape[1]);
    }

    ttnn::SmallVector<uint32_t> actual_shape_vec, final_padded_shape_vec;
    actual_shape_vec.reserve(input_rank);
    final_padded_shape_vec.reserve(input_rank);
    bool empty = false;

    // Compute actual and padded shapes for the original input rank
    for (size_t i = 0; i < input_rank; ++i) {
        TT_FATAL(
            modified_ends[i] >= modified_begins[i],
            "End {} must be greater than or equal to start {}",
            modified_ends[i],
            modified_begins[i]);
        auto val = output_dim_i(i, modified_ends);
        if (val == 0) {
            empty = true;
        }
        actual_shape_vec.push_back(val);
        final_padded_shape_vec.push_back(std::max(output_dim_i(i, padded_ends), static_cast<uint32_t>(1)));
    }
    ttnn::Shape actual_shape(actual_shape_vec);
    ttnn::Shape final_padded_shape(final_padded_shape_vec);

    if (empty) {
        TT_FATAL(
            input.storage_type() == StorageType::DEVICE, "Host tensor slice cannot return a scalar or empty tensor");
        return ttnn::empty(
            actual_shape,
            input_tensor.dtype(),
            input_tensor.layout(),
            input_tensor.mesh_device(),
            memory_config_arg.value_or(input_tensor.memory_config()));
    }

    auto res =
        tt::tt_metal::operation::run(
            SliceDeviceOperation{
                ttnn::Shape(modified_begins), ttnn::Shape(padded_ends), ttnn::Shape(modified_step), memory_config},
            {input},
            {},
            {optional_output_tensor},
            queue_id)
            .at(0);
    res = ttnn::experimental::view(res, actual_shape, final_padded_shape);

    auto dim_needs_fill = [&input_shape, &actual_shape, &final_padded_shape](int i) {
        return ((actual_shape[i] != final_padded_shape[i]) && (input_shape[i] != actual_shape[i]));
    };

    if (pad_value.has_value() && (dim_needs_fill(-1) || dim_needs_fill(-2))) {
        res = ttnn::fill_implicit_tile_padding(res, pad_value.value());
    }

    return ret_adjustment(res);
}

template <typename T>
ttnn::Tensor SliceOperation::invoke(
    const ttnn::Tensor& input_tensor,
    tt::stl::Span<const T> begins,
    tt::stl::Span<const T> ends,
    tt::stl::Span<const T> step,
    const std::optional<MemoryConfig>& memory_config_arg,
    const std::optional<Tensor>& optional_output_tensor,
    const std::optional<float>& pad_value) {
    return SliceOperation::invoke<T>(
        ttnn::DefaultQueueId, input_tensor, begins, ends, step, memory_config_arg, optional_output_tensor, pad_value);
}

template <typename T, std::size_t N>
ttnn::Tensor SliceOperation::invoke(
    QueueId queue_id,
    const ttnn::Tensor& input_tensor,
    const std::array<T, N>& output_tensor_start,
    const std::array<T, N>& output_tensor_end,
    const std::array<T, N>& step,
    const std::optional<MemoryConfig>& memory_config_arg,
    const std::optional<Tensor>& optional_output_tensor,
    const std::optional<float>& pad_value) {
    tt::stl::Span<const T> start(output_tensor_start.begin(), output_tensor_start.end());
    tt::stl::Span<const T> end(output_tensor_end.begin(), output_tensor_end.end());
    tt::stl::Span<const T> step_vec(step.begin(), step.end());
    return SliceOperation::invoke<T>(
        queue_id, input_tensor, start, end, step_vec, memory_config_arg, optional_output_tensor, pad_value);
}

template <typename T, std::size_t N>
ttnn::Tensor SliceOperation::invoke(
    const ttnn::Tensor& input_tensor,
    const std::array<T, N>& output_tensor_start,
    const std::array<T, N>& output_tensor_end,
    const std::array<T, N>& step,
    const std::optional<MemoryConfig>& memory_config_arg,
    const std::optional<Tensor>& optional_output_tensor,
    const std::optional<float>& pad_value) {
    return SliceOperation::invoke<T, N>(
        ttnn::DefaultQueueId,
        input_tensor,
        output_tensor_start,
        output_tensor_end,
        step,
        memory_config_arg,
        optional_output_tensor,
        pad_value);
}

template <typename T>
ttnn::Tensor SliceOperation::invoke(
    QueueId queue_id,
    const ttnn::Tensor& input_tensor,
    const ttnn::Tensor& output_tensor_start,
    const ttnn::Tensor& output_tensor_end,
    const std::optional<ttnn::SmallVector<T>>& step,
    const std::optional<MemoryConfig>& memory_config_arg,
    const std::optional<Tensor>& optional_output_tensor,
    const std::optional<float>& pad_value) {
    TT_FATAL(
        output_tensor_start.logical_shape().rank() == 1,
        "The start tensor for slicing must be in 1D shape, but got {}D",
        output_tensor_start.logical_shape().rank());
    TT_FATAL(
        output_tensor_end.logical_shape().rank() == 1,
        "The end tensor for slicing must be in 1D shape, but got {}D",
        output_tensor_end.logical_shape().rank());

    // convert the Tensor to Vector
    std::vector<T> output_tensor_start_vector = output_tensor_start.to_vector<T>();
    std::vector<T> output_tensor_end_vector = output_tensor_end.to_vector<T>();

    // convert the Vector to Span
    tt::stl::Span<const T> output_tensor_start_span(
        output_tensor_start_vector.data(), output_tensor_start_vector.size());
    tt::stl::Span<const T> output_tensor_end_span(output_tensor_end_vector.data(), output_tensor_end_vector.size());

    // generate the step value if it is not provided
    ttnn::SmallVector<T> step_value = step.value_or(ttnn::SmallVector<T>(output_tensor_start_span.size(), 1));

    return SliceOperation::invoke<T>(
        queue_id,
        input_tensor,
        output_tensor_start_span,
        output_tensor_end_span,
        tt::stl::Span<const T>(step_value),
        memory_config_arg,
        optional_output_tensor,
        pad_value);
}

template <typename T>
ttnn::Tensor SliceOperation::invoke(
    const ttnn::Tensor& input_tensor,
    const ttnn::Tensor& output_tensor_start,
    const ttnn::Tensor& output_tensor_end,
    const std::optional<ttnn::SmallVector<T>>& step,
    const std::optional<MemoryConfig>& memory_config_arg,
    const std::optional<Tensor>& optional_output_tensor,
    const std::optional<float>& pad_value) {
    return SliceOperation::invoke<T>(
        ttnn::DefaultQueueId,
        input_tensor,
        output_tensor_start,
        output_tensor_end,
        step,
        memory_config_arg,
        optional_output_tensor,
        pad_value);
}

template ttnn::Tensor SliceOperation::invoke<int>(
    QueueId queue_id,
    const ttnn::Tensor& input_tensor,
    tt::stl::Span<const int> begins,
    tt::stl::Span<const int> ends,
    tt::stl::Span<const int> step,
    const std::optional<MemoryConfig>& memory_config_arg,
    const std::optional<Tensor>& optional_output_tensor,
    const std::optional<float>& pad_value);

template ttnn::Tensor SliceOperation::invoke<int>(
    const ttnn::Tensor& input_tensor,
    tt::stl::Span<const int> begins,
    tt::stl::Span<const int> ends,
    tt::stl::Span<const int> step,
    const std::optional<MemoryConfig>& memory_config_arg,
    const std::optional<Tensor>& optional_output_tensor,
    const std::optional<float>& pad_value);

template ttnn::Tensor SliceOperation::invoke<uint32_t>(
    QueueId queue_id,
    const ttnn::Tensor& input_tensor,
    tt::stl::Span<const uint32_t> begins,
    tt::stl::Span<const uint32_t> ends,
    tt::stl::Span<const uint32_t> step,
    const std::optional<MemoryConfig>& memory_config_arg,
    const std::optional<Tensor>& optional_output_tensor,
    const std::optional<float>& pad_value);

template ttnn::Tensor SliceOperation::invoke<uint32_t>(
    const ttnn::Tensor& input_tensor,
    tt::stl::Span<const uint32_t> begins,
    tt::stl::Span<const uint32_t> ends,
    tt::stl::Span<const uint32_t> step,
    const std::optional<MemoryConfig>& memory_config_arg,
    const std::optional<Tensor>& optional_output_tensor,
    const std::optional<float>& pad_value);

template ttnn::Tensor SliceOperation::invoke<uint32_t, 4>(
    QueueId queue_id,
    const ttnn::Tensor& input_tensor,
    const std::array<uint32_t, 4>& output_tensor_start,
    const std::array<uint32_t, 4>& output_tensor_end,
    const std::array<uint32_t, 4>& step,
    const std::optional<MemoryConfig>& memory_config_arg,
    const std::optional<Tensor>& optional_output_tensor,
    const std::optional<float>& pad_value);

template ttnn::Tensor SliceOperation::invoke<uint32_t, 4>(
    const ttnn::Tensor& input_tensor,
    const std::array<uint32_t, 4>& output_tensor_start,
    const std::array<uint32_t, 4>& output_tensor_end,
    const std::array<uint32_t, 4>& step,
    const std::optional<MemoryConfig>& memory_config_arg,
    const std::optional<Tensor>& optional_output_tensor,
    const std::optional<float>& pad_value);

template ttnn::Tensor SliceOperation::invoke<uint32_t, 3>(
    QueueId queue_id,
    const ttnn::Tensor& input_tensor,
    const std::array<uint32_t, 3>& output_tensor_start,
    const std::array<uint32_t, 3>& output_tensor_end,
    const std::array<uint32_t, 3>& step,
    const std::optional<MemoryConfig>& memory_config_arg,
    const std::optional<Tensor>& optional_output_tensor,
    const std::optional<float>& pad_value);

template ttnn::Tensor SliceOperation::invoke<uint32_t, 3>(
    const ttnn::Tensor& input_tensor,
    const std::array<uint32_t, 3>& output_tensor_start,
    const std::array<uint32_t, 3>& output_tensor_end,
    const std::array<uint32_t, 3>& step,
    const std::optional<MemoryConfig>& memory_config_arg,
    const std::optional<Tensor>& optional_output_tensor,
    const std::optional<float>& pad_value);

template ttnn::Tensor SliceOperation::invoke<uint32_t, 1>(
    QueueId queue_id,
    const ttnn::Tensor& input_tensor,
    const std::array<uint32_t, 1>& output_tensor_start,
    const std::array<uint32_t, 1>& output_tensor_end,
    const std::array<uint32_t, 1>& step,
    const std::optional<MemoryConfig>& memory_config_arg,
    const std::optional<Tensor>& optional_output_tensor,
    const std::optional<float>& pad_value);

template ttnn::Tensor SliceOperation::invoke<uint32_t, 1>(
    const ttnn::Tensor& input_tensor,
    const std::array<uint32_t, 1>& output_tensor_start,
    const std::array<uint32_t, 1>& output_tensor_end,
    const std::array<uint32_t, 1>& step,
    const std::optional<MemoryConfig>& memory_config_arg,
    const std::optional<Tensor>& optional_output_tensor,
    const std::optional<float>& pad_value);

template ttnn::Tensor SliceOperation::invoke<uint32_t>(
    const ttnn::Tensor& input_tensor,
    const ttnn::Tensor& output_tensor_start,
    const ttnn::Tensor& output_tensor_end,
    const std::optional<ttnn::SmallVector<uint32_t>>& step,
    const std::optional<MemoryConfig>& memory_config_arg,
    const std::optional<Tensor>& optional_output_tensor,
    const std::optional<float>& pad_value);

template ttnn::Tensor SliceOperation::invoke<uint32_t>(
    QueueId queue_id,
    const ttnn::Tensor& input_tensor,
    const ttnn::Tensor& output_tensor_start,
    const ttnn::Tensor& output_tensor_end,
    const std::optional<ttnn::SmallVector<uint32_t>>& step,
    const std::optional<MemoryConfig>& memory_config_arg,
    const std::optional<Tensor>& optional_output_tensor,
    const std::optional<float>& pad_value);

}  // namespace ttnn::operations::data_movement
