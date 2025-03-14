// SPDX-FileCopyrightText: © 2023 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include "slice.hpp"
#include "device/slice_op.hpp"
#include "ttnn/common/queue_id.hpp"
#include "ttnn/run_operation.hpp"
#include "ttnn/operations/core/core.hpp"
#include "cpp/ttnn/operations/creation.hpp"
#include "cpp/ttnn/operations/data_movement/copy/copy.hpp"
#include "cpp/ttnn/operations/data_movement/unsqueeze/unsqueeze.hpp"
#include "cpp/ttnn/operations/data_movement/common/common.hpp"

namespace ttnn::operations::data_movement {

template <typename T>
ttnn::Tensor SliceOperation::invoke(
    QueueId queue_id,
    const ttnn::Tensor& input_tensor,
    tt::stl::Span<const T> begins,
    tt::stl::Span<const T> ends,
    tt::stl::Span<const T> step,
    const std::optional<MemoryConfig>& memory_config_arg,
    const std::optional<Tensor>& optional_output_tensor) {
    // Ensure start and end vectors have matching sizes and correct tensor rank

    const auto& input_shape = input_tensor.get_logical_shape();
    uint32_t input_rank = input_shape.rank();

    bool no_step = std::ranges::all_of(step, [](uint32_t s) { return s == 1; });
    bool starts_zero = std::ranges::all_of(begins, [](uint32_t s) { return s == 0; });
    bool ends_max = true;
    for (size_t i = 0; i < ends.size(); ++i) {
        ends_max &= ends[i] == input_shape[i];
        if (!ends_max) {
            break;
        }
    }

    if (no_step && starts_zero && ends_max) {
        if (input_tensor.storage_type() == StorageType::DEVICE) {
            auto memory_config = optional_output_tensor.has_value()
                                     ? optional_output_tensor.value().memory_config()
                                     : memory_config_arg.value_or(input_tensor.memory_config());
            return ttnn::to_memory_config(input_tensor, memory_config, std::nullopt);
        }
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

    bool rm_only = !no_step && input_tensor.get_layout() == Layout::TILE;
    Tensor input = input_tensor;
    if (rm_only) {
        TT_FATAL(input.get_dtype() == DataType::BFLOAT16, "Strided slice is not supported for BFLOAT8 tensors");
        input = ttnn::to_layout(input, Layout::ROW_MAJOR, std::nullopt, std::nullopt, (IDevice*)nullptr);
    }

    // Unsqueeze tensor to 4D if necessary
    if (input_rank < 4) {
        input = ttnn::unsqueeze_to_4D(input);
    }

    auto padded_shape = input.get_padded_shape();
    size_t adjusted_rank = padded_shape.rank();  // Now adjusted to 4 after unsqueeze

    // Create modified vectors with wrapped indices and adjust them to match the tensor's rank
    ttnn::SmallVector<uint32_t> modified_begins(adjusted_rank, 0);
    ttnn::SmallVector<uint32_t> modified_ends(padded_shape.cbegin(), padded_shape.cend());
    ttnn::SmallVector<uint32_t> modified_step(adjusted_rank, 1);

    size_t rank_diff = adjusted_rank - input_rank;

    // Wrap indices and adjust begins, ends, and step
    for (size_t i = 0; i < begins.size(); ++i) {
        size_t idx = i + rank_diff;

        if constexpr (std::is_signed_v<T>) {
            modified_begins[idx] = wrap_index(begins[i], input_shape[i]);
            modified_ends[idx] = wrap_index(ends[i], input_shape[i]);
            modified_step[idx] = static_cast<uint32_t>(step[i]);
        } else {
            modified_begins[idx] = begins[i];
            modified_ends[idx] = ends[i];
            modified_step[idx] = step[i];
        }
    }

    auto output_dim_i = [&modified_begins, &modified_step](size_t i, const ttnn::SmallVector<uint32_t>& modified_ends) {
        return (modified_ends[i] - modified_begins[i] + modified_step[i] - 1) / modified_step[i];
    };

    ttnn::SmallVector<uint32_t> padded_ends = modified_ends;
    if (input.layout() == Layout::TILE) {
        padded_ends[adjusted_rank - 2] = std::max(
            tt::round_up(padded_ends[adjusted_rank - 2], tt::constants::TILE_HEIGHT), tt::constants::TILE_HEIGHT);
        padded_ends[adjusted_rank - 1] = std::max(
            tt::round_up(padded_ends[adjusted_rank - 1], tt::constants::TILE_WIDTH), tt::constants::TILE_WIDTH);
    }

    ttnn::SmallVector<uint32_t> actual_shape_vec, final_padded_shape_vec;
    actual_shape_vec.reserve(input_rank);
    final_padded_shape_vec.reserve(input_rank);
    bool empty = false;

    // Compute actual and padded shapes for the original input rank
    for (size_t i = 0; i < input_rank; ++i) {
        size_t idx = i + rank_diff;
        TT_FATAL(
            modified_ends[idx] >= modified_begins[idx],
            "End {} must be greater than or equal to start {}",
            modified_ends[idx],
            modified_begins[idx]);
        auto val = output_dim_i(idx, modified_ends);
        if (val == 0) {
            empty = true;
        }
        actual_shape_vec.push_back(val);
        final_padded_shape_vec.push_back(std::max(output_dim_i(idx, padded_ends), static_cast<uint32_t>(1)));
    }
    ttnn::Shape actual_shape(actual_shape_vec);
    ttnn::Shape final_padded_shape(final_padded_shape_vec);

    if (empty) {
        TT_FATAL(
            input_tensor.storage_type() == StorageType::DEVICE,
            "Host tensor slice cannot return a scalar or empty tensor");
        return ttnn::empty(
            actual_shape,
            input_tensor.dtype(),
            input_tensor.layout(),
            input_tensor.device(),
            memory_config_arg.value_or(input_tensor.memory_config()));
    }

    // Early exit if slice is a no-op
    if (final_padded_shape == input.get_padded_shape() && no_step) {
        if (input_tensor.storage_type() == StorageType::DEVICE) {
            auto memory_config = optional_output_tensor.has_value()
                                     ? optional_output_tensor.value().memory_config()
                                     : memory_config_arg.value_or(input_tensor.memory_config());
            auto res = ttnn::to_memory_config(input_tensor, memory_config, std::nullopt);
            return ttnn::reshape(res, actual_shape, final_padded_shape);
        }
        return ttnn::reshape(input_tensor, actual_shape, final_padded_shape);
    }

    if (input_tensor.storage_type() != StorageType::DEVICE) {
        TT_FATAL(no_step, "Host tensor slice does not support strides");
        if (input_tensor.get_padded_shape() == actual_shape) {
            return input_tensor;
        } else {
            input = ttnn::to_layout(input, Layout::ROW_MAJOR, std::nullopt, std::nullopt, (IDevice*)nullptr);
            input = input.unpad(ttnn::Shape(modified_begins), ttnn::Shape(modified_ends));
            input = ttnn::to_layout(input, input_tensor.get_layout(), std::nullopt, std::nullopt, (IDevice*)nullptr);
            return ttnn::reshape(input, actual_shape, final_padded_shape);
        }
    } else {
        const auto& input_tensor_shape = input.get_padded_shape();
        auto memory_config = optional_output_tensor.has_value()
                                 ? optional_output_tensor.value().memory_config()
                                 : memory_config_arg.value_or(input_tensor.memory_config());

        if (input.is_sharded() && input.memory_config() == memory_config && input_tensor_shape.rank() > 1) {
            TT_FATAL(no_step, "Sharded tensor slice implementation does not support striding");
            uint32_t i;
            bool in_place_unpad = true;
            for (i = 0; i < input_tensor_shape.rank() - 2; ++i) {
                in_place_unpad &= modified_begins[i] == 0 && modified_ends[i] == 1 && input_tensor_shape[i] == 1;
            }
            in_place_unpad &=
                modified_begins[i] == 0 && tt::div_up(modified_ends[i], input.shard_spec().value().shape[0]) ==
                                               tt::div_up(input_tensor_shape[i], input.shard_spec().value().shape[0]);
            i++;
            in_place_unpad &= modified_begins[i] == 0 && modified_ends[i] == input_tensor_shape[i];
            if (in_place_unpad) {
                return ttnn::reshape(input_tensor, actual_shape, final_padded_shape);
            }
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
        res = ttnn::reshape(res, actual_shape, final_padded_shape);
        return rm_only ? ttnn::to_layout(res, input_tensor.get_layout(), std::nullopt, std::nullopt, (IDevice*)nullptr)
                       : res;
    }
}

template <typename T>
ttnn::Tensor SliceOperation::invoke(
    const ttnn::Tensor& input_tensor,
    tt::stl::Span<const T> begins,
    tt::stl::Span<const T> ends,
    tt::stl::Span<const T> step,
    const std::optional<MemoryConfig>& memory_config_arg,
    const std::optional<Tensor>& optional_output_tensor) {
    return SliceOperation::invoke<T>(ttnn::DefaultQueueId, input_tensor, begins, ends, step, memory_config_arg);
}

// Specialization for uint32_t and N=4
template <>
ttnn::Tensor SliceOperation::invoke<uint32_t, 4>(
    QueueId queue_id,
    const ttnn::Tensor& input_tensor,
    const std::array<uint32_t, 4>& begins,
    const std::array<uint32_t, 4>& ends,
    const std::array<uint32_t, 4>& step,
    const std::optional<MemoryConfig>& memory_config_arg,
    const std::optional<Tensor>& optional_output_tensor) {
    const auto& padded_input_shape = input_tensor.get_padded_shape();
    TT_FATAL(padded_input_shape.rank() == 4, "Input tensor must have rank 4");

    bool no_step = step[0] == 1 && step[1] == 1 && step[2] == 1 && step[3] == 1;
    bool starts_zero = begins[0] == 0 && begins[1] == 0 && begins[2] == 0 && begins[3] == 0;
    bool ends_max = ends[0] == padded_input_shape[0] && ends[1] == padded_input_shape[1] &&
                    ends[2] == padded_input_shape[2] && ends[3] == padded_input_shape[3];

    if (no_step && starts_zero && ends_max) {
        if (input_tensor.storage_type() == StorageType::DEVICE) {
            auto memory_config = optional_output_tensor.has_value()
                                     ? optional_output_tensor.value().memory_config()
                                     : memory_config_arg.value_or(input_tensor.memory_config());
            return ttnn::to_memory_config(input_tensor, memory_config, std::nullopt);
        }
        return input_tensor;
    }
    bool rm_only = !no_step && input_tensor.get_layout() == Layout::TILE;
    ttnn::Tensor input = input_tensor;
    if (rm_only) {
        input = ttnn::to_layout(input_tensor, Layout::ROW_MAJOR, std::nullopt, std::nullopt, (IDevice*)nullptr);
    }

    const bool tiled = input.get_layout() == Layout::TILE;
    bool on_device = input.storage_type() == StorageType::DEVICE;

    std::array<uint32_t, 4> actual_shape_vec;
    std::array<uint32_t, 4> padded_shape_vec;
    const std::array<uint32_t, 4> padded_ends =
        tiled ? std::array<uint32_t, 4>(
                    {ends[0],
                     ends[1],
                     std::max(tt::round_up(ends[2], tt::constants::TILE_HEIGHT), tt::constants::TILE_HEIGHT),
                     std::max(tt::round_up(ends[3], tt::constants::TILE_WIDTH), tt::constants::TILE_WIDTH)})
              : ends;
    bool empty = false;
    for (int i = 0; i < 4; ++i) {
        TT_FATAL(ends[i] >= begins[i], "End {} must be greater than or equal to start {}", ends[i], begins[i]);
        uint32_t offset = step[i] - begins[i] - 1;
        uint32_t dim_size = (ends[i] + offset) / step[i];
        empty |= dim_size == 0;
        actual_shape_vec[i] = dim_size;
        padded_shape_vec[i] = std::max((padded_ends[i] + offset) / step[i], 1u);
    }
    ttnn::Shape actual_shape(actual_shape_vec);
    ttnn::Shape padded_shape(padded_shape_vec);

    if (empty) {
        TT_FATAL(on_device, "Host tensor slice cannot return a scalar or empty tensor");
        auto memory_config = optional_output_tensor.has_value() ? optional_output_tensor.value().memory_config()
                                                                : memory_config_arg.value_or(input.memory_config());
        return ttnn::empty(actual_shape, input.dtype(), input_tensor.layout(), input.device(), memory_config);
    }

    // Early exit if slice is a no-op
    if (padded_shape == padded_input_shape && no_step) {
        if (input.storage_type() == StorageType::DEVICE) {
            auto memory_config = optional_output_tensor.has_value() ? optional_output_tensor.value().memory_config()
                                                                    : memory_config_arg.value_or(input.memory_config());
            auto res = ttnn::to_memory_config(input, memory_config, std::nullopt);
            return ttnn::reshape(res, actual_shape, padded_shape);
        }
        return ttnn::reshape(input, actual_shape, padded_shape);  // change to view
    }

    if (on_device) {
        auto memory_config = optional_output_tensor.has_value() ? optional_output_tensor.value().memory_config()
                                                                : memory_config_arg.value_or(input.memory_config());

        // Check for in-place unpad optimization
        if (input.is_sharded() && input.memory_config() == memory_config && padded_input_shape.rank() > 1) {
            TT_FATAL(no_step, "Sharded tensor slice implementation does not support striding");
            bool in_place_unpad = true;
            for (int i = 0; i < 2; ++i) {
                in_place_unpad &= begins[i] == 0 && ends[i] == 1 && padded_input_shape[i] == 1;
            }
            in_place_unpad &=
                begins[2] == 0 && tt::div_up(ends[2], input.shard_spec().value().shape[0]) ==
                                      tt::div_up(padded_input_shape[2], input.shard_spec().value().shape[0]);
            in_place_unpad &= begins[3] == 0 && ends[3] == padded_input_shape[3];
            if (in_place_unpad) {
                return ttnn::reshape(input, actual_shape, padded_shape);
            }
        }

        input = tt::tt_metal::operation::run(
            SliceDeviceOperation{ttnn::Shape(begins), ttnn::Shape(padded_ends), ttnn::Shape(step), memory_config},
            {input},
            {},
            {optional_output_tensor},
            queue_id)[0];
        input = ttnn::reshape(input, actual_shape, padded_shape);
        return rm_only ? ttnn::to_layout(input, input.get_layout(), std::nullopt, std::nullopt, (IDevice*)nullptr)
                       : input;
    }

    TT_FATAL(no_step, "Host tensor slice does not support strides");

    if (input.get_padded_shape() == actual_shape) {
        return input;
    } else {
        auto input_4d_rm = ttnn::to_layout(input, Layout::ROW_MAJOR, std::nullopt, std::nullopt, (IDevice*)nullptr);
        auto output_4d = input_4d_rm.unpad(ttnn::Shape(begins), ttnn::Shape(ends));
        auto output_4d_rm =
            ttnn::to_layout(output_4d, input.get_layout(), std::nullopt, std::nullopt, (IDevice*)nullptr);
        return ttnn::reshape(output_4d_rm, actual_shape, padded_shape);
    }
}

template <typename T, std::size_t N>
ttnn::Tensor SliceOperation::invoke(
    QueueId queue_id,
    const ttnn::Tensor& input_tensor,
    const std::array<T, N>& output_tensor_start,
    const std::array<T, N>& output_tensor_end,
    const std::array<T, N>& step,
    const std::optional<MemoryConfig>& memory_config_arg,
    const std::optional<Tensor>& optional_output_tensor) {
    tt::stl::Span<const T> start(output_tensor_start.begin(), output_tensor_start.end());
    tt::stl::Span<const T> end(output_tensor_end.begin(), output_tensor_end.end());
    tt::stl::Span<const T> step_vec(step.begin(), step.end());
    return SliceOperation::invoke<T>(queue_id, input_tensor, start, end, step_vec, memory_config_arg);
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
        ttnn::DefaultQueueId, input_tensor, output_tensor_start, output_tensor_end, step, memory_config_arg);
}

template ttnn::Tensor SliceOperation::invoke<int>(
    QueueId queue_id,
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
    QueueId queue_id,
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
    const ttnn::Tensor& input_tensor,
    const std::array<uint32_t, 4>& output_tensor_start,
    const std::array<uint32_t, 4>& output_tensor_end,
    const std::array<uint32_t, 4>& step,
    const std::optional<MemoryConfig>& memory_config_arg,
    const std::optional<Tensor>& optional_output_tensor);

template ttnn::Tensor SliceOperation::invoke<uint32_t, 1>(
    QueueId queue_id,
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
