// SPDX-FileCopyrightText: © 2023 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include "ttnn/common/constants.hpp"
#include "slice.hpp"
#include "device/slice_op.hpp"
#include "ttnn/run_operation.hpp"
#include "ttnn/operations/core/core.hpp"
#include "ttnn/cpp/ttnn/operations/creation.hpp"
#include "ttnn/common/constants.hpp"
#include "ttnn/cpp/ttnn/operations/data_movement/copy/copy.hpp"
#include "ttnn/cpp/ttnn/operations/data_movement/unsqueeze/unsqueeze.hpp"
#include "ttnn/cpp/ttnn/operations/data_movement/common/common.hpp"

namespace ttnn::operations::data_movement {

namespace {
template<typename T>
ttnn::Tensor slice_operation_invoke_impl(
    uint8_t queue_id,
    const ttnn::Tensor& input_tensor,
    std::span<const T> begins,
    std::span<const T> ends,
    std::span<const T> step,
    const std::optional<MemoryConfig>& memory_config_arg,
    const std::optional<Tensor>& optional_output_tensor) {

    // Ensure start and end vectors have matching sizes and correct tensor rank

    const auto &input_shape = input_tensor.get_logical_shape();
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
            auto memory_config = optional_output_tensor.has_value() ? optional_output_tensor.value().memory_config() : memory_config_arg.value_or(input_tensor.memory_config());
            return ttnn::to_memory_config(input_tensor, memory_config, std::nullopt);
        }
        return input_tensor;
    }

    TT_FATAL(input_rank == begins.size(), "Input rank {} and begins {} must have the same size", input_rank, begins.size());
    TT_FATAL(begins.size() == ends.size(), "Start {} and end {} must have the same size", begins.size(), ends.size());
    TT_FATAL(step.size() == begins.size(), "Step {} must have the same size as start {} and end", step.size(), begins.size());

    bool rm_only = !no_step && input_tensor.get_layout() == Layout::TILE;
    Tensor input = input_tensor;
    if (rm_only) {
        TT_FATAL(input.get_dtype() == DataType::BFLOAT16, "Strided slice is not supported for BFLOAT8 tensors");
        input = ttnn::to_layout(input, Layout::ROW_MAJOR, std::nullopt, std::nullopt, (Device *)nullptr);
    }

    // Unsqueeze tensor to 4D if necessary
    if (input_rank < 4) {
        input = ttnn::unsqueeze_to_4D(input);
    }

    auto padded_shape = input.get_padded_shape();
    size_t adjusted_rank = padded_shape.rank(); // Now adjusted to 4 after unsqueeze

    // Create modified vectors with wrapped indices and adjust them to match the tensor's rank
    std::vector<uint32_t> modified_begins(adjusted_rank, 0);
    std::vector<uint32_t> modified_ends = padded_shape.as_vector();
    std::vector<uint32_t> modified_step(adjusted_rank, 1);

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

    auto output_dim_i = [&modified_begins, &modified_step](size_t i, const std::vector<uint32_t> &modified_ends) {
        return (modified_ends[i] - modified_begins[i] + modified_step[i] - 1) / modified_step[i];
    };

    std::vector<uint32_t> padded_ends = modified_ends;
    if (input.layout() == Layout::TILE) {
        padded_ends[adjusted_rank - 2] = std::max(tt::round_up(padded_ends[adjusted_rank - 2], tt::constants::TILE_HEIGHT), tt::constants::TILE_HEIGHT);
        padded_ends[adjusted_rank - 1] = std::max(tt::round_up(padded_ends[adjusted_rank - 1], tt::constants::TILE_WIDTH), tt::constants::TILE_WIDTH);
    }
    SmallVector<uint32_t> actual_shape, padded_shape;
    actual_shape.reserve(input_rank);
    final_padded_shape.reserve(input_rank);
    bool empty = false;

    // Compute actual and padded shapes for the original input rank
    for (size_t i = 0; i < input_rank; ++i) {
        size_t idx = i + rank_diff;
        TT_FATAL(modified_ends[idx] >= modified_begins[idx], "End {} must be greater than or equal to start {}", modified_ends[idx], modified_begins[idx]);
        auto val = output_dim_i(idx, modified_ends);
        if (val == 0) {
            empty = true;
        }
        actual_shape.push_back(val);
        final_padded_shape.push_back(std::max(output_dim_i(idx, padded_ends), static_cast<uint32_t>(1)));
    }

    ttnn::Shape output_shape(actual_shape, final_padded_shape);

    if (empty) {
        TT_FATAL(input_tensor.storage_type() == StorageType::DEVICE, "Host tensor slice cannot return a scalar or empty tensor");
        return ttnn::empty(output_shape, input_tensor.dtype(), input_tensor.layout(),
            input_tensor.device(), memory_config_arg.value_or(input_tensor.memory_config()));
    }

    // Early exit if slice is a no-op
    if (ttnn::SimpleShape(final_padded_shape) == input.get_padded_shape() && no_step) {
        if (input_tensor.storage_type() == StorageType::DEVICE) {
            auto memory_config = optional_output_tensor.has_value() ? optional_output_tensor.value().memory_config() : memory_config_arg.value_or(input_tensor.memory_config());
            auto res = ttnn::to_memory_config(input_tensor, memory_config, std::nullopt);
            return ttnn::reshape(res, output_shape);
        }
        return ttnn::reshape(input_tensor, output_shape);
    }

    if (input_tensor.storage_type() != StorageType::DEVICE) {
        TT_FATAL(no_step, "Host tensor slice does not support strides");
        if (input_tensor.get_padded_shape() == actual_shape) {
            return input_tensor;
        } else {
            input = ttnn::to_layout(input, Layout::ROW_MAJOR, std::nullopt, std::nullopt, (Device *)nullptr);
            input = input.unpad(ttnn::SimpleShape(modified_begins), ttnn::SimpleShape(modified_ends));
            input = ttnn::to_layout(input, input_tensor.get_layout(), std::nullopt, std::nullopt, (Device *)nullptr);
            return ttnn::reshape(input, output_shape);
        }
    } else {
        const auto& input_tensor_shape = input.get_padded_shape();
        auto memory_config = optional_output_tensor.has_value() ? optional_output_tensor.value().memory_config() : memory_config_arg.value_or(input_tensor.memory_config());

        if (input.is_sharded() && input.memory_config() == memory_config && input_tensor_shape.rank() > 1) {
            TT_FATAL(no_step, "Sharded tensor slice implementation does not support striding");
            uint32_t i;
            bool in_place_unpad = true;
            for (i = 0; i < input_tensor_shape.rank() - 2; ++i) {
                in_place_unpad &= modified_begins[i] == 0 && modified_ends[i] == 1 && input_tensor_shape[i] == 1;
            }
            in_place_unpad &= modified_begins[i] == 0 &&
                              tt::div_up(modified_ends[i], input.shard_spec().value().shape[0]) ==
                                  tt::div_up(input_tensor_shape[i], input.shard_spec().value().shape[0]);
            i++;
            in_place_unpad &= modified_begins[i] == 0 && modified_ends[i] == input_tensor_shape[i];
            if (in_place_unpad) {
                return ttnn::reshape(input_tensor, output_shape);
            }
        }

        auto res = operation::run(
            SliceDeviceOperation{
                tt::tt_metal::LegacyShape(modified_begins),
                tt::tt_metal::LegacyShape(padded_ends),
                tt::tt_metal::LegacyShape(modified_step),
                memory_config},
            {input}, {}, {optional_output_tensor}, queue_id)
            .at(0);
        res = ttnn::reshape(res, output_shape);
        return rm_only ? ttnn::to_layout(res, input_tensor.get_layout(), std::nullopt, std::nullopt, (Device *)nullptr) : res;
    }
}
}

ttnn::Tensor SliceOperation::invoke(
        uint8_t queue_id,
        const ttnn::Tensor& input_tensor,
        std::span<const uint32_t> begins,
        std::span<const uint32_t> ends,
        std::span<const uint32_t> step,
        const std::optional<MemoryConfig>& memory_config_arg,
        const std::optional<Tensor>& optional_output_tensor) {
    return slice_operation_invoke_impl(queue_id, input_tensor, begins, ends, step, memory_config_arg, optional_output_tensor);
}

ttnn::Tensor SliceOperation::invoke(
        const ttnn::Tensor& input_tensor,
        std::span<const uint32_t> begins,
        std::span<const uint32_t> ends,
        std::span<const uint32_t> step,
        const std::optional<MemoryConfig>& memory_config_arg,
        const std::optional<Tensor>& optional_output_tensor) {
    return slice_operation_invoke_impl(ttnn::DefaultQueueId, input_tensor, begins, ends, step, memory_config_arg, optional_output_tensor);
}

ttnn::Tensor SliceOperation::invoke(
        uint8_t queue_id,
        const ttnn::Tensor& input_tensor,
        std::span<const int> begins,
        std::span<const int> ends,
        std::span<const int> step,
        const std::optional<MemoryConfig>& memory_config_arg,
        const std::optional<Tensor>& optional_output_tensor) {
    return slice_operation_invoke_impl(queue_id, input_tensor, begins, ends, step, memory_config_arg, optional_output_tensor);
}

ttnn::Tensor SliceOperation::invoke(
        const ttnn::Tensor& input_tensor,
        std::span<const int> begins,
        std::span<const int> ends,
        std::span<const int> step,
        const std::optional<MemoryConfig>& memory_config_arg,
        const std::optional<Tensor>& optional_output_tensor) {
    return slice_operation_invoke_impl(ttnn::DefaultQueueId, input_tensor, begins, ends, step, memory_config_arg, optional_output_tensor);
}
slice.cpp
}  // namespace operations
