// SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#include "device/padded_slice_op.hpp"
#include <tt-logger/tt-logger.hpp>
#include "ttnn/run_operation.hpp"
#include "ttnn/common/constants.hpp"
#include "ttnn/common/queue_id.hpp"
#include "ttnn/operations/creation.hpp"
#include "ttnn/operations/core/core.hpp"
#include "ttnn/operations/data_movement/common/common.hpp"
#include "ttnn/operations/data_movement/fill_pad/fill_pad.hpp"
#include "ttnn/operations/experimental/reshape/view.hpp"
#include "ttnn/tensor/types.hpp"
#include "padded_slice.hpp"

namespace ttnn::operations::experimental {

template <typename T>
ttnn::Tensor PaddedSliceOperation::invoke(
    QueueId queue_id,
    const ttnn::Tensor& input_tensor,
    tt::stl::Span<const T> begins,
    tt::stl::Span<const T> ends,
    tt::stl::Span<const T> step,
    const MemoryConfig& memory_config,
    const std::optional<Tensor>& optional_output_tensor,
    const std::optional<float>& pad_value) {
    // Ensure start and end vectors have matching sizes and correct tensor rank

    const auto& input_shape = input_tensor.logical_shape();
    uint32_t input_rank = input_shape.rank();
    auto input_layout = input_tensor.layout();

    TT_FATAL(input_rank == 4, "Only 4D tensors are supported for padded_slice");

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

    TT_FATAL(no_step, "Steps != 1 are not supported for padded_slice.");
    TT_FATAL(memory_config.is_sharded(), "Output Memory Config must be sharded. Use slice for non-sharded outputs.");
    TT_FATAL(!input_tensor.memory_config().is_sharded(), " padded_slice does not support sharded inputs.");

    auto ret_adjustment([&](const ttnn::Tensor& ret_input_tensor) {
        if (ret_input_tensor.storage_type() == StorageType::DEVICE) {
            auto tensor = ttnn::to_memory_config(ret_input_tensor, memory_config, std::nullopt);
            tensor = ttnn::to_layout(tensor, input_layout, std::nullopt, std::nullopt, (IDevice*)nullptr);
            return tensor;
        }
        return ret_input_tensor;
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
            modified_begins[i] = data_movement::wrap_index(begins[i], input_shape[i]);
            modified_ends[i] = data_movement::wrap_index(ends[i], input_shape[i]);
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

    ttnn::SmallVector<uint32_t> padded_ends = modified_ends;

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
            input_tensor.storage_type() == StorageType::DEVICE,
            "Host tensor slice cannot return a scalar or empty tensor");
        return ttnn::empty(
            actual_shape, input_tensor.dtype(), input_tensor.layout(), input_tensor.mesh_device(), memory_config);
    }

    auto res =
        tt::tt_metal::operation::run(
            PaddedSliceDeviceOperation{
                ttnn::Shape(modified_begins), ttnn::Shape(padded_ends), ttnn::Shape(modified_step), memory_config},
            {input_tensor},
            {},
            {optional_output_tensor},
            queue_id)
            .at(0);

    // If padded_slice should return a sharded tensor, then the op must created the sharded tensor in the requested
    // memory config
    if (res.is_sharded() && memory_config.is_sharded()) {
        TT_ASSERT(
            res.memory_config() == memory_config,
            "Memory config must match. Got {}, expecteed {}",
            res.memory_config(),
            memory_config);
        return res;
    }

    res = ttnn::experimental::view(res, actual_shape, final_padded_shape);

    return res;
}

template ttnn::Tensor PaddedSliceOperation::invoke<int>(
    QueueId queue_id,
    const ttnn::Tensor& input_tensor,
    tt::stl::Span<const int> begins,
    tt::stl::Span<const int> ends,
    tt::stl::Span<const int> step,
    const MemoryConfig& memory_config_arg,
    const std::optional<Tensor>& optional_output_tensor,
    const std::optional<float>& pad_value);

template ttnn::Tensor PaddedSliceOperation::invoke<uint32_t>(
    QueueId queue_id,
    const ttnn::Tensor& input_tensor,
    tt::stl::Span<const uint32_t> begins,
    tt::stl::Span<const uint32_t> ends,
    tt::stl::Span<const uint32_t> step,
    const MemoryConfig& memory_config_arg,
    const std::optional<Tensor>& optional_output_tensor,
    const std::optional<float>& pad_value);

}  // namespace ttnn::operations::experimental
