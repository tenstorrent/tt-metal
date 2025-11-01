// SPDX-FileCopyrightText: © 2025 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include "flip.hpp"

#include <tt-metalium/constants.hpp>

#include "ttnn/operations/core/core.hpp"
#include "ttnn/operations/copy/typecast/typecast.hpp"
#include "ttnn/operations/data_movement/flip/device/flip_device_operation.hpp"
#include "ttnn/operations/data_movement/tilize/tilize.hpp"
#include "ttnn/operations/data_movement/untilize/untilize.hpp"
#include "ttnn/operations/data_movement/unsqueeze/unsqueeze.hpp"
#include "ttnn/operations/data_movement/common/common.hpp"
#include "ttnn/operations/experimental/auto_format/auto_format.hpp"

#include "ttnn/operations/data_movement/tilize_with_val_padding/tilize_with_val_padding.hpp"
#include "ttnn/operations/data_movement/slice/slice.hpp"
#include "ttnn/operations/data_movement/slice/device/slice_op.hpp"

#include "ttnn/run_operation.hpp"
#include "ttnn/tensor/tensor_utils.hpp"

// toggle this to enable debug prints
constexpr bool debug_flip = false;
inline void flip_db_print(bool condition, const std::string& msg) {
    if constexpr (debug_flip) {
        if (condition) {
            std::cout << "[DEBUG] flip: " << msg << std::endl;
        }
    }
}

namespace ttnn::operations::data_movement {

namespace detail {

bool is_flip_nop(const ttnn::Tensor& input_tensor, const ttnn::SmallVector<uint32_t>& dims) {
    const auto& shape = input_tensor.logical_shape();
    for (auto dim : dims) {
        if (shape[dim] > 1) {
            return false;
        }
    }
    return true;  // All flip dimensions have size 1, so it's a no-op
}

ttnn::Tensor flip_impl(
    const ttnn::Tensor& input_tensor, const ttnn::SmallVector<uint32_t>& dims, const MemoryConfig& memory_config) {
    // For tensors with rank < 4, pad to 4D for device operation compatibility
    // const auto rank = input_tensor.get_logical_shape().rank();
    auto output = ttnn::prim::flip(input_tensor, dims, memory_config, std::nullopt);
    return output;
}

} // namespace detail

using OwnedFlipArgs = std::tuple<ttnn::Tensor, ttnn::SmallVector<uint32_t>>;
using MassagedFlip = MassagedOperation<ttnn::Tensor, const ttnn::Tensor&, const ttnn::SmallVector<uint32_t>&>;
using MassagedFlipParams =
    MassagedOperationParams<ttnn::Tensor, const ttnn::Tensor&, const ttnn::SmallVector<uint32_t>&>;

MassagedFlip build_untilize_rm_retilize_flip(
    const MemoryConfig& output_memory_config, ttnn::Shape& logical_output_shape) {
    return MassagedFlip(MassagedFlipParams{
        // predicate: decide whether we need untilize->rm-slice->retilize
        .predicate = [](const ttnn::Tensor& tensor, const ttnn::SmallVector<uint32_t>& /*dims*/) -> bool {
            // If padded_shape != logical_shape then input is tiled/padded and needs handling
            bool res = tensor.layout() == ttnn::TILE_LAYOUT && tensor.logical_shape() != tensor.padded_shape();
            flip_db_print(res, "untilize_rm_retilize required");
            return res;
        },
        // pre_transform: untilize -> padding-oblivious slice -> reshape to logical shape
        .pre_transform = [output_memory_config](
                             const ttnn::Tensor& input_tensor,
                             const ttnn::SmallVector<uint32_t>& dims) -> OwnedFlipArgs {
            TT_FATAL(
                input_tensor.layout() == ttnn::TILE_LAYOUT, "ttnn.flip: expected input tensor to be in tile layout");
            auto untilized_tensor = ttnn::untilize(input_tensor);

            // untilized, so now we have a padded rm tensor. we slice to remove the padding.
            const auto& input_shape = input_tensor.logical_shape();
            std::vector<uint32_t> begins_vec(input_shape.rank(), 0);
            tt::stl::Span<const uint32_t> begins = begins_vec;
            tt::stl::Span<const uint32_t> ends = input_shape.view();
            std::vector<uint32_t> steps_vec(input_shape.rank(), 1);
            tt::stl::Span<const uint32_t> steps = steps_vec;

            // perform padding-oblivious slice to remove tile padding
            // FIXME: replace with proper slice call once padding-oblivious entry point is uplifted
            untilized_tensor = tt::tt_metal::operation::run(
                SliceDeviceOperation{ttnn::Shape(begins), ttnn::Shape(ends), ttnn::Shape(steps), output_memory_config},
                {untilized_tensor},
                {},
                {std::nullopt})[0];

            untilized_tensor = ttnn::reshape(untilized_tensor, input_tensor.logical_shape());
            return std::make_tuple(untilized_tensor, dims);
        },

        // post_transform: pad back to tile-size & re-tilize, then reshape to logical output
        .post_transform = [&logical_output_shape](const ttnn::Tensor& output) -> ttnn::Tensor {
            // now we have a rm tensor, so we need ensure it's padded to tile size and re-tilize it
            if (output.layout() != ttnn::TILE_LAYOUT) {
                auto padded = pad_to_tile_vol(output, 0.0f, true, output.memory_config());
                flip_db_print(true, "[DEBUG] padded to tile layout, now tilizing.");
                auto tilized =
                    ttnn::tilize_with_val_padding(padded, padded.padded_shape(), 0.0f, output.memory_config());
                flip_db_print(true, "[DEBUG] tilized");
                // need to reshape tilized result to logical flip output shape
                return ttnn::reshape(tilized, logical_output_shape, tilized.padded_shape());
            }
            flip_db_print(true, "[DEBUG] already tilized");
            return output;
        },

        // operation: perform flip on the RM (unpadded) tensor
        .operation = [&output_memory_config](
                         const ttnn::Tensor& tensor, const ttnn::SmallVector<uint32_t>& dims) -> ttnn::Tensor {
            auto res = detail::flip_impl(tensor, dims, output_memory_config);
            return res;
        }});
}

ttnn::Tensor ExecuteFlip::invoke(
    const ttnn::Tensor& input_tensor,
    const SmallVector<int64_t>& dims,
    const std::optional<MemoryConfig>& memory_config) {
    const auto input_rank = input_tensor.logical_shape().rank();

    TT_FATAL(input_rank <= 5, "Flip operation supports tensors with rank up to 5, got rank {}", input_rank);
    TT_FATAL(!dims.empty(), "Flip dimensions cannot be empty");
    TT_FATAL(is_device_tensor(input_tensor), "Input tensor must be on device");
    TT_FATAL(
        !(input_tensor.layout() == ttnn::Layout::TILE && input_tensor.dtype() == ttnn::DataType::INT32),
        "Flip does not support INT32 tensors with tiled layout");

    // Normalize dimensions to positive indices
    SmallVector<uint32_t> normalized_dims(dims.size());
    std::transform(dims.begin(), dims.end(), normalized_dims.begin(), [input_tensor](std::int64_t idx) {
        return input_tensor.logical_shape().get_normalized_index(idx);
    });

    auto mem_conf = memory_config.value_or(input_tensor.memory_config());

    // Check for no-op case
    bool is_flip_nop = detail::is_flip_nop(input_tensor, normalized_dims);

    if (is_flip_nop) {
        return ttnn::to_memory_config(input_tensor, memory_config.value_or(input_tensor.memory_config()));
    }

    ttnn::Shape logical_output_shape = input_tensor.logical_shape();

    auto untilize_rm_retilize_flip = build_untilize_rm_retilize_flip(mem_conf, logical_output_shape);
    auto res = untilize_rm_retilize_flip(input_tensor, normalized_dims);
    return res;
}

ttnn::Tensor ExecuteFlip::invoke(const ttnn::Tensor& input_tensor, const SmallVector<int64_t>& dims) {
    return invoke(input_tensor, dims, std::nullopt);
}

} // namespace ttnn::operations::data_movement
