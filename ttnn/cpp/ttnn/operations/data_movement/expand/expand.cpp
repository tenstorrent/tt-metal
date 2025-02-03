// SPDX-FileCopyrightText: © 2024 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include "expand.hpp"

#include <optional>

#include "ttnn/operations/core/core.hpp"
#include "ttnn/operations/data_movement/expand/device/expand_device_operation.hpp"
#include "ttnn/tensor/tensor_impl.hpp"
#include "ttnn/tensor/tensor_impl_wrapper.hpp"
#include "ttnn/tensor/tensor_ops.hpp"
namespace ttnn::operations::expand {

auto infer_size(const Tensor& input, const std::vector<int32_t>& sizes) {
    const auto& input_shape = input.get_logical_shape();
    auto output_shape = SmallVector<uint32_t>(sizes.size());
    TT_FATAL(
        input_shape.rank() <= sizes.size(),
        "Input tensor shape {}({}) must be at least as large as the expansion size {}({}), which it is not",
        input_shape,
        input_shape.rank(),
        sizes,
        sizes.size());

    int in_idx = static_cast<int>(input_shape.rank()) - 1;
    for (int i = static_cast<int>(output_shape.size()) - 1; i >= 0; --i) {
        if (in_idx >= 0) {
            TT_FATAL(
                input_shape[in_idx] == sizes[i] || input_shape[in_idx] == 1 || sizes[i] == -1,
                "The size of tensor a ({}) must match the size of tensor b ({}) at non-singleton dimension {}",
                input_shape[in_idx],
                sizes[i],
                in_idx);

            if (input_shape[in_idx] == sizes[i] || sizes[i] == -1) {
                output_shape[i] = input_shape[in_idx];
            } else if (input_shape[in_idx] == 1) {
                output_shape[i] = sizes[i];
            }

            --in_idx;
        } else {
            TT_FATAL(sizes[i] != -1, "The expanded size of the tensor (-1) is not allowed in a leading dimension");
            output_shape[i] = sizes[i];
        }
    }

#ifdef DEBUG
    tt::log_debug("inferred output shape: ");
    for (int i = 0; i < output_shape.size(); ++i) {
        tt::log_debug("%d ", output_shape[i]);
    }
    tt::log_debug("\n");
#endif

    return output_shape;
}

Tensor Expand::invoke(
    const Tensor& input,
    const std::vector<int32_t>& sizes,

    const std::optional<Tensor>& output,
    const std::optional<MemoryConfig>& memory_config) {
    auto output_shape = infer_size(input, sizes);

    // Convert tile tensor to row major (lmfao)
    if (input.get_layout() == Layout::TILE) {
        // untilize/tilize is way too inaccurate for us to even remotely use.
        Tensor rm_input_dev = core::to_device(input.cpu(true).to(Layout::ROW_MAJOR), input.device(), std::nullopt);

        Tensor rm_output_dev = ttnn::prim::expand(rm_input_dev, output_shape, std::nullopt, std::nullopt);

        return core::to_device(
            rm_output_dev.cpu(true).pad_to_tile(0).to(Layout::TILE), rm_output_dev.device(), std::nullopt);
    }

    return ttnn::prim::expand(input, output_shape, output, memory_config);
}
}  // namespace ttnn::operations::expand
