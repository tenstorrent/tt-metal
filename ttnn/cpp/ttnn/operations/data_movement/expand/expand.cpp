// SPDX-FileCopyrightText: © 2024 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include "expand.hpp"

#include <cstdio>

#include "ttnn/operations/data_movement/expand/device/expand_device_operation.hpp"
namespace ttnn::operations::expand {

std::vector<uint32_t> infer_size(const Tensor& input, const std::vector<int32_t>& sizes) {
    auto input_shape = input.get_shape();
    auto output_shape = std::vector<uint32_t>(sizes.size());
    TT_FATAL(
        input_shape.size() <= sizes.size(),
        "Input tensor shape {}({}) must be at least as large as the expansion size {}({}), which it is not",
        input_shape,
        input_shape.size(),
        sizes,
        sizes.size());

    int in_idx = input_shape.size() - 1;
    for (int i = output_shape.size() - 1; i >= 0; --i) {
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
    const std::optional<MemoryConfig>& output_mem_config,
    const std::optional<DeviceComputeKernelConfig>& compute_kernel_config) {
    auto output_shape = infer_size(input, sizes);
    return ttnn::prim::expand(input, output_shape, output, output_mem_config, compute_kernel_config);
}
}  // namespace ttnn::operations::expand
