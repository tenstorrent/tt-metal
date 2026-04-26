// SPDX-FileCopyrightText: © 2026 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include "flip.hpp"

#include <tt-metalium/constants.hpp>

#include "ttnn/operations/core/core.hpp"
#include "ttnn/operations/data_movement/flip/device/flip_device_operation.hpp"
#include "ttnn/operations/data_movement/tilize_with_val_padding/tilize_with_val_padding.hpp"
#include "ttnn/operations/data_movement/untilize/untilize.hpp"
#include "ttnn/operations/data_movement/slice/slice.hpp"
#include "ttnn/operations/data_movement/common/common.hpp"
#include "ttnn/tensor/tensor_utils.hpp"

namespace ttnn::operations::data_movement::detail {

bool is_flip_nop(const ttnn::Tensor& input_tensor, const ttnn::SmallVector<uint32_t>& dims) {
    const auto& shape = input_tensor.logical_shape();
    for (auto dim : dims) {
        if (shape[dim] > 1) {
            return false;
        }
    }
    return true;
}

ttnn::Tensor flip_tiled(
    const ttnn::Tensor& input_tensor, const ttnn::SmallVector<uint32_t>& dims, const MemoryConfig& mem_conf) {
    TT_FATAL(input_tensor.layout() == ttnn::TILE_LAYOUT, "flip_tiled: expected TILE_LAYOUT input");

    const auto& logical_shape = input_tensor.logical_shape();
    const auto& padded_shape = input_tensor.padded_shape();

    // TILE to RM layout
    auto rm_padded = ttnn::untilize(input_tensor, mem_conf);

    // Slice off padding if logical != padded
    ttnn::Tensor rm_unpadded = rm_padded;
    if (logical_shape != padded_shape) {
        SmallVector<uint32_t> begins(logical_shape.rank(), 0);
        SmallVector<uint32_t> ends(logical_shape.rank());
        SmallVector<uint32_t> steps(logical_shape.rank(), 1);
        for (uint32_t i = 0; i < logical_shape.rank(); i++) {
            ends[i] = logical_shape[i];
        }
        rm_unpadded = ttnn::slice(rm_padded, begins, ends, steps, mem_conf);
    }

    auto flipped_rm = ttnn::prim::flip(rm_unpadded, dims, mem_conf, std::nullopt);

    // RM -> TILE, pad back to original padded shape
    return ttnn::tilize_with_val_padding(flipped_rm, padded_shape, 0.0f, mem_conf);
}

}  // namespace ttnn::operations::data_movement::detail

namespace ttnn {

ttnn::Tensor flip(
    const ttnn::Tensor& input_tensor,
    const SmallVector<int64_t>& dims,
    const std::optional<MemoryConfig>& memory_config) {
    const auto input_rank = input_tensor.logical_shape().rank();

    TT_FATAL(!dims.empty(), "Flip dimensions cannot be empty");
    TT_FATAL(is_device_tensor(input_tensor), "Input tensor must be on device");
    TT_FATAL(input_rank > 1, "Flip does not support tensors with rank 1");
    TT_FATAL(input_rank <= 5, "Flip operation supports tensors with rank up to 5, got rank {}", input_rank);

    // Normalize dimensions to positive indices
    SmallVector<uint32_t> normalized_dims(dims.size());
    std::transform(dims.begin(), dims.end(), normalized_dims.begin(), [&input_tensor](int64_t idx) {
        return input_tensor.logical_shape().get_normalized_index(idx);
    });

    auto mem_conf = memory_config.value_or(input_tensor.memory_config());

    // No-op: all flip dims have size 1
    if (operations::data_movement::detail::is_flip_nop(input_tensor, normalized_dims)) {
        return ttnn::to_memory_config(input_tensor, mem_conf);
    }

    // Route based on layout
    if (input_tensor.layout() == ttnn::TILE_LAYOUT) {
        return operations::data_movement::detail::flip_tiled(input_tensor, normalized_dims, mem_conf);
    }

    // ROW_MAJOR — dispatch directly to MultiCoreRowMajor
    return ttnn::prim::flip(input_tensor, normalized_dims, mem_conf, std::nullopt);
}

ttnn::Tensor flip(const ttnn::Tensor& input_tensor, const SmallVector<int64_t>& dims) {
    return flip(input_tensor, dims, std::nullopt);
}

}  // namespace ttnn
