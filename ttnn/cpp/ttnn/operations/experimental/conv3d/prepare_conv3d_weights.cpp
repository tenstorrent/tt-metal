// SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#include "prepare_conv3d_weights.hpp"
#include <cstdint>
#include <tt-logger/tt-logger.hpp>
#include "tt-metalium/shape.hpp"
#include "ttnn/operations/data_movement/permute/permute.hpp"
#include "ttnn/operations/data_movement/pad/pad.hpp"
#include "ttnn/operations/data_movement/reshape_view/reshape.hpp"
#include "ttnn/tensor/tensor.hpp"
#include "ttnn/operations/core/core.hpp"

namespace ttnn {
namespace operations::experimental::conv3d {
namespace detail {

ttnn::Tensor prepare_conv3d_weights(
    const ttnn::Tensor& weight_tensor,
    uint32_t in_channels,
    uint32_t out_channels,
    const Conv3dConfig& conv_config,
    uint32_t alignment) {
    TT_FATAL(weight_tensor.layout() == Layout::ROW_MAJOR, "Weight tensor must be in ROW_MAJOR layout");
    TT_FATAL(
        weight_tensor.logical_shape().rank() == 5, "Weight tensor must be 5D [out_channels, in_channels, kD, kH, kW]");

    auto shape = weight_tensor.logical_shape();
    uint32_t out_chan = shape[0];
    uint32_t C = shape[1];
    uint32_t kD = shape[2];
    uint32_t kH = shape[3];
    uint32_t kW = shape[4];

    TT_FATAL(out_chan == out_channels, "Weight tensor out_channels dimension mismatch");
    TT_FATAL(C == in_channels, "Weight tensor in_channels dimension mismatch");

    // Step 1: Permute [out_chan, C, kD, kH, kW] -> [kD, kH, kW, C, out_chan]
    // Permutation order: [2, 3, 4, 1, 0]
    ttnn::SmallVector<int64_t> permute_order = {2, 3, 4, 1, 0};
    auto w = ttnn::permute(weight_tensor, permute_order);

    // Step 2: Pad input channels to alignment boundary
    uint32_t ALIGN_PAD = (alignment - C % alignment) % alignment;
    uint32_t C_in_aligned = C + ALIGN_PAD;

    if (ALIGN_PAD > 0) {
        // Shape after permute is [kD, kH, kW, C, out_chan]
        // Need to pad dimension 3 (C dimension) from C to C_in_aligned
        ttnn::SmallVector<std::array<uint32_t, 2>> padding = {
            {0, 0},          // kD: no padding
            {0, 0},          // kH: no padding
            {0, 0},          // kW: no padding
            {0, ALIGN_PAD},  // C: pad ALIGN_PAD elements at the end
            {0, 0}           // out_chan: no padding
        };
        w = ttnn::pad(w, padding, 0.0f);
    }

    // Step 3: Reshape based on C_in_block configuration
    uint32_t C_in_block = conv_config.C_in_block;
    if (C_in_block == 0) {
        C_in_block = C_in_aligned;
    }

    uint32_t num_C_in_blocks = C_in_aligned / C_in_block;
    TT_FATAL(
        num_C_in_blocks * C_in_block == C_in_aligned,
        "num_C_in_blocks * C_in_block must equal C_in_aligned, got {} * {} != {}",
        num_C_in_blocks,
        C_in_block,
        C_in_aligned);

    // Reshape to [kD, kH, kW, num_C_in_blocks, C_in_block, out_chan]
    ttnn::Shape reshaped_shape({kD, kH, kW, num_C_in_blocks, C_in_block, out_chan});
    w = ttnn::reshape(w, reshaped_shape);

    // Step 4: Permute [kD, kH, kW, num_C_in_blocks, C_in_block, out_chan] ->
    //                 [num_C_in_blocks, kD, kH, kW, C_in_block, out_chan]
    // Permutation order: [3, 0, 1, 2, 4, 5]
    ttnn::SmallVector<int64_t> permute_order2 = {3, 0, 1, 2, 4, 5};
    w = ttnn::permute(w, permute_order2);

    // Step 5: Flatten to 2D [-1, out_chan]
    uint32_t dim0 = num_C_in_blocks * kD * kH * kW * C_in_block;
    ttnn::Shape final_shape({dim0, out_chan});
    w = ttnn::reshape(w, final_shape);

    // Step 6: Convert to tile layout
    return ttnn::to_layout(w, Layout::TILE);
}

ttnn::Tensor prepare_conv3d_bias(const ttnn::Tensor& bias_tensor, uint32_t out_channels) {
    TT_FATAL(bias_tensor.layout() == Layout::ROW_MAJOR, "Bias tensor must be in ROW_MAJOR layout");
    TT_FATAL(
        bias_tensor.logical_volume() == out_channels,
        "Bias tensor must have {} elements, got {}",
        out_channels,
        bias_tensor.logical_volume());

    // Reshape to [1, out_channels]
    ttnn::Shape bias_shape({1, out_channels});
    auto bias = ttnn::reshape(bias_tensor, bias_shape);

    // Convert to tile layout
    return ttnn::to_layout(bias, Layout::TILE);
}

}  // namespace detail
}  // namespace operations::experimental::conv3d
}  // namespace ttnn
