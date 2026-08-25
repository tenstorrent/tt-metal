// SPDX-FileCopyrightText: © 2025 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0
#pragma once

#include "ttnn/operations/core/core.hpp"
#include "ttnn/operations/creation/creation.hpp"
#include "ttnn/operations/data_movement/permute/permute.hpp"
#include "ttnn/operations/data_movement/pad/pad.hpp"
#include "ttnn/operations/data_movement/slice/slice.hpp"
#include "ttnn/operations/data_movement/concat/concat.hpp"
#include "ttnn/operations/data_movement/tilize_with_val_padding/tilize_with_val_padding.hpp"
#include <tt-metalium/host_api.hpp>
#include "ttnn/tensor/tensor_utils.hpp"
#include <vector>
#include <algorithm>

namespace ttnn::operations::experimental::conv3d {

// Minimal valid C_in_block for a conv3d with the given kernel volume (kD*kH*kW).
// It satisfies weight tile-alignment (kernel_vol * C_in_block divisible by
// TILE_WIDTH) and L1 alignment, and is the smallest such value -> smallest
// circular buffers. Used as the shared default in both prepare_conv3d_weights
// and conv3d so their K-row blocking always agrees (issues #42146, #47316).
uint32_t default_c_in_block(uint32_t kernel_vol);

Tensor convert_conv_weight_tensor_to_grouped_layout(
    const Tensor& conv_weight_tensor, uint32_t num_groups, DataType output_dtype);

Tensor prepare_conv3d_weights(
    const ttnn::Tensor& weights,
    uint32_t groups,
    uint32_t C_in_block = 0,
    uint32_t alignment = 32,
    MeshDevice* device = nullptr);

}  // namespace ttnn::operations::experimental::conv3d
