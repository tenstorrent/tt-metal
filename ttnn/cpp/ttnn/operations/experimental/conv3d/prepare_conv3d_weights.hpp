// SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0
#pragma once

#include "prepare_conv3d_weights.hpp"
#include "ttnn/operations/core/core.hpp"
#include "ttnn/operations/creation.hpp"
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

Tensor convert_conv_weight_tensor_to_grouped_layout(
    const Tensor& conv_weight_tensor, uint32_t num_groups, DataType output_dtype);

Tensor prepare_weights(
    const ttnn::Tensor& weights,
    uint32_t groups,
    uint32_t C_in_block = 0,
    MeshDevice* device = nullptr,
    uint32_t alignment = 32);

}  // namespace ttnn::operations::experimental::conv3d
