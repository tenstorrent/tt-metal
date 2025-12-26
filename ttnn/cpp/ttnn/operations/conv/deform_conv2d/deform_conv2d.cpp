// SPDX-FileCopyrightText: © 2025 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include "deform_conv2d.hpp"
#include "ttnn/operations/core/core.hpp"
#include "ttnn/tensor/tensor.hpp"
#include "ttnn/operations/creation.hpp"
#include "ttnn/operations/eltwise/binary/binary.hpp"
#include "ttnn/operations/eltwise/binary/binary_composite.hpp"
#include "ttnn/operations/data_movement/reshape_view/reshape.hpp"
#include "ttnn/operations/data_movement/pad/pad.hpp"
#include "ttnn/operations/data_movement/stack/stack.hpp"
#include "ttnn/operations/data_movement/unsqueeze/unsqueeze.hpp"
#include "ttnn/operations/data_movement/squeeze/squeeze.hpp"
#include "ttnn/operations/pool/grid_sample/grid_sample.hpp"
#include "ttnn/operations/data_movement/slice/slice.hpp"
#include "ttnn/operations/matmul/matmul.hpp"
#include "ttnn/operations/data_movement/concat/concat.hpp"
#include <cmath>

namespace ttnn::operations::conv::deform_conv2d {
ttnn::Tensor DeformConv2dOperation::invoke(
    const ttnn::Tensor& input_tensor,
    const ttnn::Tensor& weight_tensor,
    const ttnn::Tensor& offset_tensor,
    int stride,
    uint32_t padding,
    int dilation,
    int groups,
    int offset_groups) {
    ttnn::Tensor result;
    auto* mesh_device = dynamic_cast<tt::tt_metal::distributed::MeshDevice*>(input_tensor.device());

    auto B = input_tensor.logical_shape()[0];
    auto H = input_tensor.logical_shape()[1];
    auto W = input_tensor.logical_shape()[2];
    auto C_in = input_tensor.logical_shape()[3];

    auto kH = weight_tensor.logical_shape()[0];
    auto kW = weight_tensor.logical_shape()[1];
    auto C_in_g = weight_tensor.logical_shape()[2];
    auto C_out = weight_tensor.logical_shape()[3];

    TT_FATAL(C_in == C_in_g * groups, "C_in {} != {}. ", C_in, C_in_g * groups);

    TT_FATAL(C_out % groups == 0, "C_out {} must be divisible by groups {}. ", C_out, groups);

    auto out_H = std::floor((((H + 2 * padding) - (dilation * (kH - 1)) - 1)) / stride) + 1;
    auto out_W = std::floor((((W + 2 * padding) - (dilation * (kW - 1)) - 1)) / stride) + 1;

    ttnn::SmallVector<ttnn::operations::data_movement::PadSpecDim> padding_spec;
    padding_spec.push_back({0, 0});              // N dimension
    padding_spec.push_back({padding, padding});  // H dimension
    padding_spec.push_back({padding, padding});  // W dimension
    padding_spec.push_back({0, 0});              // C dimension

    ttnn::Tensor x_padded = ttnn::pad(input_tensor, padding_spec, 0.0f, true, std::nullopt);

    int C_in_per_group = C_in / groups;
    int C_out_per_group = C_out / groups;
    int kHkW = kH * kW;
    int groups_per_offset_group = groups / offset_groups;

    int H_padded = x_padded.logical_shape()[1];
    int W_padded = x_padded.logical_shape()[2];

    ttnn::Tensor oy = ttnn::arange(0, out_H, 1, ttnn::DataType::FLOAT32, std::ref(*mesh_device));
    oy = ttnn::reshape(oy, ttnn::Shape({out_H, 1, 1, 1}));
    oy = ttnn::multiply(oy, stride);

    ttnn::Tensor ox = ttnn::arange(0, out_W, 1, ttnn::DataType::FLOAT32, std::ref(*mesh_device));
    ox = ttnn::reshape(ox, ttnn::Shape({1, out_W, 1, 1}));
    ox = ttnn::multiply(ox, stride);

    ttnn::Tensor ky = ttnn::arange(0, kH, 1, ttnn::DataType::FLOAT32, std::ref(*mesh_device));
    ky = ttnn::reshape(ky, ttnn::Shape({1, 1, kH, 1}));
    ky = ttnn::multiply(ky, dilation);

    ttnn::Tensor kx = ttnn::arange(0, kW, 1, ttnn::DataType::FLOAT32, std::ref(*mesh_device));
    kx = ttnn::reshape(kx, ttnn::Shape({1, 1, 1, kW}));
    kx = ttnn::multiply(kx, dilation);

    ttnn::Tensor y_base = ttnn::add(oy, ky);
    ttnn::Tensor x_base = ttnn::add(ox, kx);

    std::vector<ttnn::Tensor> precomputed_grids;

    for (int i = 0; i < offset_groups; i++) {
        auto off_g = ttnn::slice(
            offset_tensor,
            ttnn::SmallVector<int>{0, 0, 0, i * 2 * kHkW},
            ttnn::SmallVector<int>{
                offset_tensor.logical_shape()[0],
                offset_tensor.logical_shape()[1],
                offset_tensor.logical_shape()[2],
                (2 * kHkW) * (i + 1)},
            ttnn::SmallVector<int>{1, 1, 1, 1});
        off_g = ttnn::reshape(off_g, ttnn::Shape({B, out_H, out_W, kH, kW, 2}));
        auto off_y = ttnn::slice(
            off_g,
            ttnn::SmallVector<int>{0, 0, 0, 0, 0, 0},
            ttnn::SmallVector<int>{
                off_g.logical_shape()[0],
                off_g.logical_shape()[1],
                off_g.logical_shape()[2],
                off_g.logical_shape()[3],
                off_g.logical_shape()[4],
                1},
            ttnn::SmallVector<int>{1, 1, 1, 1, 1, 1});
        auto off_x = ttnn::slice(
            off_g,
            ttnn::SmallVector<int>{0, 0, 0, 0, 0, 1},
            ttnn::SmallVector<int>{
                off_g.logical_shape()[0],
                off_g.logical_shape()[1],
                off_g.logical_shape()[2],
                off_g.logical_shape()[3],
                off_g.logical_shape()[4],
                2},
            ttnn::SmallVector<int>{1, 1, 1, 1, 1, 1});

        auto y = ttnn::unsqueeze(y_base, 0);
        off_y = ttnn::squeeze(off_y, -1);
        y = ttnn::add(y, off_y);
        auto x_ = ttnn::unsqueeze(x_base, 0);
        off_x = ttnn::squeeze(off_x, -1);
        x_ = ttnn::add(x_, off_x);

        auto y_add = ttnn::add(y, 0.5);
        auto y_div = ttnn::div(y_add, float(H_padded));
        auto y_mul = ttnn::multiply(y_div, 2.0);
        auto y_norm = ttnn::subtract(y_mul, 1.0);

        auto x_add = ttnn::add(x_, 0.5);
        auto x_div = ttnn::div(x_add, float(W_padded));
        auto x_mul = ttnn::multiply(x_div, 2.0);
        auto x_norm = ttnn::subtract(x_mul, 1.0);

        std::vector<ttnn::Tensor> stacks = {x_norm, y_norm};
        auto grid = ttnn::stack(stacks, -1);

        auto grid_flat = ttnn::reshape(grid, ttnn::Shape({B, out_H * out_W * kH * kW, 1, 2}));

        precomputed_grids.push_back(grid_flat);
    }

    std::vector<ttnn::Tensor> outputs;

    for (int i = 0; i < groups; i++) {
        int offset_group_idx = i / groups_per_offset_group;
        auto grid_flat = precomputed_grids[offset_group_idx];

        ttnn::SmallVector<int> left_part_1 = {0, 0, 0, i * C_in_per_group};
        ttnn::SmallVector<int> right_part_1 = {
            x_padded.logical_shape()[0],
            x_padded.logical_shape()[1],
            x_padded.logical_shape()[2],
            (i + 1) * C_in_per_group};
        ttnn::SmallVector<int> stride_vec_1 = {1, 1, 1, 1};

        auto x_g = ttnn::slice(x_padded, left_part_1, right_part_1, stride_vec_1);

        ttnn::SmallVector<int> left_part_2 = {0, 0, 0, i * C_out_per_group};
        ttnn::SmallVector<int> right_part_2 = {
            weight_tensor.logical_shape()[0],
            weight_tensor.logical_shape()[1],
            weight_tensor.logical_shape()[2],
            (i + 1) * C_out_per_group};
        ttnn::SmallVector<int> stride_vec_2 = {1, 1, 1, 1};

        auto w_g = ttnn::slice(weight_tensor, left_part_2, right_part_2, stride_vec_2);

        auto org_channels = x_g.logical_shape()[3];
        int pad_c = ((org_channels + 31) / 32) * 32 - org_channels;

        SmallVector<std::array<uint32_t, 2>> tt_padding = {
            {0, 0},
            {0, 0},
            {0, 0},
            {0, pad_c},
        };

        x_g = ttnn::pad(x_g, tt_padding, 0);

        ttnn::Tensor sampled = ttnn::grid_sample(x_g, grid_flat);

        ttnn::SmallVector<int> left_part = {0, 0, 0, 0};
        ttnn::SmallVector<int> right_part = {
            sampled.logical_shape()[0],
            sampled.logical_shape()[1],
            sampled.logical_shape()[2],
            org_channels};  // or whatever end indices are needed
        ttnn::SmallVector<int> stride_vec = {1, 1, 1, 1};

        sampled = ttnn::slice(sampled, left_part, right_part, stride_vec);

        sampled = ttnn::reshape(sampled, ttnn::Shape({B, out_H, out_W, C_in_per_group * kH * kW}));

        w_g = ttnn::reshape(w_g, ttnn::Shape({C_in_per_group * kH * kW, C_out_per_group}));

        sampled = ttnn::to_layout(sampled, ttnn::Layout::TILE);
        w_g = ttnn::to_layout(w_g, ttnn::Layout::TILE);

        result = ttnn::matmul(sampled, w_g);

        outputs.push_back(result);
    }

    result = ttnn::concat(outputs, 3);

    return result;
}
}  // namespace ttnn::operations::conv::deform_conv2d
