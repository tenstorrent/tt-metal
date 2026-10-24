// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <span>
#include <string_view>
#include <vector>

#include <tt-metalium/core_coord.hpp>
#include <tt-metalium/experimental/metal2_host_api/program_run_args.hpp>
#include <tt-metalium/experimental/metal2_host_api/program_spec.hpp>

#include "ttnn/operations/core/compute_kernel/compute_kernel_config.hpp"
#include "ttnn/operations/eltwise/unary/common/unary_op_types.hpp"
#include "ttnn/tensor/tensor.hpp"

// Work a KDA program runs on worker cores its own kernels leave idle.
namespace ttnn::experimental::prim::kda_factory_detail {

// output[..., output_column + c] = op_chain(input[..., input_column + c]) for the `columns` tile columns c of two TILE
// tensors with the same rows: the eltwise unary chain as ttnn's unary ops run it (copy into DST, the chain, pack, the
// input unpacked as ttnn::unary unpacks it), at the output's format and compute_kernel_config's DST width (ttnn::unary
// instead forces a 32-bit DST for a FLOAT32 input or output). An empty chain copies the columns, converting their
// format.
struct TileColumnActivationParams {
    std::vector<ttnn::operations::unary::EltwiseUnaryWithParam> op_chain;
    uint32_t input_column = 0;
    uint32_t output_column = 0;
    uint32_t columns = 0;
    DeviceComputeKernelConfig compute_kernel_config;
};

void check_tile_column_activation(
    const TileColumnActivationParams& params,
    const Tensor& input,
    const Tensor& output,
    std::string_view operation_name);

// Add the activation to `spec` as a work unit of its own on `cores` (outside every other work unit of the spec), the
// window's tiles split evenly over the cores in row-major order. Its kernels, buffers and tensor parameters take
// names starting with "tile_columns".
void add_tile_column_activation(
    tt::tt_metal::experimental::ProgramSpec& spec,
    tt::tt_metal::experimental::ProgramRunArgs& run_args,
    const TileColumnActivationParams& params,
    const Tensor& input,
    const Tensor& output,
    std::span<const tt::tt_metal::CoreCoord> cores);

// The cores of the `grid` block from (0, 0) that are not in `busy`, in row-major order.
std::vector<tt::tt_metal::CoreCoord> idle_cores(tt::tt_metal::CoreCoord grid, const tt::tt_metal::CoreRangeSet& busy);

}  // namespace ttnn::experimental::prim::kda_factory_detail
