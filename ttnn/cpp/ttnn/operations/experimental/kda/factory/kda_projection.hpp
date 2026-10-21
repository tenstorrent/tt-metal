// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <string_view>
#include <vector>

#include <tt-metalium/core_coord.hpp>
#include <tt-metalium/program_descriptors.hpp>

#include "ttnn/operations/core/compute_kernel/compute_kernel_config.hpp"
#include "ttnn/operations/matmul/device/config/matmul_program_config_types.hpp"
#include "ttnn/tensor/tensor.hpp"

// Programs the KDA ops run on worker cores another program of the same dispatch leaves idle.
namespace ttnn::experimental::prim::kda_factory_detail {

using ProjectionProgramConfig = ttnn::operations::matmul::MatmulMultiCoreReuseMultiCastProgramConfig;

// a @ w as ttnn.linear runs it with `program_config` (the 2D multicast matmul), on its cores: allowed_worker_cores, or
// the compute_with_storage_grid_size block from (0, 0). The output goes into output's tile columns from output_column.
tt::tt_metal::ProgramDescriptor projection_program(
    const Tensor& a,
    const Tensor& w,
    Tensor& output,
    const ProjectionProgramConfig& program_config,
    uint32_t output_column,
    const DeviceComputeKernelConfig& compute_kernel_config);

// The worker cores projection_program runs on.
tt::tt_metal::CoreRangeSet projection_cores(const ProjectionProgramConfig& program_config);

void check_projection(
    const Tensor& a, const Tensor& w, const ProjectionProgramConfig& program_config, std::string_view name);

// Copy `columns` tile columns of `source` from source_column into destination's first columns, split by tile rows over
// `cores`. Both are TILE tensors of one dtype.
tt::tt_metal::ProgramDescriptor copy_tile_columns_program(
    const Tensor& source,
    uint32_t source_column,
    const Tensor& destination,
    uint32_t columns,
    const std::vector<tt::tt_metal::CoreCoord>& cores);

}  // namespace ttnn::experimental::prim::kda_factory_detail
