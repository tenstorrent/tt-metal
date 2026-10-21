// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
// SPDX-License-Identifier: Apache-2.0

#include "ttnn/operations/experimental/kda/factory/kda_projection.hpp"

#include <algorithm>
#include <optional>
#include <string>

#include <tt-metalium/constants.hpp>
#include <tt-metalium/tensor_accessor_args.hpp>

#include "ttnn/operations/ccl/ccl_op_fusion.hpp"
#include "ttnn/operations/experimental/kda/factory/kda_factory_utils.hpp"
#include "ttnn/operations/experimental/matmul/ccl_fusion/device/ccl_fusion_mcast_2d.hpp"

namespace ttnn::experimental::prim::kda_factory_detail {

namespace {

using tt::tt_metal::CoreRangeSet;

constexpr uint32_t cb_copy = 20;
constexpr const char* kernel_dir = "ttnn/cpp/ttnn/operations/experimental/kda/device/kernels/";

}  // namespace

tt::tt_metal::CoreRangeSet projection_cores(const ProjectionProgramConfig& program_config) {
    if (program_config.allowed_worker_cores) {
        return *program_config.allowed_worker_cores;
    }
    const auto& grid = program_config.compute_with_storage_grid_size;
    return CoreRangeSet(tt::tt_metal::CoreRange({0, 0}, {grid.x - 1, grid.y - 1}));
}

void check_projection(
    const Tensor& a, const Tensor& w, const ProjectionProgramConfig& program_config, std::string_view name) {
    const uint32_t mt = a.physical_volume() / a.padded_shape()[-1] / tt::constants::TILE_HEIGHT;
    const uint32_t kt = a.padded_shape()[-1] / tt::constants::TILE_WIDTH;
    const uint32_t nt = w.padded_shape()[-1] / tt::constants::TILE_WIDTH;
    const auto box = projection_cores(program_config).bounding_box();
    const auto grid = a.device()->compute_with_storage_grid_size();
    TT_FATAL(w.padded_shape()[-2] == a.padded_shape()[-1], "{}: the projection's A and W K differ", name);
    TT_FATAL(
        kt % program_config.in0_block_w == 0,
        "{}: K={} tiles must split into {}-tile blocks",
        name,
        kt,
        program_config.in0_block_w);
    TT_FATAL(
        program_config.per_core_M * box.grid_size().y >= mt && program_config.per_core_N * box.grid_size().x >= nt,
        "{}: the projection's {} x {} output tiles do not fit its cores {}",
        name,
        mt,
        nt,
        box);
    TT_FATAL(box.end_coord.x < grid.x && box.end_coord.y < grid.y, "{}: cores {} leave the grid", name, box);
}

tt::tt_metal::ProgramDescriptor projection_program(
    const Tensor& a,
    const Tensor& w,
    Tensor& output,
    const ProjectionProgramConfig& program_config,
    uint32_t output_column,
    const DeviceComputeKernelConfig& compute_kernel_config) {
    auto config = program_config;
    config.allowed_worker_cores = projection_cores(program_config);
    tt::tt_metal::ProgramDescriptor desc;
    std::optional<ttnn::experimental::ccl::MatmulFusedOpSignaler> no_signaler;
    ttnn::prim::ccl_fusion::matmul_multi_core_reuse_mcast_2d_optimized_helper(
        desc,
        a,
        w,
        /*bias=*/std::nullopt,
        output,
        /*broadcast_batch=*/true,
        compute_kernel_config,
        config,
        /*untilize_out=*/false,
        no_signaler,
        output_column > 0 ? std::optional<uint32_t>(output_column * tt::constants::TILE_WIDTH) : std::nullopt);
    return desc;
}

tt::tt_metal::ProgramDescriptor copy_tile_columns_program(
    const Tensor& source,
    uint32_t source_column,
    const Tensor& destination,
    uint32_t columns,
    const std::vector<tt::tt_metal::CoreCoord>& cores) {
    TT_FATAL(!cores.empty(), "copy_tile_columns_program needs at least one core");
    const uint32_t rows = source.padded_shape()[-2] / tt::constants::TILE_HEIGHT;
    const std::vector<tt::tt_metal::CoreCoord> used(
        cores.begin(), cores.begin() + std::min<size_t>(rows, cores.size()));
    const uint32_t per_core = (rows + used.size() - 1) / used.size();
    const uint32_t page = tile_bytes(source);
    const CoreRangeSet set = merged_core_set(used);
    tt::tt_metal::ProgramDescriptor desc;
    desc.cbs = {
        cb_descriptor(cb_copy, columns, page, tt::tt_metal::datatype_to_dataformat_converter(source.dtype()), set)};
    std::vector<uint32_t> ct = {cb_copy, columns};
    tt::tt_metal::TensorAccessorArgs(source.buffer()).append_to(ct);
    tt::tt_metal::TensorAccessorArgs(destination.buffer()).append_to(ct);
    tt::tt_metal::KernelDescriptor kernel{
        .kernel_source = std::string(kernel_dir) + "dataflow/copy_tile_columns.cpp",
        .core_ranges = set,
        .compile_time_args = std::move(ct),
        .config = tt::tt_metal::WriterConfigDescriptor{},
    };
    for (uint32_t index = 0; index < used.size(); ++index) {
        tt::tt_metal::KernelDescriptor::RTArgList args;
        args.push_back(source.buffer());
        args.push_back(destination.buffer());
        for (uint32_t value :
             {source.padded_shape()[-1] / tt::constants::TILE_WIDTH,
              source_column,
              destination.padded_shape()[-1] / tt::constants::TILE_WIDTH,
              std::min(rows, index * per_core),
              std::min(rows, (index + 1) * per_core)}) {
            args.push_back(value);
        }
        kernel.emplace_runtime_args(used[index], args);
    }
    desc.kernels.push_back(std::move(kernel));
    return desc;
}

}  // namespace ttnn::experimental::prim::kda_factory_detail
