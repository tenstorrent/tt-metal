// SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#include "pack_fp8_rows_program_factory.hpp"

#include <algorithm>
#include <variant>
#include <vector>

#include <tt-metalium/core_coord.hpp>
#include <tt-metalium/hal.hpp>
#include <tt-metalium/host_api.hpp>
#include <tt-metalium/math.hpp>
#include <tt-metalium/tensor_accessor_args.hpp>
#include <tt-metalium/work_split.hpp>

#include "kernels/pack_fp8_rows_args.hpp"

namespace ttnn::operations::pack_fp8_rows {

using namespace tt::tt_metal;
namespace args = ::pack_fp8_rows;

namespace {

constexpr const char* kPackKernel = "ttnn/cpp/ttnn/operations/pack_fp8_rows/device/kernels/pack_fp8_rows.cpp";

}  // namespace

PackFp8RowsPlan plan_pack_fp8_rows(const PackFp8RowsParams& attrs, const Tensor& e4m3) {
    const auto& shape = e4m3.logical_shape();
    uint32_t rows = 1;
    for (uint32_t i = 0; i + 1 < shape.rank(); ++i) {
        rows *= shape[i];
    }
    const uint32_t cores = std::max(1u, std::min(attrs.grid_x * attrs.grid_y, tt::div_up(rows, args::ROWS_BLOCK)));
    return PackFp8RowsPlan{.rows = rows, .cores = cores, .grid_x = attrs.grid_x};
}

ProgramDescriptor PackFp8RowsProgramFactory::create_descriptor(
    const PackFp8RowsParams& attrs, const PackFp8RowsInputs& t, Tensor& output) {
    TT_FATAL(
        args::cb::STAGE < hal::get_num_dataflow_buffers(),
        "pack_fp8_rows: circular-buffer id {} exceeds the {} circular buffers of this architecture",
        args::cb::STAGE,
        hal::get_num_dataflow_buffers());

    const PackFp8RowsPlan plan = plan_pack_fp8_rows(attrs, t.e4m3);
    // Cores (i % grid_x, i / grid_x) for i < cores, as the Python placed them: filled row by row from (0, 0).
    const CoreRangeSet crs =
        num_cores_to_corerangeset(plan.cores, CoreCoord(attrs.grid_x, attrs.grid_y), /*row_wise=*/true);
    const uint32_t per = plan.rows / plan.cores;
    const uint32_t extra = plan.rows % plan.cores;

    ProgramDescriptor desc;
    desc.cbs.push_back(CBDescriptor{
        .total_size = args::STAGE_BYTES,
        .core_ranges = crs,
        .format_descriptors = {CBFormatDescriptor{
            .buffer_index = static_cast<uint8_t>(args::cb::STAGE),
            .data_format = tt::DataFormat::UInt8,
            .page_size = args::STAGE_BYTES}},
    });

    KernelDescriptor kernel;
    kernel.kernel_source = kPackKernel;
    kernel.core_ranges = crs;
    kernel.compile_time_args.resize(args::ct_arg::COUNT);
    kernel.compile_time_args[args::ct_arg::CB_STAGE] = args::cb::STAGE;
    kernel.compile_time_args[args::ct_arg::ROWS_BLOCK] = args::ROWS_BLOCK;
    kernel.compile_time_args[args::ct_arg::GRID_X] = plan.grid_x;
    TensorAccessorArgs(*t.e4m3.buffer()).append_to(kernel.compile_time_args);
    TensorAccessorArgs(*t.scale.buffer()).append_to(kernel.compile_time_args);
    TensorAccessorArgs(*output.buffer()).append_to(kernel.compile_time_args);

    // Pushed in common_arg order: E4M3_ADDR, SCALE_ADDR, OUT_ADDR, ROWS_PER_CORE, EXTRA_ROWS.
    KernelDescriptor::RTArgList common;
    common.push_back(t.e4m3.buffer());
    common.push_back(t.scale.buffer());
    common.push_back(output.buffer());
    common.push_back(per);
    common.push_back(extra);
    kernel.emplace_common_runtime_args(common);
    TT_FATAL(
        kernel.common_runtime_args.size() == args::common_arg::COUNT,
        "pack_fp8_rows: common runtime argument layout is out of sync with pack_fp8_rows_args.hpp");
    kernel.config = WriterConfigDescriptor{};

    desc.kernels.push_back(std::move(kernel));
    return desc;
}

void PackFp8RowsProgramFactory::override_runtime_arguments(
    Program& program,
    const PackFp8RowsParams& /*attrs*/,
    const PackFp8RowsInputs& t,
    Tensor& output,
    const std::optional<ttnn::MeshCoordinate>& /*mesh_dispatch_coordinate*/) {
    // The key pins ROWS_PER_CORE and EXTRA_ROWS (shapes and grid); only the three addresses can move.
    auto& common = GetCommonRuntimeArgs(program, args::kernel_index::PACK);
    common[args::common_arg::E4M3_ADDR] = t.e4m3.buffer()->address();
    common[args::common_arg::SCALE_ADDR] = t.scale.buffer()->address();
    common[args::common_arg::OUT_ADDR] = output.buffer()->address();
}

}  // namespace ttnn::operations::pack_fp8_rows
