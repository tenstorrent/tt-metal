// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include "ttnn/operations/transformer/sdpa/device/deepseek_v4_prefill_write_rows_device_operation.hpp"
#include "ttnn/operations/transformer/sdpa/device/kernels/deepseek_v4_prefill_write_rows_common.hpp"
#include <tt-metalium/buffer.hpp>
#include <tt-metalium/host_api.hpp>
#include <tt-metalium/program_descriptors.hpp>
#include <tt-metalium/tensor_accessor_args.hpp>
#include <algorithm>
#include <string>

namespace ttnn::prim {

tt::tt_metal::ProgramDescriptor DeepseekV4PrefillWriteRowsOperation::ProgramFactory::create_descriptor(
    const DeepseekV4PrefillWriteRowsParams& /*attrs*/, const DeepseekV4PrefillWriteRowsInputs& t, Tensor& output) {
    namespace dwr = ::deepseek_v4_prefill_write_rows;
    enum : uint32_t { cb_rows = 0, cb_data };
    tt::tt_metal::ProgramDescriptor desc;

    const uint32_t n = t.input.logical_shape()[2];
    const uint32_t row_bytes = t.input.buffer()->page_size();
    const uint32_t index_bytes = n * static_cast<uint32_t>(sizeof(uint32_t));

    auto* device = t.input.device();
    const tt::tt_metal::CoreCoord grid = device->compute_with_storage_grid_size();
    // At least one batch per core: a core's fixed cost (the row index read) outweighs a few rows.
    const uint32_t used_cores = std::min<uint32_t>(grid.x * grid.y, (n + dwr::BATCH_ROWS - 1) / dwr::BATCH_ROWS);
    const auto logical = [&](uint32_t i) { return tt::tt_metal::CoreCoord{i % grid.x, i / grid.x}; };
    std::vector<tt::tt_metal::CoreRange> ranges;
    for (uint32_t i = 0; i < used_cores; ++i) {
        ranges.emplace_back(logical(i));
    }
    const auto cores = tt::tt_metal::CoreRangeSet(ranges).merge_ranges();

    const auto cb = [&](uint32_t id, uint32_t page_size) {
        desc.cbs.push_back(tt::tt_metal::CBDescriptor{
            .total_size = page_size,
            .core_ranges = cores,
            .format_descriptors = {{tt::tt_metal::CBFormatDescriptor{
                .buffer_index = static_cast<uint8_t>(id),
                .data_format = tt::DataFormat::UInt32,
                .page_size = page_size}}},
        });
    };
    cb(cb_rows, index_bytes);
    cb(cb_data, dwr::BATCH_ROWS * row_bytes);

    auto* in_buf = t.input.buffer();
    auto* out_buf = output.buffer();
    auto* rows_buf = t.rows.buffer();
    std::vector<uint32_t> ct = {row_bytes, n, cb_rows, cb_data};
    TT_FATAL(
        ct.size() == dwr::ct_arg::END, "deepseek_v4_prefill_write_rows compile-time argument layout is out of sync");
    std::vector<uint32_t> crt;
    tt::tt_metal::TensorAccessorArgs(in_buf).append_to(ct, crt);
    tt::tt_metal::TensorAccessorArgs(out_buf).append_to(ct, crt);
    tt::tt_metal::TensorAccessorArgs(rows_buf).append_to(ct, crt);

    tt::tt_metal::KernelDescriptor kernel;
    kernel.kernel_source =
        "ttnn/cpp/ttnn/operations/transformer/sdpa/device/kernels/dataflow/deepseek_v4_prefill_write_rows.cpp";
    kernel.source_type = tt::tt_metal::KernelDescriptor::SourceType::FILE_PATH;
    kernel.core_ranges = cores;
    kernel.compile_time_args = ct;
    kernel.config = tt::tt_metal::ReaderConfigDescriptor{};
    kernel.emplace_common_runtime_args({in_buf, out_buf, rows_buf, output.logical_shape()[2]});
    TT_FATAL(
        kernel.common_runtime_args.size() == dwr::common_arg::END,
        "deepseek_v4_prefill_write_rows common runtime argument layout is out of sync");
    kernel.common_runtime_args.insert(kernel.common_runtime_args.end(), crt.begin(), crt.end());

    for (uint32_t i = 0; i < used_cores; ++i) {
        const uint32_t first = i * n / used_cores;
        const std::initializer_list<uint32_t> args = {first, (i + 1) * n / used_cores - first};
        TT_FATAL(
            args.size() == dwr::core_arg::END, "deepseek_v4_prefill_write_rows core argument layout is out of sync");
        kernel.emplace_runtime_args(logical(i), args);
    }
    desc.kernels.push_back(std::move(kernel));
    return desc;
}

void DeepseekV4PrefillWriteRowsOperation::ProgramFactory::override_runtime_arguments(
    tt::tt_metal::Program& program,
    const DeepseekV4PrefillWriteRowsParams& /*operation_attributes*/,
    const DeepseekV4PrefillWriteRowsInputs& t,
    Tensor& output,
    const std::optional<ttnn::MeshCoordinate>& /*mesh_dispatch_coordinate*/) {
    namespace dwr = ::deepseek_v4_prefill_write_rows;
    auto& args = tt::tt_metal::GetCommonRuntimeArgs(program, 0);
    args[dwr::common_arg::INPUT_ADDRESS] = t.input.buffer()->address();
    args[dwr::common_arg::OUTPUT_ADDRESS] = output.buffer()->address();
    args[dwr::common_arg::ROWS_ADDRESS] = t.rows.buffer()->address();
    args[dwr::common_arg::OUTPUT_ROWS] = output.logical_shape()[2];
}

}  // namespace ttnn::prim
