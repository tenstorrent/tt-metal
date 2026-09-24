// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
// SPDX-License-Identifier: Apache-2.0

#include "sdpa.hpp"

#include <algorithm>
#include <limits>
#include <set>
#include <tt-metalium/tensor_accessor_args.hpp>
#include "ttnn/operations/generic/generic_op.hpp"
#include "ttnn/tensor/tensor.hpp"

namespace ttnn::transformer {
using namespace tt::tt_metal;

Tensor prepare_sdpa_input(const Tensor& input, bool is_query, DataType dtype) {
    TT_FATAL(input.storage_type() == StorageType::DEVICE, "SDPA preparation requires a device tensor");
    TT_FATAL(input.device()->arch() == tt::ARCH::BLACKHOLE, "SDPA preparation currently supports Blackhole only");
    TT_FATAL(input.device()->num_devices() == 1, "SDPA preparation currently requires a single-device mesh");
    TT_FATAL(input.tensor_spec().tile() == Tile({32, 32}), "SDPA preparation requires standard 32x32 tiles");
    TT_FATAL(
        input.dtype() == DataType::BFLOAT16 && input.layout() == Layout::TILE,
        "SDPA preparation requires original tiled BF16 inputs");
    TT_FATAL(
        input.memory_config() == DRAM_MEMORY_CONFIG && input.logical_shape() == input.padded_shape(),
        "SDPA preparation requires unpadded, interleaved DRAM inputs");
    TT_FATAL(
        input.logical_shape().rank() == 4 && input.logical_shape()[3] == 128 && input.logical_volume() > 0,
        "SDPA preparation currently requires nonempty rank-four D128 inputs");
    TT_FATAL(
        dtype == DataType::BFLOAT16 || dtype == DataType::BFLOAT8_B || dtype == DataType::BFLOAT4_B,
        "SDPA preparation output must be BF16, BFP8, or BFP4");
    TT_FATAL(!is_query || dtype == DataType::BFLOAT16, "Prepared Q must retain BF16 storage");
    constexpr uint32_t batch = 4;
    TT_FATAL(
        input.logical_volume() / 1024 <= std::numeric_limits<uint32_t>::max(),
        "SDPA preparation tile count overflows uint32");
    const uint32_t tiles = input.logical_volume() / 1024;
    TT_FATAL(tiles > 0 && tiles % batch == 0, "SDPA preparation requires a multiple of four full tiles");
    const auto hardware = input.device()->compute_with_storage_grid_size();
    const uint32_t cores = std::min<uint32_t>(tiles / batch, hardware.x * hardware.y);
    std::vector<CoreCoord> coordinates;
    std::set<CoreRange> ranges;
    for (uint32_t i = 0; i < cores; ++i) {
        const CoreCoord core(i % hardware.x, i / hardware.x);
        coordinates.push_back(core);
        ranges.emplace(core, core);
    }
    const CoreRangeSet grid(ranges);
    TensorSpec spec(input.logical_shape(), TensorLayout(dtype, PageConfig(Layout::TILE), DRAM_MEMORY_CONFIG));
    auto output = create_device_tensor(spec, input.device());
    const uint32_t bytes = dtype == DataType::BFLOAT16 ? 2048 : dtype == DataType::BFLOAT8_B ? 1088 : 576;
    ProgramDescriptor program;
    program.cbs = {
        {.total_size = 2 * batch * 2048,
         .core_ranges = grid,
         .format_descriptors = {{.buffer_index = 0, .data_format = tt::DataFormat::Float16_b, .page_size = 2048}}},
        {.total_size = 2 * batch * bytes,
         .core_ranges = grid,
         .format_descriptors = {
             {.buffer_index = 16, .data_format = datatype_to_dataformat_converter(dtype), .page_size = bytes}}}};
    const std::string prefix = "ttnn/cpp/ttnn/operations/transformer/sdpa/device/kernels/";
    KernelDescriptor reader{
        .kernel_source = prefix + "dataflow/reader_prepare.cpp",
        .core_ranges = grid,
        .compile_time_args = {batch},
        .config = ReaderConfigDescriptor{}};
    TensorAccessorArgs(input.buffer()).append_to(reader.compile_time_args);
    KernelDescriptor writer{
        .kernel_source = prefix + "dataflow/writer_prepare.cpp",
        .core_ranges = grid,
        .compile_time_args = {batch},
        .config = WriterConfigDescriptor{}};
    TensorAccessorArgs(output.buffer()).append_to(writer.compile_time_args);
    KernelDescriptor compute{
        .kernel_source =
            prefix + (dtype == DataType::BFLOAT4_B ? "compute/prepare_bfp4.cpp" : "compute/prepare_significand.cpp"),
        .core_ranges = grid,
        .compile_time_args = dtype == DataType::BFLOAT4_B ? std::vector<uint32_t>{batch}
                                                          : std::vector<uint32_t>{is_query ? 7u : 5u, batch},
        .config = ComputeConfigDescriptor{
            .math_fidelity = MathFidelity::LoFi,
            .fp32_dest_acc_en = dtype != DataType::BFLOAT4_B,
            .math_approx_mode = false}};
    uint32_t offset = 0;
    for (uint32_t i = 0; i < cores; ++i) {
        const auto core = coordinates[i];
        const uint32_t count = (tiles / batch / cores + (i < tiles / batch % cores)) * batch;
        reader.runtime_args.emplace_back(
            core, KernelDescriptor::CoreRuntimeArgs{input.buffer()->address(), offset, count});
        writer.runtime_args.emplace_back(
            core, KernelDescriptor::CoreRuntimeArgs{output.buffer()->address(), offset, count});
        compute.runtime_args.emplace_back(core, KernelDescriptor::CoreRuntimeArgs{count});
        offset += count;
    }
    program.kernels = {std::move(reader), std::move(writer), std::move(compute)};
    return ttnn::generic_op({input, output}, program);
}

}  // namespace ttnn::transformer
