// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include "hello_world_program_factory.hpp"

#include <cstdint>
#include <string_view>

#include <tt-logger/tt-logger.hpp>
#include <tt-metalium/constants.hpp>
#include <tt-metalium/tensor_accessor_args.hpp>
#include <tt-metalium/work_split.hpp>

namespace ttnn::experimental::prim {
namespace {

constexpr auto kReaderKernelPath =
    "ttnn/cpp/ttnn/operations/experimental/hello_world/device/kernels/dataflow/reader_hello_world_interleaved.cpp";
constexpr auto kWriterKernelPath =
    "ttnn/cpp/ttnn/operations/experimental/hello_world/device/kernels/dataflow/writer_hello_world_interleaved.cpp";
constexpr auto kComputeKernelPath =
    "ttnn/cpp/ttnn/operations/experimental/hello_world/device/kernels/compute/hello_world_compute.cpp";

constexpr auto kSrc0CbIndex = tt::CBIndex::c_0;
constexpr auto kOutputCbIndex = tt::CBIndex::c_2;

// Create and configure a circular buffer descriptor.
inline void create_circular_buffer(
    tt::tt_metal::ProgramDescriptor& descriptor,
    const tt::tt_metal::CoreRangeSet& core_ranges,
    uint32_t cb_index,
    tt::DataFormat data_format,
    uint32_t single_tile_size,
    uint32_t num_tiles) {
    using namespace tt::tt_metal;
    descriptor.cbs.push_back(CBDescriptor{
        .total_size = num_tiles * single_tile_size,
        .core_ranges = core_ranges,
        .format_descriptors = {{CBFormatDescriptor{
            .buffer_index = cb_index,
            .data_format = data_format,
            .page_size = single_tile_size,
        }}},
    });
}

// Create a reader kernel descriptor with the given compile-time arguments.
inline tt::tt_metal::KernelDescriptor create_reader_kernel(
    const tt::tt_metal::CoreRangeSet& core_ranges,
    tt::tt_metal::KernelDescriptor::CompileTimeArgs&& compile_time_args,
    std::string_view kernel_path) {
    using namespace tt::tt_metal;
    KernelDescriptor descriptor;
    descriptor.kernel_source = kernel_path;
    descriptor.source_type = KernelDescriptor::SourceType::FILE_PATH;
    descriptor.core_ranges = core_ranges;
    descriptor.compile_time_args = std::move(compile_time_args);
    descriptor.config = ReaderConfigDescriptor{};
    return descriptor;
}

// Create a writer kernel descriptor with the given compile-time arguments.
inline tt::tt_metal::KernelDescriptor create_writer_kernel(
    const tt::tt_metal::CoreRangeSet& core_ranges,
    tt::tt_metal::KernelDescriptor::CompileTimeArgs&& compile_time_args,
    std::string_view kernel_path) {
    using namespace tt::tt_metal;
    KernelDescriptor descriptor;
    descriptor.kernel_source = kernel_path;
    descriptor.source_type = KernelDescriptor::SourceType::FILE_PATH;
    descriptor.core_ranges = core_ranges;
    descriptor.compile_time_args = std::move(compile_time_args);
    descriptor.config = WriterConfigDescriptor{};
    return descriptor;
}

// Create a compute kernel descriptor.
inline tt::tt_metal::KernelDescriptor create_compute_kernel(
    const tt::tt_metal::CoreRangeSet& core_ranges, std::string_view kernel_path, bool fp32_dest_acc_en) {
    using namespace tt::tt_metal;
    KernelDescriptor descriptor;
    descriptor.kernel_source = kernel_path;
    descriptor.source_type = KernelDescriptor::SourceType::FILE_PATH;
    descriptor.core_ranges = core_ranges;
    descriptor.config = ComputeConfigDescriptor{
        .math_fidelity = tt::tt_metal::MathFidelity::HiFi4,
        // The FPU's DST tile register is 16-bit (bf16) unless fp32_dest_acc_en is set, so a pure
        // copy of fp32 data would be quantized to bf16 on its way into DST. The unary ops enable
        // this for fp32 tensors (see operations/eltwise/unary/unary.cpp); an identity op must too.
        .fp32_dest_acc_en = fp32_dest_acc_en,
        .dst_full_sync_en = false,
        .math_approx_mode = false,
    };
    return descriptor;
}

}  // namespace

tt::tt_metal::ProgramDescriptor HelloWorldProgramFactory::create_descriptor(
    const HelloWorldParams&, const HelloWorldInputs& tensor_args, Tensor& output) {
    using namespace tt;
    using namespace tt::tt_metal;

    const auto& input = tensor_args.input;
    log_debug(
        tt::LogOp,
        "[hello_world] create_descriptor: input shape={}, dtype={}",
        input.padded_shape(),
        static_cast<int>(input.dtype()));

    ProgramDescriptor descriptor{};

    // The CB data format must match the tensor dtype.
    const tt::DataFormat data_fmt_in = datatype_to_dataformat_converter(input.dtype());
    const tt::DataFormat data_fmt_out = datatype_to_dataformat_converter(output.dtype());
    const uint32_t tile_size = tt::tile_size(data_fmt_in);

    // Distribute the input's tiles across as many cores as needed; kernels are
    // only created on the cores that get work.
    const CoreCoord grid = input.device()->compute_with_storage_grid_size();
    const uint32_t num_tiles = input.physical_volume() / constants::TILE_HW;
    auto [num_cores, all_cores, core_group_1, core_group_2, tiles_group_1, tiles_group_2] =
        split_work_to_cores(grid, num_tiles);
    log_debug(
        tt::LogOp,
        "[hello_world] create_descriptor: {} tiles split across {} cores (grid {}x{})",
        num_tiles,
        num_cores,
        grid.x,
        grid.y);

    // One tile in flight per core: single-tile ublocks, CB depth 1.
    create_circular_buffer(descriptor, all_cores, kSrc0CbIndex, data_fmt_in, tile_size, 1);
    create_circular_buffer(descriptor, all_cores, kOutputCbIndex, data_fmt_out, tile_size, 1);
    log_debug(
        tt::LogOp,
        "[hello_world] create_descriptor: circular buffers c_0 (input) and c_2 (output), page size {} bytes",
        tile_size);

    // Bind the tensors into the reader/writer kernels' compile-time args. The kernels
    // read the binding via TensorAccessorArgs<1> (index 1; index 0 is the CB index).
    // The tensor's base address is carried separately as a Buffer* runtime-arg binding
    // at runtime arg 0 (see the per-core loop below), which the program cache re-points
    // on a hit.
    auto* src_buffer = input.buffer();
    KernelDescriptor::CompileTimeArgs reader_compile_args = {static_cast<uint32_t>(kSrc0CbIndex)};
    TensorAccessorArgs(src_buffer).append_to(reader_compile_args);

    auto* dst_buffer = output.buffer();
    KernelDescriptor::CompileTimeArgs writer_compile_args = {static_cast<uint32_t>(kOutputCbIndex)};
    TensorAccessorArgs(dst_buffer).append_to(writer_compile_args);

    KernelDescriptor reader = create_reader_kernel(all_cores, std::move(reader_compile_args), kReaderKernelPath);
    KernelDescriptor writer = create_writer_kernel(all_cores, std::move(writer_compile_args), kWriterKernelPath);
    // fp32 data needs the 32-bit DST register (see create_compute_kernel); bf16 works either way.
    const bool fp32_dest_acc_en = (input.dtype() == DataType::FLOAT32);
    log_debug(tt::LogOp, "[hello_world] create_descriptor: compute kernel (fp32_dest_acc_en={})", fp32_dest_acc_en);
    KernelDescriptor compute = create_compute_kernel(all_cores, kComputeKernelPath, fp32_dest_acc_en);
    log_debug(
        tt::LogOp,
        "[hello_world] create_descriptor: kernels created (reader: DRAM -> c_0, compute: c_0 -> c_2, writer: c_2 -> "
        "DRAM)");

    // Per-core runtime args. The Buffer* entries register as buffer bindings so the
    // program cache can re-point them at the current src/dst buffers on a cache hit
    // (the output is a fresh allocation on every call).
    uint32_t tile_offset = 0;
    for (uint32_t i = 0; i < num_cores; ++i) {
        const CoreCoord core{static_cast<uint32_t>(i / grid.y), static_cast<uint32_t>(i % grid.y)};
        const uint32_t core_tiles = core_group_1.contains(core) ? tiles_group_1 : tiles_group_2;
        reader.emplace_runtime_args(core, {src_buffer, core_tiles, tile_offset});
        writer.emplace_runtime_args(core, {dst_buffer, core_tiles, tile_offset});
        // The compute kernel DPRINTs this core's coordinates (the MATH trisc has no
        // built-in coordinate getter), so they ride along as runtime args 1 and 2.
        compute.runtime_args.emplace_back(core, KernelDescriptor::CoreRuntimeArgs{core_tiles, core.x, core.y});
        tile_offset += core_tiles;
    }
    log_debug(tt::LogOp, "[hello_world] create_descriptor: per-core runtime args assigned for {} cores", num_cores);

    descriptor.kernels.push_back(std::move(reader));
    descriptor.kernels.push_back(std::move(writer));
    descriptor.kernels.push_back(std::move(compute));

    log_debug(tt::LogOp, "[hello_world] create_descriptor: program built (3 kernels)");
    return descriptor;
}

}  // namespace ttnn::experimental::prim
