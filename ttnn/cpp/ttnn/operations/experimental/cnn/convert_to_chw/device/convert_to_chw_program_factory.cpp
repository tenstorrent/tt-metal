// SPDX-FileCopyrightText: © 2024 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include "convert_to_chw_program_factory.hpp"

#include "tt-metalium/tt_backend_api_types.hpp"
#include "ttnn/tensor/types.hpp"

#include <tt-logger/tt-logger.hpp>
#include <tt-metalium/constants.hpp>
#include <tt-metalium/math.hpp>
#include <tt_stl/assert.hpp>

#include <algorithm>
#include <vector>

namespace ttnn::experimental::prim {

using namespace tt::constants;

namespace {
// Helper function to set runtime arguments for reader, writer, and compute kernels
void set_runtime_args_for_all_kernels(
    tt::tt_metal::KernelDescriptor& reader_kernel,
    tt::tt_metal::KernelDescriptor& writer_kernel,
    tt::tt_metal::KernelDescriptor& compute_kernel,
    const std::vector<tt::tt_metal::CoreCoord>& cores,
    uint32_t total_tiles_per_core) {
    const std::vector<uint32_t> runtime_args = {total_tiles_per_core};  // (num_tiles_per_core)
    std::for_each(cores.cbegin(), cores.cend(), [&](const tt::tt_metal::CoreCoord& core) {
        reader_kernel.runtime_args.emplace_back(core, runtime_args);
        writer_kernel.runtime_args.emplace_back(core, runtime_args);
        compute_kernel.runtime_args.emplace_back(core, runtime_args);
    });
}

tt::tt_metal::CBDescriptor make_chw_circular_buffer(
    const tt::tt_metal::CoreRangeSet& core_grid,
    uint32_t index,
    uint32_t total_size,
    uint32_t page_size,
    const tt::DataFormat& format,
    tt::tt_metal::Buffer* buffer) {
    log_debug(
        tt::LogType::LogOp,
        "Creating CB at index {} with total size {} B and page size {} B",
        index,
        total_size,
        page_size);
    return tt::tt_metal::CBDescriptor{
        .total_size = total_size,
        .core_ranges = core_grid,
        .format_descriptors = {{tt::tt_metal::CBFormatDescriptor{
            .buffer_index = static_cast<uint8_t>(index),
            .data_format = format,
            .page_size = page_size,
        }}},
        .buffer = buffer,
    };
}
}  // namespace

tt::tt_metal::ProgramDescriptor ConvertToCHWProgramFactory::create_descriptor(
    const ConvertToCHWParams& /*operation_attributes*/, const Tensor& tensor_args, Tensor& tensor_return_value) {
    tt::tt_metal::ProgramDescriptor desc;

    const auto& a = tensor_args;
    auto& output = tensor_return_value;

    const auto& input_shape = a.logical_shape();
    const auto input_core_grid = a.shard_spec()->grid;
    const auto input_cores = corerange_to_cores(
        input_core_grid, std::nullopt, a.shard_spec()->orientation == tt::tt_metal::ShardOrientation::ROW_MAJOR);

    const auto output_shard_shape = output.shard_spec()->shape;

    const auto HW = input_shape[2];
    const auto C = input_shape[3];

    log_debug(tt::LogType::LogOp, "Running op with HW={}, C={}, shard_shape={}", HW, C, a.shard_spec()->shape);

    TT_FATAL(C <= TILE_HEIGHT, "C must not exceed 32");
    TT_FATAL(
        tt::div_up(HW, a.shard_spec()->shape[0]) == input_cores.size(),
        "Mismatch between core grid and input/shard shapes");

    const uint32_t total_tiles = HW / TILE_HEIGHT;  // assume C < 32
    const uint32_t total_tiles_per_core = tt::div_up(total_tiles, input_cores.size());

    log_debug(tt::LogType::LogOp, "Processing {} tiles per core ({} total tiles)", total_tiles_per_core, total_tiles);

    auto* input_buffer = a.buffer();
    auto* output_buffer = output.buffer();
    TT_FATAL(input_buffer != nullptr, "Input buffer must be allocated on device");
    TT_FATAL(output_buffer != nullptr, "Output buffer must be allocated on device");

    const tt::DataFormat input_format = tt::tt_metal::datatype_to_dataformat_converter(a.dtype());
    const uint32_t input_tile_size = tt::tile_size(input_format);

    const tt::DataFormat intermediary_format = tt::DataFormat::Float16_b;
    const uint32_t intermediary_tile_size = tt::tile_size(intermediary_format);

    const uint32_t cb_in_id = tt::CBIndex::c_0;
    const uint32_t cb_in_total_size = total_tiles_per_core * input_tile_size;
    const uint32_t cb_in_page_size = input_tile_size;
    desc.cbs.push_back(make_chw_circular_buffer(
        input_core_grid, cb_in_id, cb_in_total_size, cb_in_page_size, input_format, input_buffer));

    const tt::DataFormat output_format = tt::tt_metal::datatype_to_dataformat_converter(output.dtype());
    const uint32_t cb_out_id = tt::CBIndex::c_1;
    const uint32_t element_size = tt::datum_size(output_format);
    const uint32_t cb_out_total_size = output_shard_shape[0] * output_shard_shape[1] * element_size;
    const uint32_t cb_out_page_size = output_shard_shape[1] * element_size;
    desc.cbs.push_back(make_chw_circular_buffer(
        input_core_grid, cb_out_id, cb_out_total_size, cb_out_page_size, output_format, output_buffer));

    const uint32_t cb_in_transpose_id = tt::CBIndex::c_2;
    const uint32_t cb_in_transpose_total_size = 16 * intermediary_tile_size;
    const uint32_t cb_in_transpose_page_size = intermediary_tile_size;
    desc.cbs.push_back(make_chw_circular_buffer(
        input_core_grid,
        cb_in_transpose_id,
        cb_in_transpose_total_size,
        cb_in_transpose_page_size,
        intermediary_format,
        nullptr));

    std::vector<uint32_t> reader_compile_time_args = {cb_in_id};
    std::vector<uint32_t> writer_compile_time_args = {cb_in_transpose_id, cb_out_id, C};
    std::vector<uint32_t> compute_compile_time_args = {cb_in_id, cb_in_transpose_id};

    tt::tt_metal::KernelDescriptor reader_kernel;
    reader_kernel.kernel_source =
        "ttnn/cpp/ttnn/operations/experimental/cnn/convert_to_chw/device/kernels/reader_convert_to_chw.cpp";
    reader_kernel.source_type = tt::tt_metal::KernelDescriptor::SourceType::FILE_PATH;
    reader_kernel.core_ranges = input_core_grid;
    reader_kernel.compile_time_args = std::move(reader_compile_time_args);
    reader_kernel.config = tt::tt_metal::ReaderConfigDescriptor{};

    tt::tt_metal::KernelDescriptor writer_kernel;
    writer_kernel.kernel_source =
        "ttnn/cpp/ttnn/operations/experimental/cnn/convert_to_chw/device/kernels/writer_convert_to_chw.cpp";
    writer_kernel.source_type = tt::tt_metal::KernelDescriptor::SourceType::FILE_PATH;
    writer_kernel.core_ranges = input_core_grid;
    writer_kernel.compile_time_args = std::move(writer_compile_time_args);
    writer_kernel.config = tt::tt_metal::WriterConfigDescriptor{};

    tt::tt_metal::KernelDescriptor compute_kernel;
    compute_kernel.kernel_source =
        "ttnn/cpp/ttnn/operations/experimental/cnn/convert_to_chw/device/kernels/convert_to_chw.cpp";
    compute_kernel.source_type = tt::tt_metal::KernelDescriptor::SourceType::FILE_PATH;
    compute_kernel.core_ranges = input_core_grid;
    compute_kernel.compile_time_args = std::move(compute_compile_time_args);
    compute_kernel.config = tt::tt_metal::ComputeConfigDescriptor{
        .math_fidelity = tt::tt_metal::MathFidelity::HiFi4,
        .fp32_dest_acc_en = false,
        .dst_full_sync_en = false,
        .math_approx_mode = false,
    };

    // Set initial runtime args
    set_runtime_args_for_all_kernels(reader_kernel, writer_kernel, compute_kernel, input_cores, total_tiles_per_core);

    desc.kernels.push_back(std::move(reader_kernel));
    desc.kernels.push_back(std::move(writer_kernel));
    desc.kernels.push_back(std::move(compute_kernel));
    return desc;
}

void ConvertToCHWProgramFactory::override_runtime_arguments(
    tt::tt_metal::Program& program,
    const ConvertToCHWParams& /*operation_attributes*/,
    const Tensor& tensor_args,
    Tensor& output,
    const std::optional<ttnn::MeshCoordinate>& /*mesh_dispatch_coordinate*/) {
    // buffers are not provided so we take them from output/input tensors
    auto* input_buffer = tensor_args.buffer();
    auto* output_buffer = output.buffer();
    TT_FATAL(input_buffer != nullptr, "Input buffer must be allocated on device");
    TT_FATAL(output_buffer != nullptr, "Output buffer must be allocated on device");

    // Tile count is keyed by the tensor spec; only the sharded CB addresses move.
    tt::tt_metal::ProgramDescriptor cb_addr_only;
    cb_addr_only.cbs.push_back(tt::tt_metal::CBDescriptor{.buffer = input_buffer});
    cb_addr_only.cbs.push_back(tt::tt_metal::CBDescriptor{.buffer = output_buffer});
    tt::tt_metal::apply_descriptor_runtime_args(program, cb_addr_only);  // override-rebuild-ok: cb-addr-only
}

}  // namespace ttnn::experimental::prim
