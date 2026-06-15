// SPDX-FileCopyrightText: © 2024 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include "convert_to_chw_program_factory.hpp"

#include <tt-metalium/host_api.hpp>
#include <tt-metalium/program_descriptors.hpp>
#include "tt-metalium/tt_backend_api_types.hpp"
#include "ttnn/tensor/types.hpp"

namespace ttnn::experimental::prim {

using namespace tt::constants;
using namespace tt::tt_metal;

tt::tt_metal::ProgramDescriptor ConvertToCHWProgramFactory::create_descriptor(
    const ConvertToCHWParams& /*operation_attributes*/, const Tensor& tensor_args, Tensor& tensor_return_value) {
    ProgramDescriptor desc;

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

    const tt::DataFormat input_format = tt::tt_metal::datatype_to_dataformat_converter(a.dtype());
    const uint32_t input_tile_size = tt::tile_size(input_format);

    const tt::DataFormat intermediary_format = tt::DataFormat::Float16_b;
    const uint32_t intermediary_tile_size = tt::tile_size(intermediary_format);

    // ---- Circular buffers ----
    // cb_in / cb_out are sharded onto the input / output tensor buffers; binding
    // via .buffer triggers UpdateDynamicCircularBufferAddress on cache-hit.
    const uint32_t cb_in_id = tt::CBIndex::c_0;
    const uint32_t cb_in_total_size = total_tiles_per_core * input_tile_size;
    const uint32_t cb_in_page_size = input_tile_size;
    desc.cbs.push_back(CBDescriptor{
        .total_size = cb_in_total_size,
        .core_ranges = input_core_grid,
        .format_descriptors = {{CBFormatDescriptor{
            .buffer_index = cb_in_id,
            .data_format = input_format,
            .page_size = cb_in_page_size,
        }}},
        .buffer = a.buffer(),
    });

    const tt::DataFormat output_format = tt::tt_metal::datatype_to_dataformat_converter(output.dtype());
    const uint32_t cb_out_id = tt::CBIndex::c_1;
    const uint32_t element_size = tt::datum_size(output_format);
    const uint32_t cb_out_total_size = output_shard_shape[0] * output_shard_shape[1] * element_size;
    const uint32_t cb_out_page_size = output_shard_shape[1] * element_size;
    desc.cbs.push_back(CBDescriptor{
        .total_size = cb_out_total_size,
        .core_ranges = input_core_grid,
        .format_descriptors = {{CBFormatDescriptor{
            .buffer_index = cb_out_id,
            .data_format = output_format,
            .page_size = cb_out_page_size,
        }}},
        .buffer = output.buffer(),
    });

    const uint32_t cb_in_transpose_id = tt::CBIndex::c_2;
    const uint32_t cb_in_transpose_total_size = 16 * intermediary_tile_size;
    const uint32_t cb_in_transpose_page_size = intermediary_tile_size;
    desc.cbs.push_back(CBDescriptor{
        .total_size = cb_in_transpose_total_size,
        .core_ranges = input_core_grid,
        .format_descriptors = {{CBFormatDescriptor{
            .buffer_index = cb_in_transpose_id,
            .data_format = intermediary_format,
            .page_size = cb_in_transpose_page_size,
        }}},
    });

    // ---- Kernels ----
    std::vector<uint32_t> reader_compile_time_args = {cb_in_id};
    std::vector<uint32_t> writer_compile_time_args = {cb_in_transpose_id, cb_out_id, C};
    std::vector<uint32_t> compute_compile_time_args = {cb_in_id, cb_in_transpose_id};

    KernelDescriptor reader_desc;
    reader_desc.kernel_source =
        "ttnn/cpp/ttnn/operations/experimental/cnn/convert_to_chw/device/kernels/reader_convert_to_chw.cpp";
    reader_desc.source_type = KernelDescriptor::SourceType::FILE_PATH;
    reader_desc.core_ranges = input_core_grid;
    reader_desc.compile_time_args = std::move(reader_compile_time_args);
    reader_desc.config = ReaderConfigDescriptor{};

    KernelDescriptor writer_desc;
    writer_desc.kernel_source =
        "ttnn/cpp/ttnn/operations/experimental/cnn/convert_to_chw/device/kernels/writer_convert_to_chw.cpp";
    writer_desc.source_type = KernelDescriptor::SourceType::FILE_PATH;
    writer_desc.core_ranges = input_core_grid;
    writer_desc.compile_time_args = std::move(writer_compile_time_args);
    writer_desc.config = WriterConfigDescriptor{};

    KernelDescriptor compute_desc;
    compute_desc.kernel_source =
        "ttnn/cpp/ttnn/operations/experimental/cnn/convert_to_chw/device/kernels/convert_to_chw.cpp";
    compute_desc.source_type = KernelDescriptor::SourceType::FILE_PATH;
    compute_desc.core_ranges = input_core_grid;
    compute_desc.compile_time_args = std::move(compute_compile_time_args);
    compute_desc.config = ComputeConfigDescriptor{
        .math_fidelity = MathFidelity::HiFi4,
        .fp32_dest_acc_en = false,
        .math_approx_mode = false,
    };

    // ---- Per-core runtime args ----
    // Each core gets the same (total_tiles_per_core) on reader/writer/compute.
    reader_desc.runtime_args.reserve(input_cores.size());
    writer_desc.runtime_args.reserve(input_cores.size());
    compute_desc.runtime_args.reserve(input_cores.size());
    for (const auto& core : input_cores) {
        reader_desc.runtime_args.emplace_back(core, std::vector<uint32_t>{total_tiles_per_core});
        writer_desc.runtime_args.emplace_back(core, std::vector<uint32_t>{total_tiles_per_core});
        compute_desc.runtime_args.emplace_back(core, std::vector<uint32_t>{total_tiles_per_core});
    }

    desc.kernels.push_back(std::move(reader_desc));
    desc.kernels.push_back(std::move(writer_desc));
    desc.kernels.push_back(std::move(compute_desc));

    return desc;
}

}  // namespace ttnn::experimental::prim
