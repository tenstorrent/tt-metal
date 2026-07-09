// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include "copy_device_operation.hpp"

#include <cmath>

#include <tt-metalium/constants.hpp>
#include <tt-metalium/hal.hpp>
#include <tt-metalium/host_api.hpp>
#include <tt-metalium/math.hpp>
#include <tt-metalium/program_descriptors.hpp>
#include <tt-metalium/tensor_accessor_args.hpp>
#include <tt-metalium/tt_align.hpp>
#include <tt-metalium/work_split.hpp>

namespace ttnn::prim {

using namespace tt::constants;
using namespace tt::tt_metal;

namespace {

constexpr const char* KERNEL_READER_INTERLEAVED =
    "ttnn/cpp/ttnn/operations/eltwise/unary/device/kernels/dataflow/reader_unary_interleaved_start_id.cpp";
constexpr const char* KERNEL_WRITER_INTERLEAVED =
    "ttnn/cpp/ttnn/operations/eltwise/unary/device/kernels/dataflow/writer_unary_interleaved_start_id.cpp";
constexpr const char* KERNEL_COMPUTE_ELTWISE_COPY =
    "ttnn/cpp/ttnn/operations/data_movement/sharded/device/kernels/compute/eltwise_copy.cpp";

}  // namespace

ProgramDescriptor CopyDeviceOperation::DefaultTilized::create_descriptor(
    const operation_attributes_t& /*operation_attributes*/,
    const tensor_args_t& tensor_args,
    tensor_return_value_t& output_tensor) {
    const auto& input = tensor_args.input;
    const auto& output = output_tensor;

    ProgramDescriptor desc;

    auto* device = input.device();
    auto compute_with_storage_grid_size =
        device->compute_with_storage_grid_size();  // This can be replaced with get_worker_cores in subdevices

    const auto& logical_shape = input.logical_shape();
    const auto& tile = input.tensor_spec().tile();
    const uint32_t tile_height = tile.get_height();
    const uint32_t tile_width = tile.get_width();
    const uint32_t rank = logical_shape.rank();
    uint32_t total_tiles = 1;
    if (rank >= 1) {
        total_tiles = tt::div_up(logical_shape[-1], tile_width);
    }
    if (rank >= 2) {
        total_tiles *= tt::div_up(logical_shape[-2], tile_height);
    }
    for (uint32_t i = 0; i + 2 < rank; ++i) {
        total_tiles *= logical_shape[i];
    }
    auto [num_cores, all_cores, core_group_1, core_group_2, num_tiles_per_core_group_1, num_tiles_per_core_group_2] =
        tt::tt_metal::split_work_to_cores(compute_with_storage_grid_size, total_tiles);
    std::vector<CoreCoord> ordered_cores = corerange_to_cores(all_cores, num_cores, true);

    // Configuring the CB that store input pages

    const auto input_pages_cb_index = tt::CBIndex::c_0;
    auto output_page_cb_index = input_pages_cb_index;
    const auto input_cb_data_format = tt::tt_metal::datatype_to_dataformat_converter(input.dtype());
    const auto output_cb_data_format = tt::tt_metal::datatype_to_dataformat_converter(output.dtype());

    const auto& output_tile = output.tensor_spec().tile();
    // Prefer tile-derived page size so tiny tiles are not sized as 32x32 via buffer defaults alone.
    const auto input_page_size = tile.get_tile_size(input_cb_data_format);
    const auto aligned_input_page_size = tt::align(input_page_size, input.buffer()->alignment());
    desc.cbs.push_back(CBDescriptor{
        .total_size = 2 * aligned_input_page_size,
        .core_ranges = all_cores,
        .format_descriptors = {{CBFormatDescriptor{
            .buffer_index = static_cast<uint8_t>(input_pages_cb_index),
            .data_format = input_cb_data_format,
            .page_size = aligned_input_page_size,
            .tile = TileDescriptor(tile),
        }}},
    });

    bool convert_df = input_cb_data_format != output_cb_data_format;

    if (convert_df) {
        // Configuring the CB that stores output pages if we need to convert data formats through the compute kernel
        output_page_cb_index = tt::CBIndex::c_16;
        const auto output_page_size = output_tile.get_tile_size(output_cb_data_format);
        const auto aligned_output_page_size = tt::align(output_page_size, output.buffer()->alignment());
        // Since we are double buffering, the output page_size must be
        // aligned so the noc_write reads from an aligned address in the CB

        desc.cbs.push_back(CBDescriptor{
            .total_size = 2 * aligned_output_page_size,
            .core_ranges = all_cores,
            .format_descriptors = {{CBFormatDescriptor{
                .buffer_index = static_cast<uint8_t>(output_page_cb_index),
                .data_format = output_cb_data_format,
                .page_size = aligned_output_page_size,
                .tile = TileDescriptor(output_tile),
            }}},
        });
    }
    // Reader kernel config with compile-time args
    KernelDescriptor::CompileTimeArgs reader_compile_time_args;
    tt::tt_metal::TensorAccessorArgs(input.buffer()).append_to(reader_compile_time_args);

    // Writer kernel config with compile-time args
    KernelDescriptor::CompileTimeArgs writer_compile_time_args = {static_cast<uint32_t>(output_page_cb_index)};

    tt::tt_metal::TensorAccessorArgs(output.buffer()).append_to(writer_compile_time_args);

    KernelDescriptor reader_desc;
    reader_desc.kernel_source = KERNEL_READER_INTERLEAVED;
    reader_desc.source_type = KernelDescriptor::SourceType::FILE_PATH;
    reader_desc.core_ranges = all_cores;
    reader_desc.compile_time_args = std::move(reader_compile_time_args);
    reader_desc.config = ReaderConfigDescriptor{};

    KernelDescriptor writer_desc;
    writer_desc.kernel_source = KERNEL_WRITER_INTERLEAVED;
    writer_desc.source_type = KernelDescriptor::SourceType::FILE_PATH;
    writer_desc.core_ranges = all_cores;
    writer_desc.compile_time_args = std::move(writer_compile_time_args);
    writer_desc.config = WriterConfigDescriptor{};

    KernelDescriptor compute_desc;
    const bool use_compute = convert_df;
    if (use_compute) {
        compute_desc.kernel_source = KERNEL_COMPUTE_ELTWISE_COPY;
        compute_desc.source_type = KernelDescriptor::SourceType::FILE_PATH;
        compute_desc.core_ranges = all_cores;
        compute_desc.compile_time_args = {};
        compute_desc.config = ComputeConfigDescriptor{};
    }

    // Set runtime args
    uint32_t start_tile_id = 0;
    for (const auto& core : ordered_cores) {
        uint32_t num_tiles_to_process = num_tiles_per_core_group_1;
        if (core_group_2.contains(core)) {
            num_tiles_to_process = num_tiles_per_core_group_2;
        }
        // Reader run-time args
        reader_desc.emplace_runtime_args(core, {input.buffer(), num_tiles_to_process, start_tile_id});
        writer_desc.emplace_runtime_args(core, {output.buffer(), num_tiles_to_process, start_tile_id});
        start_tile_id += num_tiles_to_process;
        // Set run-time arg
        if (use_compute) {
            compute_desc.emplace_runtime_args(core, {num_tiles_to_process});
        }
    }

    desc.kernels.push_back(std::move(reader_desc));
    desc.kernels.push_back(std::move(writer_desc));
    if (use_compute) {
        desc.kernels.push_back(std::move(compute_desc));
    }

    return desc;
}

}  // namespace ttnn::prim
