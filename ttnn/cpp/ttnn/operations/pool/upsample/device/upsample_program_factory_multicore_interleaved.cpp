// SPDX-FileCopyrightText: © 2023 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include <cmath>
#include <cstdint>
#include <string>

#include <tt-metalium/host_api.hpp>
#include <tt-metalium/hal.hpp>
#include <tt-metalium/math.hpp>
#include <tt-metalium/program_descriptors.hpp>
#include <tt-metalium/tensor_accessor_args.hpp>
#include <tt-metalium/work_split.hpp>
#include "ttnn/operations/math.hpp"
#include "ttnn/operations/pool/upsample/device/upsample_device_operation.hpp"
#include "ttnn/operations/pool/upsample/device/upsample_common.hpp"

namespace ttnn::prim {

using namespace tt::tt_metal;

ProgramDescriptor UpsampleMultiCoreInterleavedProgramFactory::create_descriptor(
    const UpsampleParams& operation_attributes, const Tensor& input_tensor, Tensor& output_tensor) {
    const auto& input = input_tensor;
    auto& output = output_tensor;
    // This factory only supports integer scale factors
    TT_FATAL(
        operations::pool::upsample::is_integer_scale(operation_attributes.scale_factor_h) &&
            operations::pool::upsample::is_integer_scale(operation_attributes.scale_factor_w),
        "Interleaved upsample factory requires integer scale factors, got scale_h={}, scale_w={}",
        operation_attributes.scale_factor_h,
        operation_attributes.scale_factor_w);
    const uint32_t scale_factor_h = static_cast<uint32_t>(operation_attributes.scale_factor_h);
    const uint32_t scale_factor_w = static_cast<uint32_t>(operation_attributes.scale_factor_w);

    const bool is_tiled_layout = (input.layout() == Layout::TILE);

    const tt::DataFormat input_cb_data_format = datatype_to_dataformat_converter(input.dtype());
    const tt::DataFormat output_cb_data_format = datatype_to_dataformat_converter(output.dtype());

    const auto& output_shape = output.padded_shape();
    IDevice* const device = output.device();

    const auto compute_with_storage_grid_size = device->compute_with_storage_grid_size();
    const uint32_t num_cores_y = compute_with_storage_grid_size.y;

    // Declare variables that will be set based on layout
    uint32_t input_unit_size;
    uint32_t output_unit_size;
    uint32_t input_cb_required_pages;
    uint32_t work_units_to_split;
    uint32_t aligned_input_unit_size;  // Size used for CB creation

    if (is_tiled_layout) {
        // Tiled layout specific calculations
        input_unit_size = tt::tile_size(input_cb_data_format);
        output_unit_size = tt::tile_size(output_cb_data_format);
        aligned_input_unit_size = input_unit_size;

        const uint32_t input_tensor_width = input.padded_shape()[-1];
        const uint32_t input_tensor_height = input.physical_volume() / input_tensor_width;

        const auto& tile_shape = input.tensor_spec().tile().get_tile_shape();
        const uint32_t tile_height = tile_shape[0];
        const uint32_t tile_width = tile_shape[1];

        const uint32_t num_input_tiles_in_row = input_tensor_width / tile_width;
        const uint32_t num_input_tiles_in_col = input_tensor_height / tile_height;

        /*
        For tiled layout, a unit of work (input wise) is a row of tiles
        */

        input_cb_required_pages = num_input_tiles_in_row;  // for CB sizing
        work_units_to_split = num_input_tiles_in_col;      // for work splitting
    } else {
        // Row-major layout specific calculations
        input_unit_size = input.padded_shape()[-1] * input.element_size();
        output_unit_size = output.padded_shape()[-1] * output.element_size();
        aligned_input_unit_size = tt::round_up(input_unit_size, hal::get_dram_alignment());

        /*
        For Row-major layout, a unit of work is one row (stick) of the input tensor
        */

        input_cb_required_pages = 1;                                               // One input unit is required in CB
        work_units_to_split = input.physical_volume() / input.padded_shape()[-1];  // N*H*W unit split
    }

    const auto [num_cores, all_cores, core_group_1, core_group_2, work_per_core_group_1, work_per_core_group_2] =
        split_work_to_cores(compute_with_storage_grid_size, work_units_to_split);

    ProgramDescriptor desc;

    // Create circular buffers
    uint32_t next_cb_index = tt::CBIndex::c_0;

    uint32_t num_pages_in_input_cb = input_cb_required_pages;
    if (work_per_core_group_1 != 1) {
        // Double buffer if the core is processing 2+ blocks
        num_pages_in_input_cb *= 2;
    }

    const uint32_t src0_cb_index = next_cb_index++;
    desc.cbs.push_back(CBDescriptor{
        .total_size = aligned_input_unit_size * num_pages_in_input_cb,
        .core_ranges = all_cores,
        .format_descriptors = {{CBFormatDescriptor{
            .buffer_index = static_cast<uint8_t>(src0_cb_index),
            .data_format = input_cb_data_format,
            .page_size = aligned_input_unit_size,
        }}},
    });

    uint32_t output_cb_index = src0_cb_index;
    if (is_tiled_layout) {
        // Separate output CB for tiled
        const uint32_t num_pages_in_output_cb = num_pages_in_input_cb;
        output_cb_index = next_cb_index++;
        desc.cbs.push_back(CBDescriptor{
            .total_size = output_unit_size * num_pages_in_output_cb,
            .core_ranges = all_cores,
            .format_descriptors = {{CBFormatDescriptor{
                .buffer_index = static_cast<uint8_t>(output_cb_index),
                .data_format = output_cb_data_format,
                .page_size = output_unit_size,
            }}},
        });
    }

    auto* const src_buffer = input.buffer();
    auto* const dst_buffer = output.buffer();

    KernelDescriptor::CompileTimeArgs reader_compile_time_args = {
        src0_cb_index,
        aligned_input_unit_size,
    };
    TensorAccessorArgs(*src_buffer).append_to(reader_compile_time_args);

    KernelDescriptor reader_desc;
    reader_desc.kernel_source =
        "ttnn/cpp/ttnn/operations/pool/upsample/device/kernels/dataflow/"
        "reader_upsample_unary_stick_layout_interleaved_start_id.cpp";
    reader_desc.source_type = KernelDescriptor::SourceType::FILE_PATH;
    reader_desc.core_ranges = all_cores;
    reader_desc.compile_time_args = std::move(reader_compile_time_args);
    reader_desc.config = ReaderConfigDescriptor{};

    // Writer compile time arguments
    const int32_t writer_unit_size = output.padded_shape()[-1] * output.element_size();

    KernelDescriptor::CompileTimeArgs writer_compile_time_args = {
        output_cb_index,
        static_cast<uint32_t>(writer_unit_size),
        scale_factor_h,
        scale_factor_w,
        output_shape[1],
        output_shape[2],
    };

    if (is_tiled_layout) {
        const auto& tile_shape = input.tensor_spec().tile().get_tile_shape();
        const uint32_t tile_height = tile_shape[0];
        const uint32_t num_input_tiles_in_row = input.padded_shape()[-1] / tile_shape[1];

        // tile_height rows need to be processed at a time
        writer_compile_time_args.push_back(tile_height);
        // whole row of tiles needs to be processed to get valid output sticks
        writer_compile_time_args.push_back(num_input_tiles_in_row);
    } else {
        const uint32_t block_height = 1;                // since input is row major, blocks are just one row tall
        const uint32_t num_units_per_output_stick = 1;  // 1 page in out_cb is needed to get a valid output stick
        writer_compile_time_args.push_back(block_height);
        writer_compile_time_args.push_back(num_units_per_output_stick);
    }

    TensorAccessorArgs(*dst_buffer).append_to(writer_compile_time_args);

    KernelDescriptor writer_desc;
    writer_desc.kernel_source =
        "ttnn/cpp/ttnn/operations/pool/upsample/device/kernels/dataflow/writer_upsample_interleaved.cpp";
    writer_desc.source_type = KernelDescriptor::SourceType::FILE_PATH;
    writer_desc.core_ranges = all_cores;
    writer_desc.compile_time_args = std::move(writer_compile_time_args);
    writer_desc.config = WriterConfigDescriptor{};

    // Compute kernel (only for tiled layout)
    std::optional<KernelDescriptor> compute_desc_g1;
    std::optional<KernelDescriptor> compute_desc_g2;
    if (is_tiled_layout) {
        const uint32_t num_input_tiles_in_row =
            input.padded_shape()[-1] / input.tensor_spec().tile().get_tile_shape()[1];

        if (core_group_1.num_cores() > 0) {
            KernelDescriptor compute_desc;
            compute_desc.kernel_source =
                "ttnn/cpp/ttnn/operations/data_movement/untilize/device/kernels/compute/untilize.cpp";
            compute_desc.source_type = KernelDescriptor::SourceType::FILE_PATH;
            compute_desc.core_ranges = core_group_1;
            compute_desc.compile_time_args = {
                work_per_core_group_1,   // per_core_block_cnt (compile-time)
                num_input_tiles_in_row,  // per_block_ntiles
                src0_cb_index,           // src_cb_id
                output_cb_index,         // out_cb_id
            };
            compute_desc.config = ComputeConfigDescriptor{};
            compute_desc_g1 = std::move(compute_desc);
        }

        if (core_group_2.num_cores() > 0) {
            KernelDescriptor compute_desc;
            compute_desc.kernel_source =
                "ttnn/cpp/ttnn/operations/data_movement/untilize/device/kernels/compute/untilize.cpp";
            compute_desc.source_type = KernelDescriptor::SourceType::FILE_PATH;
            compute_desc.core_ranges = core_group_2;
            compute_desc.compile_time_args = {
                work_per_core_group_2,
                num_input_tiles_in_row,
                src0_cb_index,
                output_cb_index,
            };
            compute_desc.config = ComputeConfigDescriptor{};
            compute_desc_g2 = std::move(compute_desc);
        }
    }

    // Per-core runtime args
    /*
    For tiled input, a block refers to a row of input tiles
    For row-major input, a block refers to a single input row (stick)
    */
    for (uint32_t i = 0, blocks_processed = 0; i < num_cores; i++) {
        const CoreCoord core = {i / num_cores_y, i % num_cores_y};
        uint32_t blocks_per_core = 0;
        if (core_group_1.contains(core)) {
            blocks_per_core = work_per_core_group_1;
        } else if (core_group_2.contains(core)) {
            blocks_per_core = work_per_core_group_2;
        } else {
            TT_ASSERT(false, "Core not in specified core ranges");
        }

        const uint32_t reader_units = blocks_per_core * input_cb_required_pages;
        const uint32_t reader_start = blocks_processed * input_cb_required_pages;

        reader_desc.emplace_runtime_args(
            core,
            {
                src_buffer,
                reader_units,
                reader_start,
            });
        writer_desc.emplace_runtime_args(
            core,
            {
                dst_buffer,
                blocks_per_core,
                blocks_processed,
            });

        blocks_processed += blocks_per_core;
    }

    desc.kernels.push_back(std::move(reader_desc));
    desc.kernels.push_back(std::move(writer_desc));
    if (compute_desc_g1) {
        desc.kernels.push_back(std::move(*compute_desc_g1));
    }
    if (compute_desc_g2) {
        desc.kernels.push_back(std::move(*compute_desc_g2));
    }

    return desc;
}

}  // namespace ttnn::prim
