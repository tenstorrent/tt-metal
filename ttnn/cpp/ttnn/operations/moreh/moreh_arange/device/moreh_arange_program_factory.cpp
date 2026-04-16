// SPDX-FileCopyrightText: © 2024 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include "moreh_arange_device_operation.hpp"
#include <tt-metalium/tensor_accessor_args.hpp>
#include <tt-metalium/work_split.hpp>

namespace ttnn::operations::moreh::moreh_arange {

using namespace tt::tt_metal;

static constexpr const char* WRITER_KERNEL_TILE =
    "ttnn/cpp/ttnn/operations/moreh/moreh_arange/device/kernels/writer_moreh_arange.cpp";
static constexpr const char* WRITER_KERNEL_RM =
    "ttnn/cpp/ttnn/operations/moreh/moreh_arange/device/kernels/writer_moreh_arange_rm.cpp";

ProgramDescriptor MorehArangeOperation::create_descriptor(
    const operation_attributes_t& operation_attributes,
    const tensor_args_t& /*tensor_args*/,
    tensor_return_value_t& output) {
    auto dtype = output.dtype();
    auto W = output.padded_shape()[-1];
    auto Wt = tt::div_up(W, tt::constants::TILE_WIDTH);

    auto start = operation_attributes.start;
    auto step = operation_attributes.step;

    auto* device = output.device();
    auto grid = device->compute_with_storage_grid_size();
    auto [num_cores, all_cores, core_group_1, core_group_2, num_tiles_per_core_group_1, num_tiles_per_core_group_2] =
        split_work_to_cores(grid, Wt);

    ProgramDescriptor desc;

    // Circular buffer
    auto out_data_format = datatype_to_dataformat_converter(dtype);
    const uint32_t out_tile_size = tile_size(out_data_format);
    constexpr uint32_t cb_id = tt::CBIndex::c_16;

    desc.cbs.push_back(CBDescriptor{
        .total_size = out_tile_size,
        .core_ranges = all_cores,
        .format_descriptors = {{CBFormatDescriptor{
            .buffer_index = cb_id,
            .data_format = out_data_format,
            .page_size = out_tile_size,
        }}},
    });

    // Writer kernel
    KernelDescriptor::Defines writer_defines;
    switch (dtype) {
        case DataType::BFLOAT16: writer_defines.emplace_back("OUTPUT_DTYPE_BFLOAT16", "1"); break;
        case DataType::INT32: writer_defines.emplace_back("OUTPUT_DTYPE_INT32", "1"); break;
        case DataType::FLOAT32: writer_defines.emplace_back("OUTPUT_DTYPE_FLOAT32", "1"); break;
        default: break;
    }

    KernelDescriptor::CompileTimeArgs writer_ct_args;
    TensorAccessorArgs(*output.buffer()).append_to(writer_ct_args);

    KernelDescriptor writer_desc;
    writer_desc.kernel_source = operation_attributes.untilize_out ? WRITER_KERNEL_RM : WRITER_KERNEL_TILE;
    writer_desc.source_type = KernelDescriptor::SourceType::FILE_PATH;
    writer_desc.core_ranges = std::move(all_cores);
    writer_desc.compile_time_args = std::move(writer_ct_args);
    writer_desc.defines = std::move(writer_defines);
    writer_desc.config = WriterConfigDescriptor{};
    writer_desc.runtime_args.reserve(num_cores);

    // Runtime args per core
    uint32_t core_h = grid.y;
    for (uint32_t i = 0, tile_offset = 0; i < num_cores; i++) {
        CoreCoord core = {i / core_h, i % core_h};
        uint32_t num_tiles_per_core;
        if (core_group_1.contains(core)) {
            num_tiles_per_core = num_tiles_per_core_group_1;
        } else if (core_group_2.contains(core)) {
            num_tiles_per_core = num_tiles_per_core_group_2;
        } else {
            TT_FATAL(false, "Core not in specified core ranges");
        }

        writer_desc.runtime_args.emplace_back(
            core,
            KernelDescriptor::CoreRuntimeArgs{
                output.buffer()->address(),
                tile_offset,
                num_tiles_per_core,
                *reinterpret_cast<uint32_t*>(&start),
                *reinterpret_cast<uint32_t*>(&step),
                output.element_size()});

        tile_offset += num_tiles_per_core;
    }

    desc.kernels.push_back(std::move(writer_desc));

    return desc;
}

}  // namespace ttnn::operations::moreh::moreh_arange
