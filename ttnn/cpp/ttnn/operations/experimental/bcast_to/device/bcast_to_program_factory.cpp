// SPDX-FileCopyrightText: © 2025 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include <cstdint>
#include <string>
#include <vector>

#include <tt-metalium/tt_backend_api_types.hpp>
#include <tt-metalium/work_split.hpp>
#include <tt-metalium/host_api.hpp>
#include "hostdevcommon/kernel_structs.h"
#include <tt-metalium/buffer.hpp>
#include <tt-metalium/circular_buffer_config.hpp>
#include <tt-metalium/kernel_types.hpp>
#include <tt-metalium/program_descriptors.hpp>
#include "ttnn/tensor/types.hpp"
#include "ttnn/operations/cb_utils.hpp"

#include "bcast_to_device_operation.hpp"
#include "bcast_to_utils.hpp"
#include <tt-metalium/tensor_accessor_args.hpp>

using namespace ttnn::operations::experimental::broadcast_to;

namespace ttnn::operations::experimental::broadcast_to {
using namespace tt::tt_metal;

std::tuple<uint32_t, uint32_t, uint32_t, uint32_t> extract_shape_dims(const Tensor& x) {
    const auto& shape = x.padded_shape();
    const auto& tile = x.tensor_spec().tile();
    return {shape[-4], shape[-3], shape[-2] / tile.get_height(), shape[-1] / tile.get_width()};
}

ProgramDescriptor BcastToOperation::BcastToTileFactory::create_descriptor(
    const operation_attributes_t& operation_attributes,
    const tensor_args_t& tensor_args,
    tensor_return_value_t& output) {
    auto input = tensor_args.input;
    tt::DataFormat input_data_format = datatype_to_dataformat_converter(input.dtype());

    uint32_t input_single_tile_size = tt::tile_size(input_data_format);

    auto* device = input.device();
    ProgramDescriptor desc;

    // we parallelize the computation across the output tiles
    auto compute_with_storage_grid_size = device->compute_with_storage_grid_size();
    uint32_t num_cores_x = compute_with_storage_grid_size.x;
    uint32_t num_cores_y = compute_with_storage_grid_size.y;
    auto all_device_cores = CoreRangeSet(CoreRange({0, 0}, {num_cores_x - 1, num_cores_y - 1}));

    // How many tiles to store per input CB (double buffer)
    constexpr uint32_t num_tiles_per_cb = 2;
    desc.cbs.push_back(CBDescriptor{
        .total_size = num_tiles_per_cb * input_single_tile_size,
        .core_ranges = all_device_cores,
        .format_descriptors = {{CBFormatDescriptor{
            .buffer_index = tt::CBIndex::c_0,
            .data_format = input_data_format,
            .page_size = input_single_tile_size,
        }}},
    });
    desc.cbs.push_back(CBDescriptor{
        .total_size = num_tiles_per_cb * input_single_tile_size,
        .core_ranges = all_device_cores,
        .format_descriptors = {{CBFormatDescriptor{
            .buffer_index = tt::CBIndex::c_1,
            .data_format = input_data_format,
            .page_size = input_single_tile_size,
        }}},
    });

    auto kernel_config = BcastToKernelConfig(operation_attributes.subtile_broadcast_type);

    // READER KERNEL
    std::vector<uint32_t> reader_compile_time_args{(uint32_t)tt::CBIndex::c_0};
    tt::tt_metal::TensorAccessorArgs(*input.buffer()).append_to(reader_compile_time_args);
    KernelDescriptor reader_desc{
        .kernel_source = get_kernel_file_path(kernel_config.reader_kernel),
        .source_type = KernelDescriptor::SourceType::FILE_PATH,
        .core_ranges = all_device_cores,
        .compile_time_args = reader_compile_time_args,
        .config = ReaderConfigDescriptor{},
    };

    // WRITER KERNEL
    uint32_t writer_cb_id = (kernel_config.writer_kernel == KernelName::WriterNoBcast) ? (uint32_t)tt::CBIndex::c_0
                                                                                       : (uint32_t)tt::CBIndex::c_1;
    std::vector<uint32_t> writer_compile_time_args{writer_cb_id};
    tt::tt_metal::TensorAccessorArgs(*output.buffer()).append_to(writer_compile_time_args);
    KernelDescriptor writer_desc{
        .kernel_source = get_kernel_file_path(kernel_config.writer_kernel),
        .source_type = KernelDescriptor::SourceType::FILE_PATH,
        .core_ranges = all_device_cores,
        .compile_time_args = writer_compile_time_args,
        .config = WriterConfigDescriptor{},
    };

    // COMPUTE KERNEL
    // Enable fp32_dest_acc_en and unpack_to_dest_mode for 32-bit formats (Float32, Int32, UInt32)
    bool is_32bit_format = input_data_format == tt::DataFormat::Float32 || input_data_format == tt::DataFormat::Int32 ||
                           input_data_format == tt::DataFormat::UInt32;
    std::vector<tt::tt_metal::UnpackToDestMode> unpack_to_dest_mode(
        NUM_CIRCULAR_BUFFERS, tt::tt_metal::UnpackToDestMode::Default);
    if (is_32bit_format) {
        unpack_to_dest_mode[(uint32_t)tt::CBIndex::c_0] = tt::tt_metal::UnpackToDestMode::UnpackToDestFp32;
    }
    KernelDescriptor compute_desc{
        .kernel_source = get_kernel_file_path(kernel_config.compute_kernel),
        .source_type = KernelDescriptor::SourceType::FILE_PATH,
        .core_ranges = all_device_cores,
        .compile_time_args = {(uint32_t)tt::CBIndex::c_0, (uint32_t)tt::CBIndex::c_1},
        .config =
            ComputeConfigDescriptor{
                .fp32_dest_acc_en = is_32bit_format,
                .unpack_to_dest_mode = unpack_to_dest_mode,
                .math_approx_mode = false,
            },
    };

    // Build per-core runtime arguments
    const auto [iN, iC, iHt, iWt] = extract_shape_dims(input);
    const auto [oN, oC, oHt, oWt] = extract_shape_dims(output);

    uint32_t num_output_tiles = output.physical_volume() / output.tensor_spec().tile().get_tile_hw();

    constexpr bool row_major = true;
    uint32_t num_cores_total = num_cores_x * num_cores_y;
    auto [num_cores, all_cores, core_group_1, core_group_2, num_tiles_per_core_group_1, num_tiles_per_core_group_2] =
        tt::tt_metal::split_work_to_cores(compute_with_storage_grid_size, num_output_tiles, row_major);

    auto* input_buffer = input.buffer();
    auto* output_buffer = output.buffer();

    auto cores = grid_to_cores(num_cores_total, num_cores_x, num_cores_y, row_major);
    for (uint32_t i = 0, start_tile_id = 0; i < num_cores_total; i++) {
        const auto& core = cores[i];

        uint32_t num_tiles_per_core;
        if (core_group_1.contains(core)) {
            num_tiles_per_core = num_tiles_per_core_group_1;
        } else if (core_group_2.contains(core)) {
            num_tiles_per_core = num_tiles_per_core_group_2;
        } else {
            // Initialize with zeros for unused cores
            reader_desc.runtime_args.emplace_back(core, KernelDescriptor::CoreRuntimeArgs(13, 0));
            writer_desc.runtime_args.emplace_back(core, KernelDescriptor::CoreRuntimeArgs(14, 0));
            compute_desc.runtime_args.emplace_back(core, KernelDescriptor::CoreRuntimeArgs(12, 0));
            continue;
        }

        uint32_t oHtWt = oHt * oWt;
        uint32_t tiles_per_batch = oHtWt * oC;
        uint32_t start_n = start_tile_id / tiles_per_batch;
        uint32_t start_remaining = start_tile_id % tiles_per_batch;
        uint32_t start_c = start_remaining / oHtWt;
        uint32_t start_t = start_remaining % oHtWt;
        uint32_t start_th = start_t / oWt;
        uint32_t start_tw = start_t % oWt;

        reader_desc.emplace_runtime_args(
            core,
            {input_buffer,
             start_n,
             start_c,
             start_t,
             start_th,
             start_tw,
             num_tiles_per_core,
             iHt * iWt * iC * (iN > 1),
             iHt * iWt * (iC > 1),
             oN,
             oC,
             oHt,
             oWt});

        writer_desc.emplace_runtime_args(
            core,
            {output_buffer,
             start_n,
             start_c,
             start_t,
             start_th,
             start_tw,
             num_tiles_per_core,
             iHt * iWt * iC * (iN > 1),
             iHt * iWt * (iC > 1),
             oN,
             oC,
             oHt,
             oWt,
             start_tile_id});

        compute_desc.runtime_args.emplace_back(
            core,
            KernelDescriptor::CoreRuntimeArgs{
                start_n,
                start_c,
                start_t,
                start_th,
                start_tw,
                num_tiles_per_core,
                iHt * iWt * iC * (iN > 1),
                iHt * iWt * (iC > 1),
                oN,
                oC,
                oHt,
                oWt});

        start_tile_id += num_tiles_per_core;
    }

    desc.kernels.push_back(std::move(reader_desc));
    desc.kernels.push_back(std::move(writer_desc));
    desc.kernels.push_back(std::move(compute_desc));
    return desc;
}
}  // namespace ttnn::operations::experimental::broadcast_to
