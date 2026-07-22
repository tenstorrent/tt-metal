// SPDX-FileCopyrightText: © 2024 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include <string>
#include <vector>

#include "moreh_mean_device_operation.hpp"
#include <tt-metalium/tensor_accessor_args.hpp>
#include <tt-metalium/work_split.hpp>
#include "ttnn/operations/moreh/moreh_helper_functions.hpp"
#include "ttnn/operations/core/compute_kernel/compute_kernel_config.hpp"
#include "ttnn/operations/reduction/generic/device/common.hpp"
#include "ttnn/operations/reduction/generic/device/reduce_op.hpp"

namespace ttnn::operations::moreh::moreh_mean {

tt::tt_metal::ProgramDescriptor MorehMeanOperation::MorehMeanNCFactory::create_descriptor(
    const operation_attributes_t& operation_attributes,
    const tensor_args_t& tensor_args,
    tensor_return_value_t& output) {
    using namespace tt;
    using namespace tt::tt_metal;

    const auto& input = tensor_args.input;
    auto dim = operation_attributes.dim;

    auto compute_kernel_config =
        init_device_compute_kernel_config(input.device()->arch(), operation_attributes.compute_kernel_config);

    auto* device = input.device();

    auto grid_coord = device->compute_with_storage_grid_size();
    const CoreRange core_range({0, 0}, {grid_coord.x - 1, grid_coord.y - 1});

    const auto cb_data_format = datatype_to_dataformat_converter(output.dtype());

    const auto& input_shape = input.padded_shape();
    const auto Ht = input_shape[-2] / constants::TILE_HEIGHT;
    const auto Wt = input_shape[-1] / constants::TILE_WIDTH;
    const auto HtWt = Ht * Wt;
    const auto num_reduce_input_tile = input_shape[dim];

    const auto rank = input_shape.rank();
    auto input_tile_stride = HtWt;
    for (int i = dim + 1; i < rank - 2; i++) {
        input_tile_stride *= input_shape[i];
    }

    uint32_t inner_size = 1;
    for (int i = dim + 1; i < rank - 2; i++) {
        inner_size *= input_shape[i];
    }

    const auto units_to_divide = output.physical_volume() / constants::TILE_HW;

    uint32_t core_h = core_range.end_coord.y - core_range.start_coord.y + 1;

    auto [num_cores, all_cores, core_group_1, core_group_2, units_per_core_group_1, units_per_core_group_2] =
        split_work_to_cores_wt_core_range(core_range, units_to_divide);

    auto [math_fidelity, math_approx_mode, fp32_dest_acc_en, packer_l1_acc, dst_full_sync_en] =
        get_compute_kernel_config_args(device->arch(), compute_kernel_config);

    ProgramDescriptor desc;

    // ---- Circular buffers ----
    desc.cbs.push_back(CBDescriptor{
        .total_size = 2 * tile_size(cb_data_format),
        .core_ranges = all_cores,
        .format_descriptors = {{CBFormatDescriptor{
            .buffer_index = static_cast<uint8_t>(CBIndex::c_0),
            .data_format = cb_data_format,
            .page_size = tile_size(cb_data_format),
        }}},
    });
    desc.cbs.push_back(CBDescriptor{
        .total_size = tile_size(cb_data_format),
        .core_ranges = all_cores,
        .format_descriptors = {{CBFormatDescriptor{
            .buffer_index = static_cast<uint8_t>(CBIndex::c_1),
            .data_format = cb_data_format,
            .page_size = tile_size(cb_data_format),
        }}},
    });
    desc.cbs.push_back(CBDescriptor{
        .total_size = tile_size(cb_data_format),
        .core_ranges = all_cores,
        .format_descriptors = {{CBFormatDescriptor{
            .buffer_index = static_cast<uint8_t>(CBIndex::c_2),
            .data_format = cb_data_format,
            .page_size = tile_size(cb_data_format),
        }}},
    });
    desc.cbs.push_back(CBDescriptor{
        .total_size = tile_size(cb_data_format),
        .core_ranges = all_cores,
        .format_descriptors = {{CBFormatDescriptor{
            .buffer_index = static_cast<uint8_t>(CBIndex::c_24),
            .data_format = cb_data_format,
            .page_size = tile_size(cb_data_format),
        }}},
    });
    desc.cbs.push_back(CBDescriptor{
        .total_size = 2 * tile_size(cb_data_format),
        .core_ranges = all_cores,
        .format_descriptors = {{CBFormatDescriptor{
            .buffer_index = static_cast<uint8_t>(CBIndex::c_16),
            .data_format = cb_data_format,
            .page_size = tile_size(cb_data_format),
        }}},
    });

    // ---- Reader kernel ----
    KernelDescriptor::CompileTimeArgs reader_compile_time_args;
    TensorAccessorArgs(*input.buffer()).append_to(reader_compile_time_args);

    KernelDescriptor reader_desc;
    reader_desc.kernel_source = "ttnn/cpp/ttnn/operations/moreh/moreh_mean/device/kernels/reader_moreh_mean_nc.cpp";
    reader_desc.source_type = KernelDescriptor::SourceType::FILE_PATH;
    reader_desc.core_ranges = all_cores;
    reader_desc.compile_time_args = std::move(reader_compile_time_args);
    reader_desc.config = ReaderConfigDescriptor{};

    // ---- Writer kernel ----
    KernelDescriptor::CompileTimeArgs writer_compile_time_args;
    TensorAccessorArgs(*output.buffer()).append_to(writer_compile_time_args);

    KernelDescriptor writer_desc;
    writer_desc.kernel_source = "ttnn/cpp/ttnn/operations/moreh/moreh_mean/device/kernels/writer_moreh_mean_nc.cpp";
    writer_desc.source_type = KernelDescriptor::SourceType::FILE_PATH;
    writer_desc.core_ranges = all_cores;
    writer_desc.compile_time_args = std::move(writer_compile_time_args);
    writer_desc.config = WriterConfigDescriptor{};

    // ---- Compute kernels (two groups) ----
    KernelDescriptor::Defines compute_defines;
    if (fp32_dest_acc_en) {
        compute_defines.emplace_back("FP32_DEST_ACC_EN", "1");
    }
    std::vector<UnpackToDestMode> unpack_to_dest_mode(NUM_CIRCULAR_BUFFERS, UnpackToDestMode::Default);

    KernelDescriptor compute_desc_1;
    compute_desc_1.kernel_source = "ttnn/cpp/ttnn/operations/moreh/moreh_mean/device/kernels/moreh_mean_nc.cpp";
    compute_desc_1.source_type = KernelDescriptor::SourceType::FILE_PATH;
    compute_desc_1.core_ranges = core_group_1;
    compute_desc_1.compile_time_args = {units_per_core_group_1};
    compute_desc_1.defines = compute_defines;
    compute_desc_1.config = ComputeConfigDescriptor{
        .math_fidelity = math_fidelity,
        .fp32_dest_acc_en = fp32_dest_acc_en,
        .dst_full_sync_en = dst_full_sync_en,
        .unpack_to_dest_mode = unpack_to_dest_mode,
        .math_approx_mode = math_approx_mode,
    };

    KernelDescriptor compute_desc_2;
    bool has_core_group_2 = !core_group_2.ranges().empty();
    if (has_core_group_2) {
        compute_desc_2.kernel_source = "ttnn/cpp/ttnn/operations/moreh/moreh_mean/device/kernels/moreh_mean_nc.cpp";
        compute_desc_2.source_type = KernelDescriptor::SourceType::FILE_PATH;
        compute_desc_2.core_ranges = core_group_2;
        compute_desc_2.compile_time_args = {units_per_core_group_2};
        compute_desc_2.defines = compute_defines;
        compute_desc_2.config = ComputeConfigDescriptor{
            .math_fidelity = math_fidelity,
            .fp32_dest_acc_en = fp32_dest_acc_en,
            .dst_full_sync_en = dst_full_sync_en,
            .unpack_to_dest_mode = unpack_to_dest_mode,
            .math_approx_mode = math_approx_mode,
        };
    }

    // ---- Runtime args per core ----
    auto* const input_buf = input.buffer();
    auto* const output_buf = output.buffer();
    for (uint32_t i = 0, tile_offset = 0; i < num_cores; ++i) {
        CoreCoord core = {i / core_h, i % core_h};

        uint32_t units_per_core;
        if (core_group_1.contains(core)) {
            units_per_core = units_per_core_group_1;
            compute_desc_1.emplace_runtime_args(core, {static_cast<uint32_t>(num_reduce_input_tile), units_per_core});
        } else if (core_group_2.contains(core)) {
            units_per_core = units_per_core_group_2;
            compute_desc_2.emplace_runtime_args(core, {static_cast<uint32_t>(num_reduce_input_tile), units_per_core});
        } else {
            TT_THROW("Core not in specified core ranges.");
        }

        reader_desc.emplace_runtime_args(
            core,
            {input_buf,
             static_cast<uint32_t>(num_reduce_input_tile),
             units_per_core,
             static_cast<uint32_t>(input_tile_stride),
             tile_offset,
             static_cast<uint32_t>(HtWt),
             inner_size});

        writer_desc.emplace_runtime_args(core, {output_buf, units_per_core, tile_offset});

        tile_offset += units_per_core;
    }

    desc.kernels.push_back(std::move(reader_desc));
    desc.kernels.push_back(std::move(writer_desc));
    desc.kernels.push_back(std::move(compute_desc_1));
    if (has_core_group_2) {
        desc.kernels.push_back(std::move(compute_desc_2));
    }

    return desc;
}

}  // namespace ttnn::operations::moreh::moreh_mean
