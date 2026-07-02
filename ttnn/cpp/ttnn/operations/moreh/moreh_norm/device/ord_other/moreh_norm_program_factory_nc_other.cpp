// SPDX-FileCopyrightText: © 2024 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include <tt-metalium/work_split.hpp>
#include "ttnn/operations/moreh/moreh_norm/device/moreh_norm_device_operation.hpp"
#include <tt-metalium/tensor_accessor_args.hpp>
#include "ttnn/operations/moreh/moreh_helper_functions.hpp"

namespace ttnn::operations::moreh::moreh_norm {

tt::tt_metal::ProgramDescriptor MorehNormOperation::ProgramFactoryNCOther::create_descriptor(
    const operation_attributes_t& operation_attributes,
    const tensor_args_t& tensor_args,
    tensor_return_value_t& output) {
    using namespace tt;
    using namespace tt::tt_metal;

    const auto& input = tensor_args.input;
    const auto dim = operation_attributes.dim;
    const auto p = operation_attributes.p;
    ////////////////////////////////////////////////////////////////////////////
    //                      Device Setup
    ////////////////////////////////////////////////////////////////////////////
    auto* device = input.device();

    ////////////////////////////////////////////////////////////////////////////
    //                         Parameters Setup
    ////////////////////////////////////////////////////////////////////////////
    const auto input_shape = input.padded_shape();
    const auto input_rank = static_cast<decltype(dim)>(input_shape.rank());

    const auto num_reduced_tiles_along_dim = input_shape[dim];
    const auto num_output_tiles = output.physical_volume() / tt::constants::TILE_HW;

    uint32_t outer_stride{1};
    for (int64_t j = dim; j < input_rank; ++j) {
        outer_stride *= input_shape[j];
    }
    outer_stride /= tt::constants::TILE_HW;

    uint32_t num_inner_tiles{1};
    for (int64_t j = dim + 1; j < input_rank; ++j) {
        num_inner_tiles *= input_shape[j];
    }
    num_inner_tiles /= tt::constants::TILE_HW;

    ////////////////////////////////////////////////////////////////////////////
    //                         Core Setup
    ////////////////////////////////////////////////////////////////////////////
    auto grid = device->compute_with_storage_grid_size();
    const auto num_cores_y = grid.y;

    auto arch = input.device()->arch();
    auto [math_fidelity, math_approx_mode, fp32_dest_acc_en, packer_l1_acc, dst_full_sync_en] =
        get_compute_kernel_config_args(arch, operation_attributes.compute_kernel_config);

    const auto
        [num_cores_to_be_used,
         all_cores,
         core_group_1,
         core_group_2,
         num_units_per_core_group_1,
         num_units_per_core_group_2] = tt::tt_metal::split_work_to_cores(grid, num_output_tiles);

    ////////////////////////////////////////////////////////////////////////////
    //                         CircularBuffer Setup
    ////////////////////////////////////////////////////////////////////////////
    const auto cb_data_format = tt::tt_metal::datatype_to_dataformat_converter(input.dtype());
    const auto intermed_data_format = fp32_dest_acc_en ? tt::DataFormat::Float32 : cb_data_format;

    const uint32_t in0_t{1};  // input
    const uint32_t in1_t{1};  // one

    const uint32_t out0_t{1};  // output

    const uint32_t im0_t{1};  // f(x)
    const uint32_t im1_t{1};  // calculate f(x) over dimensions

    ProgramDescriptor desc;

    desc.cbs.push_back(CBDescriptor{
        .total_size = in0_t * tile_size(cb_data_format),
        .core_ranges = all_cores,
        .format_descriptors = {{CBFormatDescriptor{
            .buffer_index = static_cast<uint8_t>(CBIndex::c_0),
            .data_format = cb_data_format,
            .page_size = tile_size(cb_data_format),
        }}},
    });
    desc.cbs.push_back(CBDescriptor{
        .total_size = in1_t * tile_size(cb_data_format),
        .core_ranges = all_cores,
        .format_descriptors = {{CBFormatDescriptor{
            .buffer_index = static_cast<uint8_t>(CBIndex::c_1),
            .data_format = cb_data_format,
            .page_size = tile_size(cb_data_format),
        }}},
    });
    desc.cbs.push_back(CBDescriptor{
        .total_size = out0_t * tile_size(cb_data_format),
        .core_ranges = all_cores,
        .format_descriptors = {{CBFormatDescriptor{
            .buffer_index = static_cast<uint8_t>(CBIndex::c_16),
            .data_format = cb_data_format,
            .page_size = tile_size(cb_data_format),
        }}},
    });
    desc.cbs.push_back(CBDescriptor{
        .total_size = im0_t * tile_size(intermed_data_format),
        .core_ranges = all_cores,
        .format_descriptors = {{CBFormatDescriptor{
            .buffer_index = static_cast<uint8_t>(CBIndex::c_24),
            .data_format = intermed_data_format,
            .page_size = tile_size(intermed_data_format),
        }}},
    });
    desc.cbs.push_back(CBDescriptor{
        .total_size = im1_t * tile_size(intermed_data_format),
        .core_ranges = all_cores,
        .format_descriptors = {{CBFormatDescriptor{
            .buffer_index = static_cast<uint8_t>(CBIndex::c_25),
            .data_format = intermed_data_format,
            .page_size = tile_size(intermed_data_format),
        }}},
    });

    ////////////////////////////////////////////////////////////////////////////
    //                      DataMovementKernel SetUp
    ////////////////////////////////////////////////////////////////////////////
    const auto* const reader_kernel_file =
        "ttnn/cpp/ttnn/operations/moreh/moreh_norm/device/ord_other/moreh_norm_nc/kernels/"
        "reader_moreh_norm_nc.cpp";
    const auto* const writer_kernel_file =
        "ttnn/cpp/ttnn/operations/moreh/moreh_norm/device/ord_other/moreh_norm_nc/kernels/"
        "writer_moreh_norm_nc.cpp";

    KernelDescriptor::CompileTimeArgs reader_ct_args;
    TensorAccessorArgs(*input.buffer()).append_to(reader_ct_args);
    KernelDescriptor::CompileTimeArgs writer_ct_args;
    TensorAccessorArgs(*output.buffer()).append_to(writer_ct_args);

    KernelDescriptor reader_desc;
    reader_desc.kernel_source = reader_kernel_file;
    reader_desc.source_type = KernelDescriptor::SourceType::FILE_PATH;
    reader_desc.core_ranges = all_cores;
    reader_desc.compile_time_args = std::move(reader_ct_args);
    reader_desc.config = ReaderConfigDescriptor{};

    KernelDescriptor writer_desc;
    writer_desc.kernel_source = writer_kernel_file;
    writer_desc.source_type = KernelDescriptor::SourceType::FILE_PATH;
    writer_desc.core_ranges = all_cores;
    writer_desc.compile_time_args = std::move(writer_ct_args);
    writer_desc.config = WriterConfigDescriptor{};

    ////////////////////////////////////////////////////////////////////////////
    //                      ComputeKernel SetUp
    ////////////////////////////////////////////////////////////////////////////
    std::map<std::string, std::string> compute_defines_map{};
    if (p == 0.0f) {
        compute_defines_map["IS_ZERO"] = "1";
    } else {
        if (p == -std::numeric_limits<float>::infinity()) {
            compute_defines_map["MINUS_INF"] = "1";
        }
    }
    KernelDescriptor::Defines compute_defines(compute_defines_map.begin(), compute_defines_map.end());

    const auto* const compute_kernel_file =
        "ttnn/cpp/ttnn/operations/moreh/moreh_norm/device/ord_other/moreh_norm_nc/kernels/"
        "moreh_norm_nc_kernel.cpp";

    std::vector<UnpackToDestMode> unpack_to_dest_mode(NUM_CIRCULAR_BUFFERS, UnpackToDestMode::Default);

    KernelDescriptor compute_desc_1;
    compute_desc_1.kernel_source = compute_kernel_file;
    compute_desc_1.source_type = KernelDescriptor::SourceType::FILE_PATH;
    compute_desc_1.core_ranges = core_group_1;
    compute_desc_1.compile_time_args = {};
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
        compute_desc_2.kernel_source = compute_kernel_file;
        compute_desc_2.source_type = KernelDescriptor::SourceType::FILE_PATH;
        compute_desc_2.core_ranges = core_group_2;
        compute_desc_2.compile_time_args = {};
        compute_desc_2.defines = compute_defines;
        compute_desc_2.config = ComputeConfigDescriptor{
            .math_fidelity = math_fidelity,
            .fp32_dest_acc_en = fp32_dest_acc_en,
            .dst_full_sync_en = dst_full_sync_en,
            .unpack_to_dest_mode = unpack_to_dest_mode,
            .math_approx_mode = math_approx_mode,
        };
    }

    ////////////////////////////////////////////////////////////////////////////
    //                      RuntimeArgs SetUp
    ////////////////////////////////////////////////////////////////////////////
    auto* const input_buf = input.buffer();
    auto* const output_buf = output.buffer();
    for (uint32_t i = 0, tile_offset = 0; i < num_cores_to_be_used; ++i) {
        CoreCoord core = {i / num_cores_y, i % num_cores_y};

        uint32_t num_output_tiles_per_core;
        if (core_group_1.contains(core)) {
            num_output_tiles_per_core = num_units_per_core_group_1;
            compute_desc_1.emplace_runtime_args(
                core,
                {
                    num_output_tiles_per_core,
                    static_cast<uint32_t>(num_reduced_tiles_along_dim),
                });
        } else if (core_group_2.contains(core)) {
            num_output_tiles_per_core = num_units_per_core_group_2;
            compute_desc_2.emplace_runtime_args(
                core,
                {
                    num_output_tiles_per_core,
                    static_cast<uint32_t>(num_reduced_tiles_along_dim),
                });
        } else {
            TT_THROW("Core not in specified core ranges.");
        }
        // reader
        reader_desc.emplace_runtime_args(
            core,
            {input_buf,
             static_cast<uint32_t>(is_dram(input)),
             num_output_tiles_per_core,
             tile_offset,
             outer_stride,
             num_inner_tiles,
             static_cast<uint32_t>(num_reduced_tiles_along_dim)});

        // writer
        writer_desc.emplace_runtime_args(
            core, {output_buf, static_cast<uint32_t>(is_dram(output)), num_output_tiles_per_core, tile_offset});

        tile_offset += num_output_tiles_per_core;
    }

    desc.kernels.push_back(std::move(reader_desc));
    desc.kernels.push_back(std::move(writer_desc));
    desc.kernels.push_back(std::move(compute_desc_1));
    if (has_core_group_2) {
        desc.kernels.push_back(std::move(compute_desc_2));
    }

    return desc;
}
}  // namespace ttnn::operations::moreh::moreh_norm
