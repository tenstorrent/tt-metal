// SPDX-FileCopyrightText: © 2025 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include "softmax_device_operation.hpp"

#include "ttnn/operations/moreh/moreh_helper_functions.hpp"

#include <tt-logger/tt-logger.hpp>
#include <tt-metalium/program_descriptors.hpp>
#include <tt-metalium/tensor_accessor_args.hpp>

#include <map>
#include <utility>

namespace ttnn::prim {

tt::tt_metal::ProgramDescriptor SoftmaxDeviceOperation::SoftmaxProgramFactoryGeneralCLarge::create_descriptor(
    const operation_attributes_t& attributes, const tensor_args_t& tensor_args, tensor_return_value_t& output_tensor) {
    using namespace tt::tt_metal;

    log_debug(tt::LogMetal, "SoftmaxProgramFactoryGeneralCLarge selected");

    // Constants
    const auto& input = tensor_args.input_tensor;
    const auto dim = static_cast<int>(static_cast<unsigned char>(attributes.dim));
    const auto& compute_kernel_config = attributes.compute_kernel_config;
    auto* const device = input.device();
    const auto grid_coord = device->compute_with_storage_grid_size();
    const CoreRange core_range({0, 0}, {grid_coord.x - 1, grid_coord.y - 1});
    const uint32_t tile_height = input.tensor_spec().tile().get_height();
    const uint32_t tile_width = input.tensor_spec().tile().get_width();
    const auto shape = input.padded_shape();
    const auto H = shape[-2];
    const auto W = shape[-1];
    const auto Ht = H / tile_height;
    const auto Wt = W / tile_width;

    // Work split
    const uint32_t num_tiles = input.physical_volume() / shape[dim] / H / W * Ht * Wt;
    const uint32_t core_h = core_range.end_coord.y - core_range.start_coord.y + 1;
    auto [num_cores, all_cores, core_group_1, core_group_2, num_tiles_per_core_group_1, num_tiles_per_core_group_2] =
        operations::split_work_to_cores_wt_core_range(core_range, num_tiles);

    auto arch = input.device()->arch();
    auto [math_fidelity, math_approx_mode, fp32_dest_acc_en, packer_l1_acc, dst_full_sync_en] =
        get_compute_kernel_config_args(arch, compute_kernel_config);

    if (input.dtype() == DataType::FLOAT32 && !fp32_dest_acc_en) {
        TT_THROW(
            "FP32 destination accumulation must be enabled when input tensor has FLOAT32 data type. Please update the "
            "compute kernel configuration.");
    }

    // Circular buffers
    const auto data_format = datatype_to_dataformat_converter(input.dtype());
    const auto intermed_data_format = fp32_dest_acc_en ? tt::DataFormat::Float32 : data_format;

    const uint32_t in_tile_size = tt::tile_size(data_format);
    const uint32_t intermed_tile_size = tt::tile_size(intermed_data_format);

    ProgramDescriptor desc;
    desc.cbs.push_back(CBDescriptor{
        .total_size = 2 * in_tile_size,
        .core_ranges = all_cores,
        .format_descriptors = {{CBFormatDescriptor{
            .buffer_index = static_cast<uint8_t>(tt::CBIndex::c_0),
            .data_format = data_format,
            .page_size = in_tile_size,
        }}},
    });  // input
    desc.cbs.push_back(CBDescriptor{
        .total_size = 2 * in_tile_size,
        .core_ranges = all_cores,
        .format_descriptors = {{CBFormatDescriptor{
            .buffer_index = static_cast<uint8_t>(tt::CBIndex::c_16),
            .data_format = data_format,
            .page_size = in_tile_size,
        }}},
    });  // output
    desc.cbs.push_back(CBDescriptor{
        .total_size = intermed_tile_size,
        .core_ranges = all_cores,
        .format_descriptors = {{CBFormatDescriptor{
            .buffer_index = static_cast<uint8_t>(tt::CBIndex::c_24),
            .data_format = intermed_data_format,
            .page_size = intermed_tile_size,
        }}},
    });  // exp(x)
    desc.cbs.push_back(CBDescriptor{
        .total_size = intermed_tile_size,
        .core_ranges = all_cores,
        .format_descriptors = {{CBFormatDescriptor{
            .buffer_index = static_cast<uint8_t>(tt::CBIndex::c_25),
            .data_format = intermed_data_format,
            .page_size = intermed_tile_size,
        }}},
    });  // recips
    desc.cbs.push_back(CBDescriptor{
        .total_size = 2 * intermed_tile_size,
        .core_ranges = all_cores,
        .format_descriptors = {{CBFormatDescriptor{
            .buffer_index = static_cast<uint8_t>(tt::CBIndex::c_26),
            .data_format = intermed_data_format,
            .page_size = intermed_tile_size,
        }}},
    });  // add
    desc.cbs.push_back(CBDescriptor{
        .total_size = in_tile_size,
        .core_ranges = all_cores,
        .format_descriptors = {{CBFormatDescriptor{
            .buffer_index = static_cast<uint8_t>(tt::CBIndex::c_27),
            .data_format = data_format,
            .page_size = in_tile_size,
        }}},
    });  // max
    desc.cbs.push_back(CBDescriptor{
        .total_size = intermed_tile_size,
        .core_ranges = all_cores,
        .format_descriptors = {{CBFormatDescriptor{
            .buffer_index = static_cast<uint8_t>(tt::CBIndex::c_28),
            .data_format = intermed_data_format,
            .page_size = intermed_tile_size,
        }}},
    });  // tmp

    // Data movement kernels
    std::vector<uint32_t> reader_ct_args = {};
    TensorAccessorArgs(*input.buffer()).append_to(reader_ct_args);

    KernelDescriptor reader_desc;
    reader_desc.kernel_source = std::string(SOFTMAX_KERNEL_PATH_GENERAL) + "/reader_moreh_softmax_c_large.cpp";
    reader_desc.source_type = KernelDescriptor::SourceType::FILE_PATH;
    reader_desc.core_ranges = all_cores;
    reader_desc.compile_time_args = reader_ct_args;
    reader_desc.config = ReaderConfigDescriptor{};

    std::vector<uint32_t> writer_ct_args = {};
    TensorAccessorArgs(*output_tensor.buffer()).append_to(writer_ct_args);

    KernelDescriptor writer_desc;
    writer_desc.kernel_source = std::string(SOFTMAX_KERNEL_PATH_GENERAL) + "/writer_moreh_softmax_c_large.cpp";
    writer_desc.source_type = KernelDescriptor::SourceType::FILE_PATH;
    writer_desc.core_ranges = all_cores;
    writer_desc.compile_time_args = writer_ct_args;
    writer_desc.config = WriterConfigDescriptor{};

    auto outer_stride = Ht * Wt;
    for (int i = dim; i < shape.rank() - 2; i++) {
        outer_stride *= shape[i];
    }
    const auto dim_size = shape[dim];
    const auto inner_size = outer_stride / dim_size;

    // Kernel defines
    std::map<std::string, std::string> compute_defines;
    compute_defines["SOFTMAX"] = "1";
    if (fp32_dest_acc_en) {
        compute_defines["FP32_DEST_ACC_EN"] = "1";
    }

    // Comput kernel
    KernelDescriptor compute_desc_g1;
    compute_desc_g1.kernel_source = std::string(SOFTMAX_KERNEL_PATH_GENERAL) + "/moreh_softmax_c_large.cpp";
    compute_desc_g1.source_type = KernelDescriptor::SourceType::FILE_PATH;
    compute_desc_g1.core_ranges = core_group_1;
    compute_desc_g1.compile_time_args = {num_tiles_per_core_group_1, dim_size};
    compute_desc_g1.defines = KernelDescriptor::Defines(compute_defines.begin(), compute_defines.end());
    compute_desc_g1.config = ComputeConfigDescriptor{
        .math_fidelity = math_fidelity,
        .fp32_dest_acc_en = fp32_dest_acc_en,
        .dst_full_sync_en = dst_full_sync_en,
        .math_approx_mode = math_approx_mode,
    };

    KernelDescriptor compute_desc_g2;
    if (num_tiles_per_core_group_2 > 0) {
        compute_desc_g2.kernel_source = std::string(SOFTMAX_KERNEL_PATH_GENERAL) + "/moreh_softmax_c_large.cpp";
        compute_desc_g2.source_type = KernelDescriptor::SourceType::FILE_PATH;
        compute_desc_g2.core_ranges = core_group_2;
        compute_desc_g2.compile_time_args = {num_tiles_per_core_group_2, dim_size};
        compute_desc_g2.defines = KernelDescriptor::Defines(compute_defines.begin(), compute_defines.end());
        compute_desc_g2.config = ComputeConfigDescriptor{
            .math_fidelity = math_fidelity,
            .fp32_dest_acc_en = fp32_dest_acc_en,
            .dst_full_sync_en = dst_full_sync_en,
            .math_approx_mode = math_approx_mode,
        };
    }

    // Runtime Args
    const auto core_x_offset = core_range.start_coord.x;
    const auto core_y_offset = core_range.start_coord.y;

    for (uint32_t i = 0, tile_offset = 0; i < num_cores; i++) {
        CoreCoord core = {(i / core_h) + core_x_offset, (i % core_h) + core_y_offset};
        uint32_t num_tiles_per_core = 0;
        if (core_group_1.contains(core)) {
            num_tiles_per_core = num_tiles_per_core_group_1;
        } else if (core_group_2.contains(core)) {
            num_tiles_per_core = num_tiles_per_core_group_2;
        } else {
            TT_THROW("Core not in specified core ranges");
        }

        reader_desc.emplace_runtime_args(
            core, {input.buffer(), num_tiles_per_core, tile_offset, outer_stride, inner_size, dim_size});

        writer_desc.emplace_runtime_args(
            core, {output_tensor.buffer(), num_tiles_per_core, tile_offset, outer_stride, inner_size, dim_size});

        tile_offset += num_tiles_per_core;
    }

    desc.kernels.push_back(std::move(reader_desc));
    desc.kernels.push_back(std::move(writer_desc));
    desc.kernels.push_back(std::move(compute_desc_g1));
    if (num_tiles_per_core_group_2 > 0) {
        desc.kernels.push_back(std::move(compute_desc_g2));
    }

    return desc;
}

}  // namespace ttnn::prim
