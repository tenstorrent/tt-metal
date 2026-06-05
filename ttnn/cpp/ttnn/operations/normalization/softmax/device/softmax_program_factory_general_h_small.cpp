// SPDX-FileCopyrightText: © 2025 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include "softmax_device_operation.hpp"

#include "ttnn/operations/moreh/moreh_helper_functions.hpp"

#include <tt-logger/tt-logger.hpp>
#include <tt-metalium/program_descriptors.hpp>
#include <tt-metalium/tensor_accessor_args.hpp>

#include <bit>
#include <map>
#include <utility>

namespace ttnn::prim {

tt::tt_metal::ProgramDescriptor SoftmaxDeviceOperation::SoftmaxProgramFactoryGeneralHSmall::create_descriptor(
    const operation_attributes_t& attributes, const tensor_args_t& tensor_args, tensor_return_value_t& output_tensor) {
    using namespace tt::tt_metal;

    log_debug(tt::LogMetal, "SoftmaxProgramFactoryGeneralHSmall selected");

    // Constants
    const auto& input = tensor_args.input_tensor;
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
    const auto num = input.physical_volume() / H / W;
    const uint32_t num_cols_tiles = num * Wt;
    const uint32_t core_h = core_range.end_coord.y - core_range.start_coord.y + 1;
    auto [num_cores, all_cores, core_group_1, core_group_2, num_tiles_per_core_group_1, num_tiles_per_core_group_2] =
        operations::split_work_to_cores_wt_core_range(core_range, num_cols_tiles);

    const auto arch = input.device()->arch();
    auto [math_fidelity, math_approx_mode, fp32_dest_acc_en, packer_l1_acc, dst_full_sync_en] =
        get_compute_kernel_config_args(arch, compute_kernel_config);

    if (input.dtype() == DataType::FLOAT32 && !fp32_dest_acc_en) {
        TT_THROW(
            "FP32 destination accumulation must be enabled when input tensor has FLOAT32 data type. Please update the "
            "compute kernel configuration.");
    }

    // Circular buffers
    const auto data_format = datatype_to_dataformat_converter(input.dtype());
    // Use Float16_b for intermediates when not accumulating in fp32, matching the AttentionOptimized path.
    // This avoids using Bfp8_b for intermediate computations where it lacks precision (issue #32934).
    const auto intermed_data_format = fp32_dest_acc_en ? tt::DataFormat::Float32 : tt::DataFormat::Float16_b;
    // Reader generates mask/scaler with uint16_t (1024 elements = 2048 bytes). Use Float16_b for these CBs when
    // input is Bfp8_b so tile size matches; Bfp8_b tile layout is smaller and would be overflowed (issue #32934).
    const auto mask_scaler_format = (data_format == tt::DataFormat::Bfp8_b) ? tt::DataFormat::Float16_b : data_format;

    const uint32_t in_tile_size = tt::tile_size(data_format);
    const uint32_t mask_scaler_tile_size = tt::tile_size(mask_scaler_format);
    const uint32_t intermed_tile_size = tt::tile_size(intermed_data_format);

    ProgramDescriptor desc;
    desc.cbs.push_back(CBDescriptor{
        .total_size = Ht * in_tile_size,
        .core_ranges = all_cores,
        .format_descriptors = {{CBFormatDescriptor{
            .buffer_index = static_cast<uint8_t>(tt::CBIndex::c_0),
            .data_format = data_format,
            .page_size = in_tile_size,
        }}},
    });  // input
    desc.cbs.push_back(CBDescriptor{
        .total_size = mask_scaler_tile_size,
        .core_ranges = all_cores,
        .format_descriptors = {{CBFormatDescriptor{
            .buffer_index = static_cast<uint8_t>(tt::CBIndex::c_1),
            .data_format = mask_scaler_format,
            .page_size = mask_scaler_tile_size,
        }}},
    });  // mask
    desc.cbs.push_back(CBDescriptor{
        .total_size = mask_scaler_tile_size,
        .core_ranges = all_cores,
        .format_descriptors = {{CBFormatDescriptor{
            .buffer_index = static_cast<uint8_t>(tt::CBIndex::c_2),
            .data_format = mask_scaler_format,
            .page_size = mask_scaler_tile_size,
        }}},
    });  // max scaler
    desc.cbs.push_back(CBDescriptor{
        .total_size = mask_scaler_tile_size,
        .core_ranges = all_cores,
        .format_descriptors = {{CBFormatDescriptor{
            .buffer_index = static_cast<uint8_t>(tt::CBIndex::c_3),
            .data_format = mask_scaler_format,
            .page_size = mask_scaler_tile_size,
        }}},
    });  // sum scaler
    desc.cbs.push_back(CBDescriptor{
        .total_size = Ht * in_tile_size,
        .core_ranges = all_cores,
        .format_descriptors = {{CBFormatDescriptor{
            .buffer_index = static_cast<uint8_t>(tt::CBIndex::c_16),
            .data_format = data_format,
            .page_size = in_tile_size,
        }}},
    });  // output
    desc.cbs.push_back(CBDescriptor{
        .total_size = Ht * intermed_tile_size,
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
    });  // reduce
    desc.cbs.push_back(CBDescriptor{
        .total_size = intermed_tile_size,
        .core_ranges = all_cores,
        .format_descriptors = {{CBFormatDescriptor{
            .buffer_index = static_cast<uint8_t>(tt::CBIndex::c_26),
            .data_format = intermed_data_format,
            .page_size = intermed_tile_size,
        }}},
    });  // max
    desc.cbs.push_back(CBDescriptor{
        .total_size = Ht * intermed_tile_size,
        .core_ranges = all_cores,
        .format_descriptors = {{CBFormatDescriptor{
            .buffer_index = static_cast<uint8_t>(tt::CBIndex::c_27),
            .data_format = intermed_data_format,
            .page_size = intermed_tile_size,
        }}},
    });  // x - max
    desc.cbs.push_back(CBDescriptor{
        .total_size = intermed_tile_size,
        .core_ranges = all_cores,
        .format_descriptors = {{CBFormatDescriptor{
            .buffer_index = static_cast<uint8_t>(tt::CBIndex::c_28),
            .data_format = intermed_data_format,
            .page_size = intermed_tile_size,
        }}},
    });  // tmp

    // Data movement kernel
    std::vector<uint32_t> reader_ct_args = {static_cast<uint32_t>(input.dtype() == DataType::FLOAT32)};
    TensorAccessorArgs(*input.buffer()).append_to(reader_ct_args);

    KernelDescriptor reader_desc;
    reader_desc.kernel_source = std::string(SOFTMAX_KERNEL_PATH_GENERAL) + "/reader_moreh_softmax_h.cpp";
    reader_desc.source_type = KernelDescriptor::SourceType::FILE_PATH;
    reader_desc.core_ranges = all_cores;
    reader_desc.compile_time_args = reader_ct_args;
    reader_desc.config = ReaderConfigDescriptor{};

    std::vector<uint32_t> writer_ct_args = {};
    TensorAccessorArgs(*output_tensor.buffer()).append_to(writer_ct_args);

    KernelDescriptor writer_desc;
    writer_desc.kernel_source = std::string(SOFTMAX_KERNEL_PATH_GENERAL) + "/writer_moreh_softmax_h.cpp";
    writer_desc.source_type = KernelDescriptor::SourceType::FILE_PATH;
    writer_desc.core_ranges = all_cores;
    writer_desc.compile_time_args = writer_ct_args;
    writer_desc.config = WriterConfigDescriptor{};

    // Kernel constants
    std::map<std::string, std::string> compute_defines;
    compute_defines["SOFTMAX"] = "1";
    // Enable FP32_DEST_ACC_EN for format reconfiguration in moreh compute helpers when using mixed
    // data formats (Bfp8_b input with Float16_b intermediates/mask/scaler) (issue #32934).
    if (fp32_dest_acc_en || data_format == tt::DataFormat::Bfp8_b) {
        compute_defines["FP32_DEST_ACC_EN"] = "1";
    }

    // Compute kernel
    KernelDescriptor compute_desc_g1;
    compute_desc_g1.kernel_source = std::string(SOFTMAX_KERNEL_PATH_GENERAL) + "/moreh_softmax_h.cpp";
    compute_desc_g1.source_type = KernelDescriptor::SourceType::FILE_PATH;
    compute_desc_g1.core_ranges = core_group_1;
    compute_desc_g1.compile_time_args = {num_tiles_per_core_group_1, Ht};
    compute_desc_g1.defines = KernelDescriptor::Defines(compute_defines.begin(), compute_defines.end());
    compute_desc_g1.config = ComputeConfigDescriptor{
        .math_fidelity = math_fidelity,
        .fp32_dest_acc_en = fp32_dest_acc_en,
        .dst_full_sync_en = dst_full_sync_en,
        .math_approx_mode = math_approx_mode,
    };

    KernelDescriptor compute_desc_g2;
    if (num_tiles_per_core_group_2 > 0) {
        compute_desc_g2.kernel_source = std::string(SOFTMAX_KERNEL_PATH_GENERAL) + "/moreh_softmax_h.cpp";
        compute_desc_g2.source_type = KernelDescriptor::SourceType::FILE_PATH;
        compute_desc_g2.core_ranges = core_group_2;
        compute_desc_g2.compile_time_args = {num_tiles_per_core_group_2, Ht};
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
        const CoreCoord core = {(i / core_h) + core_x_offset, (i % core_h) + core_y_offset};
        uint32_t num_tiles_per_core = 0;
        if (core_group_1.contains(core)) {
            num_tiles_per_core = num_tiles_per_core_group_1;
        } else if (core_group_2.contains(core)) {
            num_tiles_per_core = num_tiles_per_core_group_2;
        } else {
            TT_THROW("Core not in specified core ranges");
        }

        uint32_t mask_h = input.logical_shape()[-2] % tile_height;
        if (mask_h == 0) {
            mask_h = tile_height;
        }

        // Reader computes the reduce scaler in-kernel; only shape-derived args are passed.
        reader_desc.emplace_runtime_args(core, {input.buffer(), num_tiles_per_core, tile_offset, Ht, Wt, mask_h});

        writer_desc.emplace_runtime_args(core, {output_tensor.buffer(), num_tiles_per_core, tile_offset, Ht, Wt});

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
