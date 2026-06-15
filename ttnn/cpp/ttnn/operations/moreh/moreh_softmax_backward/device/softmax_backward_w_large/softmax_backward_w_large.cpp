// SPDX-FileCopyrightText: © 2023 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include <bit>
#include <string>

#include "ttnn/operations/moreh/moreh_softmax_backward/device/moreh_softmax_backward_device_operation.hpp"
#include <tt-metalium/tensor_accessor_args.hpp>
#include "ttnn/operations/moreh/moreh_helper_functions.hpp"

namespace ttnn::operations::moreh::moreh_softmax_backward {

tt::tt_metal::ProgramDescriptor MorehSoftmaxBackwardOperation::MorehSoftmaxBackwardWLargeFactory::create_descriptor(
    const operation_attributes_t& operation_attributes,
    const tensor_args_t& tensor_args,
    tensor_return_value_t& input_grad) {
    using namespace tt;
    using namespace tt::tt_metal;

    log_info(tt::LogTest, "Large tensor algorithm selected");
    const auto& output = tensor_args.output_tensor;
    const auto& output_grad = tensor_args.output_grad_tensor;
    const auto op = operation_attributes.op;
    const auto& compute_kernel_config = operation_attributes.compute_kernel_config;

    auto* device = output_grad.device();
    auto grid_coord = device->compute_with_storage_grid_size();
    const CoreRange core_range({0, 0}, {grid_coord.x - 1, grid_coord.y - 1});
    // split work
    auto shape = input_grad.padded_shape();
    auto H = shape[-2];
    auto W = shape[-1];
    auto Ht = H / tt::constants::TILE_HEIGHT;
    auto Wt = W / tt::constants::TILE_WIDTH;

    auto num = input_grad.physical_volume() / H / W;

    uint32_t num_kernel_rows = num * Ht;
    uint32_t core_h = core_range.end_coord.y - core_range.start_coord.y + 1;

    auto [num_cores, all_cores, core_group_1, core_group_2, num_tiles_per_core_group_1, num_tiles_per_core_group_2] =
        split_work_to_cores_wt_core_range(core_range, num_kernel_rows);

    auto arch = input_grad.device()->arch();
    auto [math_fidelity, math_approx_mode, fp32_dest_acc_en, packer_l1_acc, dst_full_sync_en] =
        get_compute_kernel_config_args(arch, compute_kernel_config);

    ProgramDescriptor desc;

    // create circular buffers
    tt::DataFormat data_format = tt::tt_metal::datatype_to_dataformat_converter(input_grad.dtype());
    tt::DataFormat intermed_data_format = fp32_dest_acc_en ? tt::DataFormat::Float32 : data_format;
    const uint32_t tile_size_data = tile_size(data_format);
    const uint32_t tile_size_intermed = tile_size(intermed_data_format);

    // output
    desc.cbs.push_back(CBDescriptor{
        .total_size = 2 * tile_size_data,
        .core_ranges = all_cores,
        .format_descriptors = {{CBFormatDescriptor{
            .buffer_index = static_cast<uint8_t>(CBIndex::c_0),
            .data_format = data_format,
            .page_size = tile_size_data,
        }}},
    });
    // output_grad
    desc.cbs.push_back(CBDescriptor{
        .total_size = 2 * tile_size_data,
        .core_ranges = all_cores,
        .format_descriptors = {{CBFormatDescriptor{
            .buffer_index = static_cast<uint8_t>(CBIndex::c_1),
            .data_format = data_format,
            .page_size = tile_size_data,
        }}},
    });
    // scaler
    desc.cbs.push_back(CBDescriptor{
        .total_size = 1 * tile_size_data,
        .core_ranges = all_cores,
        .format_descriptors = {{CBFormatDescriptor{
            .buffer_index = static_cast<uint8_t>(CBIndex::c_2),
            .data_format = data_format,
            .page_size = tile_size_data,
        }}},
    });
    // mask
    desc.cbs.push_back(CBDescriptor{
        .total_size = 1 * tile_size_data,
        .core_ranges = all_cores,
        .format_descriptors = {{CBFormatDescriptor{
            .buffer_index = static_cast<uint8_t>(CBIndex::c_3),
            .data_format = data_format,
            .page_size = tile_size_data,
        }}},
    });
    // input_grad
    desc.cbs.push_back(CBDescriptor{
        .total_size = 2 * tile_size_data,
        .core_ranges = all_cores,
        .format_descriptors = {{CBFormatDescriptor{
            .buffer_index = static_cast<uint8_t>(CBIndex::c_16),
            .data_format = data_format,
            .page_size = tile_size_data,
        }}},
    });
    // output * output_grad
    desc.cbs.push_back(CBDescriptor{
        .total_size = 1 * tile_size_intermed,
        .core_ranges = all_cores,
        .format_descriptors = {{CBFormatDescriptor{
            .buffer_index = static_cast<uint8_t>(CBIndex::c_24),
            .data_format = intermed_data_format,
            .page_size = tile_size_intermed,
        }}},
    });
    // reduce
    desc.cbs.push_back(CBDescriptor{
        .total_size = 1 * tile_size_intermed,
        .core_ranges = all_cores,
        .format_descriptors = {{CBFormatDescriptor{
            .buffer_index = static_cast<uint8_t>(CBIndex::c_25),
            .data_format = intermed_data_format,
            .page_size = tile_size_intermed,
        }}},
    });
    // dy - sum
    desc.cbs.push_back(CBDescriptor{
        .total_size = 1 * tile_size_intermed,
        .core_ranges = all_cores,
        .format_descriptors = {{CBFormatDescriptor{
            .buffer_index = static_cast<uint8_t>(CBIndex::c_26),
            .data_format = intermed_data_format,
            .page_size = tile_size_intermed,
        }}},
    });
    // add(output * output_grad)
    desc.cbs.push_back(CBDescriptor{
        .total_size = 2 * tile_size_intermed,
        .core_ranges = all_cores,
        .format_descriptors = {{CBFormatDescriptor{
            .buffer_index = static_cast<uint8_t>(CBIndex::c_27),
            .data_format = intermed_data_format,
            .page_size = tile_size_intermed,
        }}},
    });

    // create read/write kernel
    std::map<std::string, std::string> reader_defines_map;
    std::map<std::string, std::string> compute_defines_map;
    if (op == MorehSoftmaxBackwardOp::SOFTMAX) {
        compute_defines_map["SOFTMAX"] = "1";
    } else {
        compute_defines_map["SOFTMIN"] = "1";
    }

    if (op == MorehSoftmaxBackwardOp::LOGSOFTMAX) {
        compute_defines_map["LOG"] = "1";
        reader_defines_map["LOG"] = "1";
    }

    if (fp32_dest_acc_en) {
        compute_defines_map["FP32_DEST_ACC_EN"] = "1";
    }
    KernelDescriptor::Defines reader_defines(reader_defines_map.begin(), reader_defines_map.end());
    KernelDescriptor::Defines compute_defines(compute_defines_map.begin(), compute_defines_map.end());

    KernelDescriptor::CompileTimeArgs reader_ct_args = {};
    TensorAccessorArgs(*output.buffer()).append_to(reader_ct_args);
    TensorAccessorArgs(*output_grad.buffer()).append_to(reader_ct_args);

    KernelDescriptor reader_desc;
    reader_desc.kernel_source =
        "ttnn/cpp/ttnn/operations/moreh/moreh_softmax_backward/device/kernels/"
        "reader_moreh_softmax_backward_w_large.cpp";
    reader_desc.source_type = KernelDescriptor::SourceType::FILE_PATH;
    reader_desc.core_ranges = all_cores;
    reader_desc.compile_time_args = std::move(reader_ct_args);
    reader_desc.defines = std::move(reader_defines);
    reader_desc.config = ReaderConfigDescriptor{};

    KernelDescriptor::CompileTimeArgs writer_ct_args = {};
    TensorAccessorArgs(*input_grad.buffer()).append_to(writer_ct_args);

    KernelDescriptor writer_desc;
    writer_desc.kernel_source =
        "ttnn/cpp/ttnn/operations/moreh/moreh_softmax_backward/device/kernels/writer_moreh_softmax_w.cpp";
    writer_desc.source_type = KernelDescriptor::SourceType::FILE_PATH;
    writer_desc.core_ranges = all_cores;
    writer_desc.compile_time_args = std::move(writer_ct_args);
    writer_desc.config = WriterConfigDescriptor{};

    std::vector<UnpackToDestMode> unpack_to_dest_mode(NUM_CIRCULAR_BUFFERS, UnpackToDestMode::Default);

    // create compute kernel
    KernelDescriptor compute_desc_1;
    compute_desc_1.kernel_source =
        "ttnn/cpp/ttnn/operations/moreh/moreh_softmax_backward/device/kernels/moreh_softmax_backward_w_large.cpp";
    compute_desc_1.source_type = KernelDescriptor::SourceType::FILE_PATH;
    compute_desc_1.core_ranges = core_group_1;
    compute_desc_1.compile_time_args = {num_tiles_per_core_group_1, Wt};
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
        compute_desc_2.kernel_source =
            "ttnn/cpp/ttnn/operations/moreh/moreh_softmax_backward/device/kernels/moreh_softmax_backward_w_large.cpp";
        compute_desc_2.source_type = KernelDescriptor::SourceType::FILE_PATH;
        compute_desc_2.core_ranges = core_group_2;
        compute_desc_2.compile_time_args = {num_tiles_per_core_group_2, Wt};
        compute_desc_2.defines = compute_defines;
        compute_desc_2.config = ComputeConfigDescriptor{
            .math_fidelity = math_fidelity,
            .fp32_dest_acc_en = fp32_dest_acc_en,
            .dst_full_sync_en = dst_full_sync_en,
            .unpack_to_dest_mode = unpack_to_dest_mode,
            .math_approx_mode = math_approx_mode,
        };
    }

    // Set Runtime Args
    auto core_x_offset = core_range.start_coord.x;
    auto core_y_offset = core_range.start_coord.y;

    for (uint32_t i = 0, tile_offset = 0; i < num_cores; i++) {
        CoreCoord core = {(i / core_h) + core_x_offset, (i % core_h) + core_y_offset};
        uint32_t num_tiles_per_core;
        if (core_group_1.contains(core)) {
            num_tiles_per_core = num_tiles_per_core_group_1;
        } else if (core_group_2.contains(core)) {
            num_tiles_per_core = num_tiles_per_core_group_2;
        } else {
            TT_THROW("Core not in specified core ranges");
        }

        float scaler = 1.0f;
        uint32_t mask_w = input_grad.logical_shape()[-1] % tt::constants::TILE_WIDTH;
        if (mask_w == 0) {
            mask_w = tt::constants::TILE_WIDTH;
        }
        reader_desc.emplace_runtime_args(
            core,
            {output.buffer(),
             output_grad.buffer(),
             num_tiles_per_core,
             tile_offset,
             Wt,
             std::bit_cast<uint32_t>(scaler),
             mask_w});

        writer_desc.emplace_runtime_args(core, {input_grad.buffer(), num_tiles_per_core, tile_offset, Wt});

        tile_offset += num_tiles_per_core * Wt;
    }

    desc.kernels.push_back(std::move(reader_desc));
    desc.kernels.push_back(std::move(writer_desc));
    desc.kernels.push_back(std::move(compute_desc_1));
    if (has_core_group_2) {
        desc.kernels.push_back(std::move(compute_desc_2));
    }

    return desc;
}

}  // namespace ttnn::operations::moreh::moreh_softmax_backward
