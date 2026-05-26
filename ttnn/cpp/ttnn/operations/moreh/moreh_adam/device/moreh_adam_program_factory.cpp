// SPDX-FileCopyrightText: © 2024 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include <bit>
#include <vector>

#include "moreh_adam_device_operation.hpp"
#include <tt-metalium/work_split.hpp>
#include "ttnn/operations/core/compute_kernel/compute_kernel_config.hpp"
#include "ttnn/operations/moreh/moreh_helper_functions.hpp"
#include <tt-metalium/tensor_accessor_args.hpp>

namespace ttnn::operations::moreh::moreh_adam {

using namespace tt::tt_metal;

static constexpr const char* READER_KERNEL_PATH =
    "ttnn/cpp/ttnn/operations/moreh/moreh_adam/device/kernels/reader_moreh_adam.cpp";
static constexpr const char* WRITER_KERNEL_PATH =
    "ttnn/cpp/ttnn/operations/moreh/moreh_adam/device/kernels/writer_moreh_adam.cpp";
static constexpr const char* COMPUTE_KERNEL_PATH =
    "ttnn/cpp/ttnn/operations/moreh/moreh_adam/device/kernels/moreh_adam.cpp";

ProgramDescriptor MorehAdamOperation::create_descriptor(
    const operation_attributes_t& operation_attributes,
    const tensor_args_t& tensor_args,
    tensor_return_value_t& output_tensor) {
    const auto& param_in = tensor_args.param_in;
    const auto& grad = tensor_args.grad;
    const auto& exp_avg_in = tensor_args.exp_avg_in;
    const auto& exp_avg_sq_in = tensor_args.exp_avg_sq_in;

    auto& output_tensors = output_tensor;

    auto max_exp_avg_sq_in = tensor_args.max_exp_avg_sq_in;

    auto& param_out = output_tensors.at(0).value();
    auto& exp_avg_out = output_tensors.at(1).value();
    auto& exp_avg_sq_out = output_tensors.at(2).value();
    auto max_exp_avg_sq_out = output_tensors.at(3);

    auto lr = operation_attributes.lr;
    auto beta1 = operation_attributes.beta1;
    auto beta2 = operation_attributes.beta2;
    auto eps = operation_attributes.eps;
    auto weight_decay = operation_attributes.weight_decay;
    auto step = operation_attributes.step;
    auto amsgrad = operation_attributes.amsgrad;

    auto compute_kernel_config = operation_attributes.compute_kernel_config;

    uint32_t num_tiles = param_in.physical_volume() / tt::constants::TILE_HW;

    ////////////////////////////////////////////////////////////////////////////
    //                      Device Setup
    ////////////////////////////////////////////////////////////////////////////
    IDevice* device = param_in.device();
    auto grid = device->compute_with_storage_grid_size();
    const auto num_cores_y = grid.y;

    auto [num_cores, all_cores, core_group_1, core_group_2, num_tiles_per_core_group_1, num_tiles_per_core_group_2] =
        split_work_to_cores(grid, num_tiles);

    auto arch = param_in.device()->arch();
    auto [math_fidelity, math_approx_mode, fp32_dest_acc_en, packer_l1_acc, dst_full_sync_en] =
        get_compute_kernel_config_args(arch, compute_kernel_config);

    ////////////////////////////////////////////////////////////////////////////
    //                         CircularBuffer Setup
    ////////////////////////////////////////////////////////////////////////////
    auto data_format = datatype_to_dataformat_converter(param_in.dtype());
    auto intermed_cb_format = fp32_dest_acc_en ? tt::DataFormat::Float32 : data_format;
    const uint32_t data_tile_size = tile_size(data_format);
    const uint32_t intermed_tile_size = tile_size(intermed_cb_format);

    ProgramDescriptor desc;

    desc.cbs.push_back(CBDescriptor{
        .total_size = 1 * data_tile_size,
        .core_ranges = all_cores,
        .format_descriptors = {{CBFormatDescriptor{
            .buffer_index = tt::CBIndex::c_0, .data_format = data_format, .page_size = data_tile_size}}},
    });  // param_in
    desc.cbs.push_back(CBDescriptor{
        .total_size = 1 * data_tile_size,
        .core_ranges = all_cores,
        .format_descriptors = {{CBFormatDescriptor{
            .buffer_index = tt::CBIndex::c_1, .data_format = data_format, .page_size = data_tile_size}}},
    });  // grad
    desc.cbs.push_back(CBDescriptor{
        .total_size = 1 * data_tile_size,
        .core_ranges = all_cores,
        .format_descriptors = {{CBFormatDescriptor{
            .buffer_index = tt::CBIndex::c_2, .data_format = data_format, .page_size = data_tile_size}}},
    });  // exp_avg_in
    desc.cbs.push_back(CBDescriptor{
        .total_size = 1 * data_tile_size,
        .core_ranges = all_cores,
        .format_descriptors = {{CBFormatDescriptor{
            .buffer_index = tt::CBIndex::c_3, .data_format = data_format, .page_size = data_tile_size}}},
    });  // exp_avg_sq_in
    desc.cbs.push_back(CBDescriptor{
        .total_size = 1 * data_tile_size,
        .core_ranges = all_cores,
        .format_descriptors = {{CBFormatDescriptor{
            .buffer_index = tt::CBIndex::c_4, .data_format = data_format, .page_size = data_tile_size}}},
    });  // max_exp_avg_sq_in (optional)
    desc.cbs.push_back(CBDescriptor{
        .total_size = 5 * intermed_tile_size,
        .core_ranges = all_cores,
        .format_descriptors = {{CBFormatDescriptor{
            .buffer_index = tt::CBIndex::c_5, .data_format = intermed_cb_format, .page_size = intermed_tile_size}}},
    });  // lr, beta1, beta2, eps, weight_decay
    desc.cbs.push_back(CBDescriptor{
        .total_size = 1 * intermed_tile_size,
        .core_ranges = all_cores,
        .format_descriptors = {{CBFormatDescriptor{
            .buffer_index = tt::CBIndex::c_6, .data_format = intermed_cb_format, .page_size = intermed_tile_size}}},
    });  // 1.0f

    // Intermediate CBs (c_24 through c_31)
    for (uint8_t cb_idx = static_cast<uint8_t>(tt::CBIndex::c_24); cb_idx <= static_cast<uint8_t>(tt::CBIndex::c_31);
         ++cb_idx) {
        desc.cbs.push_back(CBDescriptor{
            .total_size = 1 * intermed_tile_size,
            .core_ranges = all_cores,
            .format_descriptors = {{CBFormatDescriptor{
                .buffer_index = cb_idx, .data_format = intermed_cb_format, .page_size = intermed_tile_size}}},
        });
    }

    desc.cbs.push_back(CBDescriptor{
        .total_size = 1 * data_tile_size,
        .core_ranges = all_cores,
        .format_descriptors = {{CBFormatDescriptor{
            .buffer_index = tt::CBIndex::c_16, .data_format = data_format, .page_size = data_tile_size}}},
    });  // param_out
    desc.cbs.push_back(CBDescriptor{
        .total_size = 1 * data_tile_size,
        .core_ranges = all_cores,
        .format_descriptors = {{CBFormatDescriptor{
            .buffer_index = tt::CBIndex::c_17, .data_format = data_format, .page_size = data_tile_size}}},
    });  // exp_avg_out
    desc.cbs.push_back(CBDescriptor{
        .total_size = 1 * data_tile_size,
        .core_ranges = all_cores,
        .format_descriptors = {{CBFormatDescriptor{
            .buffer_index = tt::CBIndex::c_18, .data_format = data_format, .page_size = data_tile_size}}},
    });  // exp_avg_sq_out
    desc.cbs.push_back(CBDescriptor{
        .total_size = 1 * data_tile_size,
        .core_ranges = all_cores,
        .format_descriptors = {{CBFormatDescriptor{
            .buffer_index = tt::CBIndex::c_19, .data_format = data_format, .page_size = data_tile_size}}},
    });  // max_exp_avg_sq_out (optional)

    ////////////////////////////////////////////////////////////////////////////
    //                      DataMovementKernel SetUp
    ////////////////////////////////////////////////////////////////////////////
    KernelDescriptor::CompileTimeArgs reader_ct_args;
    TensorAccessorArgs(*param_in.buffer()).append_to(reader_ct_args);
    TensorAccessorArgs(*grad.buffer()).append_to(reader_ct_args);
    TensorAccessorArgs(*exp_avg_in.buffer()).append_to(reader_ct_args);
    TensorAccessorArgs(*exp_avg_sq_in.buffer()).append_to(reader_ct_args);
    if (max_exp_avg_sq_in.has_value()) {
        TensorAccessorArgs(*max_exp_avg_sq_in.value().buffer()).append_to(reader_ct_args);
    }

    KernelDescriptor::CompileTimeArgs writer_ct_args;
    TensorAccessorArgs(*param_out.buffer()).append_to(writer_ct_args);
    TensorAccessorArgs(*exp_avg_out.buffer()).append_to(writer_ct_args);
    TensorAccessorArgs(*exp_avg_sq_out.buffer()).append_to(writer_ct_args);
    if (max_exp_avg_sq_out.has_value()) {
        TensorAccessorArgs(*max_exp_avg_sq_out.value().buffer()).append_to(writer_ct_args);
    }

    KernelDescriptor::Defines data_movement_defines;
    KernelDescriptor::Defines compute_defines;
    if (amsgrad) {
        data_movement_defines.emplace_back("AMSGRAD", "1");
        compute_defines.emplace_back("AMSGRAD", "1");
    }
    if (fp32_dest_acc_en) {
        data_movement_defines.emplace_back("FP32_DEST_ACC_EN", "1");
        compute_defines.emplace_back("FP32_DEST_ACC_EN", "1");
    }

    KernelDescriptor reader_desc;
    reader_desc.kernel_source = READER_KERNEL_PATH;
    reader_desc.source_type = KernelDescriptor::SourceType::FILE_PATH;
    reader_desc.core_ranges = all_cores;
    reader_desc.compile_time_args = std::move(reader_ct_args);
    reader_desc.defines = data_movement_defines;
    reader_desc.config = ReaderConfigDescriptor{};
    reader_desc.runtime_args.reserve(num_cores);

    KernelDescriptor writer_desc;
    writer_desc.kernel_source = WRITER_KERNEL_PATH;
    writer_desc.source_type = KernelDescriptor::SourceType::FILE_PATH;
    writer_desc.core_ranges = all_cores;
    writer_desc.compile_time_args = std::move(writer_ct_args);
    writer_desc.defines = data_movement_defines;
    writer_desc.config = WriterConfigDescriptor{};
    writer_desc.runtime_args.reserve(num_cores);

    ////////////////////////////////////////////////////////////////////////////
    //                      ComputeKernel SetUp
    ////////////////////////////////////////////////////////////////////////////
    ComputeConfigDescriptor compute_config{
        .math_fidelity = math_fidelity,
        .fp32_dest_acc_en = fp32_dest_acc_en,
        .math_approx_mode = math_approx_mode,
    };

    KernelDescriptor compute_desc_1;
    compute_desc_1.kernel_source = COMPUTE_KERNEL_PATH;
    compute_desc_1.source_type = KernelDescriptor::SourceType::FILE_PATH;
    compute_desc_1.core_ranges = core_group_1;
    compute_desc_1.compile_time_args = {num_tiles_per_core_group_1};
    compute_desc_1.defines = compute_defines;
    compute_desc_1.config = compute_config;

    KernelDescriptor compute_desc_2;
    bool has_core_group_2 = !core_group_2.ranges().empty();
    if (has_core_group_2) {
        compute_desc_2.kernel_source = COMPUTE_KERNEL_PATH;
        compute_desc_2.source_type = KernelDescriptor::SourceType::FILE_PATH;
        compute_desc_2.core_ranges = core_group_2;
        compute_desc_2.compile_time_args = {num_tiles_per_core_group_2};
        compute_desc_2.defines = compute_defines;
        compute_desc_2.config = compute_config;
    }

    ////////////////////////////////////////////////////////////////////////////
    //                      RuntimeArgs SetUp
    ////////////////////////////////////////////////////////////////////////////
    const auto param_in_addr = param_in.buffer()->address();
    const auto grad_addr = grad.buffer()->address();
    const auto exp_avg_in_addr = exp_avg_in.buffer()->address();
    const auto exp_avg_sq_in_addr = exp_avg_sq_in.buffer()->address();
    const auto max_exp_avg_sq_in_addr =
        max_exp_avg_sq_in.has_value() ? max_exp_avg_sq_in.value().buffer()->address() : 0;

    const auto param_out_addr = param_out.buffer()->address();
    const auto exp_avg_out_addr = exp_avg_out.buffer()->address();
    const auto exp_avg_sq_out_addr = exp_avg_sq_out.buffer()->address();
    const auto max_exp_avg_sq_out_addr = max_exp_avg_sq_out.has_value() ? max_exp_avg_sq_out->buffer()->address() : 0;

    const uint32_t f2u_lr = std::bit_cast<uint32_t>(lr);
    const uint32_t f2u_beta1 = std::bit_cast<uint32_t>(beta1);
    const uint32_t f2u_beta2 = std::bit_cast<uint32_t>(beta2);
    const uint32_t f2u_eps = std::bit_cast<uint32_t>(eps);
    const uint32_t f2u_weight_decay = std::bit_cast<uint32_t>(weight_decay);

    for (uint32_t i = 0, tile_offset = 0; i < num_cores; ++i) {
        CoreCoord core = {i / num_cores_y, i % num_cores_y};

        uint32_t num_tiles_per_core = 0;
        if (core_group_1.contains(core)) {
            num_tiles_per_core = num_tiles_per_core_group_1;
        } else if (core_group_2.contains(core)) {
            num_tiles_per_core = num_tiles_per_core_group_2;
        } else {
            TT_THROW("Core not in specified core ranges.");
        }

        reader_desc.runtime_args.emplace_back(
            core,
            KernelDescriptor::CoreRuntimeArgs{
                param_in_addr,
                grad_addr,
                exp_avg_in_addr,
                exp_avg_sq_in_addr,
                max_exp_avg_sq_in_addr,
                f2u_lr,
                f2u_beta1,
                f2u_beta2,
                f2u_eps,
                f2u_weight_decay,
                step,
                static_cast<uint32_t>(amsgrad),
                num_tiles_per_core,
                tile_offset});

        writer_desc.runtime_args.emplace_back(
            core,
            KernelDescriptor::CoreRuntimeArgs{
                param_out_addr,
                exp_avg_out_addr,
                exp_avg_sq_out_addr,
                max_exp_avg_sq_out_addr,
                num_tiles_per_core,
                tile_offset});

        // compute — runtime args go to the correct kernel descriptor
        KernelDescriptor::CoreRuntimeArgs compute_rt{step};
        if (core_group_1.contains(core)) {
            compute_desc_1.runtime_args.emplace_back(core, std::move(compute_rt));
        } else {
            compute_desc_2.runtime_args.emplace_back(core, std::move(compute_rt));
        }

        tile_offset += num_tiles_per_core;
    }

    desc.kernels.push_back(std::move(reader_desc));
    desc.kernels.push_back(std::move(writer_desc));
    desc.kernels.push_back(std::move(compute_desc_1));
    if (has_core_group_2) {
        desc.kernels.push_back(std::move(compute_desc_2));
    }

    return desc;
}

}  // namespace ttnn::operations::moreh::moreh_adam
