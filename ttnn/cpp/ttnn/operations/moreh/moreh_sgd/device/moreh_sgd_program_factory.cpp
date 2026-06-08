// SPDX-FileCopyrightText: © 2024 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include <bit>
#include <vector>

#include "moreh_sgd_device_operation.hpp"
#include <tt-metalium/work_split.hpp>
#include <tt-metalium/tensor_accessor_args.hpp>
#include "ttnn/operations/moreh/moreh_helper_functions.hpp"
#include "ttnn/operations/core/compute_kernel/compute_kernel_config.hpp"

namespace ttnn::operations::moreh::moreh_sgd {

using namespace tt::tt_metal;

static constexpr const char* READER_KERNEL_PATH =
    "ttnn/cpp/ttnn/operations/moreh/moreh_sgd/device/kernels/reader_moreh_sgd.cpp";
static constexpr const char* WRITER_KERNEL_PATH =
    "ttnn/cpp/ttnn/operations/moreh/moreh_sgd/device/kernels/writer_moreh_sgd.cpp";
static constexpr const char* COMPUTE_KERNEL_PATH =
    "ttnn/cpp/ttnn/operations/moreh/moreh_sgd/device/kernels/moreh_sgd.cpp";

ProgramDescriptor MorehSgdOperation::create_descriptor(
    const operation_attributes_t& operation_attributes,
    const tensor_args_t& tensor_args,
    tensor_return_value_t& output_tensor) {
    const auto& param_in = tensor_args.param_in;
    const auto& grad = tensor_args.grad;
    const std::optional<Tensor>& momentum_buffer_in = tensor_args.momentum_buffer_in;

    auto& output_tensors = output_tensor;
    auto& param_out = output_tensors.at(0).value();
    auto& momentum_buffer_out = output_tensors.at(1);

    auto lr = operation_attributes.lr;
    auto momentum = operation_attributes.momentum;
    auto dampening = operation_attributes.dampening;
    auto weight_decay = operation_attributes.weight_decay;
    auto nesterov = operation_attributes.nesterov;
    auto momentum_initialized = operation_attributes.momentum_initialized;

    auto compute_kernel_config = operation_attributes.compute_kernel_config;

    auto shape = param_in.logical_shape();
    auto H = shape[-2];
    auto W = shape[-1];
    auto num = param_in.physical_volume() / H / W;
    auto Ht = H / tt::constants::TILE_HEIGHT;
    auto Wt = W / tt::constants::TILE_WIDTH;

    bool has_momentum_buffer_out = momentum_buffer_out.has_value();

    ////////////////////////////////////////////////////////////////////////////
    //                      Device Setup
    ////////////////////////////////////////////////////////////////////////////
    IDevice* device = param_in.device();
    auto grid = device->compute_with_storage_grid_size();
    uint32_t units_to_divide = num * Ht * Wt;
    uint32_t core_h = grid.y;

    auto [num_cores, all_cores, core_group_1, core_group_2, num_tiles_per_core_group_1, num_tiles_per_core_group_2] =
        split_work_to_cores(grid, units_to_divide);

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
        .total_size = 2 * data_tile_size,
        .core_ranges = all_cores,
        .format_descriptors = {{CBFormatDescriptor{
            .buffer_index = tt::CBIndex::c_0, .data_format = data_format, .page_size = data_tile_size}}},
    });  // param_in
    desc.cbs.push_back(CBDescriptor{
        .total_size = 2 * data_tile_size,
        .core_ranges = all_cores,
        .format_descriptors = {{CBFormatDescriptor{
            .buffer_index = tt::CBIndex::c_1, .data_format = data_format, .page_size = data_tile_size}}},
    });  // grad
    desc.cbs.push_back(CBDescriptor{
        .total_size = 2 * data_tile_size,
        .core_ranges = all_cores,
        .format_descriptors = {{CBFormatDescriptor{
            .buffer_index = tt::CBIndex::c_2, .data_format = data_format, .page_size = data_tile_size}}},
    });  // momentum_in
    desc.cbs.push_back(CBDescriptor{
        .total_size = 2 * data_tile_size,
        .core_ranges = all_cores,
        .format_descriptors = {{CBFormatDescriptor{
            .buffer_index = tt::CBIndex::c_16, .data_format = data_format, .page_size = data_tile_size}}},
    });  // param_out
    desc.cbs.push_back(CBDescriptor{
        .total_size = 2 * data_tile_size,
        .core_ranges = all_cores,
        .format_descriptors = {{CBFormatDescriptor{
            .buffer_index = tt::CBIndex::c_17, .data_format = data_format, .page_size = data_tile_size}}},
    });  // momentum_out
    desc.cbs.push_back(CBDescriptor{
        .total_size = 5 * intermed_tile_size,
        .core_ranges = all_cores,
        .format_descriptors = {{CBFormatDescriptor{
            .buffer_index = tt::CBIndex::c_24, .data_format = intermed_cb_format, .page_size = intermed_tile_size}}},
    });  // cb_scalar_args (lr, momentum, dampening, weight_decay, one)
    desc.cbs.push_back(CBDescriptor{
        .total_size = 1 * intermed_tile_size,
        .core_ranges = all_cores,
        .format_descriptors = {{CBFormatDescriptor{
            .buffer_index = tt::CBIndex::c_25, .data_format = intermed_cb_format, .page_size = intermed_tile_size}}},
    });
    desc.cbs.push_back(CBDescriptor{
        .total_size = 1 * intermed_tile_size,
        .core_ranges = all_cores,
        .format_descriptors = {{CBFormatDescriptor{
            .buffer_index = tt::CBIndex::c_26, .data_format = intermed_cb_format, .page_size = intermed_tile_size}}},
    });
    desc.cbs.push_back(CBDescriptor{
        .total_size = 1 * intermed_tile_size,
        .core_ranges = all_cores,
        .format_descriptors = {{CBFormatDescriptor{
            .buffer_index = tt::CBIndex::c_27, .data_format = intermed_cb_format, .page_size = intermed_tile_size}}},
    });
    desc.cbs.push_back(CBDescriptor{
        .total_size = 1 * intermed_tile_size,
        .core_ranges = all_cores,
        .format_descriptors = {{CBFormatDescriptor{
            .buffer_index = tt::CBIndex::c_28, .data_format = intermed_cb_format, .page_size = intermed_tile_size}}},
    });

    ////////////////////////////////////////////////////////////////////////////
    //                         Kernels defines
    ////////////////////////////////////////////////////////////////////////////
    KernelDescriptor::Defines reader_defines;
    KernelDescriptor::Defines writer_defines;
    KernelDescriptor::Defines compute_defines;

    if (weight_decay != 0) {
        reader_defines.emplace_back("WEIGHT_DECAY", "1");
        compute_defines.emplace_back("WEIGHT_DECAY", "1");
    }

    if (momentum != 0) {
        reader_defines.emplace_back("MOMENTUM", "1");
        compute_defines.emplace_back("MOMENTUM", "1");
        writer_defines.emplace_back("MOMENTUM", "1");
    }

    if (momentum_initialized) {
        reader_defines.emplace_back("MOMENTUM_INITIALIZED", "1");
        compute_defines.emplace_back("MOMENTUM_INITIALIZED", "1");
    }

    if (nesterov) {
        reader_defines.emplace_back("NESTEROV", "1");
        compute_defines.emplace_back("NESTEROV", "1");
    }

    if (fp32_dest_acc_en) {
        reader_defines.emplace_back("FP32_DEST_ACC_EN", "1");
        compute_defines.emplace_back("FP32_DEST_ACC_EN", "1");
    }

    ////////////////////////////////////////////////////////////////////////////
    //                      DataMovementKernel SetUp
    ////////////////////////////////////////////////////////////////////////////
    KernelDescriptor::CompileTimeArgs reader_ct_args;
    TensorAccessorArgs(*param_in.buffer()).append_to(reader_ct_args);
    TensorAccessorArgs(*grad.buffer()).append_to(reader_ct_args);
    if (momentum_buffer_in.has_value()) {
        TensorAccessorArgs(*momentum_buffer_in->buffer()).append_to(reader_ct_args);
    }

    KernelDescriptor reader_desc;
    reader_desc.kernel_source = READER_KERNEL_PATH;
    reader_desc.source_type = KernelDescriptor::SourceType::FILE_PATH;
    reader_desc.core_ranges = all_cores;
    reader_desc.compile_time_args = std::move(reader_ct_args);
    reader_desc.defines = std::move(reader_defines);
    reader_desc.config = ReaderConfigDescriptor{};
    reader_desc.runtime_args.reserve(num_cores);

    KernelDescriptor::CompileTimeArgs writer_ct_args;
    TensorAccessorArgs(*param_out.buffer()).append_to(writer_ct_args);
    if (has_momentum_buffer_out) {
        TensorAccessorArgs(*momentum_buffer_out->buffer()).append_to(writer_ct_args);
    }

    KernelDescriptor writer_desc;
    writer_desc.kernel_source = WRITER_KERNEL_PATH;
    writer_desc.source_type = KernelDescriptor::SourceType::FILE_PATH;
    writer_desc.core_ranges = all_cores;
    writer_desc.compile_time_args = std::move(writer_ct_args);
    writer_desc.defines = std::move(writer_defines);
    writer_desc.config = WriterConfigDescriptor{};
    writer_desc.runtime_args.reserve(num_cores);

    ////////////////////////////////////////////////////////////////////////////
    //                      ComputeKernel SetUp
    ////////////////////////////////////////////////////////////////////////////
    ComputeConfigDescriptor compute_config{
        .math_fidelity = math_fidelity,
        .fp32_dest_acc_en = fp32_dest_acc_en,
        .dst_full_sync_en = dst_full_sync_en,
        .math_approx_mode = math_approx_mode,
    };

    // Compute kernel for core_group_1
    KernelDescriptor compute_desc_1;
    compute_desc_1.kernel_source = COMPUTE_KERNEL_PATH;
    compute_desc_1.source_type = KernelDescriptor::SourceType::FILE_PATH;
    compute_desc_1.core_ranges = core_group_1;
    compute_desc_1.compile_time_args = {num_tiles_per_core_group_1};
    compute_desc_1.defines = compute_defines;
    compute_desc_1.config = compute_config;

    // Compute kernel for core_group_2 (may be empty)
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
    const uint32_t u_lr = std::bit_cast<uint32_t>(lr);
    const uint32_t u_momentum = std::bit_cast<uint32_t>(momentum);
    const uint32_t u_dampening = std::bit_cast<uint32_t>(dampening);
    const uint32_t u_weight_decay = std::bit_cast<uint32_t>(weight_decay);
    const uint32_t u_one = std::bit_cast<uint32_t>(1.0f);

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

        reader_desc.emplace_runtime_args(
            core,
            {param_in.buffer(),
             grad.buffer(),
             momentum_buffer_in.has_value() ? momentum_buffer_in.value().buffer()->address() : 0u,
             num_tiles_per_core,
             tile_offset,
             u_lr,
             u_momentum,
             u_dampening,
             u_weight_decay,
             u_one});

        writer_desc.emplace_runtime_args(
            core,
            {param_out.buffer(),
             momentum_buffer_out.has_value() ? momentum_buffer_out.value().buffer()->address() : 0u,
             num_tiles_per_core,
             tile_offset});

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

}  // namespace ttnn::operations::moreh::moreh_sgd
