// SPDX-FileCopyrightText: © 2024 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include <optional>
#include <string>

#include <tt-metalium/constants.hpp>
#include "moreh_nll_loss_unreduced_backward_device_operation.hpp"
#include <tt-metalium/math.hpp>
#include <tt-metalium/work_split.hpp>
#include <tt-metalium/tensor_accessor_args.hpp>
#include "ttnn/operations/moreh/moreh_helper_functions.hpp"

namespace ttnn::operations::moreh::moreh_nll_loss_unreduced_backward {

using namespace tt;
using namespace tt::tt_metal;

namespace {

// Helper: append a CB with a given tile-count (skips creation when num_tiles == 0).
void push_cb(
    ProgramDescriptor& desc,
    const CoreRangeSet& core_ranges,
    uint8_t buffer_index,
    uint32_t num_tiles,
    tt::DataFormat data_format) {
    if (num_tiles == 0) {
        return;
    }
    const auto tile_sz = tt::tile_size(data_format);
    desc.cbs.push_back(CBDescriptor{
        .total_size = num_tiles * tile_sz,
        .core_ranges = core_ranges,
        .format_descriptors = {{CBFormatDescriptor{
            .buffer_index = buffer_index,
            .data_format = data_format,
            .page_size = tile_sz,
        }}},
    });
}

}  // namespace

tt::tt_metal::ProgramDescriptor moreh_nll_loss_unreduced_backward_impl_2d(
    const Tensor& target,
    const std::optional<Tensor>& weight,
    const Tensor& output_grad,
    const Tensor& input_grad,
    const uint32_t ignore_index,
    const DeviceComputeKernelConfig compute_kernel_config) {
    // split work

    // input_grad: (N, C)
    auto input_grad_shape = input_grad.padded_shape();
    auto N = input_grad_shape[0];
    uint32_t channel_size = input_grad_shape[1];

    const bool weight_has_value = weight.has_value();

    tt::tt_metal::IDevice* device = target.device();
    auto grid = device->compute_with_storage_grid_size();
    uint32_t core_h = grid.y;

    uint32_t units_to_divide = input_grad.physical_volume() / tt::constants::TILE_HEIGHT / tt::constants::TILE_WIDTH;

    auto [num_cores, all_cores, core_group_1, core_group_2, units_per_core_group_1, units_per_core_group_2] =
        split_work_to_cores(grid, units_to_divide);

    auto [math_fidelity, math_approx_mode, fp32_dest_acc_en, packer_l1_acc, dst_full_sync_en] =
        get_compute_kernel_config_args(device->arch(), compute_kernel_config);

    ProgramDescriptor desc;

    // create circular buffers
    tt::DataFormat data_format = tt::tt_metal::datatype_to_dataformat_converter(input_grad.dtype());

    auto Ct = tt::div_up(channel_size, tt::constants::TILE_WIDTH);
    auto Nt = tt::div_up(N, tt::constants::TILE_WIDTH);

    push_cb(desc, all_cores, static_cast<uint8_t>(tt::CBIndex::c_0), 1, tt::DataFormat::Int32);  // target
    push_cb(desc, all_cores, static_cast<uint8_t>(tt::CBIndex::c_1), Nt, data_format);           // output_grad
    push_cb(
        desc, all_cores, static_cast<uint8_t>(tt::CBIndex::c_2), weight_has_value ? Ct : 0u, data_format);  // weight
    push_cb(desc, all_cores, static_cast<uint8_t>(tt::CBIndex::c_16), 1, data_format);  // input_grad

    // create read/write kernel
    KernelDescriptor::CompileTimeArgs reader_compile_time_args{};
    TensorAccessorArgs(*target.buffer()).append_to(reader_compile_time_args);
    TensorAccessorArgs(*output_grad.buffer()).append_to(reader_compile_time_args);
    TensorAccessorArgs(weight.has_value() ? weight.value().buffer() : nullptr).append_to(reader_compile_time_args);

    KernelDescriptor::CompileTimeArgs writer_compile_time_args{};
    TensorAccessorArgs(*input_grad.buffer()).append_to(writer_compile_time_args);

    KernelDescriptor::Defines reader_defines;
    KernelDescriptor::Defines writer_defines;

    if (weight_has_value) {
        reader_defines.emplace_back("WEIGHT", "1");
    }

    if (fp32_dest_acc_en) {
        reader_defines.emplace_back("FP32_DEST_ACC_EN", "1");
    }

    const auto* const reader_kernel_file =
        "ttnn/cpp/ttnn/operations/moreh/moreh_nll_loss_unreduced_backward/device/kernels/"
        "reader_moreh_nll_loss_unreduced_backward_2d.cpp";
    const auto* const writer_kernel_file =
        "ttnn/cpp/ttnn/operations/moreh/moreh_nll_loss_unreduced_backward/device/kernels/"
        "writer_moreh_nll_loss_unreduced_backward.cpp";

    KernelDescriptor reader_desc;
    reader_desc.kernel_source = reader_kernel_file;
    reader_desc.source_type = KernelDescriptor::SourceType::FILE_PATH;
    reader_desc.core_ranges = all_cores;
    reader_desc.compile_time_args = std::move(reader_compile_time_args);
    reader_desc.defines = std::move(reader_defines);
    reader_desc.config = ReaderConfigDescriptor{};

    KernelDescriptor writer_desc;
    writer_desc.kernel_source = writer_kernel_file;
    writer_desc.source_type = KernelDescriptor::SourceType::FILE_PATH;
    writer_desc.core_ranges = all_cores;
    writer_desc.compile_time_args = std::move(writer_compile_time_args);
    writer_desc.defines = std::move(writer_defines);
    writer_desc.config = WriterConfigDescriptor{};

    auto* const target_buf = target.buffer();
    // Pass Buffer* (not a raw address) so the program-cache fast hit path re-patches the binding
    // when the tensor is reallocated; nullptr is fine for an absent optional (framework emits 0u).
    auto* const weight_buf = weight_has_value ? weight.value().buffer() : nullptr;
    auto* const output_grad_buf = output_grad.buffer();
    auto* const input_grad_buf = input_grad.buffer();

    // Set Runtime Args
    for (uint32_t i = 0, tile_offset = 0; i < num_cores; i++) {
        CoreCoord core = {i / core_h, i % core_h};
        uint32_t units_per_core;
        if (core_group_1.contains(core)) {
            units_per_core = units_per_core_group_1;
        } else if (core_group_2.contains(core)) {
            units_per_core = units_per_core_group_2;
        } else {
            TT_THROW("Core not in specified core ranges");
        }

        reader_desc.emplace_runtime_args(
            core,
            {
                target_buf,
                output_grad_buf,
                weight_buf,
                ignore_index,
                units_per_core,
                tile_offset,
                Nt,
                channel_size,
                Ct,
            });

        writer_desc.emplace_runtime_args(core, {input_grad_buf, units_per_core, tile_offset});

        tile_offset += units_per_core;
    }

    desc.kernels.push_back(std::move(reader_desc));
    desc.kernels.push_back(std::move(writer_desc));

    return desc;
}

tt::tt_metal::ProgramDescriptor moreh_nll_loss_unreduced_backward_impl_3d(
    const Tensor& target,
    const std::optional<Tensor>& weight,
    const Tensor& output_grad,
    const Tensor& input_grad,
    const uint32_t ignore_index,
    const DeviceComputeKernelConfig compute_kernel_config) {
    // split work

    // input_grad: (N, C, W)
    auto input_grad_shape = input_grad.padded_shape();
    uint32_t channel_size = input_grad_shape[1];

    auto W = input_grad_shape[-1];
    auto Ct = channel_size / tt::constants::TILE_HEIGHT;
    auto Wt = W / tt::constants::TILE_WIDTH;

    const bool weight_has_value = weight.has_value();

    tt::tt_metal::IDevice* device = target.device();
    auto grid = device->compute_with_storage_grid_size();
    uint32_t core_h = grid.y;

    uint32_t units_to_divide = input_grad.physical_volume() / tt::constants::TILE_HEIGHT / tt::constants::TILE_WIDTH;

    auto [num_cores, all_cores, core_group_1, core_group_2, units_per_core_group_1, units_per_core_group_2] =
        split_work_to_cores(grid, units_to_divide);

    auto [math_fidelity, math_approx_mode, fp32_dest_acc_en, packer_l1_acc, dst_full_sync_en] =
        get_compute_kernel_config_args(device->arch(), compute_kernel_config);

    ProgramDescriptor desc;

    // create circular buffers
    tt::DataFormat data_format = tt::tt_metal::datatype_to_dataformat_converter(input_grad.dtype());

    push_cb(desc, all_cores, static_cast<uint8_t>(tt::CBIndex::c_0), 1, tt::DataFormat::Int32);  // target
    push_cb(desc, all_cores, static_cast<uint8_t>(tt::CBIndex::c_1), 1, data_format);            // output_grad
    push_cb(
        desc, all_cores, static_cast<uint8_t>(tt::CBIndex::c_2), weight_has_value ? Ct : 0u, data_format);  // weight
    push_cb(desc, all_cores, static_cast<uint8_t>(tt::CBIndex::c_16), 1, data_format);  // input_grad

    // create read/write kernel
    KernelDescriptor::CompileTimeArgs reader_compile_time_args{};
    TensorAccessorArgs(*target.buffer()).append_to(reader_compile_time_args);
    TensorAccessorArgs(*output_grad.buffer()).append_to(reader_compile_time_args);
    TensorAccessorArgs(weight.has_value() ? weight.value().buffer() : nullptr).append_to(reader_compile_time_args);

    KernelDescriptor::CompileTimeArgs writer_compile_time_args{};
    TensorAccessorArgs(*input_grad.buffer()).append_to(writer_compile_time_args);

    KernelDescriptor::Defines reader_defines;
    KernelDescriptor::Defines writer_defines;

    if (weight_has_value) {
        reader_defines.emplace_back("WEIGHT", "1");
    }

    if (fp32_dest_acc_en) {
        reader_defines.emplace_back("FP32_DEST_ACC_EN", "1");
    }

    const auto* const reader_kernel_file =
        "ttnn/cpp/ttnn/operations/moreh/moreh_nll_loss_unreduced_backward/device/kernels/"
        "reader_moreh_nll_loss_unreduced_backward_3d.cpp";
    const auto* const writer_kernel_file =
        "ttnn/cpp/ttnn/operations/moreh/moreh_nll_loss_unreduced_backward/device/kernels/"
        "writer_moreh_nll_loss_unreduced_backward.cpp";

    KernelDescriptor reader_desc;
    reader_desc.kernel_source = reader_kernel_file;
    reader_desc.source_type = KernelDescriptor::SourceType::FILE_PATH;
    reader_desc.core_ranges = all_cores;
    reader_desc.compile_time_args = std::move(reader_compile_time_args);
    reader_desc.defines = std::move(reader_defines);
    reader_desc.config = ReaderConfigDescriptor{};

    KernelDescriptor writer_desc;
    writer_desc.kernel_source = writer_kernel_file;
    writer_desc.source_type = KernelDescriptor::SourceType::FILE_PATH;
    writer_desc.core_ranges = all_cores;
    writer_desc.compile_time_args = std::move(writer_compile_time_args);
    writer_desc.defines = std::move(writer_defines);
    writer_desc.config = WriterConfigDescriptor{};

    auto* const target_buf = target.buffer();
    auto* const output_grad_buf = output_grad.buffer();
    // Pass Buffer* (not a raw address) so the program-cache fast hit path re-patches the binding
    // when the tensor is reallocated; nullptr is fine for an absent optional (framework emits 0u).
    auto* const weight_buf = weight_has_value ? weight.value().buffer() : nullptr;
    auto* const input_grad_buf = input_grad.buffer();

    // Set Runtime Args
    for (uint32_t i = 0, tile_offset = 0; i < num_cores; i++) {
        CoreCoord core = {i / core_h, i % core_h};
        uint32_t units_per_core;
        if (core_group_1.contains(core)) {
            units_per_core = units_per_core_group_1;
        } else if (core_group_2.contains(core)) {
            units_per_core = units_per_core_group_2;
        } else {
            TT_THROW("Core not in specified core ranges");
        }

        reader_desc.emplace_runtime_args(
            core,
            {
                target_buf,
                output_grad_buf,
                weight_buf,
                ignore_index,
                units_per_core,
                tile_offset,
                channel_size,
                Ct,
                Wt,
            });

        writer_desc.emplace_runtime_args(core, {input_grad_buf, units_per_core, tile_offset});

        tile_offset += units_per_core;
    }

    desc.kernels.push_back(std::move(reader_desc));
    desc.kernels.push_back(std::move(writer_desc));

    return desc;
}

tt::tt_metal::ProgramDescriptor moreh_nll_loss_unreduced_backward_impl_4d(
    const Tensor& target,
    const std::optional<Tensor>& weight,
    const Tensor& output_grad,
    const Tensor& input_grad,
    const uint32_t ignore_index,
    const DeviceComputeKernelConfig compute_kernel_config) {
    // split work
    auto input_grad_shape = input_grad.padded_shape();
    auto N = input_grad_shape[0];
    uint32_t channel_size = input_grad_shape[1];

    auto Ct = tt::div_up(channel_size, tt::constants::TILE_WIDTH);

    auto H = input_grad_shape[-2];
    auto W = input_grad_shape[-1];
    auto Ht = H / tt::constants::TILE_HEIGHT;
    auto Wt = W / tt::constants::TILE_WIDTH;
    uint32_t num_inner_tile = target.physical_volume() / N / tt::constants::TILE_HEIGHT / tt::constants::TILE_WIDTH;

    const bool weight_has_value = weight.has_value();

    tt::tt_metal::IDevice* device = target.device();
    auto grid = device->compute_with_storage_grid_size();
    uint32_t core_h = grid.y;

    uint32_t units_to_divide = input_grad.physical_volume() / H / W * Ht * Wt;

    auto [num_cores, all_cores, core_group_1, core_group_2, units_per_core_group_1, units_per_core_group_2] =
        split_work_to_cores(grid, units_to_divide);

    auto [math_fidelity, math_approx_mode, fp32_dest_acc_en, packer_l1_acc, dst_full_sync_en] =
        get_compute_kernel_config_args(device->arch(), compute_kernel_config);

    ProgramDescriptor desc;

    // create circular buffers
    tt::DataFormat data_format = tt::tt_metal::datatype_to_dataformat_converter(input_grad.dtype());

    push_cb(desc, all_cores, static_cast<uint8_t>(tt::CBIndex::c_0), 1, tt::DataFormat::Int32);  // target
    push_cb(desc, all_cores, static_cast<uint8_t>(tt::CBIndex::c_1), 1, data_format);            // output_grad
    push_cb(
        desc, all_cores, static_cast<uint8_t>(tt::CBIndex::c_2), weight_has_value ? Ct : 0u, data_format);  // weight
    push_cb(desc, all_cores, static_cast<uint8_t>(tt::CBIndex::c_16), 1, data_format);  // input_grad

    // create read/write kernel
    KernelDescriptor::CompileTimeArgs reader_compile_time_args{};
    TensorAccessorArgs(*target.buffer()).append_to(reader_compile_time_args);
    TensorAccessorArgs(*output_grad.buffer()).append_to(reader_compile_time_args);
    TensorAccessorArgs(weight.has_value() ? weight.value().buffer() : nullptr).append_to(reader_compile_time_args);

    KernelDescriptor::CompileTimeArgs writer_compile_time_args{};
    TensorAccessorArgs(*input_grad.buffer()).append_to(writer_compile_time_args);

    KernelDescriptor::Defines reader_defines;
    KernelDescriptor::Defines writer_defines;

    if (weight_has_value) {
        reader_defines.emplace_back("WEIGHT", "1");
    }

    if (fp32_dest_acc_en) {
        reader_defines.emplace_back("FP32_DEST_ACC_EN", "1");
    }

    const auto* const reader_kernel_file =
        "ttnn/cpp/ttnn/operations/moreh/moreh_nll_loss_unreduced_backward/device/kernels/"
        "reader_moreh_nll_loss_unreduced_backward_4d.cpp";
    const auto* const writer_kernel_file =
        "ttnn/cpp/ttnn/operations/moreh/moreh_nll_loss_unreduced_backward/device/kernels/"
        "writer_moreh_nll_loss_unreduced_backward.cpp";

    KernelDescriptor reader_desc;
    reader_desc.kernel_source = reader_kernel_file;
    reader_desc.source_type = KernelDescriptor::SourceType::FILE_PATH;
    reader_desc.core_ranges = all_cores;
    reader_desc.compile_time_args = std::move(reader_compile_time_args);
    reader_desc.defines = std::move(reader_defines);
    reader_desc.config = ReaderConfigDescriptor{};

    KernelDescriptor writer_desc;
    writer_desc.kernel_source = writer_kernel_file;
    writer_desc.source_type = KernelDescriptor::SourceType::FILE_PATH;
    writer_desc.core_ranges = all_cores;
    writer_desc.compile_time_args = std::move(writer_compile_time_args);
    writer_desc.defines = std::move(writer_defines);
    writer_desc.config = WriterConfigDescriptor{};

    auto* const target_buf = target.buffer();
    auto* const output_grad_buf = output_grad.buffer();
    // Pass Buffer* (not a raw address) so the program-cache fast hit path re-patches the binding
    // when the tensor is reallocated; nullptr is fine for an absent optional (framework emits 0u).
    auto* const weight_buf = weight_has_value ? weight.value().buffer() : nullptr;
    auto* const input_grad_buf = input_grad.buffer();

    // Set Runtime Args
    for (uint32_t i = 0, tile_offset = 0; i < num_cores; i++) {
        CoreCoord core = {i / core_h, i % core_h};
        uint32_t units_per_core;
        if (core_group_1.contains(core)) {
            units_per_core = units_per_core_group_1;
        } else if (core_group_2.contains(core)) {
            units_per_core = units_per_core_group_2;
        } else {
            TT_THROW("Core not in specified core ranges");
        }

        reader_desc.emplace_runtime_args(
            core,
            {
                target_buf,
                output_grad_buf,
                weight_buf,
                ignore_index,
                units_per_core,
                tile_offset,
                num_inner_tile,
                channel_size,
                Ct,
            });

        writer_desc.emplace_runtime_args(core, {input_grad_buf, units_per_core, tile_offset});

        tile_offset += units_per_core;
    }

    desc.kernels.push_back(std::move(reader_desc));
    desc.kernels.push_back(std::move(writer_desc));

    return desc;
}

tt::tt_metal::ProgramDescriptor MorehNllLossUnreducedBackwardDeviceOperation::Factory::create_descriptor(
    const operation_attributes_t& operation_attributes,
    const tensor_args_t& tensor_args,
    tensor_return_value_t& tensor_return_value) {
    using namespace tt;
    using namespace tt::tt_metal;

    const Tensor& target = tensor_args.target_tensor;
    const std::optional<Tensor>& weight = tensor_args.weight_tensor;
    const Tensor& output_grad = tensor_args.output_grad_tensor;

    const uint32_t ignore_index = operation_attributes.ignore_index;
    const DeviceComputeKernelConfig compute_kernel_config = operation_attributes.compute_kernel_config;

    const Tensor& input_grad = tensor_return_value;

    // split work
    const auto& input_grad_shape = input_grad.logical_shape();
    auto input_grad_rank = input_grad_shape.rank();

    if (input_grad_rank == 2) {
        return moreh_nll_loss_unreduced_backward_impl_2d(
            target, weight, output_grad, input_grad, ignore_index, compute_kernel_config);
    }

    if (input_grad_rank == 3) {
        return moreh_nll_loss_unreduced_backward_impl_3d(
            target, weight, output_grad, input_grad, ignore_index, compute_kernel_config);
    }

    return moreh_nll_loss_unreduced_backward_impl_4d(
        target, weight, output_grad, input_grad, ignore_index, compute_kernel_config);
}

}  // namespace ttnn::operations::moreh::moreh_nll_loss_unreduced_backward
