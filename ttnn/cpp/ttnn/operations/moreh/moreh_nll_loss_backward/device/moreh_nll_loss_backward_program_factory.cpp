// SPDX-FileCopyrightText: © 2024 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include <string>

#include "moreh_nll_loss_backward_device_operation.hpp"
#include <tt-metalium/work_split.hpp>
#include <tt-metalium/tensor_accessor_args.hpp>
#include "ttnn/operations/moreh/moreh_helper_functions.hpp"

using namespace tt;
using namespace tt::tt_metal;

namespace ttnn::operations::moreh::moreh_nll_loss_backward {

namespace {
void push_cb(
    ProgramDescriptor& desc,
    uint32_t num_tiles,
    const CoreRangeSet& cores,
    uint32_t cb_index,
    tt::DataFormat data_format,
    uint32_t tile_sz) {
    if (num_tiles > 0) {
        desc.cbs.push_back(CBDescriptor{
            .total_size = num_tiles * tile_sz,
            .core_ranges = cores,
            .format_descriptors = {{CBFormatDescriptor{
                .buffer_index = cb_index, .data_format = data_format, .page_size = tile_sz}}},
        });
    }
}

struct CommonSetup {
    ProgramDescriptor desc;
    KernelDescriptor reader_desc;
    KernelDescriptor writer_desc;
    KernelDescriptor compute_desc_1;
    KernelDescriptor compute_desc_2;
    bool has_core_group_2;
    uint32_t num_cores;
    uint32_t core_h;
    CoreRangeSet core_group_1;
    CoreRangeSet core_group_2;
    uint32_t units_per_core_group_1;
    uint32_t units_per_core_group_2;
    uint32_t target_addr;
    uint32_t weight_addr;
    uint32_t divisor_addr;
    uint32_t output_grad_addr;
    uint32_t input_grad_addr;
    uint32_t element_size;
};

CommonSetup setup_common(
    const Tensor& target,
    const std::optional<Tensor>& weight,
    const std::optional<Tensor>& divisor,
    const Tensor& output_grad,
    const Tensor& input_grad,
    const DeviceComputeKernelConfig& compute_kernel_config,
    uint32_t units_to_divide,
    uint32_t weight_cb_tiles,
    const char* reader_path,
    bool add_scratch_cb_c8) {
    const bool weight_has_value = weight.has_value();
    const bool divisor_has_value = divisor.has_value();

    IDevice* device = target.device();
    auto grid = device->compute_with_storage_grid_size();
    uint32_t core_h = grid.y;

    auto [num_cores, all_cores, core_group_1, core_group_2, units_per_core_group_1, units_per_core_group_2] =
        split_work_to_cores(grid, units_to_divide);

    auto [math_fidelity, math_approx_mode, fp32_dest_acc_en, packer_l1_acc, dst_full_sync_en] =
        get_compute_kernel_config_args(device->arch(), compute_kernel_config);

    tt::DataFormat data_format = datatype_to_dataformat_converter(input_grad.dtype());
    auto fp32_format = fp32_dest_acc_en ? tt::DataFormat::Float32 : data_format;
    const uint32_t data_ts = tile_size(data_format);
    const uint32_t fp32_ts = tile_size(fp32_format);
    const uint32_t int32_ts = tile_size(tt::DataFormat::Int32);

    CommonSetup s;

    // CBs
    push_cb(s.desc, 1, all_cores, tt::CBIndex::c_0, data_format, data_ts);                          // output_grad
    push_cb(s.desc, 1, all_cores, tt::CBIndex::c_1, tt::DataFormat::Int32, int32_ts);               // target
    push_cb(s.desc, weight_cb_tiles, all_cores, tt::CBIndex::c_2, data_format, data_ts);            // weight
    push_cb(s.desc, divisor_has_value ? 1 : 0, all_cores, tt::CBIndex::c_3, data_format, data_ts);  // divisor
    push_cb(s.desc, 1, all_cores, tt::CBIndex::c_24, fp32_format, fp32_ts);                         // tmp_weight
    push_cb(s.desc, 1, all_cores, tt::CBIndex::c_25, fp32_format, fp32_ts);                         // tmp1
    push_cb(s.desc, 1, all_cores, tt::CBIndex::c_26, fp32_format, fp32_ts);                         // tmp2
    push_cb(s.desc, 1, all_cores, tt::CBIndex::c_16, data_format, data_ts);                         // input_grad

    if (weight_has_value) {
        // This CB will be used as scratch storage when reading data from DRAM into L1,
        // since the two have different alignment requirements on some architectures.
        // Need space for only a single tile in scratch CB, because content is read immediately after writing.
        push_cb(s.desc, 1, all_cores, tt::CBIndex::c_7, data_format, data_ts);
    }
    if (add_scratch_cb_c8) {
        // Need another scratch CB for output_grad reading data from DRAM into L1.
        push_cb(s.desc, 1, all_cores, tt::CBIndex::c_8, data_format, data_ts);
    }

    // Defines
    KernelDescriptor::Defines reader_defines;
    KernelDescriptor::Defines compute_defines;
    if (weight_has_value) {
        reader_defines.emplace_back("WEIGHT", "1");
        compute_defines.emplace_back("WEIGHT", "1");
    }
    if (divisor_has_value) {
        reader_defines.emplace_back("DIVISOR", "1");
        compute_defines.emplace_back("DIVISOR", "1");
    }
    if (fp32_dest_acc_en) {
        reader_defines.emplace_back("FP32_DEST_ACC_EN", "1");
        compute_defines.emplace_back("FP32_DEST_ACC_EN", "1");
    }

    // Reader kernel
    KernelDescriptor::CompileTimeArgs reader_ct_args;
    TensorAccessorArgs(target.buffer()).append_to(reader_ct_args);
    TensorAccessorArgs(weight.has_value() ? weight.value().buffer() : nullptr).append_to(reader_ct_args);
    TensorAccessorArgs(divisor.has_value() ? divisor.value().buffer() : nullptr).append_to(reader_ct_args);
    TensorAccessorArgs(output_grad.buffer()).append_to(reader_ct_args);

    s.reader_desc.kernel_source = reader_path;
    s.reader_desc.source_type = KernelDescriptor::SourceType::FILE_PATH;
    s.reader_desc.core_ranges = all_cores;
    s.reader_desc.compile_time_args = std::move(reader_ct_args);
    s.reader_desc.defines = std::move(reader_defines);
    s.reader_desc.config = ReaderConfigDescriptor{};
    s.reader_desc.runtime_args.reserve(num_cores);

    // Writer kernel
    static constexpr const char* WRITER_KERNEL =
        "ttnn/cpp/ttnn/operations/moreh/moreh_nll_loss_backward/device/kernels/writer_moreh_nll_loss_backward.cpp";
    KernelDescriptor::CompileTimeArgs writer_ct_args;
    TensorAccessorArgs(input_grad.buffer()).append_to(writer_ct_args);

    s.writer_desc.kernel_source = WRITER_KERNEL;
    s.writer_desc.source_type = KernelDescriptor::SourceType::FILE_PATH;
    s.writer_desc.core_ranges = std::move(all_cores);
    s.writer_desc.compile_time_args = std::move(writer_ct_args);
    s.writer_desc.config = WriterConfigDescriptor{};
    s.writer_desc.runtime_args.reserve(num_cores);

    // Compute kernels (dual core groups)
    static constexpr const char* COMPUTE_KERNEL =
        "ttnn/cpp/ttnn/operations/moreh/moreh_nll_loss_backward/device/kernels/moreh_nll_loss_backward_kernel.cpp";

    ComputeConfigDescriptor compute_config{
        .math_fidelity = math_fidelity,
        .fp32_dest_acc_en = fp32_dest_acc_en,
        .math_approx_mode = math_approx_mode,
    };

    s.compute_desc_1.kernel_source = COMPUTE_KERNEL;
    s.compute_desc_1.source_type = KernelDescriptor::SourceType::FILE_PATH;
    s.compute_desc_1.core_ranges = core_group_1;
    s.compute_desc_1.compile_time_args = {units_per_core_group_1, static_cast<uint32_t>(divisor_has_value)};
    s.compute_desc_1.defines = compute_defines;
    s.compute_desc_1.config = compute_config;

    s.has_core_group_2 = !core_group_2.ranges().empty();
    if (s.has_core_group_2) {
        s.compute_desc_2.kernel_source = COMPUTE_KERNEL;
        s.compute_desc_2.source_type = KernelDescriptor::SourceType::FILE_PATH;
        s.compute_desc_2.core_ranges = core_group_2;
        s.compute_desc_2.compile_time_args = {units_per_core_group_2, static_cast<uint32_t>(divisor_has_value)};
        s.compute_desc_2.defines = compute_defines;
        s.compute_desc_2.config = compute_config;
    }

    s.num_cores = num_cores;
    s.core_h = core_h;
    s.core_group_1 = core_group_1;
    s.core_group_2 = core_group_2;
    s.units_per_core_group_1 = units_per_core_group_1;
    s.units_per_core_group_2 = units_per_core_group_2;
    s.target_addr = target.buffer()->address();
    s.weight_addr = weight_has_value ? weight.value().buffer()->address() : 0;
    s.divisor_addr = divisor_has_value ? divisor.value().buffer()->address() : 0;
    s.output_grad_addr = output_grad.buffer()->address();
    s.input_grad_addr = input_grad.buffer()->address();
    s.element_size = weight_has_value ? weight.value().element_size() : 0;

    return s;
}

ProgramDescriptor finalize(CommonSetup& s) {
    s.desc.kernels.push_back(std::move(s.reader_desc));
    s.desc.kernels.push_back(std::move(s.writer_desc));
    s.desc.kernels.push_back(std::move(s.compute_desc_1));
    if (s.has_core_group_2) {
        s.desc.kernels.push_back(std::move(s.compute_desc_2));
    }
    return std::move(s.desc);
}
}  // namespace

static ProgramDescriptor moreh_nll_loss_backward_impl_2d(
    const Tensor& target,
    const std::optional<Tensor>& weight,
    const std::optional<Tensor>& divisor,
    const Tensor& output_grad,
    const Tensor& input_grad,
    const uint32_t ignore_index,
    const DeviceComputeKernelConfig& compute_kernel_config) {
    auto input_grad_shape = input_grad.padded_shape();
    auto channel_size = input_grad_shape[1];
    uint32_t weight_num_tile = tt::div_up(channel_size, tt::constants::TILE_WIDTH);
    uint32_t units_to_divide = input_grad.physical_volume() / tt::constants::TILE_HEIGHT / tt::constants::TILE_WIDTH;

    auto s = setup_common(
        target,
        weight,
        divisor,
        output_grad,
        input_grad,
        compute_kernel_config,
        units_to_divide,
        weight.has_value() ? weight_num_tile : 0,
        "ttnn/cpp/ttnn/operations/moreh/moreh_nll_loss_backward/device/kernels/reader_moreh_nll_loss_backward_2d.cpp",
        true);  // add_scratch_cb_c8 = true for 2d

    for (uint32_t i = 0, tile_offset = 0; i < s.num_cores; i++) {
        CoreCoord core = {i / s.core_h, i % s.core_h};
        uint32_t units_per_core = s.core_group_1.contains(core) ? s.units_per_core_group_1 : s.units_per_core_group_2;

        s.reader_desc.runtime_args.emplace_back(
            core,
            KernelDescriptor::CoreRuntimeArgs{
                s.target_addr,
                s.output_grad_addr,
                s.weight_addr,
                s.divisor_addr,
                ignore_index,
                units_per_core,
                tile_offset,
                channel_size,
                weight_num_tile,
                s.element_size});

        s.writer_desc.runtime_args.emplace_back(
            core, KernelDescriptor::CoreRuntimeArgs{s.input_grad_addr, units_per_core, tile_offset});

        KernelDescriptor::CoreRuntimeArgs compute_rt{units_per_core, tile_offset};
        if (s.core_group_1.contains(core)) {
            s.compute_desc_1.runtime_args.emplace_back(core, std::move(compute_rt));
        } else {
            s.compute_desc_2.runtime_args.emplace_back(core, std::move(compute_rt));
        }
        tile_offset += units_per_core;
    }
    return finalize(s);
}

static ProgramDescriptor moreh_nll_loss_backward_impl_3d(
    const Tensor& target,
    const std::optional<Tensor>& weight,
    const std::optional<Tensor>& divisor,
    const Tensor& output_grad,
    const Tensor& input_grad,
    const uint32_t ignore_index,
    const DeviceComputeKernelConfig& compute_kernel_config) {
    auto input_grad_shape = input_grad.padded_shape();
    auto channel_size = input_grad_shape[1];
    auto target_shape = target.padded_shape();
    auto num_inner_tile = target_shape[-1] / tt::constants::TILE_WIDTH;
    uint32_t weight_num_tile = tt::div_up(channel_size, tt::constants::TILE_WIDTH);
    uint32_t units_to_divide = input_grad.physical_volume() / tt::constants::TILE_HEIGHT / tt::constants::TILE_WIDTH;

    auto s = setup_common(
        target,
        weight,
        divisor,
        output_grad,
        input_grad,
        compute_kernel_config,
        units_to_divide,
        weight.has_value() ? weight_num_tile : 0,
        "ttnn/cpp/ttnn/operations/moreh/moreh_nll_loss_backward/device/kernels/reader_moreh_nll_loss_backward_3d.cpp",
        false);

    for (uint32_t i = 0, tile_offset = 0; i < s.num_cores; i++) {
        CoreCoord core = {i / s.core_h, i % s.core_h};
        uint32_t units_per_core = s.core_group_1.contains(core) ? s.units_per_core_group_1 : s.units_per_core_group_2;

        s.reader_desc.runtime_args.emplace_back(
            core,
            KernelDescriptor::CoreRuntimeArgs{
                s.target_addr,
                s.output_grad_addr,
                s.weight_addr,
                s.divisor_addr,
                ignore_index,
                units_per_core,
                tile_offset,
                channel_size,
                num_inner_tile,
                weight_num_tile,
                s.element_size});

        s.writer_desc.runtime_args.emplace_back(
            core, KernelDescriptor::CoreRuntimeArgs{s.input_grad_addr, units_per_core, tile_offset});

        KernelDescriptor::CoreRuntimeArgs compute_rt{units_per_core, tile_offset};
        if (s.core_group_1.contains(core)) {
            s.compute_desc_1.runtime_args.emplace_back(core, std::move(compute_rt));
        } else {
            s.compute_desc_2.runtime_args.emplace_back(core, std::move(compute_rt));
        }
        tile_offset += units_per_core;
    }
    return finalize(s);
}

static ProgramDescriptor moreh_nll_loss_backward_impl_4d(
    const Tensor& target,
    const std::optional<Tensor>& weight,
    const std::optional<Tensor>& divisor,
    const Tensor& output_grad,
    const Tensor& input_grad,
    const uint32_t ignore_index,
    const DeviceComputeKernelConfig& compute_kernel_config) {
    auto input_grad_shape = input_grad.padded_shape();
    auto N = input_grad_shape[0];
    auto channel_size = input_grad_shape[1];
    auto H = input_grad_shape[-2];
    auto W = input_grad_shape[-1];
    auto Ht = H / tt::constants::TILE_HEIGHT;
    auto Wt = W / tt::constants::TILE_WIDTH;
    auto num_inner_tile = target.physical_volume() / N / tt::constants::TILE_HEIGHT / tt::constants::TILE_WIDTH;
    uint32_t weight_num_tile = tt::div_up(channel_size, tt::constants::TILE_WIDTH);
    uint32_t units_to_divide = input_grad.physical_volume() / H / W * Ht * Wt;

    auto s = setup_common(
        target,
        weight,
        divisor,
        output_grad,
        input_grad,
        compute_kernel_config,
        units_to_divide,
        weight.has_value() ? weight_num_tile : 0,
        "ttnn/cpp/ttnn/operations/moreh/moreh_nll_loss_backward/device/kernels/reader_moreh_nll_loss_backward_4d.cpp",
        false);

    for (uint32_t i = 0, tile_offset = 0; i < s.num_cores; i++) {
        CoreCoord core = {i / s.core_h, i % s.core_h};
        uint32_t units_per_core = s.core_group_1.contains(core) ? s.units_per_core_group_1 : s.units_per_core_group_2;

        s.reader_desc.runtime_args.emplace_back(
            core,
            KernelDescriptor::CoreRuntimeArgs{
                s.target_addr,
                s.output_grad_addr,
                s.weight_addr,
                s.divisor_addr,
                ignore_index,
                units_per_core,
                tile_offset,
                channel_size,
                num_inner_tile,
                weight_num_tile,
                s.element_size});

        s.writer_desc.runtime_args.emplace_back(
            core, KernelDescriptor::CoreRuntimeArgs{s.input_grad_addr, units_per_core, tile_offset});

        KernelDescriptor::CoreRuntimeArgs compute_rt{units_per_core, tile_offset};
        if (s.core_group_1.contains(core)) {
            s.compute_desc_1.runtime_args.emplace_back(core, std::move(compute_rt));
        } else {
            s.compute_desc_2.runtime_args.emplace_back(core, std::move(compute_rt));
        }
        tile_offset += units_per_core;
    }
    return finalize(s);
}

ProgramDescriptor MorehNllLossBackwardDeviceOperation::create_descriptor(
    const operation_attributes_t& operation_attributes,
    const tensor_args_t& tensor_args,
    tensor_return_value_t& tensor_return_value) {
    const Tensor& target = tensor_args.target_tensor;
    const Tensor& output_grad = tensor_args.output_grad_tensor;
    const std::optional<Tensor>& weight = tensor_args.weight_tensor;
    const std::optional<Tensor>& divisor = tensor_args.divisor_tensor;

    const uint32_t ignore_index = operation_attributes.ignore_index;
    const DeviceComputeKernelConfig compute_kernel_config = operation_attributes.compute_kernel_config;

    const Tensor& input_grad = tensor_return_value;

    const auto& input_grad_shape = input_grad.logical_shape();
    auto input_grad_rank = input_grad_shape.rank();

    if (input_grad_rank == 2) {
        return moreh_nll_loss_backward_impl_2d(
            target, weight, divisor, output_grad, input_grad, ignore_index, compute_kernel_config);
    }

    if (input_grad_rank == 3) {
        return moreh_nll_loss_backward_impl_3d(
            target, weight, divisor, output_grad, input_grad, ignore_index, compute_kernel_config);
    }

    return moreh_nll_loss_backward_impl_4d(
        target, weight, divisor, output_grad, input_grad, ignore_index, compute_kernel_config);
}

}  // namespace ttnn::operations::moreh::moreh_nll_loss_backward
