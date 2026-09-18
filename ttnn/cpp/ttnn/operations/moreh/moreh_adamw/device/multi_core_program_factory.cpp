// SPDX-FileCopyrightText: © 2024 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include <array>
#include <bit>
#include <cmath>
#include <optional>

#include "moreh_adamw_device_operation.hpp"
#include <tt-metalium/host_api.hpp>
#include <tt-metalium/work_split.hpp>
#include "ttnn/operations/moreh/moreh_helper_functions.hpp"
#include <tt-metalium/tensor_accessor_args.hpp>

namespace ttnn::operations::moreh::moreh_adamw {

using namespace tt;
using namespace tt::tt_metal;

static constexpr const char* READER_KERNEL_PATH =
    "ttnn/cpp/ttnn/operations/moreh/moreh_adamw/device/kernels/reader_moreh_adamw.cpp";
static constexpr const char* WRITER_KERNEL_PATH =
    "ttnn/cpp/ttnn/operations/moreh/moreh_adamw/device/kernels/writer_moreh_adamw.cpp";
static constexpr const char* COMPUTE_KERNEL_PATH =
    "ttnn/cpp/ttnn/operations/moreh/moreh_adamw/device/kernels/moreh_adamw.cpp";

namespace {

// Work split used by create_descriptor to derive the core list and group membership.
struct AdamwWorkSplit {
    uint32_t num_cores = 0;
    uint32_t num_cores_y = 0;
    CoreRangeSet all_cores;
    CoreRangeSet core_group_1;
    CoreRangeSet core_group_2;
    uint32_t num_units_per_core_group_1 = 0;
    uint32_t num_units_per_core_group_2 = 0;
};

AdamwWorkSplit compute_adamw_work_split(const Tensor& param_in) {
    auto grid = param_in.device()->compute_with_storage_grid_size();
    uint32_t num_units = param_in.physical_volume() / tt::constants::TILE_HW;
    auto [num_cores, all_cores, core_group_1, core_group_2, num_units_per_core_group_1, num_units_per_core_group_2] =
        split_work_to_cores(grid, num_units);
    return {
        num_cores,
        grid.y,
        all_cores,
        core_group_1,
        core_group_2,
        num_units_per_core_group_1,
        num_units_per_core_group_2};
}

}  // namespace

ProgramDescriptor MorehAdamWDeviceOperation::MorehAdamWProgramFactory::create_descriptor(
    const operation_attributes_t& operation_attributes,
    const tensor_args_t& tensor_args,
    tensor_return_value_t& tensor_return_value) {
    const Tensor& param_in = tensor_args.param_in;
    const Tensor& grad = tensor_args.grad;
    const Tensor& exp_avg_in = tensor_args.exp_avg_in;
    const Tensor& exp_avg_sq_in = tensor_args.exp_avg_sq_in;

    float lr = operation_attributes.lr;
    float beta1 = operation_attributes.beta1;
    float beta2 = operation_attributes.beta2;
    float eps = operation_attributes.eps;
    float weight_decay = operation_attributes.weight_decay;
    uint32_t step = operation_attributes.step;
    bool amsgrad = operation_attributes.amsgrad;

    const std::optional<Tensor>& max_exp_avg_sq_in = tensor_args.max_exp_avg_sq_in;

    // It's guarantee that param_out, exp_avg_out, exp_avg_sq_out are created.
    const Tensor& param_out = tensor_return_value.at(0).value();
    const Tensor& exp_avg_out = tensor_return_value.at(1).value();
    const Tensor& exp_avg_sq_out = tensor_return_value.at(2).value();
    const std::optional<Tensor> max_exp_avg_sq_out = amsgrad ? tensor_return_value.at(3) : std::nullopt;

    DeviceComputeKernelConfig compute_kernel_config = operation_attributes.compute_kernel_config;

    ////////////////////////////////////////////////////////////////////////////
    //                      Device Setup
    ////////////////////////////////////////////////////////////////////////////
    const auto
        [num_cores,
         num_cores_y,
         all_cores,
         core_group_1,
         core_group_2,
         num_units_per_core_group_1,
         num_units_per_core_group_2] = compute_adamw_work_split(param_in);

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

    // Intermediate CBs
    for (uint32_t cb_idx = tt::CBIndex::c_24; cb_idx <= tt::CBIndex::c_31; ++cb_idx) {
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
    compute_desc_1.compile_time_args = {num_units_per_core_group_1};
    compute_desc_1.defines = compute_defines;
    compute_desc_1.config = compute_config;

    KernelDescriptor compute_desc_2;
    bool has_core_group_2 = !core_group_2.ranges().empty();
    if (has_core_group_2) {
        compute_desc_2.kernel_source = COMPUTE_KERNEL_PATH;
        compute_desc_2.source_type = KernelDescriptor::SourceType::FILE_PATH;
        compute_desc_2.core_ranges = core_group_2;
        compute_desc_2.compile_time_args = {num_units_per_core_group_2};
        compute_desc_2.defines = compute_defines;
        compute_desc_2.config = compute_config;
    }

    ////////////////////////////////////////////////////////////////////////////
    //                      RuntimeArgs SetUp
    ////////////////////////////////////////////////////////////////////////////
    auto* const param_in_buf = param_in.buffer();
    auto* const grad_buf = grad.buffer();
    auto* const exp_avg_in_buf = exp_avg_in.buffer();
    auto* const exp_avg_sq_in_buf = exp_avg_sq_in.buffer();
    // Register max_exp_avg_sq as a BufferBinding so the framework patches its address on
    // cache hit. A raw Buffer::address() write here would go stale across cache hits because
    // the program hash zeros out step+lr, so the same cached program is reused with new tensors.
    auto* const max_exp_avg_sq_in_buf = max_exp_avg_sq_in.has_value() ? max_exp_avg_sq_in.value().buffer() : nullptr;

    auto* const param_out_buf = param_out.buffer();
    auto* const exp_avg_out_buf = exp_avg_out.buffer();
    auto* const exp_avg_sq_out_buf = exp_avg_sq_out.buffer();
    auto* const max_exp_avg_sq_out_buf = max_exp_avg_sq_out.has_value() ? max_exp_avg_sq_out.value().buffer() : nullptr;
    float beta1_exponent = std::pow(beta1, step);
    float beta2_exponent = std::pow(beta2, step);

    const uint32_t f2u_lr = std::bit_cast<uint32_t>(lr);
    const uint32_t f2u_beta1 = std::bit_cast<uint32_t>(beta1);
    const uint32_t f2u_beta2 = std::bit_cast<uint32_t>(beta2);
    const uint32_t f2u_eps = std::bit_cast<uint32_t>(eps);
    const uint32_t f2u_weight_decay = std::bit_cast<uint32_t>(weight_decay);
    const uint32_t f2u_beta1_exponent = std::bit_cast<uint32_t>(beta1_exponent);
    const uint32_t f2u_beta2_exponent = std::bit_cast<uint32_t>(beta2_exponent);

    for (uint32_t i = 0, tile_offset = 0; i < num_cores; ++i) {
        CoreCoord core = {i / num_cores_y, i % num_cores_y};

        uint32_t num_tiles_per_core = 0;
        if (core_group_1.contains(core)) {
            num_tiles_per_core = num_units_per_core_group_1;
        } else if (core_group_2.contains(core)) {
            num_tiles_per_core = num_units_per_core_group_2;
        } else {
            TT_THROW("Core not in specified core ranges.");
        }

        reader_desc.emplace_runtime_args(
            core,
            {param_in_buf,
             grad_buf,
             exp_avg_in_buf,
             exp_avg_sq_in_buf,
             max_exp_avg_sq_in_buf,
             f2u_lr,
             f2u_beta1,
             f2u_beta2,
             f2u_eps,
             f2u_weight_decay,
             f2u_beta1_exponent,
             f2u_beta2_exponent,
             step,
             static_cast<uint32_t>(amsgrad),
             num_tiles_per_core,
             tile_offset});

        writer_desc.emplace_runtime_args(
            core,
            {param_out_buf,
             exp_avg_out_buf,
             exp_avg_sq_out_buf,
             max_exp_avg_sq_out_buf,
             num_tiles_per_core,
             tile_offset});

        // compute — runtime args go to the correct kernel descriptor
        if (core_group_1.contains(core)) {
            compute_desc_1.emplace_runtime_args(core, {step});
        } else {
            compute_desc_2.emplace_runtime_args(core, {step});
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

void MorehAdamWDeviceOperation::MorehAdamWProgramFactory::override_runtime_arguments(
    tt::tt_metal::Program& program,
    const operation_attributes_t& operation_attributes,
    const tensor_args_t& tensor_args,
    tensor_return_value_t& tensor_return_value,
    const std::optional<ttnn::MeshCoordinate>& /*mesh_dispatch_coordinate*/) {
    // attribute_names excludes lr and step, so those two plus the beta exponents derived from step and
    // the buffer addresses are all that vary per dispatch; the work split is keyed by the shapes.
    const auto& param_in = tensor_args.param_in;
    const auto& max_exp_avg_sq_in = tensor_args.max_exp_avg_sq_in;
    auto& param_out = tensor_return_value.at(0).value();
    auto& exp_avg_out = tensor_return_value.at(1).value();
    auto& exp_avg_sq_out = tensor_return_value.at(2).value();
    // create_descriptor drops this output when amsgrad is off; the hit path has to match or it patches
    // an address the miss path never baked.
    const std::optional<Tensor> max_exp_avg_sq_out =
        operation_attributes.amsgrad ? tensor_return_value.at(3) : std::nullopt;

    constexpr uint32_t kReaderKernelIdx = 0, kWriterKernelIdx = 1, kComputeKernelIdx = 2;
    constexpr uint32_t kLrIdx = 5, kBeta1ExpIdx = 10, kBeta2ExpIdx = 11, kReaderStepIdx = 12;

    const std::array<uint32_t, 5> reader_addrs{
        param_in.buffer()->address(),
        tensor_args.grad.buffer()->address(),
        tensor_args.exp_avg_in.buffer()->address(),
        tensor_args.exp_avg_sq_in.buffer()->address(),
        max_exp_avg_sq_in.has_value() ? max_exp_avg_sq_in->buffer()->address() : 0u};
    const std::array<uint32_t, 4> writer_addrs{
        param_out.buffer()->address(),
        exp_avg_out.buffer()->address(),
        exp_avg_sq_out.buffer()->address(),
        max_exp_avg_sq_out.has_value() ? max_exp_avg_sq_out->buffer()->address() : 0u};

    const uint32_t step = operation_attributes.step;
    const uint32_t f2u_lr = std::bit_cast<uint32_t>(operation_attributes.lr);
    const uint32_t f2u_beta1_exponent =
        std::bit_cast<uint32_t>(static_cast<float>(std::pow(operation_attributes.beta1, step)));
    const uint32_t f2u_beta2_exponent =
        std::bit_cast<uint32_t>(static_cast<float>(std::pow(operation_attributes.beta2, step)));

    for (auto& col : tt::tt_metal::GetRuntimeArgs(program, kReaderKernelIdx)) {
        for (auto& a : col) {
            if (a.size() <= kReaderStepIdx) {
                continue;
            }
            for (uint32_t i = 0; i < reader_addrs.size(); ++i) {
                a[i] = reader_addrs[i];
            }
            a[kLrIdx] = f2u_lr;
            a[kBeta1ExpIdx] = f2u_beta1_exponent;
            a[kBeta2ExpIdx] = f2u_beta2_exponent;
            a[kReaderStepIdx] = step;
        }
    }
    for (auto& col : tt::tt_metal::GetRuntimeArgs(program, kWriterKernelIdx)) {
        for (auto& a : col) {
            for (uint32_t i = 0; i < writer_addrs.size() && i < a.size(); ++i) {
                a[i] = writer_addrs[i];
            }
        }
    }

    // step also feeds the compute kernel(s); the second one exists only when the split has a remainder
    // group, so reuse create_descriptor's own work-split helper rather than rebuilding the descriptor.
    const auto split = compute_adamw_work_split(param_in);
    const uint32_t num_compute_kernels = split.core_group_2.ranges().empty() ? 1u : 2u;
    for (uint32_t k = 0; k < num_compute_kernels; ++k) {
        for (auto& col : tt::tt_metal::GetRuntimeArgs(program, kComputeKernelIdx + k)) {
            for (auto& a : col) {
                if (a.size() > 0) {
                    a[0] = step;
                }
            }
        }
    }
}

}  // namespace ttnn::operations::moreh::moreh_adamw
