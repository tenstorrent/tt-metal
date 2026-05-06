// SPDX-FileCopyrightText: © 2024 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include "moreh_abs_pow_device_operation.hpp"
#include "ttnn/operations/moreh/moreh_helper_functions.hpp"
#include <tt-metalium/work_split.hpp>
#include <tt-metalium/tensor_accessor_args.hpp>

#include <bit>

namespace ttnn::operations::moreh::moreh_abs_pow {

using namespace tt::tt_metal;

static constexpr const char* READER_KERNEL_PATH =
    "ttnn/cpp/ttnn/operations/moreh/moreh_abs_pow/device/kernels/reader_moreh_abs_pow.cpp";
static constexpr const char* WRITER_KERNEL_PATH =
    "ttnn/cpp/ttnn/operations/moreh/moreh_abs_pow/device/kernels/writer_moreh_abs_pow.cpp";
static constexpr const char* COMPUTE_KERNEL_PATH =
    "ttnn/cpp/ttnn/operations/moreh/moreh_abs_pow/device/kernels/moreh_abs_pow_kernel.cpp";

ProgramDescriptor MorehAbsPowOperation::create_descriptor(
    const operation_attributes_t& operation_attributes,
    const tensor_args_t& tensor_args,
    tensor_return_value_t& output) {
    const auto& input = tensor_args.input;
    const auto p = operation_attributes.p;

    ////////////////////////////////////////////////////////////////////////////
    //                      Device Setup
    ////////////////////////////////////////////////////////////////////////////
    auto* device = input.device();

    ////////////////////////////////////////////////////////////////////////////
    //                         Parameters Setup
    ////////////////////////////////////////////////////////////////////////////
    const auto input_shape = input.padded_shape();
    const auto input_rank = input_shape.rank();

    const auto H = input_shape[-2];
    const auto W = input_shape[-1];

    const auto Ht = H / tt::constants::TILE_HEIGHT;
    const auto Wt = W / tt::constants::TILE_WIDTH;

    const auto num_units = input.physical_volume() / H / W * Ht;

    const auto origin_w = input.logical_shape()[input_rank - 1];

    auto [floored_p, decimal, p_is_negative] = get_floored_p_and_decimal_and_p_is_negative(p);

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
         num_units_per_core_group_2] = split_work_to_cores(grid, num_units);

    ////////////////////////////////////////////////////////////////////////////
    //                         CircularBuffer Setup
    ////////////////////////////////////////////////////////////////////////////
    const auto cb_data_format = datatype_to_dataformat_converter(input.dtype());
    const auto intermed_data_format = fp32_dest_acc_en ? tt::DataFormat::Float32 : cb_data_format;
    const uint32_t cb_tile_size = tile_size(cb_data_format);
    const uint32_t intermed_tile_size = tile_size(intermed_data_format);

    const uint32_t in0_t{1};  // input
    const uint32_t in1_t{1};  // one
    const uint32_t in2_t{1};  // recip_p_decimal
    const uint32_t in3_t{1};  // mask_w

    const uint32_t out0_t{1};  // output

    const uint32_t im0_t{1};  // |x|
    const uint32_t im1_t{1};  // log(|x|)
    const uint32_t im2_t{1};  // exp(log(|x|) * decimal)
    const uint32_t im3_t{1};  // |x|^p

    ProgramDescriptor desc;

    desc.cbs.push_back(CBDescriptor{
        .total_size = in0_t * cb_tile_size,
        .core_ranges = all_cores,
        .format_descriptors = {{CBFormatDescriptor{
            .buffer_index = tt::CBIndex::c_0,
            .data_format = cb_data_format,
            .page_size = cb_tile_size,
        }}},
    });
    desc.cbs.push_back(CBDescriptor{
        .total_size = in1_t * cb_tile_size,
        .core_ranges = all_cores,
        .format_descriptors = {{CBFormatDescriptor{
            .buffer_index = tt::CBIndex::c_1,
            .data_format = cb_data_format,
            .page_size = cb_tile_size,
        }}},
    });
    desc.cbs.push_back(CBDescriptor{
        .total_size = in2_t * cb_tile_size,
        .core_ranges = all_cores,
        .format_descriptors = {{CBFormatDescriptor{
            .buffer_index = tt::CBIndex::c_2,
            .data_format = cb_data_format,
            .page_size = cb_tile_size,
        }}},
    });
    desc.cbs.push_back(CBDescriptor{
        .total_size = in3_t * cb_tile_size,
        .core_ranges = all_cores,
        .format_descriptors = {{CBFormatDescriptor{
            .buffer_index = tt::CBIndex::c_3,
            .data_format = cb_data_format,
            .page_size = cb_tile_size,
        }}},
    });
    desc.cbs.push_back(CBDescriptor{
        .total_size = out0_t * cb_tile_size,
        .core_ranges = all_cores,
        .format_descriptors = {{CBFormatDescriptor{
            .buffer_index = tt::CBIndex::c_16,
            .data_format = cb_data_format,
            .page_size = cb_tile_size,
        }}},
    });
    desc.cbs.push_back(CBDescriptor{
        .total_size = im0_t * intermed_tile_size,
        .core_ranges = all_cores,
        .format_descriptors = {{CBFormatDescriptor{
            .buffer_index = tt::CBIndex::c_24,
            .data_format = intermed_data_format,
            .page_size = intermed_tile_size,
        }}},
    });
    desc.cbs.push_back(CBDescriptor{
        .total_size = im1_t * intermed_tile_size,
        .core_ranges = all_cores,
        .format_descriptors = {{CBFormatDescriptor{
            .buffer_index = tt::CBIndex::c_25,
            .data_format = intermed_data_format,
            .page_size = intermed_tile_size,
        }}},
    });
    desc.cbs.push_back(CBDescriptor{
        .total_size = im2_t * intermed_tile_size,
        .core_ranges = all_cores,
        .format_descriptors = {{CBFormatDescriptor{
            .buffer_index = tt::CBIndex::c_26,
            .data_format = intermed_data_format,
            .page_size = intermed_tile_size,
        }}},
    });
    desc.cbs.push_back(CBDescriptor{
        .total_size = im3_t * intermed_tile_size,
        .core_ranges = all_cores,
        .format_descriptors = {{CBFormatDescriptor{
            .buffer_index = tt::CBIndex::c_27,
            .data_format = intermed_data_format,
            .page_size = intermed_tile_size,
        }}},
    });

    ////////////////////////////////////////////////////////////////////////////
    //                      DataMovementKernel SetUp
    ////////////////////////////////////////////////////////////////////////////
    KernelDescriptor::CompileTimeArgs reader_ct_args;
    TensorAccessorArgs(*input.buffer()).append_to(reader_ct_args);

    KernelDescriptor reader_desc;
    reader_desc.kernel_source = READER_KERNEL_PATH;
    reader_desc.source_type = KernelDescriptor::SourceType::FILE_PATH;
    reader_desc.core_ranges = all_cores;
    reader_desc.compile_time_args = std::move(reader_ct_args);
    reader_desc.config = ReaderConfigDescriptor{};
    reader_desc.runtime_args.reserve(num_cores_to_be_used);

    KernelDescriptor::CompileTimeArgs writer_ct_args;
    TensorAccessorArgs(*output.buffer()).append_to(writer_ct_args);

    KernelDescriptor writer_desc;
    writer_desc.kernel_source = WRITER_KERNEL_PATH;
    writer_desc.source_type = KernelDescriptor::SourceType::FILE_PATH;
    writer_desc.core_ranges = all_cores;
    writer_desc.compile_time_args = std::move(writer_ct_args);
    writer_desc.config = WriterConfigDescriptor{};
    writer_desc.runtime_args.reserve(num_cores_to_be_used);

    ////////////////////////////////////////////////////////////////////////////
    //                      ComputeKernel SetUp
    ////////////////////////////////////////////////////////////////////////////
    ComputeConfigDescriptor compute_config{
        .math_fidelity = math_fidelity,
        .fp32_dest_acc_en = fp32_dest_acc_en,
        .math_approx_mode = math_approx_mode,
    };

    // Compute kernel for core_group_1
    KernelDescriptor compute_desc_1;
    compute_desc_1.kernel_source = COMPUTE_KERNEL_PATH;
    compute_desc_1.source_type = KernelDescriptor::SourceType::FILE_PATH;
    compute_desc_1.core_ranges = core_group_1;
    compute_desc_1.compile_time_args = {num_units_per_core_group_1};
    compute_desc_1.config = compute_config;

    // Compute kernel for core_group_2 (may be empty)
    KernelDescriptor compute_desc_2;
    bool has_core_group_2 = !core_group_2.ranges().empty();
    if (has_core_group_2) {
        compute_desc_2.kernel_source = COMPUTE_KERNEL_PATH;
        compute_desc_2.source_type = KernelDescriptor::SourceType::FILE_PATH;
        compute_desc_2.core_ranges = core_group_2;
        compute_desc_2.compile_time_args = {num_units_per_core_group_2};
        compute_desc_2.config = compute_config;
    }

    ////////////////////////////////////////////////////////////////////////////
    //                      RuntimeArgs SetUp
    ////////////////////////////////////////////////////////////////////////////
    for (uint32_t i = 0, tile_offset = 0; i < num_cores_to_be_used; ++i) {
        CoreCoord core = {i / num_cores_y, i % num_cores_y};

        uint32_t num_units_per_core;
        if (core_group_1.contains(core)) {
            num_units_per_core = num_units_per_core_group_1;
        } else if (core_group_2.contains(core)) {
            num_units_per_core = num_units_per_core_group_2;
        } else {
            TT_THROW("Core not in specified core ranges.");
        }

        // reader
        reader_desc.runtime_args.emplace_back(
            core,
            KernelDescriptor::CoreRuntimeArgs{
                input.buffer()->address(),
                static_cast<uint32_t>(is_dram(input)),
                std::bit_cast<uint32_t>(decimal),
                num_units_per_core,
                Wt,
                tile_offset,
                origin_w});

        // writer
        writer_desc.runtime_args.emplace_back(
            core,
            KernelDescriptor::CoreRuntimeArgs{
                output.buffer()->address(),
                static_cast<uint32_t>(is_dram(output)),
                num_units_per_core,
                Wt,
                tile_offset});

        // compute — runtime args go to the correct kernel descriptor
        KernelDescriptor::CoreRuntimeArgs compute_rt{
            num_units_per_core, Wt, origin_w, floored_p, static_cast<uint32_t>(p_is_negative)};

        if (core_group_1.contains(core)) {
            compute_desc_1.runtime_args.emplace_back(core, std::move(compute_rt));
        } else {
            compute_desc_2.runtime_args.emplace_back(core, std::move(compute_rt));
        }

        tile_offset += num_units_per_core * Wt;
    }

    desc.kernels.push_back(std::move(reader_desc));
    desc.kernels.push_back(std::move(writer_desc));
    desc.kernels.push_back(std::move(compute_desc_1));
    if (has_core_group_2) {
        desc.kernels.push_back(std::move(compute_desc_2));
    }

    return desc;
}

}  // namespace ttnn::operations::moreh::moreh_abs_pow
