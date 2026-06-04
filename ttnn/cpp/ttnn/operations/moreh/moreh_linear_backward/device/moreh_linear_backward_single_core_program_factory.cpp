// SPDX-FileCopyrightText: © 2024 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include <string>
#include <vector>

#include "moreh_linear_backward_device_operation.hpp"
#include <tt-metalium/tensor_accessor_args.hpp>
#include "ttnn/operations/moreh/moreh_helper_functions.hpp"
#include "ttnn/operations/core/compute_kernel/compute_kernel_config.hpp"

namespace ttnn::operations::moreh::moreh_linear_backward {

tt::tt_metal::ProgramDescriptor MorehBiasAddBackwardOperation::SingleCoreProgramFactory::create_descriptor(
    const operation_attributes_t& operation_attributes,
    const tensor_args_t& tensor_args,
    tensor_return_value_t& bias_grad) {
    using namespace tt;
    using namespace tt::tt_metal;

    const auto& output_grad = tensor_args.output_grad;

    const auto& output_grad_shape_wo_padding = output_grad.logical_shape();

    auto compute_kernel_config = operation_attributes.compute_kernel_config;

    const bool do_mask_h = (output_grad_shape_wo_padding[-2] % constants::TILE_HEIGHT) != 0;
    const uint32_t mask_h =
        do_mask_h ? output_grad_shape_wo_padding[-2] % constants::TILE_HEIGHT : constants::TILE_HEIGHT;
    const bool do_mask_w = (output_grad_shape_wo_padding[-1] % constants::TILE_WIDTH) != 0;
    const uint32_t mask_w =
        do_mask_w ? output_grad_shape_wo_padding[-1] % constants::TILE_WIDTH : constants::TILE_WIDTH;

    const auto& output_grad_shape = output_grad.padded_shape();
    uint32_t batch_num = output_grad.physical_volume() / output_grad_shape[-2] / output_grad_shape[-1];
    uint32_t Ht = output_grad_shape[-2] / constants::TILE_HEIGHT;
    uint32_t Wt = output_grad_shape[-1] / constants::TILE_WIDTH;
    uint32_t num_tiles = output_grad.physical_volume() / constants::TILE_HW;

    const uint32_t in0_t = 2;
    const uint32_t in1_t = 1;
    const uint32_t in2_t = (do_mask_h || do_mask_w) ? 2 : 0;  // mask_h_w

    const uint32_t out0_t = 1;
    const uint32_t im0_t = 1;
    const uint32_t im1_t = 1;

    ////////////////////////////////////////////////////////////////////////////
    //                      Device Setup
    ////////////////////////////////////////////////////////////////////////////
    CoreCoord core = {0, 0};
    const CoreRangeSet core_set{CoreRange(core, core)};

    IDevice* device = output_grad.device();
    auto arch = device->arch();
    auto [math_fidelity, math_approx_mode, fp32_dest_acc_en, packer_l1_acc, dst_full_sync_en] =
        get_compute_kernel_config_args(arch, compute_kernel_config);

    ////////////////////////////////////////////////////////////////////////////
    //                         CircularBuffer Setup
    ////////////////////////////////////////////////////////////////////////////
    auto cb_data_format = datatype_to_dataformat_converter(output_grad.dtype());
    auto fp32_dest_acc_en_data_format = fp32_dest_acc_en ? tt::DataFormat::Float32 : cb_data_format;

    ProgramDescriptor desc;

    desc.cbs.push_back(CBDescriptor{
        .total_size = in0_t * tile_size(cb_data_format),
        .core_ranges = core_set,
        .format_descriptors = {{CBFormatDescriptor{
            .buffer_index = static_cast<uint8_t>(CBIndex::c_0),
            .data_format = cb_data_format,
            .page_size = tile_size(cb_data_format),
        }}},
    });  // output_grad
    desc.cbs.push_back(CBDescriptor{
        .total_size = in1_t * tile_size(cb_data_format),
        .core_ranges = core_set,
        .format_descriptors = {{CBFormatDescriptor{
            .buffer_index = static_cast<uint8_t>(CBIndex::c_1),
            .data_format = cb_data_format,
            .page_size = tile_size(cb_data_format),
        }}},
    });  // scaler
    if (in2_t > 0) {
        desc.cbs.push_back(CBDescriptor{
            .total_size = in2_t * tile_size(cb_data_format),
            .core_ranges = core_set,
            .format_descriptors = {{CBFormatDescriptor{
                .buffer_index = static_cast<uint8_t>(CBIndex::c_2),
                .data_format = cb_data_format,
                .page_size = tile_size(cb_data_format),
            }}},
        });  // mask_h_w
    }
    desc.cbs.push_back(CBDescriptor{
        .total_size = out0_t * tile_size(cb_data_format),
        .core_ranges = core_set,
        .format_descriptors = {{CBFormatDescriptor{
            .buffer_index = static_cast<uint8_t>(CBIndex::c_16),
            .data_format = cb_data_format,
            .page_size = tile_size(cb_data_format),
        }}},
    });  // bias_grad
    desc.cbs.push_back(CBDescriptor{
        .total_size = im0_t * tile_size(cb_data_format),
        .core_ranges = core_set,
        .format_descriptors = {{CBFormatDescriptor{
            .buffer_index = static_cast<uint8_t>(CBIndex::c_24),
            .data_format = cb_data_format,
            .page_size = tile_size(cb_data_format),
        }}},
    });
    desc.cbs.push_back(CBDescriptor{
        .total_size = im1_t * tile_size(fp32_dest_acc_en_data_format),
        .core_ranges = core_set,
        .format_descriptors = {{CBFormatDescriptor{
            .buffer_index = static_cast<uint8_t>(CBIndex::c_25),
            .data_format = fp32_dest_acc_en_data_format,
            .page_size = tile_size(fp32_dest_acc_en_data_format),
        }}},
    });

    ////////////////////////////////////////////////////////////////////////////
    //                      DataMovementKernel SetUp
    ////////////////////////////////////////////////////////////////////////////
    KernelDescriptor::CompileTimeArgs reader_compile_time_args =
        TensorAccessorArgs(*output_grad.buffer()).get_compile_time_args();
    KernelDescriptor::CompileTimeArgs writer_compile_time_args =
        TensorAccessorArgs(*bias_grad.buffer()).get_compile_time_args();

    KernelDescriptor reader_desc;
    reader_desc.kernel_source =
        "ttnn/cpp/ttnn/operations/moreh/moreh_linear_backward/device/kernels/reader_moreh_bias_backward_hw.cpp";
    reader_desc.source_type = KernelDescriptor::SourceType::FILE_PATH;
    reader_desc.core_ranges = core_set;
    reader_desc.compile_time_args = std::move(reader_compile_time_args);
    reader_desc.config = ReaderConfigDescriptor{};

    KernelDescriptor writer_desc;
    writer_desc.kernel_source =
        "ttnn/cpp/ttnn/operations/moreh/moreh_linear_backward/device/kernels/writer_moreh_bias_backward.cpp";
    writer_desc.source_type = KernelDescriptor::SourceType::FILE_PATH;
    writer_desc.core_ranges = core_set;
    writer_desc.compile_time_args = std::move(writer_compile_time_args);
    writer_desc.config = WriterConfigDescriptor{};

    ////////////////////////////////////////////////////////////////////////////
    //                      ComputeKernel SetUp
    ////////////////////////////////////////////////////////////////////////////
    std::map<std::string, std::string> compute_defines_map;
    compute_defines_map["REDUCE_OP"] = "PoolType::SUM";
    compute_defines_map["REDUCE_DIM"] = "ReduceDim::REDUCE_SCALAR";

    if (fp32_dest_acc_en) {
        compute_defines_map["FP32_DEST_ACC_EN"] = "1";
    }
    KernelDescriptor::Defines compute_defines(compute_defines_map.begin(), compute_defines_map.end());
    std::vector<UnpackToDestMode> unpack_to_dest_mode(NUM_CIRCULAR_BUFFERS, UnpackToDestMode::Default);

    KernelDescriptor compute_desc;
    compute_desc.kernel_source =
        "ttnn/cpp/ttnn/operations/moreh/moreh_linear_backward/device/kernels/moreh_bias_backward_single_core_hw.cpp";
    compute_desc.source_type = KernelDescriptor::SourceType::FILE_PATH;
    compute_desc.core_ranges = core_set;
    compute_desc.compile_time_args = {};
    compute_desc.defines = compute_defines;
    compute_desc.config = ComputeConfigDescriptor{
        .math_fidelity = math_fidelity,
        .fp32_dest_acc_en = fp32_dest_acc_en,
        .dst_full_sync_en = dst_full_sync_en,
        .unpack_to_dest_mode = unpack_to_dest_mode,
        .math_approx_mode = math_approx_mode,
    };

    ////////////////////////////////////////////////////////////////////////////
    //                      RuntimeArgs SetUp
    ////////////////////////////////////////////////////////////////////////////
    reader_desc.emplace_runtime_args(
        core,
        {output_grad.buffer(),
         num_tiles,
         0u,
         mask_h,
         mask_w,
         static_cast<uint32_t>(do_mask_h),
         static_cast<uint32_t>(do_mask_w)});
    writer_desc.emplace_runtime_args(core, {bias_grad.buffer(), 1u, 0u});
    compute_desc.emplace_runtime_args(
        core, {batch_num, Ht, Wt, static_cast<uint32_t>(do_mask_h), static_cast<uint32_t>(do_mask_w)});

    desc.kernels.push_back(std::move(reader_desc));
    desc.kernels.push_back(std::move(writer_desc));
    desc.kernels.push_back(std::move(compute_desc));

    return desc;
}

}  // namespace ttnn::operations::moreh::moreh_linear_backward
