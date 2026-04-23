// SPDX-FileCopyrightText: © 2023 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include "moreh_dot_device_operation.hpp"
#include <tt-metalium/constants.hpp>
#include <tt-metalium/tensor_accessor_args.hpp>

namespace ttnn::operations::moreh::moreh_dot {

using namespace tt::tt_metal;

static constexpr const char* READER_KERNEL_PATH =
    "ttnn/cpp/ttnn/operations/moreh/moreh_dot/device/kernels/reader_moreh_dot.cpp";
static constexpr const char* WRITER_KERNEL_PATH =
    "ttnn/cpp/ttnn/operations/moreh/moreh_dot/device/kernels/writer_moreh_dot.cpp";
static constexpr const char* COMPUTE_KERNEL_PATH =
    "ttnn/cpp/ttnn/operations/moreh/moreh_dot/device/kernels/moreh_dot.cpp";

ProgramDescriptor MorehDotOperation::create_descriptor(
    const operation_attributes_t& operation_attributes,
    const tensor_args_t& tensor_args,
    tensor_return_value_t& output) {
    const auto& input_a = tensor_args.input_a;
    const auto& input_b = tensor_args.input_b;

    auto* src0_buffer = input_a.buffer();
    auto* src1_buffer = input_b.buffer();
    auto* dst_buffer = output.buffer();
    float scaler = 1.0f;

    tt::DataFormat cb_data_format = datatype_to_dataformat_converter(input_a.dtype());
    const uint32_t cb_tile_size = tile_size(cb_data_format);

    uint32_t num_tiles = input_a.physical_volume() / tt::constants::TILE_HW;
    const auto& a_shape_wo_padding = input_a.logical_shape();
    uint32_t pad_h = a_shape_wo_padding[2] % tt::constants::TILE_HEIGHT;
    uint32_t pad_w = a_shape_wo_padding[3] % tt::constants::TILE_WIDTH;
    uint32_t mask_h = (pad_h == 0) ? (tt::constants::TILE_HEIGHT) : (pad_h);
    uint32_t mask_w = (pad_w == 0) ? (tt::constants::TILE_WIDTH) : (pad_w);

    IDevice* device = input_a.device();

    auto [math_fidelity, math_approx_mode, fp32_dest_acc_en, packer_l1_acc, dst_full_sync_en] =
        get_compute_kernel_config_args(device->arch(), operation_attributes.compute_kernel_config);

    const uint32_t in0_t = 2;   // a
    const uint32_t in1_t = 2;   // b
    const uint32_t in2_t = 1;   // scaler
    const uint32_t out0_t = 2;  // out
    const uint32_t im0_t = 1;
    const uint32_t im1_t = 1;

    CoreCoord core = {0, 0};
    CoreRangeSet core_set(CoreRange(core, core));

    ProgramDescriptor desc;

    // Circular buffers — all share same data format and single core
    desc.cbs.push_back(CBDescriptor{
        .total_size = in0_t * cb_tile_size,
        .core_ranges = core_set,
        .format_descriptors = {{CBFormatDescriptor{
            .buffer_index = tt::CBIndex::c_0,
            .data_format = cb_data_format,
            .page_size = cb_tile_size,
        }}},
    });
    desc.cbs.push_back(CBDescriptor{
        .total_size = in1_t * cb_tile_size,
        .core_ranges = core_set,
        .format_descriptors = {{CBFormatDescriptor{
            .buffer_index = tt::CBIndex::c_1,
            .data_format = cb_data_format,
            .page_size = cb_tile_size,
        }}},
    });
    desc.cbs.push_back(CBDescriptor{
        .total_size = in2_t * cb_tile_size,
        .core_ranges = core_set,
        .format_descriptors = {{CBFormatDescriptor{
            .buffer_index = tt::CBIndex::c_2,
            .data_format = cb_data_format,
            .page_size = cb_tile_size,
        }}},
    });
    desc.cbs.push_back(CBDescriptor{
        .total_size = out0_t * cb_tile_size,
        .core_ranges = core_set,
        .format_descriptors = {{CBFormatDescriptor{
            .buffer_index = tt::CBIndex::c_16,
            .data_format = cb_data_format,
            .page_size = cb_tile_size,
        }}},
    });
    desc.cbs.push_back(CBDescriptor{
        .total_size = im0_t * cb_tile_size,
        .core_ranges = core_set,
        .format_descriptors = {{CBFormatDescriptor{
            .buffer_index = tt::CBIndex::c_24,
            .data_format = cb_data_format,
            .page_size = cb_tile_size,
        }}},
    });
    desc.cbs.push_back(CBDescriptor{
        .total_size = im1_t * cb_tile_size,
        .core_ranges = core_set,
        .format_descriptors = {{CBFormatDescriptor{
            .buffer_index = tt::CBIndex::c_25,
            .data_format = cb_data_format,
            .page_size = cb_tile_size,
        }}},
    });

    // Reader kernel
    KernelDescriptor::CompileTimeArgs reader_ct_args = {*reinterpret_cast<uint32_t*>(&scaler)};
    TensorAccessorArgs(src0_buffer).append_to(reader_ct_args);
    TensorAccessorArgs(src1_buffer).append_to(reader_ct_args);

    KernelDescriptor reader_desc;
    reader_desc.kernel_source = READER_KERNEL_PATH;
    reader_desc.source_type = KernelDescriptor::SourceType::FILE_PATH;
    reader_desc.core_ranges = core_set;
    reader_desc.compile_time_args = std::move(reader_ct_args);
    reader_desc.config = ReaderConfigDescriptor{};
    reader_desc.runtime_args.emplace_back(
        core,
        KernelDescriptor::CoreRuntimeArgs{
            src0_buffer->address(), src1_buffer->address(), num_tiles, 0u, mask_h, mask_w});

    // Writer kernel
    KernelDescriptor::CompileTimeArgs writer_ct_args = {static_cast<uint32_t>(tt::CBIndex::c_16)};
    TensorAccessorArgs(dst_buffer).append_to(writer_ct_args);

    KernelDescriptor writer_desc;
    writer_desc.kernel_source = WRITER_KERNEL_PATH;
    writer_desc.source_type = KernelDescriptor::SourceType::FILE_PATH;
    writer_desc.core_ranges = core_set;
    writer_desc.compile_time_args = std::move(writer_ct_args);
    writer_desc.config = WriterConfigDescriptor{};
    writer_desc.runtime_args.emplace_back(core, KernelDescriptor::CoreRuntimeArgs{dst_buffer->address(), 1u, 0u});

    // Compute kernel
    KernelDescriptor compute_desc;
    compute_desc.kernel_source = COMPUTE_KERNEL_PATH;
    compute_desc.source_type = KernelDescriptor::SourceType::FILE_PATH;
    compute_desc.core_ranges = std::move(core_set);
    compute_desc.defines = {{"REDUCE_OP", "PoolType::SUM"}, {"REDUCE_DIM", "ReduceDim::REDUCE_ROW"}};
    compute_desc.config = ComputeConfigDescriptor{
        .math_fidelity = math_fidelity,
        .fp32_dest_acc_en = fp32_dest_acc_en,
        .math_approx_mode = math_approx_mode,
    };
    compute_desc.runtime_args.emplace_back(core, KernelDescriptor::CoreRuntimeArgs{num_tiles, 1u});

    desc.kernels.push_back(std::move(reader_desc));
    desc.kernels.push_back(std::move(writer_desc));
    desc.kernels.push_back(std::move(compute_desc));

    return desc;
}

}  // namespace ttnn::operations::moreh::moreh_dot
