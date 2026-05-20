// SPDX-FileCopyrightText: © 2024 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include <vector>

#include "fold_device_operation.hpp"
#include <tt-metalium/work_split.hpp>
#include <tt-metalium/tensor_accessor_args.hpp>
#include <tt-metalium/tt_align.hpp>
#include <tt-metalium/hal.hpp>

namespace ttnn::operations::moreh::moreh_fold {

using namespace tt::tt_metal;

static constexpr const char* READER_KERNEL_PATH =
    "ttnn/cpp/ttnn/operations/moreh/moreh_fold/device/kernels/reader_fold_rm.cpp";
static constexpr const char* WRITER_KERNEL_PATH =
    "ttnn/cpp/ttnn/operations/moreh/moreh_fold/device/kernels/writer_fold_rm.cpp";

ProgramDescriptor MorehFoldOperation::create_descriptor(
    const operation_attributes_t& operation_attributes,
    const tensor_args_t& tensor_args,
    tensor_return_value_t& output) {
    const auto& input = tensor_args.input;

    auto output_size = operation_attributes.output_size;
    auto kernel_size = operation_attributes.kernel_size;
    auto dilation = operation_attributes.dilation;
    auto padding = operation_attributes.padding;
    auto stride = operation_attributes.stride;
    auto output_shape = output.logical_shape();
    auto output_shape_rank = output.logical_shape().rank();

    std::vector<uint32_t> ls;
    for (uint32_t i = 0; i < 2; ++i) {
        uint32_t l = (((output_size[i] + 2 * padding[i] - dilation[i] * (kernel_size[i] - 1) - 1) / stride[i]) + 1);
        ls.push_back(l);
    }
    uint32_t N = output_shape_rank == 4 ? output_shape[0] : 1;
    uint32_t C = output_shape_rank == 4 ? output_shape[1] : output_shape[0];
    uint32_t H = output_shape_rank == 4 ? output_shape[2] : output_shape[1];
    uint32_t W = output_shape_rank == 4 ? output_shape[3] : output_shape[2];
    uint32_t kernel_size_h = kernel_size[0];
    uint32_t kernel_size_w = kernel_size[1];
    uint32_t stride_h = stride[0];
    uint32_t stride_w = stride[1];
    uint32_t padding_h = padding[0];
    uint32_t padding_w = padding[1];
    uint32_t dilation_h = dilation[0];
    uint32_t dilation_w = dilation[1];
    uint32_t LH = ls[0];
    uint32_t LW = ls[1];

    IDevice* device = input.device();

    uint32_t num_units = output.logical_volume() / output.logical_shape()[-1];

    auto grid = device->compute_with_storage_grid_size();
    uint32_t num_cores_x = grid.x;
    uint32_t num_cores_y = grid.y;
    auto [num_cores, all_cores, core_group_1, core_group_2, num_units_per_core_group_1, num_units_per_core_group_2] =
        split_work_to_cores(grid, num_units);

    auto data_format = datatype_to_dataformat_converter(input.dtype());

    uint32_t unit_size = input.element_size();
    uint32_t input_cb_page_size = unit_size * input.logical_shape()[-1];
    uint32_t output_cb_page_size = unit_size * output.logical_shape()[-1];

    // For L1 circular buffer alignment
    uint32_t aligned_input_cb_page_size = round_up_to_mul32(input_cb_page_size);
    uint32_t aligned_output_cb_page_size = round_up_to_mul32(output_cb_page_size);

    // For DRAM reads, we need DRAM-aligned size
    bool src_is_dram = input.buffer()->buffer_type() == BufferType::DRAM;
    bool is_blackhole = (device->arch() == tt::ARCH::BLACKHOLE);
    uint32_t dram_alignment = hal::get_dram_alignment();
    uint32_t dram_aligned_input_cb_page_size = tt::align(input_cb_page_size, dram_alignment);

    uint32_t input_cb_index = tt::CBIndex::c_0;    // input
    uint32_t scratch_cb_index = tt::CBIndex::c_1;  // scratch for DRAM alignment
    uint32_t output_cb_index = tt::CBIndex::c_16;  // output

    ProgramDescriptor desc;

    // Input CB
    desc.cbs.push_back(CBDescriptor{
        .total_size = aligned_input_cb_page_size * 2,
        .core_ranges = all_cores,
        .format_descriptors = {{CBFormatDescriptor{
            .buffer_index = input_cb_index,
            .data_format = data_format,
            .page_size = aligned_input_cb_page_size,
        }}},
    });

    // Scratch CB for DRAM alignment
    // On Blackhole, always use two-step read for DRAM
    if ((src_is_dram && (input_cb_page_size % dram_alignment != 0)) || is_blackhole) {
        uint32_t scratch_cb_page_size = dram_aligned_input_cb_page_size;
        desc.cbs.push_back(CBDescriptor{
            .total_size = 4 * scratch_cb_page_size,
            .core_ranges = all_cores,
            .format_descriptors = {{CBFormatDescriptor{
                .buffer_index = scratch_cb_index,
                .data_format = data_format,
                .page_size = scratch_cb_page_size,
            }}},
        });
    }

    // Output CB
    desc.cbs.push_back(CBDescriptor{
        .total_size = aligned_output_cb_page_size * 2,
        .core_ranges = all_cores,
        .format_descriptors = {{CBFormatDescriptor{
            .buffer_index = output_cb_index,
            .data_format = data_format,
            .page_size = aligned_output_cb_page_size,
        }}},
    });

    // Kernel defines
    KernelDescriptor::Defines reader_defines;
    switch (input.dtype()) {
        case DataType::BFLOAT16: reader_defines.emplace_back("DTYPE_BFLOAT16", "1"); break;
        case DataType::FLOAT32: reader_defines.emplace_back("DTYPE_FLOAT32", "1"); break;
        default: break;
    }

    // Reader kernel
    KernelDescriptor::CompileTimeArgs reader_ct_args{input_cb_index, output_cb_index, scratch_cb_index};
    TensorAccessorArgs(input.buffer()).append_to(reader_ct_args);

    KernelDescriptor reader_desc;
    reader_desc.kernel_source = READER_KERNEL_PATH;
    reader_desc.source_type = KernelDescriptor::SourceType::FILE_PATH;
    reader_desc.core_ranges = all_cores;
    reader_desc.compile_time_args = std::move(reader_ct_args);
    reader_desc.defines = std::move(reader_defines);
    reader_desc.config = ReaderConfigDescriptor{};

    // Writer kernel
    KernelDescriptor::CompileTimeArgs writer_ct_args{output_cb_index};
    TensorAccessorArgs(output.buffer()).append_to(writer_ct_args);

    KernelDescriptor writer_desc;
    writer_desc.kernel_source = WRITER_KERNEL_PATH;
    writer_desc.source_type = KernelDescriptor::SourceType::FILE_PATH;
    writer_desc.core_ranges = std::move(all_cores);
    writer_desc.compile_time_args = std::move(writer_ct_args);
    writer_desc.config = WriterConfigDescriptor{};

    // Runtime args per core
    auto cores = grid_to_cores(num_cores, num_cores_x, num_cores_y, false);
    uint32_t g1_numcores = core_group_1.num_cores();
    reader_desc.runtime_args.reserve(cores.size());
    writer_desc.runtime_args.reserve(cores.size());

    uint32_t start_id = 0;
    for (uint32_t i = 0; i < cores.size(); ++i) {
        const CoreCoord& core = cores.at(i);
        uint32_t num_units_per_core = i < g1_numcores ? num_units_per_core_group_1 : num_units_per_core_group_2;

        // Calculate alignment info for the kernel
        // On Blackhole, always use two-step read for DRAM
        uint32_t aligned = (src_is_dram ? (input_cb_page_size % dram_alignment == 0) : 1);
        aligned = aligned && !is_blackhole;

        reader_desc.runtime_args.emplace_back(
            core,
            KernelDescriptor::CoreRuntimeArgs{
                input.buffer()->address(),
                N,
                C,
                H,
                W,
                kernel_size_h,
                kernel_size_w,
                stride_h,
                stride_w,
                padding_h,
                padding_w,
                dilation_h,
                dilation_w,
                LH,
                LW,
                input_cb_page_size,
                dram_aligned_input_cb_page_size,
                aligned_output_cb_page_size,
                start_id,
                num_units_per_core,
                aligned});

        writer_desc.runtime_args.emplace_back(
            core,
            KernelDescriptor::CoreRuntimeArgs{
                output.buffer()->address(), aligned_output_cb_page_size, start_id, num_units_per_core});

        start_id += num_units_per_core;
    }

    desc.kernels.push_back(std::move(reader_desc));
    desc.kernels.push_back(std::move(writer_desc));

    return desc;
}

}  // namespace ttnn::operations::moreh::moreh_fold
