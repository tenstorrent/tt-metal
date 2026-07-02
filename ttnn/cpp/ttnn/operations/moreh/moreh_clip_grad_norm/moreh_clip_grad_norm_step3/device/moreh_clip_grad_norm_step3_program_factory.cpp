// SPDX-FileCopyrightText: © 2024 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include <bit>
#include <vector>

#include "moreh_clip_grad_norm_step3_device_operation.hpp"
#include <tt_stl/assert.hpp>
#include <tt-metalium/work_split.hpp>
#include "ttnn/operations/core/compute_kernel/compute_kernel_config.hpp"
#include "ttnn/operations/moreh/moreh_helper_functions.hpp"
#include <tt-metalium/tensor_accessor_args.hpp>

namespace ttnn::operations::moreh::moreh_clip_grad_norm_step3 {

using namespace tt::tt_metal;

static constexpr const char* READER_KERNEL_PATH =
    "ttnn/cpp/ttnn/operations/moreh/moreh_clip_grad_norm/moreh_clip_grad_norm_step3/device/kernels/"
    "reader_moreh_clip_grad_norm_step3.cpp";
static constexpr const char* WRITER_KERNEL_PATH =
    "ttnn/cpp/ttnn/operations/moreh/moreh_clip_grad_norm/moreh_clip_grad_norm_step3/device/kernels/"
    "writer_moreh_clip_grad_norm_step3.cpp";
static constexpr const char* COMPUTE_KERNEL_PATH =
    "ttnn/cpp/ttnn/operations/moreh/moreh_clip_grad_norm/moreh_clip_grad_norm_step3/device/kernels/"
    "moreh_clip_grad_norm_step3_kernel.cpp";

ProgramDescriptor MorehClipGradNormStep3Operation::create_descriptor(
    const operation_attributes_t& /*operation_attributes*/,
    const tensor_args_t& tensor_args,
    tensor_return_value_t& inputs) {
    const auto& clip_coef_clamped = tensor_args.clip_coef_clamped;

    ////////////////////////////////////////////////////////////////////////////
    //                      Device Setup
    ////////////////////////////////////////////////////////////////////////////
    auto* device = inputs.at(0).device();

    ////////////////////////////////////////////////////////////////////////////
    //                         Parameters Setup
    ////////////////////////////////////////////////////////////////////////////
    const auto num_inputs = static_cast<uint32_t>(inputs.size());

    ////////////////////////////////////////////////////////////////////////////
    //                         Core Setup
    ////////////////////////////////////////////////////////////////////////////
    auto grid = device->compute_with_storage_grid_size();
    const auto num_cores_x = grid.x;
    const auto num_cores_y = grid.y;

    const auto
        [num_cores_to_be_used,
         all_cores,
         core_group_1,
         core_group_2,
         num_inputs_per_core_group_1,
         num_inputs_per_core_group_2] = split_work_to_cores(grid, num_inputs);
    TT_FATAL(core_group_2.ranges().empty(), "core_group_2 must be empty");
    TT_FATAL(num_inputs_per_core_group_1 == 1, "num_inputs_per_core_group_1 must be 1");
    TT_FATAL(num_inputs_per_core_group_2 == 0, "num_inputs_per_core_group_2 must be 0");

    ////////////////////////////////////////////////////////////////////////////
    //                         CircularBuffer Setup
    ////////////////////////////////////////////////////////////////////////////
    const uint32_t in0_t = 1;   // input(inplace)
    const uint32_t in1_t = 1;   // clip_coef_clamped
    const uint32_t out0_t = 1;  // output(inplace)

    const auto cb_data_format = datatype_to_dataformat_converter(inputs.at(0).dtype());
    const uint32_t cb_tile_size = tile_size(cb_data_format);

    ProgramDescriptor desc;

    desc.cbs.push_back(CBDescriptor{
        .total_size = in0_t * cb_tile_size,
        .core_ranges = core_group_1,
        .format_descriptors = {{CBFormatDescriptor{
            .buffer_index = tt::CBIndex::c_0, .data_format = cb_data_format, .page_size = cb_tile_size}}},
    });  // input(inplace)
    desc.cbs.push_back(CBDescriptor{
        .total_size = in1_t * cb_tile_size,
        .core_ranges = core_group_1,
        .format_descriptors = {{CBFormatDescriptor{
            .buffer_index = tt::CBIndex::c_1, .data_format = cb_data_format, .page_size = cb_tile_size}}},
    });  // clip_coef_clamped
    desc.cbs.push_back(CBDescriptor{
        .total_size = out0_t * cb_tile_size,
        .core_ranges = core_group_1,
        .format_descriptors = {{CBFormatDescriptor{
            .buffer_index = tt::CBIndex::c_16, .data_format = cb_data_format, .page_size = cb_tile_size}}},
    });  // output(inplace)

    ////////////////////////////////////////////////////////////////////////////
    //                      DataMovementKernel SetUp
    ////////////////////////////////////////////////////////////////////////////
    // Use inputs.at(0) for compile-time accessor args (all inputs share same buffer layout)
    KernelDescriptor::CompileTimeArgs reader_ct_args;
    TensorAccessorArgs(*inputs.at(0).buffer()).append_to(reader_ct_args);
    TensorAccessorArgs(*clip_coef_clamped.buffer()).append_to(reader_ct_args);

    KernelDescriptor reader_desc;
    reader_desc.kernel_source = READER_KERNEL_PATH;
    reader_desc.source_type = KernelDescriptor::SourceType::FILE_PATH;
    reader_desc.core_ranges = core_group_1;
    reader_desc.compile_time_args = std::move(reader_ct_args);
    reader_desc.config = ReaderConfigDescriptor{};
    reader_desc.runtime_args.reserve(num_cores_to_be_used);

    KernelDescriptor::CompileTimeArgs writer_ct_args;
    TensorAccessorArgs(*inputs.at(0).buffer()).append_to(writer_ct_args);

    KernelDescriptor writer_desc;
    writer_desc.kernel_source = WRITER_KERNEL_PATH;
    writer_desc.source_type = KernelDescriptor::SourceType::FILE_PATH;
    writer_desc.core_ranges = core_group_1;
    writer_desc.compile_time_args = std::move(writer_ct_args);
    writer_desc.config = WriterConfigDescriptor{};
    writer_desc.runtime_args.reserve(num_cores_to_be_used);

    ////////////////////////////////////////////////////////////////////////////
    //                      ComputeKernel SetUp
    ////////////////////////////////////////////////////////////////////////////
    KernelDescriptor compute_desc;
    compute_desc.kernel_source = COMPUTE_KERNEL_PATH;
    compute_desc.source_type = KernelDescriptor::SourceType::FILE_PATH;
    compute_desc.core_ranges = core_group_1;
    compute_desc.compile_time_args = {num_inputs_per_core_group_1};
    compute_desc.config = ComputeConfigDescriptor{};
    compute_desc.runtime_args.reserve(num_cores_to_be_used);

    ////////////////////////////////////////////////////////////////////////////
    //                      RuntimeArgs SetUp
    ////////////////////////////////////////////////////////////////////////////
    auto cores = grid_to_cores(num_cores_to_be_used, num_cores_x, num_cores_y, false);
    const auto clip_coef_clamped_addr = clip_coef_clamped.buffer()->address();

    for (uint32_t i = 0; i < cores.size(); ++i) {
        const CoreCoord& core = cores.at(i);

        const auto& input = inputs.at(i);
        const auto input_addr = input.buffer()->address();
        const auto num_tiles = static_cast<uint32_t>(input.physical_volume()) / tt::constants::TILE_HW;

        // reader
        reader_desc.runtime_args.emplace_back(
            core, KernelDescriptor::CoreRuntimeArgs{input_addr, clip_coef_clamped_addr, num_tiles});

        // writer
        writer_desc.runtime_args.emplace_back(core, KernelDescriptor::CoreRuntimeArgs{input_addr, num_tiles});

        // compute
        compute_desc.runtime_args.emplace_back(core, KernelDescriptor::CoreRuntimeArgs{num_tiles});
    }

    desc.kernels.push_back(std::move(reader_desc));
    desc.kernels.push_back(std::move(writer_desc));
    desc.kernels.push_back(std::move(compute_desc));

    return desc;
}

}  // namespace ttnn::operations::moreh::moreh_clip_grad_norm_step3
