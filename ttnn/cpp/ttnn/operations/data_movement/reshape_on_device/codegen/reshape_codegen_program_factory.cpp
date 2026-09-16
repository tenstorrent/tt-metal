// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include "ttnn/operations/data_movement/reshape_on_device/codegen/reshape_codegen_program_factory.hpp"

#include <cstdint>
#include <utility>
#include <vector>

#include <tt_stl/assert.hpp>

#include <tt-metalium/allocator.hpp>
#include <tt-metalium/constants.hpp>
#include <tt-metalium/core_coord.hpp>
#include <tt-metalium/device.hpp>
#include <tt-metalium/hal.hpp>
#include <tt-metalium/program_descriptors.hpp>
#include <tt-metalium/tensor_accessor_args.hpp>
#include <tt-metalium/tt_backend_api_types.hpp>
#include <tt-metalium/work_split.hpp>

#include "ttnn/tensor/tensor.hpp"

using namespace tt::tt_metal;

namespace ttnn::prim {

namespace {

struct CoreSplit {
    CoreRangeSet all_cores;
    std::vector<CoreCoord> cores_in_order;
    CoreRangeSet core_group_1;
    CoreRangeSet core_group_2;
    uint32_t work_per_core_1 = 0;
    uint32_t work_per_core_2 = 0;
};

CoreSplit split_work(const Tensor& input, uint32_t total_work) {
    IDevice* device = input.device();
    auto grid_size = device->compute_with_storage_grid_size();
    // row_wise=false (column-major core enumeration) matches the generator's split_cores(), which
    // always calls its work splitter at its row_wise=False default.
    auto [num_cores, all_cores, core_group_1, core_group_2, work_per_core_1, work_per_core_2] =
        tt::tt_metal::split_work_to_cores(grid_size, total_work, /*row_wise=*/false);
    return CoreSplit{
        .all_cores = all_cores,
        .cores_in_order = corerange_to_cores(all_cores, num_cores, /*row_wise=*/false),
        .core_group_1 = core_group_1,
        .core_group_2 = core_group_2,
        .work_per_core_1 = work_per_core_1,
        .work_per_core_2 = work_per_core_2,
    };
}

uint32_t work_for_core(const CoreSplit& split, const CoreCoord& core) {
    if (split.core_group_1.contains(core)) {
        return split.work_per_core_1;
    }
    if (split.core_group_2.contains(core)) {
        return split.work_per_core_2;
    }
    return 0;
}

}  // namespace

ProgramDescriptor ReshapeCodegenProgramFactory::create_descriptor(
    const ReshapeCodegenParams& /*operation_attributes*/,
    const ReshapeCodegenInputs& tensor_args,
    Tensor& tensor_return_value) {
    const Tensor& input = tensor_args.input;
    Tensor& output = tensor_return_value;
    Buffer* src_buffer = input.buffer();
    Buffer* dst_buffer = output.buffer();
    TT_FATAL(src_buffer != nullptr, "ReshapeCodegen input must be allocated on device!");
    TT_FATAL(dst_buffer != nullptr, "ReshapeCodegen output must be allocated on device!");

    IDevice* device = input.device();
    const uint32_t elem_size = input.element_size();
    const uint32_t old_stick_bytes = static_cast<uint32_t>(input.logical_shape()[-1]) * elem_size;
    const uint32_t new_stick_bytes = static_cast<uint32_t>(output.logical_shape()[-1]) * elem_size;
    uint32_t num_new_sticks = 1;
    for (int i = 0; i < static_cast<int>(output.logical_shape().rank()) - 1; ++i) {
        num_new_sticks *= static_cast<uint32_t>(output.logical_shape()[i]);
    }

    const uint32_t noc_max_burst_bytes = tt::tt_metal::hal::get_noc_max_burst_size_bytes();
    // TensorAccessorArgs is the authority on physical page pitch; DRAM and interleaved-L1 buffers
    // carry different alignments (architecture DRAM alignment vs. a fixed 16B L1 pitch), and using
    // the wrong one would make an "aligned" read cross a narrower page's boundary.
    const uint32_t input_alignment = static_cast<uint32_t>(src_buffer->alignment());
    const uint32_t output_alignment = static_cast<uint32_t>(dst_buffer->alignment());
    const uint32_t usable_l1_bytes =
        device->l1_size_per_core() - device->allocator()->get_base_allocator_addr(tt::tt_metal::HalMemType::L1);

    const ReshapeRmArbitraryPlan plan = plan_reshape_rm_arbitrary_transport(
        old_stick_bytes,
        new_stick_bytes,
        num_new_sticks,
        input_alignment,
        output_alignment,
        noc_max_burst_bytes,
        usable_l1_bytes);
    TT_FATAL(
        plan.nabatch > 0,
        "ReshapeCodegen: transport plan does not fit L1 for this shape; supported_by_codegen() should have "
        "rejected it before reaching program creation");

    const CoreSplit split = split_work(input, plan.total_units);
    tt::DataFormat cb_data_format = tt::tt_metal::datatype_to_dataformat_converter(input.dtype());

    ProgramDescriptor desc;

    constexpr uint32_t kCbOut = 0;
    constexpr uint32_t kCbScratch = 1;

    desc.cbs.push_back(CBDescriptor{
        .total_size = 2 * plan.nabatch * plan.slab_slot_bytes,
        .core_ranges = split.all_cores,
        .format_descriptors = {{CBFormatDescriptor{
            .buffer_index = kCbOut,
            .data_format = cb_data_format,
            .page_size = plan.slab_slot_bytes,
        }}},
    });
    desc.cbs.push_back(CBDescriptor{
        .total_size = plan.nabatch * plan.region_stride,
        .core_ranges = split.all_cores,
        .format_descriptors = {{CBFormatDescriptor{
            .buffer_index = kCbScratch,
            .data_format = cb_data_format,
            .page_size = plan.region_stride,
        }}},
    });

    std::vector<uint32_t> reader_ct_args = {
        kCbOut,
        kCbScratch,
        plan.old_stick_bytes,
        plan.old_page_bytes,
        plan.new_stick_bytes,
        plan.slab_bytes,
        plan.slab_slot_bytes,
        plan.slabs_per_output,
        plan.input_alignment,
        plan.noc_max_burst_bytes,
        plan.new_page_bytes,
        plan.nabatch,
        plan.region_stride,
    };
    TensorAccessorArgs(*src_buffer).append_to(reader_ct_args);

    KernelDescriptor reader_desc;
    reader_desc.kernel_source =
        "ttnn/cpp/ttnn/operations/data_movement/reshape_on_device/codegen/kernels/reader_reshape_rm_arbitrary.cpp";
    reader_desc.source_type = KernelDescriptor::SourceType::FILE_PATH;
    reader_desc.core_ranges = split.all_cores;
    reader_desc.compile_time_args = std::move(reader_ct_args);
    reader_desc.config = ReaderConfigDescriptor{};

    std::vector<uint32_t> writer_ct_args = {
        kCbOut,
        plan.new_stick_bytes,
        plan.new_page_bytes,
        plan.slab_bytes,
        plan.slabs_per_output,
        plan.noc_max_burst_bytes,
        plan.nabatch,
        plan.slab_slot_bytes,
    };
    TensorAccessorArgs(*dst_buffer).append_to(writer_ct_args);

    KernelDescriptor writer_desc;
    writer_desc.kernel_source =
        "ttnn/cpp/ttnn/operations/data_movement/reshape_on_device/codegen/kernels/writer_reshape_rm_arbitrary.cpp";
    writer_desc.source_type = KernelDescriptor::SourceType::FILE_PATH;
    writer_desc.core_ranges = split.all_cores;
    writer_desc.compile_time_args = std::move(writer_ct_args);
    writer_desc.config = WriterConfigDescriptor{};

    uint32_t start = 0;
    for (const auto& core : split.cores_in_order) {
        const uint32_t n = work_for_core(split, core);
        reader_desc.emplace_runtime_args(core, {src_buffer, start, n});
        writer_desc.emplace_runtime_args(core, {dst_buffer, start, n});
        start += n;
    }

    desc.kernels.push_back(std::move(reader_desc));
    desc.kernels.push_back(std::move(writer_desc));
    return desc;
}

}  // namespace ttnn::prim
