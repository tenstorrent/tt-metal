// SPDX-FileCopyrightText: © 2024 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include "reshape_rm_program_factory.hpp"

#include <tt-metalium/constants.hpp>
#include <tt-metalium/hal.hpp>
#include <tt-metalium/host_api.hpp>
#include <tt-metalium/program_descriptors.hpp>
#include <tt-metalium/tensor_accessor_args.hpp>
#include <tt-metalium/work_split.hpp>

#include "ttnn/operation.hpp"
#include "ttnn/operations/math.hpp"

namespace ttnn::prim {

using namespace tt::tt_metal;

ProgramDescriptor ReshapeRMProgramFactory::create_descriptor(
    const ttnn::prim::ReshapeOnDeviceParams& /*operation_attributes*/,
    const ttnn::prim::ReshapeOnDeviceInputs& tensor_args,
    tt::tt_metal::Tensor& output_tensor) {
    const auto& input_tensor = tensor_args.input_tensor;
    TT_FATAL(
        input_tensor.dtype() == output_tensor.dtype(),
        "Input tensor dtype ({}) must match output tensor dtype ({})",
        input_tensor.dtype(),
        output_tensor.dtype());

    IDevice* device = input_tensor.device();

    auto output_shape = output_tensor.padded_shape();
    Buffer* src0_buffer = input_tensor.buffer();
    Buffer* dst_buffer = output_tensor.buffer();

    tt::DataFormat cb_data_format = datatype_to_dataformat_converter(input_tensor.dtype());

    uint32_t num_old_sticks =
        input_tensor.padded_shape()[0] * input_tensor.padded_shape()[1] * input_tensor.padded_shape()[2];
    uint32_t num_new_sticks = output_shape[0] * output_shape[1] * output_shape[2];

    uint32_t old_stick_size = input_tensor.padded_shape()[3] * input_tensor.element_size();
    uint32_t new_stick_size = output_shape[3] * output_tensor.element_size();

    TT_FATAL(
        std::max(old_stick_size, new_stick_size) % std::min(old_stick_size, new_stick_size) == 0,
        "Last dimension of the old shape ({}) should be divisible by the last dimension of the new shape ({}) or vice "
        "versa",
        old_stick_size,
        new_stick_size);

    auto compute_with_storage_grid_size = device->compute_with_storage_grid_size();
    uint32_t num_cores_x = compute_with_storage_grid_size.x;
    uint32_t num_cores_y = compute_with_storage_grid_size.y;
    uint32_t num_cores_total = num_cores_x * num_cores_y;
    CoreRange total_cores({0, 0}, {num_cores_x - 1, num_cores_y - 1});
    const CoreRangeSet total_core_ranges{total_cores};

    bool split_work_by_old_sticks = old_stick_size > new_stick_size;

    auto [num_cores, all_cores, core_group_1, core_group_2, num_sticks_per_core_group_1, num_sticks_per_core_group_2] =
        tt::tt_metal::split_work_to_cores(
            compute_with_storage_grid_size, split_work_by_old_sticks ? num_old_sticks : num_new_sticks);

    constexpr uint32_t src0_cb_index = 0;
    auto num_pages = num_sticks_per_core_group_1 > num_sticks_per_core_group_2 ? num_sticks_per_core_group_1
                                                                               : num_sticks_per_core_group_2;
    auto max_page_size = old_stick_size > new_stick_size ? old_stick_size : new_stick_size;
    auto page_size = new_stick_size;

    ProgramDescriptor desc;

    // CB sized for the largest page across the whole grid; page_size set to new_stick_size
    // so the writer drains output-sized chunks.
    desc.cbs.push_back(CBDescriptor{
        .total_size = num_pages * max_page_size,
        .core_ranges = total_core_ranges,
        .format_descriptors = {{CBFormatDescriptor{
            .buffer_index = static_cast<uint8_t>(src0_cb_index),
            .data_format = cb_data_format,
            .page_size = page_size,
        }}},
    });

    // Reader compile-time args
    std::vector<uint32_t> reader_ct_args = {old_stick_size};
    TensorAccessorArgs(*src0_buffer).append_to(reader_ct_args);

    // Writer compile-time args
    std::vector<uint32_t> writer_ct_args = {src0_cb_index, new_stick_size};
    TensorAccessorArgs(*dst_buffer).append_to(writer_ct_args);

    KernelDescriptor reader_desc;
    reader_desc.kernel_source =
        "ttnn/cpp/ttnn/operations/data_movement/reshape_on_device/device/kernels/dataflow/"
        "reader_unary_reshape_stick_layout_interleaved_multi_core.cpp";
    reader_desc.source_type = KernelDescriptor::SourceType::FILE_PATH;
    reader_desc.core_ranges = total_core_ranges;
    reader_desc.compile_time_args = std::move(reader_ct_args);
    reader_desc.config = ReaderConfigDescriptor{};

    KernelDescriptor writer_desc;
    writer_desc.kernel_source =
        "ttnn/cpp/ttnn/operations/data_movement/reshape_on_device/device/kernels/dataflow/"
        "writer_unary_reshape_stick_layout_interleaved_multi_core.cpp";
    writer_desc.source_type = KernelDescriptor::SourceType::FILE_PATH;
    writer_desc.core_ranges = total_core_ranges;
    writer_desc.compile_time_args = std::move(writer_ct_args);
    writer_desc.config = WriterConfigDescriptor{};

    constexpr uint32_t max_read_size = 2048;
    uint32_t curr_sticks_read = 0;
    uint32_t curr_sticks_write = 0;
    for (uint32_t i = 0; i < num_cores_total; i++) {
        CoreCoord core = {i / num_cores_y, i % num_cores_y};
        uint32_t num_new_sticks_per_core = 0;
        uint32_t num_old_sticks_per_core = 0;
        if (split_work_by_old_sticks) {
            if (core_group_1.contains(core)) {
                num_old_sticks_per_core = num_sticks_per_core_group_1;
            } else if (core_group_2.contains(core)) {
                num_old_sticks_per_core = num_sticks_per_core_group_2;
            }
        } else {
            if (core_group_1.contains(core)) {
                num_new_sticks_per_core = num_sticks_per_core_group_1;
            } else if (core_group_2.contains(core)) {
                num_new_sticks_per_core = num_sticks_per_core_group_2;
            }
        }

        uint32_t old_new_stick_size_ratio =
            old_stick_size > new_stick_size ? (old_stick_size / new_stick_size) : (new_stick_size / old_stick_size);
        if (split_work_by_old_sticks) {
            num_new_sticks_per_core = num_old_sticks_per_core * old_new_stick_size_ratio;
        } else {
            num_old_sticks_per_core = num_new_sticks_per_core * old_new_stick_size_ratio;
        }

        // issue more reads before calling barrier
        uint32_t num_old_sticks_per_core_read = 0;
        uint32_t num_old_sticks_read_per_barrier = 0;
        uint32_t num_old_sticks_per_cb_push = 0;
        uint32_t num_new_sticks_per_core_read = 0;
        uint32_t num_new_sticks_read_per_barrier = 0;
        uint32_t num_new_sticks_per_cb_push = 0;
        if (old_stick_size > new_stick_size) {
            if (num_old_sticks_per_core != 0) {
                num_old_sticks_per_core_read =
                    tt::tt_metal::merge_num_sticks_to_read(num_old_sticks_per_core, old_stick_size, max_read_size);
                num_old_sticks_read_per_barrier = num_old_sticks_per_core / num_old_sticks_per_core_read;
                num_old_sticks_per_cb_push = num_old_sticks_read_per_barrier * old_new_stick_size_ratio;

                num_new_sticks_per_cb_push = num_old_sticks_per_cb_push;
                num_new_sticks_read_per_barrier = num_old_sticks_per_cb_push;
                num_new_sticks_per_core_read = num_new_sticks_per_core / num_new_sticks_read_per_barrier;
            }
        } else {
            if (num_new_sticks_per_core != 0) {
                num_new_sticks_per_core_read =
                    tt::tt_metal::merge_num_sticks_to_read(num_new_sticks_per_core, new_stick_size, max_read_size);
                num_new_sticks_read_per_barrier = num_new_sticks_per_core / num_new_sticks_per_core_read;
                num_new_sticks_per_cb_push = num_new_sticks_read_per_barrier;

                num_old_sticks_per_cb_push = num_new_sticks_per_cb_push;
                num_old_sticks_read_per_barrier = num_new_sticks_per_cb_push * old_new_stick_size_ratio;
                num_old_sticks_per_core_read = num_old_sticks_per_core / num_old_sticks_read_per_barrier;
            }
        }

        // Per-core runtime args. Slot 0 holds the buffer address; idle cores (no sticks
        // scheduled) get 0u so we skip BufferBinding registration — the kernel short-circuits
        // when its sticks-to-read count is zero and never dereferences the address.
        const bool reader_idle = num_old_sticks_per_core == 0;
        if (reader_idle) {
            reader_desc.emplace_runtime_args(
                core,
                {0u,
                 num_old_sticks_per_core_read,
                 num_old_sticks_read_per_barrier,
                 num_old_sticks_per_cb_push,
                 curr_sticks_read});
        } else {
            reader_desc.emplace_runtime_args(
                core,
                {src0_buffer,
                 num_old_sticks_per_core_read,
                 num_old_sticks_read_per_barrier,
                 num_old_sticks_per_cb_push,
                 curr_sticks_read});
        }

        const bool writer_idle = num_new_sticks_per_core == 0;
        if (writer_idle) {
            writer_desc.emplace_runtime_args(
                core,
                {0u,
                 num_new_sticks_per_core_read,
                 num_new_sticks_read_per_barrier,
                 num_new_sticks_per_cb_push,
                 curr_sticks_write});
        } else {
            writer_desc.emplace_runtime_args(
                core,
                {dst_buffer,
                 num_new_sticks_per_core_read,
                 num_new_sticks_read_per_barrier,
                 num_new_sticks_per_cb_push,
                 curr_sticks_write});
        }

        curr_sticks_read += num_old_sticks_per_core;
        curr_sticks_write += num_new_sticks_per_core;
    }

    desc.kernels.push_back(std::move(reader_desc));
    desc.kernels.push_back(std::move(writer_desc));

    return desc;
}

}  // namespace ttnn::prim
