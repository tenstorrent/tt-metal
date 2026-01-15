// SPDX-FileCopyrightText: © 2024 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include "reshape_rm_program_factory.hpp"
#include <tt-metalium/work_split.hpp>
#include "ttnn/operations/math.hpp"
#include <tt-metalium/host_api.hpp>
#include <tt-metalium/hal.hpp>
#include <tt-metalium/constants.hpp>
#include <tt-metalium/tensor_accessor_args.hpp>
#include "ttnn/operation.hpp"

using namespace tt::tt_metal;

namespace ttnn::operations::data_movement::reshape_on_device {

namespace {
std::vector<std::pair<std::vector<uint32_t>, std::vector<uint32_t>>> get_runtime_args_rm_multi_core(
    const Tensor& input_tensor,
    Tensor& output_tensor,
    uint32_t num_cores_total,
    uint32_t num_cores_y,
    const CoreRangeSet& core_group_1,
    uint32_t num_w_sticks_per_core_group_1,
    const CoreRangeSet& core_group_2,
    uint32_t num_w_sticks_per_core_group_2,
    bool split_work_by_old_sticks) {
    auto* input_buffer = input_tensor.buffer();
    auto* output_buffer = output_tensor.buffer();
    auto input_shape = input_tensor.padded_shape();
    auto output_shape = output_tensor.padded_shape();

    uint32_t old_stick_size = input_shape[3] * input_tensor.element_size();
    uint32_t new_stick_size = output_shape[3] * output_tensor.element_size();

    std::vector<std::pair<std::vector<uint32_t>, std::vector<uint32_t>>> ret_val(num_cores_total);

    uint32_t max_read_size = 2048;
    for (uint32_t i = 0, curr_sticks_read = 0, curr_sticks_write = 0; i < num_cores_total; i++) {
        CoreCoord core = {i / num_cores_y, i % num_cores_y};
        uint32_t num_new_sticks_per_core = 0, num_old_sticks_per_core = 0;
        if (split_work_by_old_sticks) {
            if (core_group_1.contains(core)) {
                num_old_sticks_per_core = num_w_sticks_per_core_group_1;
            } else if (core_group_2.contains(core)) {
                num_old_sticks_per_core = num_w_sticks_per_core_group_2;
            }
        } else {
            if (core_group_1.contains(core)) {
                num_new_sticks_per_core = num_w_sticks_per_core_group_1;
            } else if (core_group_2.contains(core)) {
                num_new_sticks_per_core = num_w_sticks_per_core_group_2;
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
        uint32_t num_old_sticks_per_core_read = 0, num_old_sticks_read_per_barrier = 0, num_old_sticks_per_cb_push = 0;
        uint32_t num_new_sticks_per_core_read = 0, num_new_sticks_read_per_barrier = 0, num_new_sticks_per_cb_push = 0;
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
                num_old_sticks_read_per_barrier = num_old_sticks_per_cb_push * old_new_stick_size_ratio;
                num_old_sticks_per_core_read = num_old_sticks_per_core / num_old_sticks_read_per_barrier;
            }
        }

        // reader
        std::vector<uint32_t> reader_runtime_args = {
            input_buffer->address(),
            num_old_sticks_per_core_read,
            num_old_sticks_read_per_barrier,
            num_old_sticks_per_cb_push,
            curr_sticks_read};

        // writer
        std::vector<uint32_t> writer_runtime_args = {
            output_buffer->address(),
            num_new_sticks_per_core_read,
            num_new_sticks_read_per_barrier,
            num_new_sticks_per_cb_push,
            curr_sticks_write};

        ret_val[i] = {reader_runtime_args, writer_runtime_args};

        curr_sticks_read += num_old_sticks_per_core;
        curr_sticks_write += num_new_sticks_per_core;
    }

    return ret_val;
}
}  // namespace

ReshapeRMProgramFactory::cached_program_t ReshapeRMProgramFactory::create(
    const reshape_on_device::ReshapeOnDeviceParams& /*operation_attributes*/,
    const reshape_on_device::ReshapeOnDeviceInputs& tensor_args,
    reshape_on_device::tensor_return_value_t& output_tensor) {
    const auto& input_tensor = tensor_args.input_tensor;
    TT_FATAL(
        input_tensor.dtype() == output_tensor.dtype(),
        "Input tensor dtype ({}) must match output tensor dtype ({})",
        input_tensor.dtype(),
        output_tensor.dtype());

    tt::tt_metal::Program program = tt::tt_metal::CreateProgram();

    tt::tt_metal::IDevice* device = input_tensor.device();

    auto output_shape = output_tensor.padded_shape();
    tt::tt_metal::Buffer* src0_buffer = input_tensor.buffer();
    tt::tt_metal::Buffer* dst_buffer = output_tensor.buffer();

    tt::DataFormat cb_data_format = tt::tt_metal::datatype_to_dataformat_converter(input_tensor.dtype());

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

    bool split_work_by_old_sticks = old_stick_size > new_stick_size;

    auto [num_cores, all_cores, core_group_1, core_group_2, num_sticks_per_core_group_1, num_sticks_per_core_group_2] =
        tt::tt_metal::split_work_to_cores(
            compute_with_storage_grid_size, old_stick_size > new_stick_size ? num_old_sticks : num_new_sticks);

    uint32_t src0_cb_index = 0;
    auto num_pages = num_sticks_per_core_group_1 > num_sticks_per_core_group_2 ? num_sticks_per_core_group_1
                                                                               : num_sticks_per_core_group_2;
    auto max_page_size = old_stick_size > new_stick_size ? old_stick_size : new_stick_size;
    auto page_size = new_stick_size;
    tt::tt_metal::CircularBufferConfig cb_src0_config =
        tt::tt_metal::CircularBufferConfig(num_pages * max_page_size, {{src0_cb_index, cb_data_format}})
            .set_page_size(src0_cb_index, page_size);
    tt::tt_metal::CreateCircularBuffer(program, total_cores, cb_src0_config);

    // Reader compile-time args
    std::vector<uint32_t> reader_ct_args = {old_stick_size};
    tt::tt_metal::TensorAccessorArgs(*src0_buffer).append_to(reader_ct_args);

    // Writer compile-time args
    std::vector<uint32_t> writer_ct_args = {src0_cb_index, new_stick_size};
    tt::tt_metal::TensorAccessorArgs(*dst_buffer).append_to(writer_ct_args);

    tt::tt_metal::KernelHandle reader_kernel_id = tt::tt_metal::CreateKernel(
        program,
        "ttnn/cpp/ttnn/operations/data_movement/reshape_on_device/device/kernels/dataflow/"
        "reader_unary_reshape_stick_layout_interleaved_multi_core.cpp",
        total_cores,
        tt::tt_metal::ReaderDataMovementConfig(reader_ct_args));

    tt::tt_metal::KernelHandle writer_kernel_id = tt::tt_metal::CreateKernel(
        program,
        "ttnn/cpp/ttnn/operations/data_movement/reshape_on_device/device/kernels/dataflow/"
        "writer_unary_reshape_stick_layout_interleaved_multi_core.cpp",
        total_cores,
        tt::tt_metal::WriterDataMovementConfig(writer_ct_args));

    auto all_runtime_args = get_runtime_args_rm_multi_core(
        input_tensor,
        output_tensor,
        num_cores_total,
        num_cores_y,
        core_group_1,
        num_sticks_per_core_group_1,
        core_group_2,
        num_sticks_per_core_group_2,
        split_work_by_old_sticks);

    for (uint32_t i = 0; i < num_cores_total; i++) {
        CoreCoord core = {i / num_cores_y, i % num_cores_y};
        tt::tt_metal::SetRuntimeArgs(program, reader_kernel_id, core, all_runtime_args[i].first);

        tt::tt_metal::SetRuntimeArgs(
            program, writer_kernel_id, core, all_runtime_args[i].second

        );
    }

    return {std::move(program), {reader_kernel_id, writer_kernel_id, compute_with_storage_grid_size}};
}

void ReshapeRMProgramFactory::override_runtime_arguments(
    cached_program_t& cached_program,
    const reshape_on_device::ReshapeOnDeviceParams& /*operation_attributes*/,
    const reshape_on_device::ReshapeOnDeviceInputs& tensor_args,
    reshape_on_device::tensor_return_value_t& output_tensor) {
    const auto& src_tensor = tensor_args.input_tensor;
    auto& dst_tensor = output_tensor;

    auto& program = cached_program.program;
    auto& reader_kernel_id = cached_program.shared_variables.reader_kernel_id;
    auto& writer_kernel_id = cached_program.shared_variables.writer_kernel_id;
    auto& compute_with_storage_grid_size = cached_program.shared_variables.compute_with_storage_grid_size;

    uint32_t num_cores_x = compute_with_storage_grid_size.x;
    uint32_t num_cores_y = compute_with_storage_grid_size.y;

    uint32_t num_cores_total = num_cores_x * num_cores_y;

    auto output_shape = dst_tensor.logical_shape();

    uint32_t num_old_sticks =
        src_tensor.padded_shape()[0] * src_tensor.padded_shape()[1] * src_tensor.padded_shape()[2];
    uint32_t num_new_sticks = output_shape[0] * output_shape[1] * output_shape[2];

    uint32_t old_stick_size = src_tensor.padded_shape()[3] * src_tensor.element_size();
    uint32_t new_stick_size = output_shape[3] * dst_tensor.element_size();

    bool split_work_by_old_sticks = old_stick_size > new_stick_size;

    auto [num_cores, all_cores, core_group_1, core_group_2, num_sticks_per_core_group_1, num_sticks_per_core_group_2] =
        tt::tt_metal::split_work_to_cores(
            compute_with_storage_grid_size, old_stick_size > new_stick_size ? num_old_sticks : num_new_sticks);
    auto all_runtime_args = get_runtime_args_rm_multi_core(
        src_tensor,
        dst_tensor,
        num_cores_total,
        num_cores_y,
        core_group_1,
        num_sticks_per_core_group_1,
        core_group_2,
        num_sticks_per_core_group_2,
        split_work_by_old_sticks);

    for (uint32_t i = 0; i < num_cores_total; i++) {
        CoreCoord core = {i / num_cores_y, i % num_cores_y};

        {
            SetRuntimeArgs(program, reader_kernel_id, core, all_runtime_args[i].first);
        }

        {
            SetRuntimeArgs(program, writer_kernel_id, core, all_runtime_args[i].second);
        }
    }
}

}  // namespace ttnn::operations::data_movement::reshape_on_device
