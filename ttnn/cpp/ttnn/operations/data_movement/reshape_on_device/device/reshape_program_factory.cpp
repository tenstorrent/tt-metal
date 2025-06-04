// SPDX-FileCopyrightText: © 2024 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include <tt-metalium/work_split.hpp>
#include "ttnn/operations/math.hpp"
#include <tt-metalium/host_api.hpp>
#include <tt-metalium/hal.hpp>
#include <tt-metalium/constants.hpp>
#include "ttnn/operation.hpp"

using namespace tt::tt_metal;

namespace ttnn::operations::data_movement::detail {

operation::ProgramWithCallbacks reshape_tile_single_core(const Tensor& a, Tensor& output) {
    tt::tt_metal::Program program = tt::tt_metal::CreateProgram();

    CoreRange core({0, 0}, {0, 0});

    tt::DataFormat cb_data_format = tt::tt_metal::datatype_to_dataformat_converter(a.get_dtype());
    uint32_t single_tile_size = tt::tt_metal::detail::TileSize(cb_data_format);

    tt::tt_metal::Buffer* src0_buffer = a.buffer();

    uint32_t num_tiles = a.volume() / tt::constants::TILE_HW;

    // This should allocate a DRAM buffer on the device
    tt::tt_metal::IDevice* device = a.device();

    auto output_shape = output.get_padded_shape();

    tt::tt_metal::Buffer* dst_buffer = output.buffer();
    TT_ASSERT(dst_buffer != nullptr, "Output buffer should be allocated on device!");

    uint32_t src0_cb_index = 0;
    uint32_t num_input_tiles = 2;
    tt::tt_metal::CircularBufferConfig cb_src0_config =
        tt::tt_metal::CircularBufferConfig(num_input_tiles * single_tile_size, {{src0_cb_index, cb_data_format}})
            .set_page_size(src0_cb_index, single_tile_size);
    auto cb_src0 = tt::tt_metal::CreateCircularBuffer(program, core, cb_src0_config);

    bool src0_is_dram = src0_buffer->buffer_type() == tt::tt_metal::BufferType::DRAM;
    uint32_t alignment = src0_is_dram ? hal::get_dram_alignment() : hal::get_l1_alignment();

    std::vector<uint32_t> reader_compile_time_args = {(std::uint32_t)src0_is_dram, alignment};

    bool dst_is_dram = dst_buffer->buffer_type() == tt::tt_metal::BufferType::DRAM;
    std::vector<uint32_t> writer_compile_time_args = {(std::uint32_t)src0_cb_index, (std::uint32_t)dst_is_dram};

    if (alignment > (tt::constants::FACE_WIDTH * a.element_size())) {
        uint32_t src1_cb_index = 1;
        tt::tt_metal::CircularBufferConfig cb_src1_config =
            tt::tt_metal::CircularBufferConfig(alignment, {{src1_cb_index, cb_data_format}})
                .set_page_size(src1_cb_index, alignment);
        auto cb_src1 = tt::tt_metal::CreateCircularBuffer(program, core, cb_src1_config);
    }

    tt::tt_metal::KernelHandle unary_reader_kernel_id = tt::tt_metal::CreateKernel(
        program,
        "ttnn/cpp/ttnn/operations/data_movement/reshape_on_device/device/kernels/dataflow/"
        "reader_unary_reshape_interleaved.cpp",
        core,
        tt::tt_metal::ReaderDataMovementConfig(reader_compile_time_args));

    tt::tt_metal::KernelHandle unary_writer_kernel_id = tt::tt_metal::CreateKernel(
        program,
        "ttnn/cpp/ttnn/operations/eltwise/unary/device/kernels/dataflow/writer_unary_interleaved_start_id.cpp",
        core,
        tt::tt_metal::WriterDataMovementConfig(writer_compile_time_args));

    tt::tt_metal::SetRuntimeArgs(
        program,
        unary_reader_kernel_id,
        core,
        {src0_buffer->address(),
         a.get_padded_shape()[3] / tt::constants::TILE_WIDTH,
         (uint32_t)output_shape[0],
         (uint32_t)output_shape[1],
         (uint32_t)output_shape[2] / tt::constants::TILE_HEIGHT,
         (uint32_t)output_shape[3] / tt::constants::TILE_WIDTH});

    tt::tt_metal::SetRuntimeArgs(program, unary_writer_kernel_id, core, {dst_buffer->address(), num_tiles, 0});

    auto override_runtime_args_callback = [unary_reader_kernel_id, unary_writer_kernel_id](
                                              const void* operation,
                                              Program& program,
                                              const std::vector<Tensor>& input_tensors,
                                              const std::vector<std::optional<const Tensor>>& optional_tensors,
                                              const std::vector<Tensor>& output_tensors) {
        auto src_buffer = input_tensors.at(0).buffer();

        auto dst_buffer = output_tensors.at(0).buffer();

        CoreCoord core = {0, 0};

        {
            auto& runtime_args = GetRuntimeArgs(program, unary_reader_kernel_id, core);
            runtime_args[0] = src_buffer->address();
        }

        {
            auto& runtime_args = GetRuntimeArgs(program, unary_writer_kernel_id, core);
            runtime_args[0] = dst_buffer->address();
        }
    };

    return {std::move(program), override_runtime_args_callback};
}

std::vector<std::pair<std::vector<uint32_t>, std::vector<uint32_t>>> get_runtime_args_rm_multi_core(
    const Tensor& input_tensor,
    Tensor& output_tensor,
    uint32_t num_cores_total,
    uint32_t num_cores,
    uint32_t num_cores_y,
    const CoreRangeSet& core_group_1,
    uint32_t num_w_sticks_per_core_group_1,
    const CoreRangeSet& core_group_2,
    uint32_t num_w_sticks_per_core_group_2,
    bool split_work_by_old_sticks) {
    auto input_buffer = input_tensor.buffer();
    auto output_buffer = output_tensor.buffer();
    auto input_shape = input_tensor.get_padded_shape();
    auto output_shape = output_tensor.get_padded_shape();

    uint32_t old_stick_size = input_shape[3] * input_tensor.element_size();
    uint32_t new_stick_size = output_shape[3] * output_tensor.element_size();

    std::vector<std::pair<std::vector<uint32_t>, std::vector<uint32_t>>> ret_val(num_cores_total);

    uint32_t max_read_size = 2048;
    uint32_t curr_c = 0, curr_h = 0, curr_n = 0;
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

operation::ProgramWithCallbacks reshape_rm_multi_core(const Tensor& a, Tensor& output) {
    TT_FATAL(a.get_dtype() == output.get_dtype(), "Error");

    tt::tt_metal::Program program = tt::tt_metal::CreateProgram();

    tt::tt_metal::IDevice* device = a.device();

    auto output_shape = output.get_padded_shape();
    tt::tt_metal::Buffer* src0_buffer = a.buffer();
    tt::tt_metal::Buffer* dst_buffer = output.buffer();

    tt::DataFormat cb_data_format = tt::tt_metal::datatype_to_dataformat_converter(a.get_dtype());

    uint32_t num_old_sticks = a.get_padded_shape()[0] * a.get_padded_shape()[1] * a.get_padded_shape()[2];
    uint32_t num_new_sticks = output_shape[0] * output_shape[1] * output_shape[2];

    uint32_t old_stick_size = a.get_padded_shape()[3] * a.element_size();
    uint32_t new_stick_size = output_shape[3] * output.element_size();

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
    auto cb_src0 = tt::tt_metal::CreateCircularBuffer(program, total_cores, cb_src0_config);

    // Reader compile-time args
    bool src0_is_dram = src0_buffer->buffer_type() == tt::tt_metal::BufferType::DRAM;
    bool old_stick_size_is_power_of_two = tt::tt_metal::is_power_of_two_at_least_32(old_stick_size);
    uint32_t old_log2_stick_size = old_stick_size_is_power_of_two ? (std::uint32_t)std::log2(old_stick_size) : 0;
    bool is_new_stick_larger = new_stick_size > old_stick_size;
    uint32_t new_old_stick_size_ratio =
        new_stick_size > old_stick_size ? new_stick_size / old_stick_size : old_stick_size / new_stick_size;
    std::vector<uint32_t> reader_ct_args = {
        (std::uint32_t)src0_is_dram,
        (std::uint32_t)old_stick_size,
        (std::uint32_t)old_stick_size_is_power_of_two,
        (std::uint32_t)old_stick_size_is_power_of_two ? old_log2_stick_size : old_stick_size};

    // Writer compile-time args
    bool dst_is_dram = dst_buffer->buffer_type() == tt::tt_metal::BufferType::DRAM;
    bool new_stick_size_is_power_of_two = tt::tt_metal::is_power_of_two_at_least_32(new_stick_size);
    uint32_t new_log2_stick_size = new_stick_size_is_power_of_two ? (std::uint32_t)std::log2(new_stick_size) : 0;
    std::vector<uint32_t> writer_ct_args = {
        (std::uint32_t)src0_cb_index,
        (std::uint32_t)dst_is_dram,
        (std::uint32_t)new_stick_size,
        (std::uint32_t)new_stick_size_is_power_of_two,
        (std::uint32_t)new_stick_size_is_power_of_two ? new_log2_stick_size : new_stick_size};

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
        a,
        output,
        num_cores_total,
        num_cores,
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

    auto override_runtime_args_callback = [reader_kernel_id, writer_kernel_id, compute_with_storage_grid_size](
                                              const void* operation,
                                              const Program& program,
                                              const std::vector<Tensor>& input_tensors,
                                              const std::vector<std::optional<const Tensor>>&,
                                              const std::vector<Tensor>& output_tensors) {
        auto src_tensor = input_tensors.at(0);

        auto dst_tensor = output_tensors.at(0);

        uint32_t num_cores_x = compute_with_storage_grid_size.x;
        uint32_t num_cores_y = compute_with_storage_grid_size.y;

        uint32_t num_cores_total = num_cores_x * num_cores_y;

        auto output_shape = dst_tensor.get_logical_shape();

        uint32_t num_old_sticks =
            src_tensor.get_padded_shape()[0] * src_tensor.get_padded_shape()[1] * src_tensor.get_padded_shape()[2];
        uint32_t num_new_sticks = output_shape[0] * output_shape[1] * output_shape[2];

        uint32_t old_stick_size = src_tensor.get_padded_shape()[3] * src_tensor.element_size();
        uint32_t new_stick_size = output_shape[3] * dst_tensor.element_size();

        bool split_work_by_old_sticks = old_stick_size > new_stick_size;

        auto
            [num_cores,
             all_cores,
             core_group_1,
             core_group_2,
             num_sticks_per_core_group_1,
             num_sticks_per_core_group_2] =
                tt::tt_metal::split_work_to_cores(
                    compute_with_storage_grid_size, old_stick_size > new_stick_size ? num_old_sticks : num_new_sticks);
        auto all_runtime_args = get_runtime_args_rm_multi_core(
            src_tensor,
            dst_tensor,
            num_cores_total,
            num_cores,
            num_cores_y,
            core_group_1,
            num_sticks_per_core_group_1,
            core_group_2,
            num_sticks_per_core_group_2,
            split_work_by_old_sticks);

        for (uint32_t i = 0; i < num_cores_total; i++) {
            CoreCoord core = {i / num_cores_y, i % num_cores_y};

            { SetRuntimeArgs(program, reader_kernel_id, core, all_runtime_args[i].first); }

            { SetRuntimeArgs(program, writer_kernel_id, core, all_runtime_args[i].second); }
        }
    };

    return {.program = std::move(program), .override_runtime_arguments_callback = override_runtime_args_callback};
}

}  // namespace ttnn::operations::data_movement::detail
