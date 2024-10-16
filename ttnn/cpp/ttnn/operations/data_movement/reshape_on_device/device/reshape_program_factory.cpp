// SPDX-FileCopyrightText: © 2024 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include "tt_metal/common/work_split.hpp"
#include "ttnn/operations/math.hpp"
#include "tt_metal/host_api.hpp"
#include "tt_metal/common/constants.hpp"
#include "ttnn/operation.hpp"

namespace ttnn::operations::data_movement::detail {

operation::ProgramWithCallbacks reshape_tile_single_core(const Tensor &a, Tensor &output, int N, int C, int H, int W) {
    tt::tt_metal::Program program = tt::tt_metal::CreateProgram();

    CoreRange core({0, 0}, {0, 0});

    tt::DataFormat cb_data_format = tt::tt_metal::datatype_to_dataformat_converter(a.get_dtype());
    uint32_t single_tile_size = tt::tt_metal::detail::TileSize(cb_data_format);

    tt::tt_metal::Buffer *src0_buffer = a.buffer();

    uint32_t num_tiles = a.volume() / tt::constants::TILE_HW;

    // This should allocate a DRAM buffer on the device
    tt::tt_metal::Device *device = a.device();

    tt::tt_metal::LegacyShape output_shape = output.get_legacy_shape();

    tt::tt_metal::Buffer *dst_buffer = output.buffer();
    TT_ASSERT(dst_buffer != nullptr, "Output buffer should be allocated on device!");

    uint32_t src0_cb_index = 0;
    uint32_t num_input_tiles = 2;
    tt::tt_metal::CircularBufferConfig cb_src0_config =
        tt::tt_metal::CircularBufferConfig(num_input_tiles * single_tile_size, {{src0_cb_index, cb_data_format}})
            .set_page_size(src0_cb_index, single_tile_size);
    auto cb_src0 = tt::tt_metal::CreateCircularBuffer(program, core, cb_src0_config);

    bool src0_is_dram = src0_buffer->buffer_type() == tt::tt_metal::BufferType::DRAM ? 1 : 0;
    std::vector<uint32_t> reader_compile_time_args = {(std::uint32_t)src0_is_dram};

    bool dst_is_dram = dst_buffer->buffer_type() == tt::tt_metal::BufferType::DRAM ? 1 : 0;
    std::vector<uint32_t> writer_compile_time_args = {(std::uint32_t)src0_cb_index, (std::uint32_t)dst_is_dram};

    tt::tt_metal::KernelHandle unary_reader_kernel_id =
        tt::tt_metal::CreateKernel(program,
                                   "ttnn/cpp/ttnn/operations/data_movement/reshape_on_device/device/kernels/dataflow/"
                                   "reader_unary_reshape_interleaved.cpp",
                                   core,
                                   tt::tt_metal::ReaderDataMovementConfig(reader_compile_time_args));

    tt::tt_metal::KernelHandle unary_writer_kernel_id = tt::tt_metal::CreateKernel(
        program,
        "ttnn/cpp/ttnn/operations/eltwise/unary/device/kernels/dataflow/writer_unary_interleaved_start_id.cpp",
        core,
        tt::tt_metal::WriterDataMovementConfig(writer_compile_time_args));

    tt::tt_metal::SetRuntimeArgs(program,
                                 unary_reader_kernel_id,
                                 core,
                                 {src0_buffer->address(),
                                  a.get_legacy_shape()[3] / tt::constants::TILE_WIDTH,
                                  (uint32_t)output_shape[0],
                                  (uint32_t)output_shape[1],
                                  (uint32_t)output_shape[2] / tt::constants::TILE_HEIGHT,
                                  (uint32_t)output_shape[3] / tt::constants::TILE_WIDTH});

    tt::tt_metal::SetRuntimeArgs(program, unary_writer_kernel_id, core, {dst_buffer->address(), num_tiles, 0});

    auto override_runtime_args_callback = [unary_reader_kernel_id, unary_writer_kernel_id](
                                              const Program &program,
                                              const std::vector<Buffer *> &input_buffers,
                                              const std::vector<Buffer *> &output_buffers) {
        auto src_buffer = input_buffers.at(0);

        auto dst_buffer = output_buffers.at(0);

        CoreCoord core = {0, 0};

        {
            auto &runtime_args = GetRuntimeArgs(program, unary_reader_kernel_id, core);
            runtime_args[0] = src_buffer->address();
        }

        {
            auto &runtime_args = GetRuntimeArgs(program, unary_writer_kernel_id, core);
            runtime_args[0] = dst_buffer->address();
        }
    };

    return {std::move(program), override_runtime_args_callback};
}

operation::ProgramWithCallbacks reshape_rm_single_core(const Tensor &a, Tensor &output, int N, int C, int H, int W) {
    tt::tt_metal::Program program = tt::tt_metal::CreateProgram();
    CoreRange core({0, 0}, {0, 0});

    // This should allocate a DRAM buffer on the device
    tt::tt_metal::Device *device = a.device();
    tt::tt_metal::LegacyShape output_shape = output.get_legacy_shape();
    tt::tt_metal::Buffer *src0_buffer = a.buffer();
    tt::tt_metal::Buffer *dst_buffer = output.buffer();

    uint32_t num_old_sticks = a.get_legacy_shape()[0] * a.get_legacy_shape()[1] * a.get_legacy_shape()[2];
    uint32_t num_new_sticks = output_shape[0] * output_shape[1] * output_shape[2];

    uint32_t old_stick_size = a.get_legacy_shape()[3] * 2;  // Assuming bfloat16 data format
    uint32_t new_stick_size = output_shape[3] * 2;          // Assuming bfloat16 data format

    tt::DataFormat cb_data_format = tt::tt_metal::datatype_to_dataformat_converter(a.get_dtype());
    uint32_t single_tile_size = tt::tt_metal::detail::TileSize(cb_data_format);
    uint32_t src0_cb_index = 0;
    uint32_t num_input_tiles =
        (a.get_legacy_shape()[1] * a.get_legacy_shape()[2] * a.get_legacy_shape()[3] / tt::constants::TILE_HW);
    uint32_t num_output_tiles = (output_shape[1] * output_shape[2] * output_shape[3] / tt::constants::TILE_HW);

    // Currently added to support Bert large, TODO: Make op more generic, parallelize
    uint32_t available_l1 = device->l1_size_per_core() - device->get_base_allocator_addr(HalMemType::L1);
    if (num_input_tiles * single_tile_size + num_output_tiles * single_tile_size > available_l1) {
        if (old_stick_size >= new_stick_size) {
            if (old_stick_size % new_stick_size == 0) {
                // Maximize L1 usage. Is this needed or do we just need to double buffer 32 sticks (64)
                // Evenly divide L1 between input/output
                uint32_t w_tiles = a.get_legacy_shape()[3] / tt::constants::TILE_WIDTH;
                num_input_tiles = ((available_l1 / 2) / single_tile_size) / w_tiles * w_tiles;
                num_output_tiles = num_input_tiles;
            } else {
                // Not needed for Bert large at the moment so will trigger L1 OOM assert
            }
        } else {
            if (new_stick_size % old_stick_size == 0) {
                // Maximize L1 usage. Is this needed or do we just need to double buffer 32 sticks (64)
                // Evenly divide L1 between input/output
                uint32_t w_tiles = (output_shape[3] / tt::constants::TILE_WIDTH);
                num_output_tiles = ((available_l1 / 2) / single_tile_size) / w_tiles * w_tiles;
                num_input_tiles = num_output_tiles;
            } else {
                // Not needed for Bert large at the moment so will trigger L1 OOM assert
            }
        }
        TT_ASSERT(num_input_tiles > 0 && num_output_tiles > 0, "Cannot fit input/output rows into L1");
    }

    tt::tt_metal::CircularBufferConfig cb_src0_config =
        tt::tt_metal::CircularBufferConfig(num_input_tiles * single_tile_size, {{src0_cb_index, cb_data_format}})
            .set_page_size(src0_cb_index, single_tile_size);
    auto cb_src0 = tt::tt_metal::CreateCircularBuffer(program, core, cb_src0_config);

    uint32_t output_cb_index = 16;  // output operands start at index 16
    tt::tt_metal::CircularBufferConfig cb_output_config =
        tt::tt_metal::CircularBufferConfig(num_output_tiles * single_tile_size, {{output_cb_index, cb_data_format}})
            .set_page_size(output_cb_index, single_tile_size);
    auto cb_output = tt::tt_metal::CreateCircularBuffer(program, core, cb_output_config);

    // Reader compile-time args
    bool src0_is_dram = src0_buffer->buffer_type() == tt::tt_metal::BufferType::DRAM ? 1 : 0;
    bool old_stick_size_is_power_of_two = tt::tt_metal::is_power_of_two_at_least_32(old_stick_size);
    vector<uint32_t> reader_kernel_args = {src0_buffer->address(), num_old_sticks, old_stick_size};
    std::vector<uint32_t> reader_compile_time_args = {src0_is_dram};
    if (old_stick_size_is_power_of_two) {
        reader_kernel_args.push_back(log2(old_stick_size));

        // Use the fast stick size power of 2 path (get noc addr uses just shift operations, no slow multiply algorithm)
        reader_compile_time_args.push_back(1);
    } else {
        reader_compile_time_args.push_back(0);
    }

    // Writer compile-time args
    bool dst_is_dram = dst_buffer->buffer_type() == tt::tt_metal::BufferType::DRAM ? 1 : 0;
    bool new_stick_size_is_power_of_two = tt::tt_metal::is_power_of_two_at_least_32(new_stick_size);
    vector<uint32_t> writer_kernel_args = {dst_buffer->address(), num_new_sticks, new_stick_size};
    std::vector<uint32_t> writer_compile_time_args{dst_is_dram};
    if (new_stick_size_is_power_of_two) {
        writer_kernel_args.push_back(log2(new_stick_size));
        writer_compile_time_args.push_back(1);
    } else {
        writer_compile_time_args.push_back(0);
    }

    tt::tt_metal::KernelHandle unary_reader_kernel_id =
        tt::tt_metal::CreateKernel(program,
                                   "ttnn/cpp/ttnn/operations/data_movement/reshape_on_device/device/kernels/dataflow/"
                                   "reader_unary_reshape_stick_layout_interleaved.cpp",
                                   core,
                                   tt::tt_metal::ReaderDataMovementConfig(reader_compile_time_args));

    tt::tt_metal::KernelHandle unary_writer_kernel_id =
        tt::tt_metal::CreateKernel(program,
                                   "ttnn/cpp/ttnn/operations/data_movement/reshape_on_device/device/kernels/dataflow/"
                                   "writer_unary_reshape_stick_layout_interleaved.cpp",
                                   core,
                                   tt::tt_metal::WriterDataMovementConfig(writer_compile_time_args));

    // No compute required, so using blank kernel
    vector<uint32_t> compute_args = {
        uint(a.volume() / tt::constants::TILE_HW),  // per_core_block_cnt
        1                                           // per_core_block_size
    };

    auto eltwise_unary_kernel_id =
        tt::tt_metal::CreateKernel(program,
                                   "ttnn/cpp/ttnn/deprecated/tt_dnn/kernels/compute/eltwise_copy.cpp",
                                   core,
                                   tt::tt_metal::ComputeConfig{.compile_args = compute_args});

    tt::tt_metal::SetRuntimeArgs(program, unary_reader_kernel_id, core, reader_kernel_args);

    tt::tt_metal::SetRuntimeArgs(program, unary_writer_kernel_id, core, writer_kernel_args);

    auto override_runtime_args_callback = [unary_reader_kernel_id, unary_writer_kernel_id](
                                              const Program &program,
                                              const std::vector<Buffer *> &input_buffers,
                                              const std::vector<Buffer *> &output_buffers) {
        auto src_buffer = input_buffers.at(0);

        auto dst_buffer = output_buffers.at(0);

        CoreCoord core = {0, 0};

        {
            auto &runtime_args = GetRuntimeArgs(program, unary_reader_kernel_id, core);
            runtime_args[0] = src_buffer->address();
        }

        {
            auto &runtime_args = GetRuntimeArgs(program, unary_writer_kernel_id, core);
            runtime_args[0] = dst_buffer->address();
        }
    };

    return {std::move(program), override_runtime_args_callback};
}

std::vector<std::pair<std::vector<uint32_t>, std::vector<uint32_t>>> get_runtime_args_rm_multi_core(
    const Tensor &input_tensor,
    Tensor &output_tensor,
    uint32_t num_cores_total,
    uint32_t num_cores,
    uint32_t num_cores_y,
    CoreRangeSet core_group_1,
    uint32_t num_w_sticks_per_core_group_1,
    CoreRangeSet core_group_2,
    uint32_t num_w_sticks_per_core_group_2,
    bool split_work_by_old_sticks) {
    auto input_buffer = input_tensor.buffer();
    auto output_buffer = output_tensor.buffer();
    auto input_shape = input_tensor.get_legacy_shape();
    auto output_shape = output_tensor.get_legacy_shape();

    uint32_t old_stick_size = input_shape[3] * input_tensor.element_size();
    uint32_t new_stick_size = output_shape[3] * output_tensor.element_size();

    std::vector<std::pair<std::vector<uint32_t>, std::vector<uint32_t>>> ret_val(num_cores_total);

    uint32_t max_read_size = 2048;
    uint32_t curr_c = 0, curr_h = 0, curr_n = 0;
    for (uint32_t i = 0, curr_sticks_read = 0, curr_sticks_write = 0; i < num_cores_total; i++) {
        CoreCoord core = {i / num_cores_y, i % num_cores_y};
        uint32_t num_new_sticks_per_core = 0, num_old_sticks_per_core = 0;
        if (split_work_by_old_sticks) {
            if (core_group_1.core_coord_in_core_ranges(core)) {
                num_old_sticks_per_core = num_w_sticks_per_core_group_1;
            } else if (core_group_2.core_coord_in_core_ranges(core)) {
                num_old_sticks_per_core = num_w_sticks_per_core_group_2;
            }
        } else {
            if (core_group_1.core_coord_in_core_ranges(core)) {
                num_new_sticks_per_core = num_w_sticks_per_core_group_1;
            } else if (core_group_2.core_coord_in_core_ranges(core)) {
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
        std::vector<uint32_t> reader_runtime_args = {input_buffer->address(),
                                                     num_old_sticks_per_core_read,
                                                     num_old_sticks_read_per_barrier,
                                                     num_old_sticks_per_cb_push,
                                                     curr_sticks_read};

        // writer
        std::vector<uint32_t> writer_runtime_args = {output_buffer->address(),
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

operation::ProgramWithCallbacks reshape_rm_multi_core(const Tensor &a, Tensor &output, int N, int C, int H, int W) {
    TT_FATAL(a.get_dtype() == output.get_dtype(), "Error");

    tt::tt_metal::Program program = tt::tt_metal::CreateProgram();

    tt::tt_metal::Device *device = a.device();

    tt::tt_metal::LegacyShape output_shape = output.get_legacy_shape();
    tt::tt_metal::Buffer *src0_buffer = a.buffer();
    tt::tt_metal::Buffer *dst_buffer = output.buffer();

    tt::DataFormat cb_data_format = tt::tt_metal::datatype_to_dataformat_converter(a.get_dtype());

    uint32_t num_old_sticks = a.get_legacy_shape()[0] * a.get_legacy_shape()[1] * a.get_legacy_shape()[2];
    uint32_t num_new_sticks = output_shape[0] * output_shape[1] * output_shape[2];

    uint32_t old_stick_size = a.get_legacy_shape()[3] * a.element_size();
    uint32_t new_stick_size = output_shape[3] * output.element_size();

    if (old_stick_size > new_stick_size) {
        TT_FATAL(old_stick_size % new_stick_size == 0, "Error");
    } else {
        TT_FATAL(new_stick_size % old_stick_size == 0, "Error");
    }

    auto compute_with_storage_grid_size = device->compute_with_storage_grid_size();
    uint32_t num_cores_x = compute_with_storage_grid_size.x;
    uint32_t num_cores_y = compute_with_storage_grid_size.y;
    uint32_t num_cores_total = num_cores_x * num_cores_y;
    CoreRange total_cores({0, 0}, {num_cores_x - 1, num_cores_y - 1});

    bool split_work_by_old_sticks = old_stick_size > new_stick_size;

    auto [num_cores, all_cores, core_group_1, core_group_2, num_sticks_per_core_group_1, num_sticks_per_core_group_2] =
        tt::tt_metal::split_work_to_cores(compute_with_storage_grid_size,
                                          old_stick_size > new_stick_size ? num_old_sticks : num_new_sticks);

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
    bool src0_is_dram = src0_buffer->buffer_type() == tt::tt_metal::BufferType::DRAM ? 1 : 0;
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
    bool dst_is_dram = dst_buffer->buffer_type() == tt::tt_metal::BufferType::DRAM ? 1 : 0;
    bool new_stick_size_is_power_of_two = tt::tt_metal::is_power_of_two_at_least_32(new_stick_size);
    uint32_t new_log2_stick_size = new_stick_size_is_power_of_two ? (std::uint32_t)std::log2(new_stick_size) : 0;
    std::vector<uint32_t> writer_ct_args = {
        (std::uint32_t)src0_cb_index,
        (std::uint32_t)dst_is_dram,
        (std::uint32_t)new_stick_size,
        (std::uint32_t)new_stick_size_is_power_of_two,
        (std::uint32_t)new_stick_size_is_power_of_two ? new_log2_stick_size : new_stick_size};

    tt::tt_metal::KernelHandle reader_kernel_id =
        tt::tt_metal::CreateKernel(program,
                                   "ttnn/cpp/ttnn/operations/data_movement/reshape_on_device/device/kernels/dataflow/"
                                   "reader_unary_reshape_stick_layout_interleaved_multi_core.cpp",
                                   total_cores,
                                   tt::tt_metal::ReaderDataMovementConfig(reader_ct_args));

    tt::tt_metal::KernelHandle writer_kernel_id =
        tt::tt_metal::CreateKernel(program,
                                   "ttnn/cpp/ttnn/operations/data_movement/reshape_on_device/device/kernels/dataflow/"
                                   "writer_unary_reshape_stick_layout_interleaved_multi_core.cpp",
                                   total_cores,
                                   tt::tt_metal::WriterDataMovementConfig(writer_ct_args));

    auto all_runtime_args = get_runtime_args_rm_multi_core(a,
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

        tt::tt_metal::SetRuntimeArgs(program, writer_kernel_id, core, all_runtime_args[i].second

        );
    }

    auto override_runtime_args_callback = [reader_kernel_id, writer_kernel_id, compute_with_storage_grid_size](
                                              const void *operation,
                                              const Program &program,
                                              const std::vector<Tensor> &input_tensors,
                                              const std::vector<std::optional<const Tensor>> &,
                                              const std::vector<Tensor> &output_tensors) {
        auto src_tensor = input_tensors.at(0);

        auto dst_tensor = output_tensors.at(0);

        uint32_t num_cores_x = compute_with_storage_grid_size.x;
        uint32_t num_cores_y = compute_with_storage_grid_size.y;

        uint32_t num_cores_total = num_cores_x * num_cores_y;

        auto output_shape = dst_tensor.shape();

        uint32_t num_old_sticks =
            src_tensor.get_legacy_shape()[0] * src_tensor.get_legacy_shape()[1] * src_tensor.get_legacy_shape()[2];
        uint32_t num_new_sticks = output_shape[0] * output_shape[1] * output_shape[2];

        uint32_t old_stick_size = src_tensor.get_legacy_shape()[3] * src_tensor.element_size();
        uint32_t new_stick_size = output_shape[3] * dst_tensor.element_size();

        bool split_work_by_old_sticks = old_stick_size > new_stick_size;

        auto [num_cores,
              all_cores,
              core_group_1,
              core_group_2,
              num_sticks_per_core_group_1,
              num_sticks_per_core_group_2] =
            tt::tt_metal::split_work_to_cores(compute_with_storage_grid_size,
                                              old_stick_size > new_stick_size ? num_old_sticks : num_new_sticks);
        auto all_runtime_args = get_runtime_args_rm_multi_core(src_tensor,
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
