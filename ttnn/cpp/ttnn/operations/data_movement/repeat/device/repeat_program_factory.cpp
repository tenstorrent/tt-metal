// SPDX-FileCopyrightText: © 2023 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include "tt_metal/common/work_split.hpp"
#include "tt_metal/detail/util.hpp"
#include "tt_metal/host_api.hpp"

using namespace tt::constants;


namespace ttnn::operations::data_movement::detail {

operation::ProgramWithCallbacks repeat_multi_core(
    const Tensor &input_tensor, const uint32_t repeat_dim, const uint32_t num_repeats, const Tensor &output) {
    auto program = tt::tt_metal::CreateProgram();

    tt::tt_metal::Device *device = output.device();

    const tt::DataFormat cb_data_format = tt::tt_metal::datatype_to_dataformat_converter(output.get_dtype());

    const bool rm_layout = output.get_layout() == Layout::ROW_MAJOR;

    constexpr bool rm_orientation = false;

    uint32_t num_output_pages;
    uint32_t single_page_size;
    if (rm_layout) {
        num_output_pages = output.volume() / output.get_legacy_shape()[-1];
        single_page_size = align(output.element_size() * output.get_legacy_shape()[-1], output.buffer()->alignment());
    } else {
        num_output_pages = output.volume() / TILE_HW;
        single_page_size = tt::tt_metal::detail::TileSize(cb_data_format);
    }

    auto compute_with_storage_grid_size = device->compute_with_storage_grid_size();
    uint32_t num_cores_x = compute_with_storage_grid_size.x;
    uint32_t num_cores_y = compute_with_storage_grid_size.y;
    auto [num_cores, all_cores, core_group_1, core_group_2, num_tiles_per_core_group_1, num_tiles_per_core_group_2] =
        tt::tt_metal::split_work_to_cores(compute_with_storage_grid_size, num_output_pages, rm_orientation);

    tt::tt_metal::Buffer *dst_buffer = output.buffer();
    TT_ASSERT(dst_buffer != nullptr, "Output buffer should be allocated on device!");

    uint32_t src0_cb_index = 0;
    uint32_t num_input_pages = 2;
    tt::tt_metal::CircularBufferConfig cb_src0_config =
        tt::tt_metal::CircularBufferConfig(num_input_pages * single_page_size, {{src0_cb_index, cb_data_format}})
            .set_page_size(src0_cb_index, single_page_size);
    auto cb_src0 = tt::tt_metal::CreateCircularBuffer(program, all_cores, cb_src0_config);

    uint32_t num_dims = output.get_legacy_shape().rank();

    auto input_buffer = input_tensor.buffer();
    uint32_t src_addr = input_buffer->address();
    uint32_t src_is_dram = input_buffer->buffer_type() == tt::tt_metal::BufferType::DRAM ? 1 : 0;
    uint32_t src_page_size = input_buffer->page_size();
    uint32_t num_pages_per_block;

    uint32_t num_accum_pages = 1;
    uint32_t scale_factor = 1;

    // RM is special cased in the loop (dim_units = 1 for last dim else it's the dim size)
    if (!rm_layout) {
        if (repeat_dim == num_dims - 2) {
            scale_factor = TILE_HEIGHT;
        } else if (repeat_dim == num_dims - 1) {
            scale_factor = TILE_WIDTH;
        }
    }

    for (uint32_t i = repeat_dim + 1; i < num_dims; ++i) {
        num_accum_pages *= output.get_legacy_shape()[i];
    }
    if (rm_layout) {
        if (num_dims > 1 && repeat_dim < num_dims - 1) {
            num_accum_pages /= output.get_legacy_shape()[-1];
        }
    } else {
        if (repeat_dim < num_dims - 2) {
            num_accum_pages /= TILE_HW;
        } else if (repeat_dim == num_dims - 2) {
            num_accum_pages /= TILE_WIDTH;
        }
    }

    if (rm_layout) {
        if (repeat_dim == num_dims - 1) {
            num_pages_per_block = num_accum_pages;
        } else {
            uint32_t dim_pages = input_tensor.get_legacy_shape()[repeat_dim];
            num_pages_per_block = num_accum_pages * dim_pages;
        }
    } else {
        uint32_t dim_pages = input_tensor.get_legacy_shape()[repeat_dim] / scale_factor;
        num_pages_per_block = num_accum_pages * dim_pages;
    }

    vector<uint32_t> reader_kernel_args = {src_addr, 0, num_pages_per_block, 0, 0, 0, 0};
    if (rm_layout) {
        reader_kernel_args.push_back(src_page_size);
    }

    // Reader compile-time args
    // Data is 32 byte aligned
    bool dst_is_dram = dst_buffer->buffer_type() == tt::tt_metal::BufferType::DRAM ? 1 : 0;
    std::vector<uint32_t> reader_compile_time_args = {// interleaved accessor args
                                                      (std::uint32_t)src0_cb_index,
                                                      (std::uint32_t)src_is_dram,
                                                      (std::uint32_t)num_repeats};

    std::map<string, string> repeat_defines;

    if (rm_layout && repeat_dim == num_dims - 1) {
        repeat_defines["WIDTH_REPEAT"] = "1";
    }

    std::vector<uint32_t> writer_compile_time_args = {// interleaved accessor args
                                                      (std::uint32_t)src0_cb_index,
                                                      (std::uint32_t)dst_is_dram};

    tt::tt_metal::KernelHandle unary_reader_kernel_id = tt::tt_metal::CreateKernel(
        program,
        rm_layout
            ? "ttnn/cpp/ttnn/operations/data_movement/repeat/device/kernels/dataflow/reader_repeat_stick_layout_interleaved_start_id.cpp"
            : "ttnn/cpp/ttnn/operations/data_movement/repeat/device/kernels/dataflow/reader_repeat_interleaved_start_id.cpp",
        all_cores,
        tt::tt_metal::ReaderDataMovementConfig(reader_compile_time_args, repeat_defines));

    tt::tt_metal::KernelHandle unary_writer_kernel_id = tt::tt_metal::CreateKernel(
        program,
        rm_layout ? "ttnn/cpp/ttnn/deprecated/tt_dnn/kernels/dataflow/writer_unary_stick_layout_interleaved_start_id.cpp"
                  : "ttnn/cpp/ttnn/operations/eltwise/unary/device/kernels/dataflow/writer_unary_interleaved_start_id.cpp",
        all_cores,
        tt::tt_metal::WriterDataMovementConfig(writer_compile_time_args));

    const auto cores = grid_to_cores(num_cores, num_cores_x, num_cores_y, rm_orientation);
    uint32_t g1_num_cores = core_group_1.num_cores();
    for (uint32_t i = 0, num_pages_written = 0; i < cores.size(); ++i) {
        const CoreCoord &core = cores[i];
        uint32_t num_pages_per_core = 0;
        if (i < g1_num_cores) {
            num_pages_per_core = num_tiles_per_core_group_1;
        } else {
            num_pages_per_core = num_tiles_per_core_group_2;
        }
        uint32_t curr_repeat_idx = 0;
        uint32_t curr_idx_in_block = 0;
        uint32_t curr_block_start_id = 0;
        uint32_t curr_id = 0;
        if (rm_layout && repeat_dim == num_dims - 1) {
            curr_id = num_pages_written;
        } else {
            curr_repeat_idx = num_pages_written / num_pages_per_block % num_repeats;
            curr_idx_in_block = num_pages_written % num_pages_per_block;
            curr_block_start_id = num_pages_written / (num_pages_per_block * num_repeats) * num_pages_per_block;
            curr_id = curr_block_start_id + curr_idx_in_block;
        }

        reader_kernel_args[1] = num_pages_per_core;
        reader_kernel_args[3] = curr_repeat_idx;
        reader_kernel_args[4] = curr_idx_in_block;
        reader_kernel_args[5] = curr_block_start_id;
        reader_kernel_args[6] = curr_id;

        vector<uint32_t> writer_kernel_args;
        if (rm_layout) {
            writer_kernel_args = {
                dst_buffer->address(), output.buffer()->page_size(), num_pages_per_core, num_pages_written};
        } else {
            writer_kernel_args = {dst_buffer->address(), num_pages_per_core, num_pages_written};
        }
        tt::tt_metal::SetRuntimeArgs(program, unary_reader_kernel_id, core, reader_kernel_args);

        tt::tt_metal::SetRuntimeArgs(program, unary_writer_kernel_id, core, writer_kernel_args);
        num_pages_written += num_pages_per_core;
    }

    auto override_runtime_args_callback = [unary_reader_kernel_id, unary_writer_kernel_id, cores](
                                              const ProgramHandle program,
                                              const std::vector<Buffer *> &input_buffers,
                                              const std::vector<Buffer *> &output_buffers) {
        auto src_buffer = input_buffers.at(0);

        auto dst_buffer = output_buffers.at(0);

        for (const auto &core : cores) {
            {
                auto &runtime_args = GetRuntimeArgs(program, unary_reader_kernel_id, core);
                runtime_args[0] = src_buffer->address();
            }

            {
                auto &runtime_args = GetRuntimeArgs(program, unary_writer_kernel_id, core);
                runtime_args[0] = dst_buffer->address();
            }
        }
    };

    return {program, override_runtime_args_callback};
}

}
