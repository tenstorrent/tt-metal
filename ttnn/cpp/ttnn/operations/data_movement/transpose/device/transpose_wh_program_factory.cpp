// SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC.
// SPDX-License-Identifier: Apache-2.0

#include "transpose_wh_program_factory.hpp"

#include <tt_stl/assert.hpp>
#include <tt-metalium/constants.hpp>
#include <tt-metalium/host_api.hpp>
#include <tt-metalium/tensor_accessor_args.hpp>
#include <tt-metalium/work_split.hpp>

using namespace tt::constants;
using namespace tt::tt_metal;

namespace ttnn::prim {

namespace {

void set_runtime_args_wh_tiled(
    Program& program,
    KernelHandle reader_kernel_id,
    KernelHandle compute_kernel_id,
    KernelHandle writer_kernel_id,
    const Tensor& input_tensor,
    Tensor& output_tensor,
    uint32_t num_cores_total,
    uint32_t num_cores_y,
    const CoreRangeSet& core_group_1,
    uint32_t num_tiles_per_core_group_1,
    const CoreRangeSet& core_group_2,
    uint32_t num_tiles_per_core_group_2,
    bool is_create) {
    auto input_shape = input_tensor.padded_shape();

    uint32_t W = input_shape[3], H = input_shape[2];

    uint32_t Wt = W / TILE_WIDTH;
    uint32_t Ht = H / TILE_HEIGHT;

    auto HtWt = Ht * Wt;

    auto& cached_reader_args = GetRuntimeArgs(program, reader_kernel_id);
    auto& cached_compute_args = GetRuntimeArgs(program, compute_kernel_id);
    auto& cached_writer_args = GetRuntimeArgs(program, writer_kernel_id);

    for (uint32_t i = 0, num_tiles_read = 0; i < num_cores_total; i++) {
        CoreCoord core = {i / num_cores_y, i % num_cores_y};
        uint32_t num_tiles_per_core;

        if (core_group_1.contains(core)) {
            num_tiles_per_core = num_tiles_per_core_group_1;
        } else if (core_group_2.contains(core)) {
            num_tiles_per_core = num_tiles_per_core_group_2;
        } else {
            num_tiles_per_core = 0;

            if (!is_create) {
                auto& reader_args = cached_reader_args.at(core.x).at(core.y);
                auto& compute_args = cached_compute_args.at(core.x).at(core.y);
                auto& writer_args = cached_writer_args.at(core.x).at(core.y);

                reader_args[1] = 0;
                compute_args[0] = 0;
                writer_args[1] = 0;
                continue;
            }
        }

        uint32_t h = num_tiles_read % Ht;
        uint32_t w = num_tiles_read / Ht % Wt;

        if (is_create) {
            SetRuntimeArgs(
                program,
                reader_kernel_id,
                core,
                {input_tensor.buffer()->address(),
                 num_tiles_per_core,
                 tt::round_down(num_tiles_read, HtWt) + (h * Wt) + w,
                 h,
                 w,
                 Ht,
                 Wt,
                 HtWt});

            SetRuntimeArgs(program, compute_kernel_id, core, {num_tiles_per_core});

            SetRuntimeArgs(
                program,
                writer_kernel_id,
                core,
                {output_tensor.buffer()->address(), num_tiles_per_core, num_tiles_read});
        } else {
            auto& reader_args = cached_reader_args.at(core.x).at(core.y);
            auto& compute_args = cached_compute_args.at(core.x).at(core.y);
            auto& writer_args = cached_writer_args.at(core.x).at(core.y);

            reader_args[0] = input_tensor.buffer()->address();
            reader_args[1] = num_tiles_per_core;
            reader_args[2] = tt::round_down(num_tiles_read, HtWt) + h * Wt + w;
            reader_args[3] = h;
            reader_args[4] = w;
            reader_args[5] = Ht;
            reader_args[6] = Wt;
            reader_args[7] = HtWt;

            compute_args[0] = num_tiles_per_core;

            writer_args[0] = output_tensor.buffer()->address();
            writer_args[1] = num_tiles_per_core;
            writer_args[2] = num_tiles_read;
        }

        num_tiles_read += num_tiles_per_core;
    }
}

void set_runtime_args_wh_rm(
    Program& program,
    KernelHandle reader_kernel_id,
    KernelHandle compute_kernel_id,
    KernelHandle writer_kernel_id,
    const Tensor& input_tensor,
    Tensor& output_tensor,
    uint32_t num_cores_total,
    uint32_t num_cores_y,
    const CoreRangeSet& core_group_1,
    uint32_t num_hw_blocks_per_core_group_1,
    const CoreRangeSet& core_group_2,
    uint32_t num_hw_blocks_per_core_group_2,
    bool is_create) {
    auto input_shape = input_tensor.logical_shape();

    uint32_t W = input_shape[3], H = input_shape[2];

    auto& cached_reader_args = GetRuntimeArgs(program, reader_kernel_id);
    auto& cached_compute_args = GetRuntimeArgs(program, compute_kernel_id);
    auto& cached_writer_args = GetRuntimeArgs(program, writer_kernel_id);

    for (uint32_t i = 0, num_sticks_read = 0, num_sticks_write = 0; i < num_cores_total; i++) {
        CoreCoord core = {i / num_cores_y, i % num_cores_y};
        uint32_t num_hw_blocks_per_core;

        if (core_group_1.contains(core)) {
            num_hw_blocks_per_core = num_hw_blocks_per_core_group_1;
        } else if (core_group_2.contains(core)) {
            num_hw_blocks_per_core = num_hw_blocks_per_core_group_2;
        } else {
            num_hw_blocks_per_core = 0;
        }

        if (is_create) {
            SetRuntimeArgs(
                program,
                reader_kernel_id,
                core,
                {input_tensor.buffer()->address(), num_sticks_read, num_hw_blocks_per_core});

            SetRuntimeArgs(program, compute_kernel_id, core, {num_hw_blocks_per_core});

            SetRuntimeArgs(
                program,
                writer_kernel_id,
                core,
                {output_tensor.buffer()->address(), num_sticks_write, num_hw_blocks_per_core});
        } else {
            auto& reader_args = cached_reader_args.at(core.x).at(core.y);
            auto& compute_args = cached_compute_args.at(core.x).at(core.y);
            auto& writer_args = cached_writer_args.at(core.x).at(core.y);

            reader_args[0] = input_tensor.buffer()->address();
            reader_args[1] = num_sticks_read;
            reader_args[2] = num_hw_blocks_per_core;

            compute_args[0] = num_hw_blocks_per_core;

            writer_args[0] = output_tensor.buffer()->address();
            writer_args[1] = num_sticks_write;
            writer_args[2] = num_hw_blocks_per_core;
        }

        num_sticks_read += num_hw_blocks_per_core * H;
        num_sticks_write += num_hw_blocks_per_core * W;
    }
}

}  // namespace

TransposeWHProgramFactory::cached_program_t TransposeWHProgramFactory::create(
    const TransposeParams& /*operation_attributes*/, const TransposeInputs& tensor_args, Tensor& output_tensor) {
    const auto& input_tensor = tensor_args.input;

    TT_ASSERT(input_tensor.storage_type() == StorageType::DEVICE, "Operand to transpose_wh needs to be on device!");
    TT_ASSERT(input_tensor.buffer() != nullptr, "Operand to transpose_wh needs to be allocated in a buffer on device!");

    uint32_t num_tensor_tiles = input_tensor.physical_volume() / TILE_HW;
    uint32_t W = input_tensor.logical_shape()[3], H = input_tensor.logical_shape()[2];
    uint32_t NC = input_tensor.logical_shape()[1] * input_tensor.logical_shape()[0];
    bool row_major = input_tensor.layout() == Layout::ROW_MAJOR;

    uint32_t ht = (H + TILE_HEIGHT - 1) / TILE_HEIGHT;
    uint32_t wt = (W + TILE_WIDTH - 1) / TILE_WIDTH;

    Program program = CreateProgram();

    tt::DataFormat src0_cb_data_format = datatype_to_dataformat_converter(input_tensor.dtype());
    uint32_t src0_single_tile_size = tt::tile_size(src0_cb_data_format);
    tt::DataFormat dst_cb_data_format = datatype_to_dataformat_converter(output_tensor.dtype());
    uint32_t dst_single_tile_size = tt::tile_size(dst_cb_data_format);

    Buffer* src0_buffer = input_tensor.buffer();
    IDevice* device = input_tensor.device();

    bool fp32_dest_acc_en = src0_cb_data_format == tt::DataFormat::Float32;

    auto compute_with_storage_grid_size = device->compute_with_storage_grid_size();
    uint32_t num_cores_x = compute_with_storage_grid_size.x;
    uint32_t num_cores_y = compute_with_storage_grid_size.y;
    uint32_t num_cores_total = num_cores_x * num_cores_y;
    CoreRange total_cores({0, 0}, {num_cores_x - 1, num_cores_y - 1});

    auto [num_cores, all_cores, core_group_1, core_group_2, num_tiles_per_core_group_1, num_tiles_per_core_group_2] =
        split_work_to_cores(compute_with_storage_grid_size, row_major ? NC : num_tensor_tiles);

    Buffer* dst_buffer = output_tensor.buffer();
    TT_ASSERT(dst_buffer != nullptr, "Output buffer should be allocated on device!");

    uint32_t src0_cb_index = 0;
    uint32_t num_input_tiles = row_major ? wt * 2 : 2;
    CircularBufferConfig cb_src0_config =
        CircularBufferConfig(num_input_tiles * src0_single_tile_size, {{src0_cb_index, src0_cb_data_format}})
            .set_page_size(src0_cb_index, src0_single_tile_size);
    CreateCircularBuffer(program, total_cores, cb_src0_config);

    uint32_t output_cb_index = tt::CBIndex::c_16;
    uint32_t num_output_tiles = row_major ? ht * 2 : 2;
    CircularBufferConfig cb_output_config =
        CircularBufferConfig(num_output_tiles * dst_single_tile_size, {{output_cb_index, dst_cb_data_format}})
            .set_page_size(output_cb_index, dst_single_tile_size);
    CreateCircularBuffer(program, total_cores, cb_output_config);

    if (row_major) {
        uint32_t im_cb_index = 24;
        uint32_t num_im_tiles = ht * wt;
        CircularBufferConfig cb_im_config =
            CircularBufferConfig(num_im_tiles * src0_single_tile_size, {{im_cb_index, src0_cb_data_format}})
                .set_page_size(im_cb_index, src0_single_tile_size);
        CreateCircularBuffer(program, total_cores, cb_im_config);

        uint32_t im2_cb_index = 25;
        uint32_t num_im2_tiles = ht;
        CircularBufferConfig cb_im2_config =
            CircularBufferConfig(num_im2_tiles * dst_single_tile_size, {{im2_cb_index, dst_cb_data_format}})
                .set_page_size(im2_cb_index, dst_single_tile_size);
        CreateCircularBuffer(program, total_cores, cb_im2_config);
    }

    std::vector<uint32_t> reader_compile_time_args;
    if (row_major) {
        reader_compile_time_args.push_back(ht);
        reader_compile_time_args.push_back(H > TILE_HEIGHT ? TILE_HEIGHT : H % TILE_HEIGHT);
        reader_compile_time_args.push_back(H % TILE_HEIGHT == 0 ? TILE_HEIGHT : H % TILE_HEIGHT);
        reader_compile_time_args.push_back(wt);
        reader_compile_time_args.push_back(W);
        reader_compile_time_args.push_back(ht * wt);
        reader_compile_time_args.push_back(W * input_tensor.element_size());
        reader_compile_time_args.push_back(wt * input_tensor.element_size() * TILE_WIDTH);
        reader_compile_time_args.push_back(W * input_tensor.element_size());
    }
    TensorAccessorArgs(*src0_buffer).append_to(reader_compile_time_args);

    std::vector<uint32_t> writer_compile_time_args = {output_cb_index};
    if (row_major) {
        writer_compile_time_args.push_back(ht);
        writer_compile_time_args.push_back(H);
        writer_compile_time_args.push_back(wt);
        writer_compile_time_args.push_back(W > TILE_WIDTH ? TILE_WIDTH : W % TILE_WIDTH);
        writer_compile_time_args.push_back(W % TILE_WIDTH == 0 ? TILE_WIDTH : W % TILE_WIDTH);
        writer_compile_time_args.push_back(ht * wt);
        writer_compile_time_args.push_back(H * output_tensor.element_size());
        writer_compile_time_args.push_back(ht * output_tensor.element_size() * TILE_HEIGHT);
        writer_compile_time_args.push_back(H * output_tensor.element_size());
    }
    TensorAccessorArgs(*dst_buffer).append_to(writer_compile_time_args);

    KernelHandle reader_kernel_id = CreateKernel(
        program,
        row_major ? "ttnn/cpp/ttnn/operations/data_movement/transpose/device/kernels/dataflow/"
                    "reader_unary_transpose_wh_interleaved_start_id_rm.cpp"
                  : "ttnn/cpp/ttnn/operations/data_movement/transpose/device/kernels/dataflow/"
                    "reader_unary_transpose_wh_interleaved_start_id.cpp",
        total_cores,
        ReaderDataMovementConfig(reader_compile_time_args));

    KernelHandle writer_kernel_id = CreateKernel(
        program,
        row_major
            ? "ttnn/cpp/ttnn/operations/data_movement/transpose/device/kernels/dataflow/"
              "writer_unary_transpose_wh_interleaved_start_id_rm.cpp"
            : "ttnn/cpp/ttnn/operations/eltwise/unary/device/kernels/dataflow/writer_unary_interleaved_start_id.cpp",
        total_cores,
        WriterDataMovementConfig(writer_compile_time_args));

    std::vector<uint32_t> compute_kernel_args = {};
    if (row_major) {
        compute_kernel_args.push_back(ht);
        compute_kernel_args.push_back(wt);
        compute_kernel_args.push_back(ht * wt);
    }
    auto compute_kernel_id = CreateKernel(
        program,
        row_major ? "ttnn/cpp/ttnn/operations/data_movement/transpose/device/kernels/compute/transpose_wh_rm.cpp"
                  : "ttnn/cpp/ttnn/operations/data_movement/transpose/device/kernels/compute/transpose_wh.cpp",
        total_cores,
        ComputeConfig{.fp32_dest_acc_en = fp32_dest_acc_en, .compile_args = compute_kernel_args});

    if (row_major) {
        set_runtime_args_wh_rm(
            program,
            reader_kernel_id,
            compute_kernel_id,
            writer_kernel_id,
            input_tensor,
            output_tensor,
            num_cores_total,
            num_cores_y,
            core_group_1,
            num_tiles_per_core_group_1,
            core_group_2,
            num_tiles_per_core_group_2,
            true);
    } else {
        set_runtime_args_wh_tiled(
            program,
            reader_kernel_id,
            compute_kernel_id,
            writer_kernel_id,
            input_tensor,
            output_tensor,
            num_cores_total,
            num_cores_y,
            core_group_1,
            num_tiles_per_core_group_1,
            core_group_2,
            num_tiles_per_core_group_2,
            true);
    }

    return {
        std::move(program),
        {.reader_kernel_id = reader_kernel_id,
         .compute_kernel_id = compute_kernel_id,
         .writer_kernel_id = writer_kernel_id,
         .core_group_1 = core_group_1,
         .core_group_2 = core_group_2,
         .num_cores_total = num_cores_total,
         .num_cores_y = num_cores_y,
         .num_tiles_per_core_group_1 = num_tiles_per_core_group_1,
         .num_tiles_per_core_group_2 = num_tiles_per_core_group_2,
         .is_row_major = row_major}};
}

void TransposeWHProgramFactory::override_runtime_arguments(
    cached_program_t& cached_program,
    const TransposeParams& /*operation_attributes*/,
    const TransposeInputs& tensor_args,
    Tensor& output_tensor) {
    auto& program = cached_program.program;
    auto& shared_variables = cached_program.shared_variables;

    if (shared_variables.is_row_major) {
        set_runtime_args_wh_rm(
            program,
            shared_variables.reader_kernel_id,
            shared_variables.compute_kernel_id,
            shared_variables.writer_kernel_id,
            tensor_args.input,
            output_tensor,
            shared_variables.num_cores_total,
            shared_variables.num_cores_y,
            shared_variables.core_group_1,
            shared_variables.num_tiles_per_core_group_1,
            shared_variables.core_group_2,
            shared_variables.num_tiles_per_core_group_2,
            false);
    } else {
        set_runtime_args_wh_tiled(
            program,
            shared_variables.reader_kernel_id,
            shared_variables.compute_kernel_id,
            shared_variables.writer_kernel_id,
            tensor_args.input,
            output_tensor,
            shared_variables.num_cores_total,
            shared_variables.num_cores_y,
            shared_variables.core_group_1,
            shared_variables.num_tiles_per_core_group_1,
            shared_variables.core_group_2,
            shared_variables.num_tiles_per_core_group_2,
            false);
    }
}

}  // namespace ttnn::prim
