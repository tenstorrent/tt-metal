// SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC.
// SPDX-License-Identifier: Apache-2.0

#include "transpose_hc_tiled_program_factory.hpp"

#include <tt_stl/assert.hpp>
#include <tt-metalium/constants.hpp>
#include <tt-metalium/hal.hpp>
#include <tt-metalium/host_api.hpp>
#include <tt-metalium/tensor_accessor_args.hpp>
#include <tt-metalium/work_split.hpp>
#include <tt-logger/tt-logger.hpp>

using namespace tt::constants;
using namespace tt::tt_metal;

namespace ttnn::prim {

namespace {

void set_runtime_args_hc_tiled(
    Program& program,
    KernelHandle reader_kernel_id,
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
    auto* input_buffer = input_tensor.buffer();
    auto* output_buffer = output_tensor.buffer();
    auto input_shape = input_tensor.padded_shape();

    uint32_t W = input_shape[3], H = input_shape[2], C = input_shape[1];
    uint32_t HW = H * W;
    uint32_t HW_bytes = HW * input_tensor.element_size();
    uint32_t CHW_bytes = C * HW * input_tensor.element_size();

    uint32_t Wt = W / TILE_WIDTH;
    uint32_t Ct = C / TILE_HEIGHT;
    uint32_t CtHWt = Ct * H * Wt;
    uint32_t CtWt = Ct * Wt;

    auto& cached_reader_args = GetRuntimeArgs(program, reader_kernel_id);
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
        }

        uint32_t h = num_tiles_read / CtWt % H;
        uint32_t ct = num_tiles_read / Wt % Ct;

        if (is_create) {
            SetRuntimeArgs(
                program,
                reader_kernel_id,
                core,
                {input_buffer->address(),
                 Wt,
                 H,
                 Ct,
                 HW_bytes,
                 CHW_bytes,
                 num_tiles_read,
                 num_tiles_per_core,
                 num_tiles_read / CtHWt * CHW_bytes,
                 h,
                 h / TILE_HEIGHT * Wt,
                 ct,
                 ct * TILE_HEIGHT * HW_bytes,
                 num_tiles_read % Wt});

            SetRuntimeArgs(
                program, writer_kernel_id, core, {output_buffer->address(), num_tiles_per_core, num_tiles_read});
        } else {
            auto& reader_args = cached_reader_args.at(core.x).at(core.y);
            auto& writer_args = cached_writer_args.at(core.x).at(core.y);

            reader_args[0] = input_buffer->address();
            reader_args[1] = Wt;
            reader_args[2] = H;
            reader_args[3] = Ct;
            reader_args[4] = HW_bytes;
            reader_args[5] = CHW_bytes;
            reader_args[6] = num_tiles_read;
            reader_args[7] = num_tiles_per_core;
            reader_args[8] = num_tiles_read / CtHWt * CHW_bytes;
            reader_args[9] = h;
            reader_args[10] = h / TILE_HEIGHT * Wt;
            reader_args[11] = ct;
            reader_args[12] = ct * TILE_HEIGHT * HW_bytes;
            reader_args[13] = num_tiles_read % Wt;

            writer_args[0] = output_buffer->address();
            writer_args[1] = num_tiles_per_core;
            writer_args[2] = num_tiles_read;
        }

        num_tiles_read += num_tiles_per_core;
    }
}

}  // namespace

TransposeHCTiledProgramFactory::cached_program_t TransposeHCTiledProgramFactory::create(
    const TransposeParams& /*operation_attributes*/, const TransposeInputs& tensor_args, Tensor& output_tensor) {
    const auto& input_tensor = tensor_args.input;

    TT_ASSERT(input_tensor.storage_type() == StorageType::DEVICE, "Operand to transpose_hc needs to be on device!");
    TT_ASSERT(input_tensor.buffer() != nullptr, "Operand to transpose_hc needs to be allocated in a buffer on device!");

    uint32_t sub_tile_line_bytes = 16 * input_tensor.element_size();
    uint32_t num_tensor_tiles = input_tensor.physical_volume() / TILE_HW;

    Program program = CreateProgram();

    tt::DataFormat cb_data_format = datatype_to_dataformat_converter(input_tensor.dtype());
    uint32_t single_tile_size = tt::tile_size(cb_data_format);

    log_debug(tt::LogOp, "transpose_hc_tiled");
    log_debug(tt::LogOp, "sub_tile_line_bytes: {}", sub_tile_line_bytes);
    log_debug(tt::LogOp, "cb_data_format: {}", cb_data_format);
    log_debug(tt::LogOp, "single_tile_size: {}", single_tile_size);

    IDevice* device = input_tensor.device();
    auto compute_with_storage_grid_size = device->compute_with_storage_grid_size();
    uint32_t num_cores_x = compute_with_storage_grid_size.x;
    uint32_t num_cores_y = compute_with_storage_grid_size.y;
    uint32_t num_cores_total = num_cores_x * num_cores_y;
    CoreRange total_cores({0, 0}, {num_cores_x - 1, num_cores_y - 1});

    auto [num_cores, all_cores, core_group_1, core_group_2, num_tiles_per_core_group_1, num_tiles_per_core_group_2] =
        split_work_to_cores(compute_with_storage_grid_size, num_tensor_tiles);

    Buffer* dst_buffer = output_tensor.buffer();
    TT_ASSERT(dst_buffer != nullptr, "Output buffer should be allocated on device!");

    uint32_t src0_cb_index = 0;
    // check if we need to allocate a scratch buffer
    // The kernel reads several 16 element face lines (32B for BFLOAT16) from different input tiles to form a single
    // output tile, one output tile at a time Each face line is 32 bytes, so if our minimum read alignment is greater
    // than that (64B for Blackhole) then we will have reads from unaligned face-lines into differently aligned
    // destination face-lines
    // TODO: noc_async_write only require 16B alignment for both DRAM and L1 for Blackhole, so instead of reading in
    // face-lines from C tiles to form a single tile, we can load a single tile and then write out its face-lines to C
    // tiles
    uint32_t alignment = dst_buffer->alignment();
    bool misaligned = alignment > sub_tile_line_bytes;

    uint32_t num_input_tiles = 2;
    CircularBufferConfig cb_src0_config =
        CircularBufferConfig(num_input_tiles * single_tile_size, {{src0_cb_index, cb_data_format}})
            .set_page_size(src0_cb_index, single_tile_size);
    CreateCircularBuffer(program, total_cores, cb_src0_config);

    // need some scratch memory here - if we need data from a misaligned address then we need to read from the
    // nearest aligned address and then copy the data to the correct location
    if (misaligned) {
        uint32_t src1_cb_index = 1;
        CircularBufferConfig cb_src1_config =
            CircularBufferConfig(alignment, {{src1_cb_index, cb_data_format}}).set_page_size(src1_cb_index, alignment);
        CreateCircularBuffer(program, total_cores, cb_src1_config);
    }

    Buffer* src0_buffer = input_tensor.buffer();
    std::vector<uint32_t> reader_compile_time_args;
    reader_compile_time_args.push_back(sub_tile_line_bytes);
    reader_compile_time_args.push_back(cb_data_format == tt::DataFormat::Float32 ? 1 : 0);
    reader_compile_time_args.push_back(alignment);
    TensorAccessorArgs(*src0_buffer).append_to(reader_compile_time_args);

    std::vector<uint32_t> writer_compile_time_args = {src0_cb_index};
    TensorAccessorArgs(*dst_buffer).append_to(writer_compile_time_args);

    KernelHandle reader_kernel_id = CreateKernel(
        program,
        "ttnn/cpp/ttnn/operations/data_movement/transpose/device/kernels/dataflow/"
        "reader_unary_transpose_hc_interleaved_partitioned.cpp",
        total_cores,
        ReaderDataMovementConfig(reader_compile_time_args));

    KernelHandle writer_kernel_id = CreateKernel(
        program,
        "ttnn/cpp/ttnn/operations/eltwise/unary/device/kernels/dataflow/writer_unary_interleaved_start_id.cpp",
        total_cores,
        WriterDataMovementConfig(writer_compile_time_args));

    set_runtime_args_hc_tiled(
        program,
        reader_kernel_id,
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

    return {
        std::move(program),
        {.reader_kernel_id = reader_kernel_id,
         .writer_kernel_id = writer_kernel_id,
         .core_group_1 = core_group_1,
         .core_group_2 = core_group_2,
         .num_cores_total = num_cores_total,
         .num_cores_y = num_cores_y,
         .num_tiles_per_core_group_1 = num_tiles_per_core_group_1,
         .num_tiles_per_core_group_2 = num_tiles_per_core_group_2}};
}

void TransposeHCTiledProgramFactory::override_runtime_arguments(
    cached_program_t& cached_program,
    const TransposeParams& /*operation_attributes*/,
    const TransposeInputs& tensor_args,
    Tensor& output_tensor) {
    auto& program = cached_program.program;
    auto& shared_variables = cached_program.shared_variables;

    set_runtime_args_hc_tiled(
        program,
        shared_variables.reader_kernel_id,
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

}  // namespace ttnn::prim
