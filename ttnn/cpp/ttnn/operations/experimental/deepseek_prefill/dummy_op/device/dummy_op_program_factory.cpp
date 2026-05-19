// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include "dummy_op_program_factory.hpp"

#include <cstdint>
#include <utility>

#include <tt-metalium/constants.hpp>
#include <tt-metalium/core_coord.hpp>
#include <tt-metalium/host_api.hpp>
#include <tt-metalium/tensor_accessor_args.hpp>

#include "ttnn/operation.hpp"

namespace ttnn::operations::experimental::deepseek_prefill::dummy_op {

DummyOpProgramFactory::cached_program_t DummyOpProgramFactory::create(
    const DummyOpParams& operation_attributes, const DummyOpInputs& tensor_args, Tensor& /*tensor_return_value*/) {
    tt::tt_metal::Program program = tt::tt_metal::CreateProgram();

    const auto& input_tensor = tensor_args.input_tensor;
    auto* input_buffer = input_tensor.buffer();

    const tt::DataFormat tile_data_format = tt::tt_metal::datatype_to_dataformat_converter(input_tensor.dtype());
    const uint32_t single_tile_size = tt::tile_size(tile_data_format);
    const uint32_t total_tiles = input_buffer->num_pages();

    // Single-row CoreRangeSet from operation_attributes (validated in
    // dummy_op_device_operation.cpp).
    const CoreRangeSet& core_range_set = operation_attributes.worker_core_range_set;
    const auto& range = *core_range_set.ranges().begin();
    const uint32_t row_y = range.start_coord.y;
    const uint32_t x_start = range.start_coord.x;
    const uint32_t num_cores = range.end_coord.x - x_start + 1;
    std::vector<CoreCoord> cores;
    cores.reserve(num_cores);
    for (uint32_t i = 0; i < num_cores; ++i) {
        cores.emplace_back(x_start + i, row_y);
    }

    constexpr uint32_t cb_tile = tt::CBIndex::c_0;

    // Deep enough that the reader can race ahead of the writer (or vice versa)
    // without stalling at cb_reserve_back / cb_wait_front each tile.
    constexpr uint32_t tile_buffering = 32;
    tt::tt_metal::CircularBufferConfig cb_tile_config =
        tt::tt_metal::CircularBufferConfig(tile_buffering * single_tile_size, {{cb_tile, tile_data_format}})
            .set_page_size(cb_tile, single_tile_size);
    tt::tt_metal::CreateCircularBuffer(program, core_range_set, cb_tile_config);

    std::vector<uint32_t> reader_compile_time_args = {
        cb_tile,
        operation_attributes.num_iter,
        num_cores,
    };
    tt::tt_metal::TensorAccessorArgs(input_buffer).append_to(reader_compile_time_args);

    std::vector<uint32_t> writer_compile_time_args = {
        cb_tile,
        operation_attributes.num_iter,
        num_cores,
    };
    tt::tt_metal::TensorAccessorArgs(input_buffer).append_to(writer_compile_time_args);

    auto reader_kernel_id = tt::tt_metal::CreateKernel(
        program,
        "ttnn/cpp/ttnn/operations/experimental/deepseek_prefill/dummy_op/device/kernels/dataflow/reader_dummy_op.cpp",
        core_range_set,
        tt::tt_metal::ReaderDataMovementConfig(reader_compile_time_args));

    auto writer_kernel_id = tt::tt_metal::CreateKernel(
        program,
        "ttnn/cpp/ttnn/operations/experimental/deepseek_prefill/dummy_op/device/kernels/dataflow/writer_dummy_op.cpp",
        core_range_set,
        tt::tt_metal::WriterDataMovementConfig(writer_compile_time_args));

    for (uint32_t core_id = 0; core_id < num_cores; ++core_id) {
        const auto& core = cores[core_id];
        tt::tt_metal::SetRuntimeArgs(
            program,
            reader_kernel_id,
            core,
            {input_buffer->address(), total_tiles, core_id, operation_attributes.global_semaphore_address});
        tt::tt_metal::SetRuntimeArgs(program, writer_kernel_id, core, {input_buffer->address(), total_tiles, core_id});
    }

    return cached_program_t{
        std::move(program),
        {.reader_kernel_id = reader_kernel_id, .writer_kernel_id = writer_kernel_id, .cores = std::move(cores)}};
}

void DummyOpProgramFactory::override_runtime_arguments(
    cached_program_t& cached_program,
    const DummyOpParams& operation_attributes,
    const DummyOpInputs& tensor_args,
    Tensor& /*tensor_return_value*/) {
    auto& program = cached_program.program;
    const auto reader_kernel_id = cached_program.shared_variables.reader_kernel_id;
    const auto writer_kernel_id = cached_program.shared_variables.writer_kernel_id;
    const auto& cores = cached_program.shared_variables.cores;

    const uint32_t input_addr = tensor_args.input_tensor.buffer()->address();
    const uint32_t sem_addr = operation_attributes.global_semaphore_address;

    for (const auto& core : cores) {
        auto& reader_args = tt::tt_metal::GetRuntimeArgs(program, reader_kernel_id, core);
        reader_args[0] = input_addr;
        reader_args[3] = sem_addr;

        auto& writer_args = tt::tt_metal::GetRuntimeArgs(program, writer_kernel_id, core);
        writer_args[0] = input_addr;
    }
}

}  // namespace ttnn::operations::experimental::deepseek_prefill::dummy_op
