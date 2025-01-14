// SPDX-FileCopyrightText: © 2024 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include "consolidate_cache_program_factory.hpp"

#include <algorithm>

#include <tt-metalium/core_coord.hpp>
#include <tt-metalium/buffer.hpp>
#include "ttnn/tensor/tensor_impl.hpp"
#include "ttnn/operations/math.hpp"
#include <tt-metalium/work_split.hpp>
#include <tt-metalium/constants.hpp>
#include <tt-metalium/util.hpp>
#include <tt-metalium/host_api.hpp>

#include <sstream>
#include <type_traits>
#include <ranges>

#include <optional>
using namespace tt;
using namespace tt::constants;
using namespace tt::tt_metal;

namespace ttnn::operations::experimental::speculative_execution::detail {

// implementation of softmax with optional scale/mask (see the header for input_tensor more detailed description)
operation::ProgramWithCallbacks consolidate_cache(
    const Tensor& input_tensor,
    const Tensor& other_tensor,
    const Tensor& priority_tensor,
    const Tensor& other_priority_tensor,
    const Tensor& output_tensor) {
    Program program = CreateProgram();

    auto input_shape = input_tensor.get_logical_shape();
    tt::DataFormat input_df = tt_metal::datatype_to_dataformat_converter(input_tensor.get_dtype());
    auto input_element_size = input_tensor.element_size();
    auto input_num_tiles = input_tensor.buffer()->num_pages();
    const uint32_t input_is_dram = input_tensor.buffer()->buffer_type() == BufferType::DRAM ? 1 : 0;
    const uint32_t B = priority_tensor.get_logical_shape()[2];

    CoreCoord worker_core = {0, 0};  // TODO: assume single core for now

    uint32_t tensor_cb_index = CBIndex::c_0;
    CircularBufferConfig cb_tensor_config =
        CircularBufferConfig(input_num_tiles * input_element_size, {{tensor_cb_index, input_df}})
            .set_page_size(tensor_cb_index, input_element_size);
    auto cb_tensor = tt_metal::CreateCircularBuffer(program, worker_core, cb_tensor_config);

    // priority cb
    auto priority_buffer = priority_tensor.buffer();
    tt::DataFormat priority_df = tt_metal::datatype_to_dataformat_converter(priority_tensor.get_dtype());
    uint32_t priority_tensor_tile_size = tt_metal::detail::TileSize(priority_df);
    uint32_t priority_stick_size = priority_buffer->aligned_page_size();

    uint32_t priority_cb_index = CBIndex::c_1;
    auto priority_cb_config = CircularBufferConfig(priority_stick_size * 2 * B, {{priority_cb_index, priority_df}})
                                  .set_page_size(priority_cb_index, priority_stick_size);
    auto cb_priority = CreateCircularBuffer(program, worker_core, priority_cb_config);

    // kernel
    tt::log_debug("input_num_tiles: {}", input_num_tiles);
    tt::log_debug("input_is_dram: {}", input_is_dram);
    tt::log_debug("priority_stick_size: {}", priority_stick_size);
    tt::log_debug("B: {}", B);
    std::vector<uint32_t> reader_compile_time_args = {input_num_tiles, input_is_dram, priority_stick_size, B};
    auto reader_kernel_id = tt_metal::CreateKernel(
        program,
        "ttnn/cpp/ttnn/operations/experimental/speculative_execution/consolidate_cache/device/kernels/"
        "reader_consolidate_cache.cpp",
        worker_core,
        tt_metal::ReaderDataMovementConfig(reader_compile_time_args));

    std::vector<uint32_t> reader_runtime_args = {
        input_tensor.buffer()->address(),
        other_tensor.buffer()->address(),
        priority_tensor.buffer()->address(),
        other_priority_tensor.buffer()->address(),
        output_tensor.buffer()->address()};

    tt_metal::SetRuntimeArgs(program, reader_kernel_id, worker_core, reader_runtime_args);

    auto override_runtime_arguments_callback =
        [reader_kernel_id, worker_core](
            const void* operation,
            Program& program,
            const std::vector<Tensor>& input_tensors,
            const std::vector<std::optional<const Tensor>>& optional_input_tensors,
            const std::vector<Tensor>& output_tensors) {
            auto in_addr = input_tensors.at(0).buffer()->address();
            auto other_addr = input_tensors.at(1).buffer()->address();
            auto priority_addr = input_tensors.at(2).buffer()->address();
            auto other_priority_addr = input_tensors.at(3).buffer()->address();
            auto out_addr = output_tensors.at(0).buffer()->address();

            // Update runtime args
            auto& worker_reader_runtime_args_by_core = GetRuntimeArgs(program, reader_kernel_id);
            // reader
            auto& worker_reader_runtime_args = worker_reader_runtime_args_by_core[worker_core.x][worker_core.y];
            worker_reader_runtime_args.at(0) = in_addr;
            worker_reader_runtime_args.at(1) = other_addr;
            worker_reader_runtime_args.at(2) = priority_addr;
            worker_reader_runtime_args.at(3) = other_priority_addr;
            worker_reader_runtime_args.at(4) = out_addr;
        };

    return {.program = std::move(program), .override_runtime_arguments_callback = override_runtime_arguments_callback};
}

}  // namespace ttnn::operations::experimental::speculative_execution::detail
