// SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#include "concat_nd_sharded_program_factory.hpp"
#include "ttnn/cpp/ttnn/operations/data_movement/concat/device/kernels/dataflow/concat_nd_sharded_args.hpp"

#include <algorithm>
#include <numeric>
#include <vector>

#include "ttnn/tensor/tensor.hpp"
#include <tt-metalium/host_api.hpp>
#include <tt-metalium/constants.hpp>
#include <tt-metalium/buffer_distribution_spec.hpp>
#include <tt-metalium/buffer_page_mapping.hpp>
#include <tt-metalium/tensor_accessor_args.hpp>

namespace ttnn::prim {

namespace {

using namespace tt::tt_metal;

constexpr uint32_t CONCAT_ND_SHARDED_MAX_NUM_INPUTS = ttnn::kernel::CONCAT_ND_SHARDED_MAX_NUM_INPUTS;

// Returns the device address of the buffer on the given core (start of this core's shard).
// For sharded buffers, uses the buffer's page mapping to get device_start_page for the core.
uint32_t get_buffer_address_for_core(Buffer* buffer, const CoreCoord& core) {
    TT_FATAL(buffer->buffer_distribution_spec().has_value(), "ND sharded concat expects sharded buffers");
    const auto& page_mapping = buffer->get_buffer_page_mapping();
    auto it = page_mapping->core_to_core_id.find(core);
    TT_FATAL(it != page_mapping->core_to_core_id.end(), "Core not found in buffer page mapping");
    uint32_t core_id = it->second;
    TT_FATAL(
        core_id < page_mapping->core_page_mappings.size() && !page_mapping->core_page_mappings[core_id].empty(),
        "No page mapping for core");
    const auto& core_mapping = page_mapping->core_page_mappings[core_id][0];
    uint32_t address = buffer->address() + core_mapping.device_start_page * buffer->aligned_page_size();
    if (buffer->is_dram()) {
        address += buffer->device()->allocator()->get_bank_offset(
            BufferType::DRAM, buffer->device()->dram_channel_from_logical_core(core));
    }
    return address;
}

// Unroll grid into a vector of CoreCoord in row-major order.
std::vector<CoreCoord> grid_to_coresA(const CoreRangeSet& grid) {
    std::vector<CoreCoord> cores;
    for (const CoreRange& range : grid.ranges()) {
        for (uint32_t y = range.start_coord.y; y <= range.end_coord.y; ++y) {
            for (uint32_t x = range.start_coord.x; x <= range.end_coord.x; ++x) {
                cores.push_back(CoreCoord{x, y});
            }
        }
    }
    return cores;
}

}  // namespace

// Creates the device program for ND sharded concat: a single reader kernel on each core that
// copies this core's input shards (in concat order) into this core's output shard using
// TensorAccessors. No writer kernel; the reader does read-from-inputs and write-to-output.
ConcatNDShardedProgramFactory::cached_program_t ConcatNDShardedProgramFactory::create(
    const ConcatParams& /*operation_attributes*/, const ConcatInputs& tensor_args, Tensor& tensor_return_value) {
    using namespace tt::constants;
    using namespace tt::tt_metal;

    const std::vector<Tensor>& input_tensors = tensor_args.input_tensors;
    Tensor& output = tensor_return_value;
    const uint32_t num_input_tensors = static_cast<uint32_t>(input_tensors.size());

    TT_FATAL(
        num_input_tensors <= CONCAT_ND_SHARDED_MAX_NUM_INPUTS,
        "ND sharded concat supports at most {} inputs, got {}",
        CONCAT_ND_SHARDED_MAX_NUM_INPUTS,
        num_input_tensors);

    const auto& nd_shard_spec = input_tensors[0].nd_shard_spec().value();
    const CoreRangeSet all_cores = nd_shard_spec.grid;
    const std::vector<CoreCoord> cores = grid_to_coresA(all_cores);

    Program program = CreateProgram();

    // Single scratch CB for the reader: one page for copy (read page -> write to output).
    const uint32_t page_size = output.buffer()->aligned_page_size();
    const uint32_t cb_id_scratch = 0;
    CircularBufferConfig cb_config =
        CircularBufferConfig(page_size, {{cb_id_scratch, datatype_to_dataformat_converter(output.dtype())}})
            .set_page_size(cb_id_scratch, page_size);
    CreateCircularBuffer(program, all_cores, cb_config);

    // Compile-time args: num_input_tensors, then 17x page_size (for make_tensor_accessor_tuple), then
    // TensorAccessorArgs.
    std::vector<uint32_t> reader_compile_time_args;
    reader_compile_time_args.push_back(num_input_tensors);
    for (uint32_t i = 0; i < 1u + CONCAT_ND_SHARDED_MAX_NUM_INPUTS; ++i) {
        reader_compile_time_args.push_back(static_cast<uint32_t>(page_size));
    }

    TensorAccessorArgs(output.buffer()).append_to(reader_compile_time_args);
    for (uint32_t i = 0; i < num_input_tensors; ++i) {
        TensorAccessorArgs(input_tensors[i].buffer()).append_to(reader_compile_time_args);
    }
    for (uint32_t i = num_input_tensors; i < CONCAT_ND_SHARDED_MAX_NUM_INPUTS; ++i) {
        TensorAccessorArgs(input_tensors[0].buffer()).append_to(reader_compile_time_args);
    }

    KernelHandle reader_kernel_id = CreateKernel(
        program,
        "ttnn/cpp/ttnn/operations/data_movement/concat/device/kernels/dataflow/reader_concat_nd_sharded.cpp",
        all_cores,
        ReaderDataMovementConfig(reader_compile_time_args));

    // Per-core runtime args: [output_addr, input0_addr, ..., input15_addr, shard_id] (fixed 18 args)
    for (size_t c = 0; c < cores.size(); ++c) {
        const CoreCoord& core = cores[c];
        std::vector<uint32_t> runtime_args;
        runtime_args.reserve(18u);
        runtime_args.push_back(get_buffer_address_for_core(output.buffer(), core));
        for (uint32_t i = 0; i < num_input_tensors; ++i) {
            runtime_args.push_back(get_buffer_address_for_core(input_tensors[i].buffer(), core));
        }
        for (uint32_t i = num_input_tensors; i < CONCAT_ND_SHARDED_MAX_NUM_INPUTS; ++i) {
            runtime_args.push_back(0u);  // unused; kernel only uses first 1 + num_input_tensors
        }
        runtime_args.push_back(static_cast<uint32_t>(c));  // shard_id
        SetRuntimeArgs(program, reader_kernel_id, core, runtime_args);
    }

    ConcatNDShardedSharedVariables shared_vars;
    shared_vars.num_input_tensors = num_input_tensors;
    shared_vars.reader_kernel_id = reader_kernel_id;
    shared_vars.all_cores = all_cores;
    shared_vars.cores = cores;
    shared_vars.cb_inputs.clear();
    shared_vars.cb_output = 0;
    shared_vars.writer_kernel_id = 0;

    return cached_program_t{std::move(program), std::move(shared_vars)};
}

// Updates buffer addresses in the cached program when the same program is reused with
// different tensor pointers (e.g. program cache hit with new tensor allocations).
void ConcatNDShardedProgramFactory::override_runtime_arguments(
    cached_program_t& cached_program,
    const ConcatParams& /*operation_attributes*/,
    const ConcatInputs& tensor_args,
    Tensor& tensor_return_value) {
    auto& program = cached_program.program;
    const auto& shared_vars = cached_program.shared_variables;
    const uint32_t num_input_tensors = shared_vars.num_input_tensors;
    const std::vector<CoreCoord>& cores = shared_vars.cores;

    for (size_t c = 0; c < cores.size(); ++c) {
        const CoreCoord& core = cores[c];
        std::vector<uint32_t> runtime_args;
        runtime_args.reserve(18u);
        runtime_args.push_back(get_buffer_address_for_core(tensor_return_value.buffer(), core));
        for (uint32_t i = 0; i < num_input_tensors; ++i) {
            runtime_args.push_back(get_buffer_address_for_core(tensor_args.input_tensors[i].buffer(), core));
        }
        for (uint32_t i = num_input_tensors; i < CONCAT_ND_SHARDED_MAX_NUM_INPUTS; ++i) {
            runtime_args.push_back(0u);
        }
        runtime_args.push_back(static_cast<uint32_t>(c));  // shard_id
        SetRuntimeArgs(program, shared_vars.reader_kernel_id, core, runtime_args);
    }
}

}  // namespace ttnn::prim
