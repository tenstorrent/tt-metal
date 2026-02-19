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
    const std::vector<CoreCoord> cores =
        corerange_to_cores(all_cores, std::nullopt, nd_shard_spec.orientation == ShardOrientation::ROW_MAJOR);

    Program program = CreateProgram();

    // Page sizes: output and each of 16 input slots (fill absent from first input)
    const uint32_t output_page_size = output.buffer()->aligned_page_size();
    std::array<uint32_t, CONCAT_ND_SHARDED_MAX_NUM_INPUTS> input_page_sizes;
    for (uint32_t i = 0; i < CONCAT_ND_SHARDED_MAX_NUM_INPUTS; ++i) {
        const Buffer* buf = (i < num_input_tensors) ? input_tensors[i].buffer() : input_tensors[0].buffer();
        input_page_sizes[i] = buf->aligned_page_size();
    }

    // Scratch CB (index 0) for copy_tensor_data: one page, size = max of all page sizes.
    const uint32_t scratch_page_size =
        std::max(output_page_size, *std::max_element(input_page_sizes.begin(), input_page_sizes.end()));
    constexpr uint8_t cb_scratch_id = 0;
    auto scratch_cb_config = CircularBufferConfig(scratch_page_size, {{cb_scratch_id, tt::DataFormat::RawUInt32}})
                                 .set_page_size(cb_scratch_id, scratch_page_size);
    CreateCircularBuffer(program, all_cores, scratch_cb_config);

    // Compile-time args: num_input_tensors, output_page_size, input_page_sizes[0..15],
    // then output TensorAccessorArgs, then 16 input TensorAccessorArgs (absent filled from first input).
    std::vector<uint32_t> reader_compile_time_args = {
        num_input_tensors,
        output_page_size,
    };
    for (uint32_t i = 0; i < CONCAT_ND_SHARDED_MAX_NUM_INPUTS; ++i) {
        reader_compile_time_args.push_back(input_page_sizes[i]);
    }

    // TensorAccessor parameters come from tensor buffers only; no Circular Buffer required.
    TensorAccessorArgs(*output.buffer()).append_to(reader_compile_time_args);
    for (uint32_t i = 0; i < CONCAT_ND_SHARDED_MAX_NUM_INPUTS; ++i) {
        const Buffer* buf = (i < num_input_tensors) ? input_tensors[i].buffer() : input_tensors[0].buffer();
        TensorAccessorArgs(*buf).append_to(reader_compile_time_args);
    }

    KernelHandle reader_kernel_id = CreateKernel(
        program,
        "ttnn/cpp/ttnn/operations/data_movement/concat/device/kernels/dataflow/reader_concat_nd_sharded.cpp",
        all_cores,
        ReaderDataMovementConfig(reader_compile_time_args));

    // Runtime args per core: 17 buffer addresses (output, in0..in15), then shard_id
    for (size_t c = 0; c < cores.size(); ++c) {
        std::vector<uint32_t> runtime_args;
        runtime_args.push_back(output.buffer()->address());
        for (uint32_t i = 0; i < CONCAT_ND_SHARDED_MAX_NUM_INPUTS; ++i) {
            const Buffer* buf = (i < num_input_tensors) ? input_tensors[i].buffer() : input_tensors[0].buffer();
            runtime_args.push_back(buf->address());
        }
        runtime_args.push_back(static_cast<uint32_t>(c));
        SetRuntimeArgs(program, reader_kernel_id, cores[c], runtime_args);
    }

    return cached_program_t{
        std::move(program),
        {.num_input_tensors = num_input_tensors,
         .cb_inputs = {},
         .cb_output = 0,
         .reader_kernel_id = reader_kernel_id,
         .writer_kernel_id = 0,
         .all_cores = all_cores,
         .cores = cores}};
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
    const std::vector<Tensor>& input_tensors = tensor_args.input_tensors;
    Tensor& output = tensor_return_value;

    for (size_t c = 0; c < cores.size(); ++c) {
        std::vector<uint32_t> runtime_args;
        runtime_args.push_back(output.buffer()->address());
        for (uint32_t i = 0; i < CONCAT_ND_SHARDED_MAX_NUM_INPUTS; ++i) {
            const Buffer* buf = (i < num_input_tensors) ? input_tensors[i].buffer() : input_tensors[0].buffer();
            runtime_args.push_back(buf->address());
        }
        runtime_args.push_back(static_cast<uint32_t>(c));
        SetRuntimeArgs(program, shared_vars.reader_kernel_id, cores[c], runtime_args);
    }
}

}  // namespace ttnn::prim
