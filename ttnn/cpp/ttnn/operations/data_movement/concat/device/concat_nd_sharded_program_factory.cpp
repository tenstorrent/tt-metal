// SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#include "concat_nd_sharded_program_factory.hpp"

#include <algorithm>
#include <numeric>

#include "ttnn/tensor/tensor.hpp"
#include <tt-metalium/host_api.hpp>
#include <tt-metalium/constants.hpp>
#include <tt-metalium/buffer_distribution_spec.hpp>

namespace ttnn::prim {

namespace {

// Returns the index of `core` in `cores` vector, or cores.size() if not found.
size_t find_core_index(const CoreCoord& core, const std::vector<CoreCoord>& cores) {
    auto it = std::find(cores.begin(), cores.end(), core);
    return static_cast<size_t>(it - cores.begin());
}

}  // namespace

// Creates the device program for ND sharded concat: one program with reader and writer kernels
// that copy each core's input shards (in concat order) into the output shard. Reader and writer
// split the input range so both RISC-V processors can run in parallel.
ConcatNDShardedProgramFactory::cached_program_t ConcatNDShardedProgramFactory::create(
    const ConcatParams& /*operation_attributes*/, const ConcatInputs& tensor_args, Tensor& tensor_return_value) {
    using namespace tt::constants;
    using namespace tt::tt_metal;

    const std::vector<Tensor>& input_tensors = tensor_args.input_tensors;
    Tensor& output = tensor_return_value;
    const uint32_t num_input_tensors = static_cast<uint32_t>(input_tensors.size());

    TT_FATAL(!input_tensors.empty(), "ND sharded concat requires at least one input");
    TT_FATAL(input_tensors[0].nd_shard_spec().has_value(), "First input must be ND sharded");
    TT_FATAL(output.nd_shard_spec().has_value(), "Output must be ND sharded");

    const NdShardSpec& first_nd_spec = input_tensors[0].nd_shard_spec().value();
    const CoreRangeSet all_cores = first_nd_spec.grid;
    const ShardOrientation orientation = first_nd_spec.orientation;
    const bool is_row_major = (orientation == ShardOrientation::ROW_MAJOR);

    // Core list order must match the order used by each buffer's distribution spec for correct core_idx.
    std::vector<CoreCoord> cores = corerange_to_cores(all_cores, std::nullopt, is_row_major);

    Program program = CreateProgram();

    // All inputs and output must share the same aligned page size; the kernels use a single page_size.
    // (Tensor buffers can have different page sizes when shard shapes or layouts differ.)
    const uint32_t page_size = input_tensors[0].buffer()->aligned_page_size();
    for (uint32_t i = 1; i < num_input_tensors; ++i) {
        TT_FATAL(
            input_tensors[i].buffer()->aligned_page_size() == page_size,
            "ND sharded concat requires all inputs to have the same aligned page size (input 0: {}, input {}: {})",
            page_size,
            i,
            input_tensors[i].buffer()->aligned_page_size());
    }
    TT_FATAL(
        output.buffer()->aligned_page_size() == page_size,
        "ND sharded concat requires output to have the same aligned page size as inputs (inputs: {}, output: {})",
        page_size,
        output.buffer()->aligned_page_size());

    const tt::DataFormat cb_data_format = datatype_to_dataformat_converter(output.dtype());
    const uint32_t cb_dst_id =
        16;  // TODO:Z decide if it must be 31(wormhole) or 61(blackhole) or constant from somewhere
    TT_FATAL(num_input_tensors <= cb_dst_id, "ND sharded concat supports at most {} inputs", cb_dst_id);

    // Distribution specs from buffers (already set when tensors are allocated with ND sharding).
    std::vector<const BufferDistributionSpec*> input_dist_specs(num_input_tensors);
    for (uint32_t i = 0; i < num_input_tensors; ++i) {
        TT_FATAL(
            input_tensors[i].buffer()->buffer_distribution_spec().has_value(),
            "ND sharded input {} must have buffer distribution spec",
            i);
        input_dist_specs[i] = &input_tensors[i].buffer()->buffer_distribution_spec().value();
    }
    TT_FATAL(
        output.buffer()->buffer_distribution_spec().has_value(),
        "ND sharded output must have buffer distribution spec");
    const BufferDistributionSpec& output_dist_spec = output.buffer()->buffer_distribution_spec().value();

    // Input CBs: each backed by the corresponding input buffer so each core sees only its shard.
    std::vector<CBHandle> cb_inputs(num_input_tensors);
    for (uint32_t input_id = 0; input_id < num_input_tensors; ++input_id) {
        const BufferDistributionSpec& spec = *input_dist_specs[input_id];
        uint32_t max_pages = static_cast<uint32_t>(spec.max_num_dev_pages_per_core());
        const CircularBufferConfig input_cb_config =
            CircularBufferConfig(page_size * max_pages, {{input_id, cb_data_format}})
                .set_page_size(input_id, page_size)
                .set_globally_allocated_address(*input_tensors[input_id].buffer());
        cb_inputs[input_id] = CreateCircularBuffer(program, all_cores, input_cb_config);
    }

    // Output CB: backed by output buffer; each core writes its output shard.
    uint32_t output_max_pages = static_cast<uint32_t>(output_dist_spec.max_num_dev_pages_per_core());
    const CircularBufferConfig output_cb_config =
        CircularBufferConfig(page_size * output_max_pages, {{cb_dst_id, cb_data_format}})
            .set_page_size(cb_dst_id, page_size)
            .set_globally_allocated_address(*output.buffer());
    CBHandle cb_output = CreateCircularBuffer(program, all_cores, output_cb_config);

    // Compile-time args for reader/writer: output CB id, page size, number of inputs.
    std::vector<uint32_t> compile_time_args = {cb_dst_id, page_size, num_input_tensors};

    // Reader: reads from input CBs (in concat order) and writes into output CB.
    // Writer: same kernel, different runtime args to split work (first half of inputs vs second half).
    KernelHandle reader_kernel_id = CreateKernel(
        program,
        "ttnn/cpp/ttnn/operations/data_movement/concat/device/kernels/dataflow/reader_concat_nd_sharded.cpp",
        all_cores,
        ReaderDataMovementConfig(compile_time_args));

    KernelHandle writer_kernel_id = CreateKernel(
        program,
        "ttnn/cpp/ttnn/operations/data_movement/concat/device/kernels/dataflow/writer_concat_nd_sharded.cpp",
        all_cores,
        WriterDataMovementConfig(compile_time_args));

    // Per-core runtime args: [start_input_id, end_input_id, (num_pages, write_offset_in_pages) for each input in
    // [start,end)). Reader handles inputs [0, mid), writer handles [mid, num_input_tensors).
    const uint32_t mid = (num_input_tensors + 1) / 2;

    std::cout << "cores amount " << cores.size() << " output cores amount " << output_dist_spec.cores().size() << "\n";

    uint32_t cn = 0;
    for (const CoreCoord& core : cores) {
        const size_t core_idx = find_core_index(core, output_dist_spec.cores());
        if (core_idx >= output_dist_spec.num_cores()) {
            continue;
        }
        std::cout << "core x :" << ++cn << "\n";

        uint32_t output_write_offset_pages = 0;
        std::vector<uint32_t> input_num_pages(num_input_tensors);
        std::vector<uint32_t> input_write_offsets_pages(num_input_tensors);

        for (uint32_t input_id = 0; input_id < num_input_tensors; ++input_id) {
            const BufferDistributionSpec& in_spec = *input_dist_specs[input_id];
            const size_t in_core_idx = find_core_index(core, in_spec.cores());
            const uint32_t num_pages = (in_core_idx < in_spec.num_cores())
                                           ? static_cast<uint32_t>(in_spec.num_dev_pages_per_core(in_core_idx))
                                           : 0;
            input_num_pages[input_id] = num_pages;
            input_write_offsets_pages[input_id] = output_write_offset_pages;
            output_write_offset_pages += num_pages;
        }

        std::vector<uint32_t> reader_runtime_args = {0u, mid};
        for (uint32_t input_id = 0; input_id < mid; ++input_id) {
            reader_runtime_args.push_back(input_num_pages[input_id]);
            reader_runtime_args.push_back(input_write_offsets_pages[input_id]);
        }
        std::vector<uint32_t> writer_runtime_args = {mid, num_input_tensors};
        for (uint32_t input_id = mid; input_id < num_input_tensors; ++input_id) {
            writer_runtime_args.push_back(input_num_pages[input_id]);
            writer_runtime_args.push_back(input_write_offsets_pages[input_id]);
        }

        SetRuntimeArgs(program, reader_kernel_id, core, reader_runtime_args);
        SetRuntimeArgs(program, writer_kernel_id, core, writer_runtime_args);
    }  // for cores

    return {
        std::move(program),
        {.num_input_tensors = num_input_tensors,
         .cb_inputs = cb_inputs,
         .cb_output = cb_output,
         .reader_kernel_id = reader_kernel_id,
         .writer_kernel_id = writer_kernel_id,
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

    for (uint32_t input_id = 0; input_id < shared_vars.num_input_tensors; ++input_id) {
        UpdateDynamicCircularBufferAddress(
            program, shared_vars.cb_inputs[input_id], *tensor_args.input_tensors[input_id].buffer());
    }
    UpdateDynamicCircularBufferAddress(program, shared_vars.cb_output, *tensor_return_value.buffer());
}

}  // namespace ttnn::prim
