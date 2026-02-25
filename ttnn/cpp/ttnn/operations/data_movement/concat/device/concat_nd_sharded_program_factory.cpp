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
#include <tt-metalium/buffer.hpp>
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
    const ConcatParams& operation_attributes, const ConcatInputs& tensor_args, Tensor& tensor_return_value) {
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

    const Tensor& input_t0 = input_tensors[0];
    const auto& nd_shard_spec = input_t0.nd_shard_spec().value();
    const CoreRangeSet all_cores = nd_shard_spec.grid;
    const std::vector<CoreCoord> cores =
        corerange_to_cores(all_cores, std::nullopt, nd_shard_spec.orientation == ShardOrientation::ROW_MAJOR);

    // Pick the core with biggest coordinates (lexicographic: x then y) and run only on that core
    const auto max_core_it = std::max_element(cores.begin(), cores.end(), [](const CoreCoord& a, const CoreCoord& b) {
        return a.x < b.x || (a.x == b.x && a.y < b.y);
    });
    TT_FATAL(max_core_it != cores.end(), "cores must be non-empty");
    const CoreCoord max_core = *max_core_it;
    const size_t max_core_index = static_cast<size_t>(std::distance(cores.begin(), max_core_it));
    const CoreRangeSet single_core_set = CoreRangeSet(CoreRange(max_core));
    std::cout << "max_core_it " << max_core.x << " " << max_core.y << "\n";

    Program program = CreateProgram();

    // Page sizes: output and each of 16 input slots (fill absent from first input)
    const uint32_t output_page_size = output.buffer()->aligned_page_size();
    std::array<uint32_t, CONCAT_ND_SHARDED_MAX_NUM_INPUTS> input_page_sizes;
    for (uint32_t i = 0; i < CONCAT_ND_SHARDED_MAX_NUM_INPUTS; ++i) {
        const Buffer* buf = (i < num_input_tensors) ? input_tensors[i].buffer() : input_t0.buffer();
        input_page_sizes[i] = buf->aligned_page_size();
        // TODO - move to the verification part
        TT_ASSERT(
            input_page_sizes[i] == input_page_sizes[0],
            "input page sizes should match {}:{} != 0:{}",
            i,
            input_page_sizes[i],
            input_page_sizes[0]);
    }

    // some debug information about output tensor
    {
        const Buffer* out_buf = output.buffer();
        const tt::tt_metal::Shape& output_logical_shape = output.logical_shape();
        std::cout << "[concat_nd_sharded] Output tensor: total pages = " << out_buf->num_pages();
        std::cout << ", logical_shape = " << output_logical_shape << ", padded_shape = " << output.padded_shape();
        std::cout << ", rank = " << output_logical_shape.rank() << "\n";

        const size_t sz_to_fill = output.logical_volume() * output.element_size();
        std::cout << " size to fill in output " << sz_to_fill;

        const NdShardSpec& nd_spec = *output.nd_shard_spec();
        std::cout << ", nd_shard_spec: shard_shape = " << nd_spec.shard_shape
                  << ", grid num_cores = " << nd_spec.grid.num_cores();
        std::cout << "\n";
    }
    std::array<uint32_t, CONCAT_ND_SHARDED_MAX_NUM_INPUTS> initial_offsets_bytes{0};
    std::array<uint32_t, CONCAT_ND_SHARDED_MAX_NUM_INPUTS> to_write_bytes{0};
    //    std::array<uint32_t, CONCAT_ND_SHARDED_MAX_NUM_INPUTS> to_skip_bytes{0};
    {
        for (uint32_t i = 0; i < num_input_tensors; ++i) {
            const uint32_t sz_to_fill =
                static_cast<uint32_t>(input_tensors[i].logical_volume() * input_tensors[i].element_size());
            to_write_bytes[i] = sz_to_fill;
            std::cout << " tensor " << i << " size to fill in output " << sz_to_fill;

            if (i + 1 < num_input_tensors) {
                initial_offsets_bytes[i + 1] = initial_offsets_bytes[i] + sz_to_fill;
            }
        }
        std::cout << "\n";
    }

    // Host-allocated L1 scratch buffer (one page, max page size across all tensors) for copy_tensor_data.
    const uint32_t scratch_page_size =
        std::max(output_page_size, *std::max_element(input_page_sizes.begin(), input_page_sizes.end()));
    ShardSpecBuffer scratch_shard_spec(single_core_set, {1, 1}, ShardOrientation::ROW_MAJOR, {1, 1}, {1, 1});
    ShardedBufferConfig scratch_l1_config{
        .device = output.device(),
        .size = scratch_page_size,
        .page_size = scratch_page_size,
        .buffer_type = BufferType::L1,
        .buffer_layout = TensorMemoryLayout::HEIGHT_SHARDED,
        .shard_parameters = std::move(scratch_shard_spec),
    };
    std::shared_ptr<Buffer> scratch_l1_buffer = CreateBuffer(scratch_l1_config);
    AssignGlobalBufferToProgram(scratch_l1_buffer, program);
    const uint32_t scratch_l1_addr = static_cast<uint32_t>(scratch_l1_buffer->page_address(0, 0));

    // Compile-time args: num_input_tensors, concat dim, output_page_size, input_page_sizes[0..15],
    // then output TensorAccessorArgs, then 16 input TensorAccessorArgs (absent filled from first input).
    std::vector<uint32_t> reader_compile_time_args = {
        num_input_tensors,
        static_cast<uint32_t>(operation_attributes.dim),
        output_page_size,
    };
    for (uint32_t i = 0; i < CONCAT_ND_SHARDED_MAX_NUM_INPUTS; ++i) {
        reader_compile_time_args.push_back(input_page_sizes[i]);
    }

    // TensorAccessor parameters come from tensor buffers only; no Circular Buffer required.
    TensorAccessorArgs(*output.buffer()).append_to(reader_compile_time_args);
    for (uint32_t i = 0; i < CONCAT_ND_SHARDED_MAX_NUM_INPUTS; ++i) {
        const Buffer* buf = (i < num_input_tensors) ? input_tensors[i].buffer() : input_t0.buffer();
        TensorAccessorArgs(*buf).append_to(reader_compile_time_args);
    }

    KernelHandle reader_kernel_id = CreateKernel(
        program,
        "ttnn/cpp/ttnn/operations/data_movement/concat/device/kernels/dataflow/reader_concat_nd_sharded.cpp",
        single_core_set,
        ReaderDataMovementConfig(reader_compile_time_args));

    // Runtime args for the single core: scratch_l1_addr, output addr, then per-input (addr, absolute_offset_to_start,
    // amount_to_write, amount_to_skip), then shard_id
    {
        uint32_t dim_pages{0};  // how many pages I need to write from every Tensor, sequentially
        const uint32_t num_dims = output.padded_shape().rank();
        const uint32_t dim = operation_attributes.dim;
        const bool rm_layout = (output.layout() == Layout::ROW_MAJOR);
        uint32_t scale_factor = 1;
        if (!rm_layout) {
            if (dim == num_dims - 2) {
                scale_factor = TILE_HEIGHT;
            } else if (dim == num_dims - 1) {
                scale_factor = TILE_WIDTH;
            }
        }
        uint32_t num_accum_pages = 1;
        for (uint32_t i = dim + 1; i < num_dims; ++i) {
            num_accum_pages *= output.padded_shape()[i];
            std::cout << "num_acum_pages at " << i << ": " << num_accum_pages;
        }
        std::cout << "\n";

        if (rm_layout) {
            if (num_dims > 1 && dim < num_dims - 1) {
                num_accum_pages /= output.padded_shape()[-1];
            }
        } else {
            if (dim < num_dims - 2) {
                num_accum_pages /= TILE_HW;
            } else if (dim == num_dims - 2) {
                num_accum_pages /= TILE_WIDTH;
            }
        }
        std::cout << "result acum_pages: " << num_accum_pages << "\n";
        uint32_t num_output_pages_per_block_total = 0;

        if (rm_layout) {
            dim_pages = input_t0.padded_shape()[dim];
        } else {
            dim_pages = input_t0.padded_shape()[dim] / scale_factor;
        }
        if (rm_layout && dim == num_dims - 1) {
            num_output_pages_per_block_total += num_accum_pages;
        } else {
            num_output_pages_per_block_total += num_accum_pages * dim_pages;
        }
        if (rm_layout && dim == num_dims - 1) {
            num_output_pages_per_block_total = 1;
        }

        std::cout << "dim pages: " << dim_pages << ". pages per block: " << num_output_pages_per_block_total << "\n";

        std::vector<uint32_t> runtime_args;
        runtime_args.push_back(scratch_l1_addr);
        runtime_args.push_back(output.buffer()->address());
        const uint32_t per_tensor_to_write = num_output_pages_per_block_total * input_page_sizes[0];
        const uint32_t per_tensor_to_skip = per_tensor_to_write * (num_input_tensors - 1);
        std::cout << "per tensor write " << per_tensor_to_write << "  per tensor to skip " << per_tensor_to_skip
                  << "\n";
        for (uint32_t i = 0; i < CONCAT_ND_SHARDED_MAX_NUM_INPUTS; ++i) {
            const Buffer* buf = (i < num_input_tensors) ? input_tensors[i].buffer() : input_t0.buffer();
            runtime_args.push_back(buf->address());

            runtime_args.push_back(i * per_tensor_to_write);
            runtime_args.push_back(per_tensor_to_write);
            runtime_args.push_back(per_tensor_to_skip);
            std::cout << i << ". ao " << i * per_tensor_to_write << "\n";

            // runtime_args.push_back(initial_offsets_bytes[i]);
            // runtime_args.push_back(to_write_bytes[i]);
            // runtime_args.push_back(to_skip_bytes[i]);
        }
        runtime_args.push_back(static_cast<uint32_t>(max_core_index));
        SetRuntimeArgs(program, reader_kernel_id, max_core, runtime_args);
    }

    return cached_program_t{
        std::move(program),
        {.num_input_tensors = num_input_tensors,
         .cb_inputs = {},
         .cb_output = 0,
         .reader_kernel_id = reader_kernel_id,
         .writer_kernel_id = 0,
         .all_cores = single_core_set,
         .cores = {max_core},
         .scratch_l1_buffer = std::move(scratch_l1_buffer)}};
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

    // Recompute full core list to get shard index for our single core
    const auto& nd_shard_spec = input_tensors[0].nd_shard_spec().value();
    const std::vector<CoreCoord> all_cores_list =
        corerange_to_cores(nd_shard_spec.grid, std::nullopt, nd_shard_spec.orientation == ShardOrientation::ROW_MAJOR);
    const CoreCoord& single_core = cores[0];
    const auto shard_id_it = std::find(all_cores_list.begin(), all_cores_list.end(), single_core);
    TT_FATAL(shard_id_it != all_cores_list.end(), "Single core must be in grid");
    const uint32_t shard_id = static_cast<uint32_t>(std::distance(all_cores_list.begin(), shard_id_it));

    TT_FATAL(shared_vars.scratch_l1_buffer, "Scratch L1 buffer must be set");
    const uint32_t scratch_l1_addr = static_cast<uint32_t>(shared_vars.scratch_l1_buffer->page_address(0, 0));

    std::vector<uint32_t> runtime_args;
    runtime_args.push_back(scratch_l1_addr);
    runtime_args.push_back(output.buffer()->address());
    for (uint32_t i = 0; i < CONCAT_ND_SHARDED_MAX_NUM_INPUTS; ++i) {
        const Buffer* buf = (i < num_input_tensors) ? input_tensors[i].buffer() : input_tensors[0].buffer();
        runtime_args.push_back(buf->address());
        uint32_t absolute_offset_to_start{0};  //  - to fill later
        uint32_t amount_to_write{0};           //  - to fill later
        uint32_t amount_to_skip{0};            //  - to fill later
        runtime_args.push_back(absolute_offset_to_start);
        runtime_args.push_back(amount_to_write);
        runtime_args.push_back(amount_to_skip);
    }
    runtime_args.push_back(shard_id);
    SetRuntimeArgs(program, shared_vars.reader_kernel_id, single_core, runtime_args);
}

}  // namespace ttnn::prim
