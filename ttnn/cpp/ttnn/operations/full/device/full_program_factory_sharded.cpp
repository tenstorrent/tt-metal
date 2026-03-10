// SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC.
//
// SPDX-License-Identifier: Apache-2.0

#include <cstdint>

#include <tt-metalium/constants.hpp>
#include <tt-metalium/host_api.hpp>
#include <tt-metalium/circular_buffer_config.hpp>
#include <tt-metalium/work_split.hpp>
#include <tt-metalium/tensor_accessor_args.hpp>
#include <tt-metalium/buffer_distribution_spec.hpp>
#include "full_program_factory_sharded.hpp"
#include "full_program_factory_common.hpp"

namespace ttnn::operations::full {

using namespace tt;
using namespace tt::constants;
using namespace tt::tt_metal::detail;

FullShardedProgramFactory::cached_program_t FullShardedProgramFactory::create(
    const operation_attributes_t& operation_attributes,
    const tensor_args_t& /*tensor_args*/,
    tensor_return_value_t& output) {
    auto fill_value = operation_attributes.fill_value;
    DataType dtype{operation_attributes.dtype};
    MemoryConfig memory_config{operation_attributes.memory_config};

    Program program{};

    auto data_format = datatype_to_dataformat_converter(dtype);

    uint32_t tensor_width = output.padded_shape()[-1];
    uint32_t tensor_height = output.physical_volume() / tensor_width;
    const auto& output_shard_spec = output.shard_spec().value();
    uint32_t shard_height = output_shard_spec.shape[0];
    uint32_t shard_width = output_shard_spec.shape[1];
    uint32_t num_compute_cores = output_shard_spec.grid.num_cores();
    uint32_t tensor_width_in_pages = output.buffer()->shard_spec().tensor2d_shape_in_pages[1];

    uint32_t num_input_blocks_across_width = tt::div_up(tensor_width, shard_width);
    uint32_t num_shards_height = tt::div_up(tensor_height, shard_height);
    uint32_t num_shards = num_shards_height * num_input_blocks_across_width;

    CoreRangeSet compute_core_range;
    std::vector<CoreCoord> runtime_cores;
    if (memory_config.is_dram()) {  // For DRAM sharded tensors, we take one core that is optimal for each DRAM bank
                                    // with a shard to use as our compute cores.
        num_compute_cores =
            std::min(num_compute_cores, num_shards);  // If the number of banks to shard over is more than the number
                                                      // of shards, only num_shards DRAM banks will have data.
        auto all_dram_workers =
            output.device()->get_optimal_dram_bank_to_logical_worker_assignment(tt::tt_metal::NOC::RISCV_0_default);
        runtime_cores.assign(
            all_dram_workers.begin(),
            all_dram_workers.begin() +
                num_compute_cores);  // TODO: We can optimize worker cores chosen for each DRAM bank. We may not
                                     // necessarily specify the DRAM bank starting at (0,0), so this may not always
                                     // return the most optimal worker core assignments for the DRAM banks with shards.
        compute_core_range = CoreRangeSet(tt::stl::Span<const CoreCoord>(runtime_cores));
    } else {
        if (num_compute_cores >
            num_shards) {  // For L1 sharding, the user may specify a core grid larger than the number of shards. In
                           // this case, we need to determine which cores have data on them so that we are not running
                           // programs on cores with no data being processed.
            runtime_cores = output.get_cores_with_shards();
            compute_core_range = CoreRangeSet(tt::stl::Span<const CoreCoord>(runtime_cores));
        } else {
            compute_core_range =
                output_shard_spec.grid;  // If the user specified the same number of compute cores as the number of
                                         // shards, then we can directly use the core grid specified in the shard_spec.
            bool is_row_major = (output_shard_spec.orientation == ShardOrientation::ROW_MAJOR);
            runtime_cores = corerange_to_cores(compute_core_range, std::nullopt, is_row_major);
        }
    }

    const auto& aligned_page_size = output.buffer()->aligned_page_size();
    const auto& page_size = output.buffer()->page_size();

    constexpr CBIndex cb_fill_value_id = CBIndex::c_24;

    auto cb_value_config = tt::tt_metal::CircularBufferConfig(page_size, {{cb_fill_value_id, data_format}})
                               .set_page_size(cb_fill_value_id, page_size);
    CreateCircularBuffer(program, compute_core_range, cb_value_config);
    auto writer_defines = get_writer_defines(dtype);
    auto u = encode_fill_value(fill_value, dtype);

    uint32_t elems_per_page = page_size / datum_size(data_format);
    std::vector<uint32_t> writer_compile_time_args = {
        (uint32_t)cb_fill_value_id, elems_per_page, page_size, aligned_page_size, tensor_width_in_pages};
    tt::tt_metal::TensorAccessorArgs(output.buffer()).append_to(writer_compile_time_args);

    auto writer_id = CreateKernel(
        program,
        "ttnn/cpp/ttnn/operations/full/device/kernels/writer_full_sharded.cpp",
        compute_core_range,
        tt::tt_metal::WriterDataMovementConfig(writer_compile_time_args, writer_defines));

    uint32_t shard_height_in_pages = output.buffer()->shard_spec().shape_in_pages()[0];
    uint32_t shard_width_in_pages = output.buffer()->shard_spec().shape_in_pages()[1];
    uint32_t tensor_height_in_pages = output.buffer()->shard_spec().tensor2d_shape_in_pages[0];
    uint32_t num_shards_across_width = tt::div_up(tensor_width_in_pages, shard_width_in_pages);
    uint32_t num_shards_across_height = tt::div_up(tensor_height_in_pages, shard_height_in_pages);

    for (uint32_t i = 0; i < runtime_cores.size(); i++) {
        const auto& core = runtime_cores[i];

        uint32_t shard_row_idx = i / num_shards_across_width;
        uint32_t shard_col_idx = i % num_shards_across_width;

        uint32_t first_page_id =
            (shard_row_idx * shard_height_in_pages * tensor_width_in_pages) + (shard_col_idx * shard_width_in_pages);

        uint32_t valid_pages_width = (shard_col_idx == num_shards_across_width - 1)
                                         ? (tensor_width_in_pages - (shard_col_idx * shard_width_in_pages))
                                         : shard_width_in_pages;

        uint32_t valid_pages_height = (shard_row_idx == num_shards_across_height - 1)
                                          ? (tensor_height_in_pages - (shard_row_idx * shard_height_in_pages))
                                          : shard_height_in_pages;
        SetRuntimeArgs(
            program,
            writer_id,
            core,
            {output.buffer()->address(), u.u32, first_page_id, valid_pages_width, valid_pages_height});
    }

    return {std::move(program), {writer_id, runtime_cores}};
}

void FullShardedProgramFactory::override_runtime_arguments(
    cached_program_t& cached_program,
    const operation_attributes_t& /*operation_attributes*/,
    const tensor_args_t& /*tensor_args*/,
    tensor_return_value_t& output) {
    auto& program = cached_program.program;
    auto& writer_kernel_id = cached_program.shared_variables.writer_kernel_id;
    auto& cores_with_runtime_args = cached_program.shared_variables.cores_with_runtime_args;

    auto output_buffer_address = output.buffer()->address();
    for (const auto& core : cores_with_runtime_args) {
        auto& runtime_args = GetRuntimeArgs(program, writer_kernel_id, core);
        runtime_args[0] = output_buffer_address;
    }
}

}  // namespace ttnn::operations::full
