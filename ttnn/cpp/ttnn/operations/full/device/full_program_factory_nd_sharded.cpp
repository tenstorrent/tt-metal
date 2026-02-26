// SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC.
//
// SPDX-License-Identifier: Apache-2.0

#include <cstdint>

#include <tt-metalium/constants.hpp>
#include <tt-metalium/host_api.hpp>
#include <tt-metalium/circular_buffer_config.hpp>
#include <tt-metalium/work_split.hpp>
#include <tt-metalium/tensor_accessor_args.hpp>
#include "full_program_factory_nd_sharded.hpp"
#include "full_program_factory_common.hpp"

namespace ttnn::operations::full {

using namespace tt;
using namespace tt::constants;
using namespace tt::tt_metal::detail;

FullNDShardedProgramFactory::cached_program_t FullNDShardedProgramFactory::create(
    const operation_attributes_t& operation_attributes,
    const tensor_args_t& /*tensor_args*/,
    tensor_return_value_t& output) {
    auto fill_value = operation_attributes.fill_value;
    DataType dtype{operation_attributes.dtype};
    MemoryConfig memory_config{operation_attributes.memory_config};

    Program program{};

    auto data_format = datatype_to_dataformat_converter(dtype);
    const auto& distribution_spec = output.buffer()->buffer_distribution_spec().value();
    uint32_t num_shards = distribution_spec.num_shards();
    std::vector<CoreCoord> ordered_cores_with_data;
    uint32_t num_compute_cores = distribution_spec.cores_with_data().size();

    if (memory_config.is_dram()) {  // For DRAM sharded tensors, we take one core that is optimal for each DRAM bank
                                    // with a shard to use as our compute cores.
        auto all_dram_workers =
            output.device()->get_optimal_dram_bank_to_logical_worker_assignment(tt::tt_metal::NOC::RISCV_0_default);
        ordered_cores_with_data.assign(all_dram_workers.begin(), all_dram_workers.begin() + num_compute_cores);
    } else {
        ordered_cores_with_data = distribution_spec.cores_with_data();
    }
    const auto& compute_core_range = CoreRangeSet(tt::stl::Span<const CoreCoord>(ordered_cores_with_data));
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
        (uint32_t)cb_fill_value_id, elems_per_page, page_size, aligned_page_size, num_shards, num_compute_cores};
    tt::tt_metal::TensorAccessorArgs(output.buffer()).append_to(writer_compile_time_args);

    auto writer_id = CreateKernel(
        program,
        "ttnn/cpp/ttnn/operations/full/device/kernels/writer_full_nd_sharded.cpp",
        compute_core_range,
        tt::tt_metal::WriterDataMovementConfig(writer_compile_time_args, writer_defines));

    uint32_t start_shard_id = 0;
    for (auto core : ordered_cores_with_data) {
        SetRuntimeArgs(program, writer_id, core, {output.buffer()->address(), u.u32, start_shard_id});
        start_shard_id++;
    }

    return {std::move(program), {writer_id, ordered_cores_with_data}};
}

void FullNDShardedProgramFactory::override_runtime_arguments(
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
