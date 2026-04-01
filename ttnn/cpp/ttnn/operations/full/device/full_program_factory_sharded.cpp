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
#include "ttnn/tensor/tensor_utils.hpp"

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

    Program program{};

    auto data_format = datatype_to_dataformat_converter(dtype);

    uint32_t tensor_width_in_pages = output.buffer()->shard_spec().tensor2d_shape_in_pages[1];

    std::vector<CoreCoord> runtime_cores = get_optimal_worker_cores_for_sharded_tensor(output);
    const auto& compute_core_range = CoreRangeSet(ttsl::Span<const CoreCoord>(runtime_cores));

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
