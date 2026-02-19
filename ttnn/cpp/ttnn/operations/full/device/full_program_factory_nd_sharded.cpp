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
    int32_t num_shards = distribution_spec.num_shards();
    const auto page_mapping = distribution_spec.compute_page_mapping();
    const auto& ordered_cores_with_data = distribution_spec.cores_with_data();
    uint32_t num_compute_cores = ordered_cores_with_data.size();

    std::vector<CoreCoord> worker_cores;
    if (memory_config.is_dram()) {
        const auto* device = output.device();
        const auto grid_size = device->compute_with_storage_grid_size();
        for (uint32_t i = 0; i < num_compute_cores; i++) {
            worker_cores.push_back(CoreCoord(i % grid_size.x, i / grid_size.x));
        }
    }
    const auto& compute_core_range = memory_config.is_dram()
                                         ? CoreRangeSet(tt::stl::Span<const CoreCoord>(worker_cores))
                                         : CoreRangeSet(tt::stl::Span<const CoreCoord>(ordered_cores_with_data));
    const auto& aligned_page_size = output.buffer()->aligned_page_size();
    const auto& page_size = output.buffer()->page_size();

    constexpr CBIndex cb_fill_value_id = CBIndex::c_24;

    auto cb_value_config = tt::tt_metal::CircularBufferConfig(page_size, {{cb_fill_value_id, data_format}})
                               .set_page_size(cb_fill_value_id, page_size);
    CreateCircularBuffer(program, compute_core_range, cb_value_config);
    std::map<std::string, std::string> writer_defines;

    switch (dtype) {
        case DataType::BFLOAT16: writer_defines["OUTPUT_DTYPE_BFLOAT16"] = "1"; break;
        case DataType::INT32: writer_defines["OUTPUT_DTYPE_INT32"] = "1"; break;
        case DataType::FLOAT32: writer_defines["OUTPUT_DTYPE_FLOAT32"] = "1"; break;
        default: break;
    }

    datatype u;
    if (std::holds_alternative<int>(fill_value)) {
        u.u32 = std::get<int>(fill_value);
    } else if (std::holds_alternative<float>(fill_value)) {
        auto float_fill_value = std::get<float>(fill_value);
        if (dtype == DataType::BFLOAT16) {
            u.u32 = get_bfloat16_rounded(float_fill_value);
        } else {
            u.f32 = float_fill_value;
        }
    }

    uint32_t elems_per_page = page_size / datum_size(data_format);
    std::vector<uint32_t> writer_compile_time_args = {
        (uint32_t)cb_fill_value_id, elems_per_page, page_size, aligned_page_size, num_shards, num_compute_cores};
    tt::tt_metal::TensorAccessorArgs(output.buffer()).append_to(writer_compile_time_args);

    auto writer_id = CreateKernel(
        program,
        "ttnn/cpp/ttnn/operations/full/device/kernels/writer_full_nd_sharded.cpp",
        compute_core_range,
        tt::tt_metal::WriterDataMovementConfig(writer_compile_time_args, writer_defines));

    const auto& runtime_cores = memory_config.is_dram() ? worker_cores : ordered_cores_with_data;
    uint32_t start_shard_id = 0;
    for (auto core : runtime_cores) {
        SetRuntimeArgs(program, writer_id, core, {output.buffer()->address(), u.u32, start_shard_id});
        start_shard_id++;
    }

    return {std::move(program), {writer_id, runtime_cores}};
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
