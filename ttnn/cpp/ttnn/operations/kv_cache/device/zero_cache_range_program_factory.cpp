// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include <cstdint>

#include <tt-metalium/work_split.hpp>
#include <tt-metalium/host_api.hpp>
#include <tt-metalium/constants.hpp>
#include <tt-metalium/tensor_accessor_args.hpp>
#include "zero_cache_range_program_factory.hpp"

using namespace tt::tt_metal;

namespace ttnn::prim {

ZeroCacheRangeProgramFactory::cached_program_t ZeroCacheRangeProgramFactory::create(
    const ZeroCacheRangeParams& operation_attributes,
    const ZeroCacheRangeInputs& tensor_args,
    Tensor& /*tensor_return_value*/) {
    const auto& cache_tensor = tensor_args.cache;
    const auto start_page = operation_attributes.start_page;
    const auto end_page = operation_attributes.end_page;
    const uint32_t total_pages = end_page - start_page;

    Program program{};

    auto* device = cache_tensor.device();
    auto* dst_buffer = cache_tensor.buffer();
    const uint32_t aligned_page_size = static_cast<uint32_t>(dst_buffer->aligned_page_size());

    // Determine NOC max burst size based on architecture
    uint32_t noc_max_burst_size;
    const auto arch = device->arch();
    if (arch == tt::ARCH::BLACKHOLE) {
        noc_max_burst_size = 16384;
    } else if (arch == tt::ARCH::WORMHOLE_B0) {
        noc_max_burst_size = 8192;
    } else {
        TT_THROW("Unsupported architecture for zero cache range: {}", arch);
    }

    // Distribute work across cores
    auto compute_with_storage_grid_size = device->compute_with_storage_grid_size();
    uint32_t num_cores_x = compute_with_storage_grid_size.x;
    uint32_t num_cores_y = compute_with_storage_grid_size.y;

    uint32_t num_cores, num_pages_per_core_group_1, num_pages_per_core_group_2;
    CoreRangeSet all_cores, core_group_1, core_group_2;
    bool row_major = true;

    std::tie(num_cores, all_cores, core_group_1, core_group_2, num_pages_per_core_group_1, num_pages_per_core_group_2) =
        tt::tt_metal::split_work_to_cores(compute_with_storage_grid_size, total_pages, row_major);

    // Create CB for zero buffer
    uint32_t cb_zero_id = 0;
    tt::tt_metal::CircularBufferConfig cb_config =
        tt::tt_metal::CircularBufferConfig(noc_max_burst_size, {{cb_zero_id, tt::DataFormat::UInt8}})
            .set_page_size(cb_zero_id, noc_max_burst_size);
    tt::tt_metal::CreateCircularBuffer(program, all_cores, cb_config);

    // Build compile-time args
    std::vector<uint32_t> compile_time_args = {
        aligned_page_size,
        cb_zero_id,
    };
    tt::tt_metal::TensorAccessorArgs(*dst_buffer).append_to(compile_time_args);

    // Create kernel
    auto writer_kernel_id = tt::tt_metal::CreateKernel(
        program,
        "ttnn/cpp/ttnn/operations/kv_cache/device/kernels/dataflow/zero_cache_writer.cpp",
        all_cores,
        tt::tt_metal::DataMovementConfig{
            .processor = tt::tt_metal::DataMovementProcessor::RISCV_0,
            .noc = tt::tt_metal::detail::preferred_noc_for_dram_write(arch),
            .compile_args = compile_time_args});

    // Set runtime args per core
    uint32_t g1_numcores = core_group_1.num_cores();
    const auto& cores = grid_to_cores(num_cores, num_cores_x, num_cores_y, row_major);

    for (uint32_t i = 0, pages_written = 0; i < num_cores; i++) {
        const CoreCoord& core = cores.at(i);
        uint32_t num_pages_per_core = (i < g1_numcores) ? num_pages_per_core_group_1 : num_pages_per_core_group_2;

        uint32_t core_page_start = start_page + pages_written;
        uint32_t core_page_end = core_page_start + num_pages_per_core;

        tt::tt_metal::SetRuntimeArgs(
            program,
            writer_kernel_id,
            core,
            {
                dst_buffer->address(),
                core_page_start,
                core_page_end,
            });

        pages_written += num_pages_per_core;
    }

    return cached_program_t{
        std::move(program),
        shared_variables_t{
            .writer_kernel_id = writer_kernel_id,
            .cores = cores,
            .g1_numcores = g1_numcores,
            .num_pages_per_core_group_1 = num_pages_per_core_group_1,
            .num_pages_per_core_group_2 = num_pages_per_core_group_2,
        }};
}

void ZeroCacheRangeProgramFactory::override_runtime_arguments(
    cached_program_t& cached_program,
    const ZeroCacheRangeParams& operation_attributes,
    const ZeroCacheRangeInputs& tensor_args,
    Tensor& /*tensor_return_value*/) {
    auto& program = cached_program.program;
    const auto& writer_kernel_id = cached_program.shared_variables.writer_kernel_id;
    const auto& cores = cached_program.shared_variables.cores;
    const auto g1_numcores = cached_program.shared_variables.g1_numcores;
    const auto num_pages_per_core_group_1 = cached_program.shared_variables.num_pages_per_core_group_1;
    const auto num_pages_per_core_group_2 = cached_program.shared_variables.num_pages_per_core_group_2;

    const auto start_page = operation_attributes.start_page;
    auto* dst_buffer = tensor_args.cache.buffer();

    for (uint32_t i = 0, pages_written = 0; i < cores.size(); i++) {
        const CoreCoord& core = cores.at(i);
        uint32_t num_pages_per_core = (i < g1_numcores) ? num_pages_per_core_group_1 : num_pages_per_core_group_2;

        uint32_t core_page_start = start_page + pages_written;
        uint32_t core_page_end = core_page_start + num_pages_per_core;

        auto& runtime_args = GetRuntimeArgs(program, writer_kernel_id, core);
        runtime_args[0] = dst_buffer->address();
        runtime_args[1] = core_page_start;
        runtime_args[2] = core_page_end;

        pages_written += num_pages_per_core;
    }
}

}  // namespace ttnn::prim
