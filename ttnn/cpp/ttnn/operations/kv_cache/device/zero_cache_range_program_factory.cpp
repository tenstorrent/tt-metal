// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include <cstdint>
#include <map>
#include <set>

#include <tt-metalium/host_api.hpp>
#include <tt-metalium/constants.hpp>
#include <tt-metalium/tensor_accessor_args.hpp>
#include "zero_cache_range_program_factory.hpp"

using namespace tt::tt_metal;

namespace ttnn::prim {

// Compute per-bank page ranges for a given [start_page, end_page) range.
// Returns a vector indexed by bank_id, where each entry is (page_start, page_end).
// Banks with no work have page_start == page_end == 0.
static std::vector<std::pair<uint32_t, uint32_t>> compute_bank_page_ranges(
    uint32_t start_page, uint32_t end_page, uint32_t pages_per_shard, uint32_t num_dram_banks) {
    std::vector<std::pair<uint32_t, uint32_t>> bank_ranges(num_dram_banks, {0, 0});

    const uint32_t first_shard = start_page / pages_per_shard;
    const uint32_t last_shard = (end_page + pages_per_shard - 1) / pages_per_shard;

    for (uint32_t shard = first_shard; shard < last_shard; shard++) {
        uint32_t bank_id = shard % num_dram_banks;
        uint32_t shard_page_start = std::max(shard * pages_per_shard, start_page);
        uint32_t shard_page_end = std::min(shard * pages_per_shard + pages_per_shard, end_page);

        if (bank_ranges[bank_id].first == 0 && bank_ranges[bank_id].second == 0) {
            bank_ranges[bank_id] = {shard_page_start, shard_page_end};
        } else {
            bank_ranges[bank_id].first = std::min(bank_ranges[bank_id].first, shard_page_start);
            bank_ranges[bank_id].second = std::max(bank_ranges[bank_id].second, shard_page_end);
        }
    }

    return bank_ranges;
}

ZeroCacheRangeProgramFactory::cached_program_t ZeroCacheRangeProgramFactory::create(
    const ZeroCacheRangeParams& operation_attributes,
    const ZeroCacheRangeInputs& tensor_args,
    Tensor& /*tensor_return_value*/) {
    const auto& cache_tensor = tensor_args.cache;
    const auto start_page = operation_attributes.start_page;
    const auto end_page = operation_attributes.end_page;

    Program program{};

    auto* device = cache_tensor.device();
    auto* dst_buffer = cache_tensor.buffer();
    const uint32_t aligned_page_size = static_cast<uint32_t>(dst_buffer->aligned_page_size());
    const auto arch = device->arch();

    // Determine NOC max burst size based on architecture
    uint32_t noc_max_burst_size;
    if (arch == tt::ARCH::BLACKHOLE) {
        noc_max_burst_size = 16384;
    } else if (arch == tt::ARCH::WORMHOLE_B0) {
        noc_max_burst_size = 8192;
    } else {
        TT_THROW("Unsupported architecture for zero cache range: {}", arch);
    }

    const auto noc = tt::tt_metal::detail::preferred_noc_for_dram_write(arch);

    // Get optimal core-to-DRAM-bank mapping — one core per bank for all banks
    const auto num_dram_banks = static_cast<uint32_t>(device->num_dram_channels());
    const auto optimal_cores = device->get_optimal_dram_bank_to_logical_worker_assignment(noc);

    // Pages per shard = embedding_dim / tile_width (e.g. 576 / 32 = 18)
    const uint32_t pages_per_shard = cache_tensor.padded_shape()[-1] / tt::constants::TILE_WIDTH;

    // Compute per-bank page ranges
    auto bank_ranges = compute_bank_page_ranges(start_page, end_page, pages_per_shard, num_dram_banks);

    // Always allocate all bank cores so the program structure is stable for caching.
    // Cores with no work get page_start == page_end and the kernel loop does nothing.
    std::vector<CoreCoord> cores;
    std::vector<uint32_t> page_starts;
    std::vector<uint32_t> page_ends;
    std::set<CoreRange> core_ranges;

    for (uint32_t bank_id = 0; bank_id < num_dram_banks; bank_id++) {
        const auto& core = optimal_cores[bank_id];
        cores.push_back(core);
        page_starts.push_back(bank_ranges[bank_id].first);
        page_ends.push_back(bank_ranges[bank_id].second);
        core_ranges.insert(CoreRange(core));
    }

    CoreRangeSet all_cores(core_ranges);

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

    // Create kernel on all bank cores
    auto writer_kernel_id = tt::tt_metal::CreateKernel(
        program,
        "ttnn/cpp/ttnn/operations/kv_cache/device/kernels/dataflow/zero_cache_writer.cpp",
        all_cores,
        tt::tt_metal::DataMovementConfig{
            .processor = tt::tt_metal::DataMovementProcessor::RISCV_0, .noc = noc, .compile_args = compile_time_args});

    // Set runtime args per core
    for (uint32_t i = 0; i < cores.size(); i++) {
        tt::tt_metal::SetRuntimeArgs(
            program,
            writer_kernel_id,
            cores[i],
            {
                dst_buffer->address(),
                page_starts[i],
                page_ends[i],
            });
    }

    return cached_program_t{
        std::move(program),
        shared_variables_t{
            .writer_kernel_id = writer_kernel_id,
            .cores = cores,
            .page_starts = page_starts,
            .page_ends = page_ends,
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
    auto& page_starts = cached_program.shared_variables.page_starts;
    auto& page_ends = cached_program.shared_variables.page_ends;

    auto* dst_buffer = tensor_args.cache.buffer();
    const uint32_t pages_per_shard = tensor_args.cache.padded_shape()[-1] / tt::constants::TILE_WIDTH;
    const auto num_dram_banks = static_cast<uint32_t>(cores.size());

    auto bank_ranges = compute_bank_page_ranges(
        operation_attributes.start_page, operation_attributes.end_page, pages_per_shard, num_dram_banks);

    for (uint32_t i = 0; i < cores.size(); i++) {
        page_starts[i] = bank_ranges[i].first;
        page_ends[i] = bank_ranges[i].second;

        auto& runtime_args = GetRuntimeArgs(program, writer_kernel_id, cores[i]);
        runtime_args[0] = dst_buffer->address();
        runtime_args[1] = page_starts[i];
        runtime_args[2] = page_ends[i];
    }
}

}  // namespace ttnn::prim
