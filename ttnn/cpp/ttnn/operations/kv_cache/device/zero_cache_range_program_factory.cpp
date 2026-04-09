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

    // Get optimal core-to-DRAM-bank mapping
    const auto num_dram_banks = static_cast<uint32_t>(device->num_dram_channels());
    const auto optimal_cores = device->get_optimal_dram_bank_to_logical_worker_assignment(noc);

    // Pages per shard = embedding_dim / tile_width (e.g. 576 / 32 = 18)
    const uint32_t pages_per_shard = cache_tensor.padded_shape()[-1] / tt::constants::TILE_WIDTH;

    // Determine which shards need zeroing and group by DRAM bank
    const uint32_t first_shard = start_page / pages_per_shard;
    const uint32_t last_shard = (end_page + pages_per_shard - 1) / pages_per_shard;

    // Map: bank_id -> list of (page_start, page_end) for contiguous shard ranges on that bank
    std::map<uint32_t, std::pair<uint32_t, uint32_t>> bank_page_ranges;
    for (uint32_t shard = first_shard; shard < last_shard; shard++) {
        uint32_t bank_id = shard % num_dram_banks;
        uint32_t shard_page_start = shard * pages_per_shard;
        uint32_t shard_page_end = shard_page_start + pages_per_shard;

        // Clamp to the requested range
        shard_page_start = std::max(shard_page_start, start_page);
        shard_page_end = std::min(shard_page_end, end_page);

        if (bank_page_ranges.find(bank_id) == bank_page_ranges.end()) {
            bank_page_ranges[bank_id] = {shard_page_start, shard_page_end};
        } else {
            // Extend range (pages for same bank may not be contiguous in global page space,
            // but the kernel iterates page-by-page so it handles gaps via TensorAccessor)
            bank_page_ranges[bank_id].first = std::min(bank_page_ranges[bank_id].first, shard_page_start);
            bank_page_ranges[bank_id].second = std::max(bank_page_ranges[bank_id].second, shard_page_end);
        }
    }

    // Build core list and page assignments - one optimal core per active bank
    std::vector<CoreCoord> cores;
    std::vector<uint32_t> page_starts;
    std::vector<uint32_t> page_ends;
    std::set<CoreRange> core_ranges;

    for (const auto& [bank_id, range] : bank_page_ranges) {
        const auto& core = optimal_cores[bank_id];
        cores.push_back(core);
        page_starts.push_back(range.first);
        page_ends.push_back(range.second);
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

    // Create kernel on optimal cores only
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

    // Recompute page assignments from the new page range
    const auto start_page = operation_attributes.start_page;
    const auto end_page = operation_attributes.end_page;
    const uint32_t pages_per_shard = tensor_args.cache.padded_shape()[-1] / tt::constants::TILE_WIDTH;
    const auto num_dram_banks = static_cast<uint32_t>(tensor_args.cache.device()->num_dram_channels());

    const uint32_t first_shard = start_page / pages_per_shard;
    const uint32_t last_shard = (end_page + pages_per_shard - 1) / pages_per_shard;

    std::map<uint32_t, std::pair<uint32_t, uint32_t>> bank_page_ranges;
    for (uint32_t shard = first_shard; shard < last_shard; shard++) {
        uint32_t bank_id = shard % num_dram_banks;
        uint32_t shard_page_start = std::max(shard * pages_per_shard, start_page);
        uint32_t shard_page_end = std::min(shard * pages_per_shard + pages_per_shard, end_page);

        if (bank_page_ranges.find(bank_id) == bank_page_ranges.end()) {
            bank_page_ranges[bank_id] = {shard_page_start, shard_page_end};
        } else {
            bank_page_ranges[bank_id].first = std::min(bank_page_ranges[bank_id].first, shard_page_start);
            bank_page_ranges[bank_id].second = std::max(bank_page_ranges[bank_id].second, shard_page_end);
        }
    }

    uint32_t i = 0;
    for (const auto& [bank_id, range] : bank_page_ranges) {
        if (i < cores.size()) {
            page_starts[i] = range.first;
            page_ends[i] = range.second;

            auto& runtime_args = GetRuntimeArgs(program, writer_kernel_id, cores[i]);
            runtime_args[0] = dst_buffer->address();
            runtime_args[1] = page_starts[i];
            runtime_args[2] = page_ends[i];
        }
        i++;
    }

    // Zero out remaining cores if fewer banks needed this time
    for (; i < cores.size(); i++) {
        page_starts[i] = 0;
        page_ends[i] = 0;
        auto& runtime_args = GetRuntimeArgs(program, writer_kernel_id, cores[i]);
        runtime_args[0] = dst_buffer->address();
        runtime_args[1] = 0;
        runtime_args[2] = 0;
    }
}

}  // namespace ttnn::prim
