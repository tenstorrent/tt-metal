// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include "sample_candidates_program_factory.hpp"

#include <algorithm>
#include <bit>
#include <vector>

#include <tt-metalium/constants.hpp>
#include <tt-metalium/host_api.hpp>
#include <tt-metalium/math.hpp>
#include <tt-metalium/tensor_accessor_args.hpp>

namespace ttnn::operations::experimental::deepseek::sample_candidates {

namespace {

using namespace tt::tt_metal;

constexpr uint32_t kCbRows = tt::CBIndex::c_0;
constexpr uint32_t kCbList = tt::CBIndex::c_1;
constexpr uint32_t kCbGather = tt::CBIndex::c_2;
constexpr uint32_t kCbOut = tt::CBIndex::c_3;
// Each tile's row 0 is read as its two 64 B face-row reads (columns 0-15 and 16-31, plus row 1 of each face).
constexpr uint32_t kRowReadBytes = 64;
constexpr uint32_t kBf16TileBytes = tt::constants::TILE_HW * 2;

constexpr char kKernelPath[] =
    "ttnn/cpp/ttnn/operations/experimental/deepseek/sample_candidates/device/kernels/dataflow/"
    "sample_candidates.cpp";

void add_cb(Program& program, const CoreRangeSet& cores, uint32_t index, uint32_t bytes) {
    CreateCircularBuffer(
        program, cores, CircularBufferConfig(bytes, {{index, tt::DataFormat::UInt32}}).set_page_size(index, bytes));
}

// fp32 -> bf16, round to nearest even.
uint32_t bf16_bits(float v) {
    const uint32_t b = std::bit_cast<uint32_t>(v);
    return (b + 0x7FFFu + ((b >> 16) & 1u)) >> 16;
}

}  // namespace

SampleCandidatesProgramFactory::cached_program_t SampleCandidatesProgramFactory::create(
    const SampleCandidatesParams& attrs, const SampleCandidatesInputs& args, SampleCandidatesTensorReturn& outputs) {
    Program program{};
    const auto& logits = args.logits;
    auto* device = logits.device();
    const auto grid = device->compute_with_storage_grid_size();
    const uint32_t v = logits.logical_shape()[3];
    const uint32_t tiles = tt::div_up(v, tt::constants::TILE_WIDTH);
    // Every worker gets at least ceil(k / 32) tiles, so it always has k candidates.
    const uint32_t min_tiles = tt::div_up(attrs.k, tt::constants::TILE_WIDTH);
    const uint32_t workers =
        std::min<uint32_t>({attrs.num_workers, tiles / min_tiles, static_cast<uint32_t>(grid.x * grid.y)});
    TT_FATAL(workers >= 1, "decode_sample_candidates: no worker cores");

    std::vector<CoreCoord> cores;
    std::set<CoreRange> ranges;
    for (uint32_t w = 0; w < workers; ++w) {
        cores.emplace_back(w % grid.x, w / grid.x);
        ranges.insert(CoreRange(cores.back()));
    }
    const CoreRangeSet all_cores(ranges);
    const uint32_t max_tiles = tt::div_up(tiles, workers);
    const uint32_t list_bytes = attrs.k * 2 * sizeof(uint32_t);
    const uint32_t out_tiles = 2 * attrs.k / tt::constants::TILE_WIDTH;
    add_cb(program, all_cores, kCbRows, max_tiles * 2 * kRowReadBytes);
    add_cb(program, all_cores, kCbList, tt::round_up(list_bytes, 64u));
    add_cb(program, all_cores, kCbGather, workers * tt::round_up(list_bytes, 64u));
    // The values' tiles, the indices' row, then 64 B each for inv_temp and index_offset.
    add_cb(program, all_cores, kCbOut, out_tiles * kBf16TileBytes + tt::round_up(2 * attrs.k * 4, 64u) + 128);
    const uint32_t sem = CreateSemaphore(program, all_cores, 0);

    std::vector<uint32_t> ct = {
        attrs.k,
        workers,
        v,
        kCbRows,
        kCbList,
        kCbGather,
        kCbOut,
        sem,
        bf16_bits(attrs.fill_value),
        tt::round_up(list_bytes, 64u)};
    TensorAccessorArgs(logits.buffer()).append_to(ct);
    TensorAccessorArgs(args.inv_temp.buffer()).append_to(ct);
    TensorAccessorArgs(args.index_offset.buffer()).append_to(ct);
    TensorAccessorArgs(outputs[0].buffer()).append_to(ct);
    TensorAccessorArgs(outputs[1].buffer()).append_to(ct);
    const KernelHandle kernel_id = CreateKernel(program, kKernelPath, all_cores, ReaderDataMovementConfig(ct));

    const CoreCoord hub = device->worker_core_from_logical_core(cores[0]);
    for (uint32_t w = 0; w < workers; ++w) {
        const uint32_t t0 = w * tiles / workers;
        const uint32_t t1 = (w + 1) * tiles / workers;
        SetRuntimeArgs(
            program,
            kernel_id,
            cores[w],
            {logits.buffer()->address(),
             args.inv_temp.buffer()->address(),
             args.index_offset.buffer()->address(),
             outputs[0].buffer()->address(),
             outputs[1].buffer()->address(),
             w,
             t0,
             t1,
             static_cast<uint32_t>(hub.x),
             static_cast<uint32_t>(hub.y)});
    }
    return cached_program_t{std::move(program), {.kernel_id = kernel_id, .cores = cores}};
}

void SampleCandidatesProgramFactory::override_runtime_arguments(
    cached_program_t& cached_program,
    const SampleCandidatesParams& /*attrs*/,
    const SampleCandidatesInputs& args,
    SampleCandidatesTensorReturn& outputs) {
    auto& program = cached_program.program;
    const auto& shared = cached_program.shared_variables;
    for (const auto& core : shared.cores) {
        auto& a = GetRuntimeArgs(program, shared.kernel_id, core);
        a[0] = args.logits.buffer()->address();
        a[1] = args.inv_temp.buffer()->address();
        a[2] = args.index_offset.buffer()->address();
        a[3] = outputs[0].buffer()->address();
        a[4] = outputs[1].buffer()->address();
    }
}

}  // namespace ttnn::operations::experimental::deepseek::sample_candidates
