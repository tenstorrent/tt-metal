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
// A RISC's rows of each tile are read as whole 64 B reads (two face-rows) of faces 0 and 1.
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
    const uint32_t rows = logits.logical_shape()[2];
    const uint32_t tiles = tt::div_up(v, tt::constants::TILE_WIDTH);
    // Every worker gets at least ceil(k / 32) tiles, so it always has k candidates.
    const uint32_t min_tiles = tt::div_up(attrs.k, tt::constants::TILE_WIDTH);
    const uint32_t scanners =
        std::min<uint32_t>({attrs.num_workers, tiles / min_tiles, static_cast<uint32_t>(grid.x * grid.y)});
    TT_FATAL(scanners >= 1, "decode_sample_candidates: no worker cores");
    // Core r merges row r: with few tiles, cores past the scanners only take part in that (empty lists).
    const uint32_t workers = std::max(scanners, rows);
    TT_FATAL(workers <= grid.x * grid.y, "decode_sample_candidates: {} rows need {} cores", rows, workers);
    // Rows [0, split) run on the reader RISC, [split, rows) on the writer RISC.
    const uint32_t split = (rows + 1) / 2;
    auto read_bytes = [](uint32_t r0, uint32_t r1) { return tt::round_up((r1 - (r0 & ~1u)) * 32, kRowReadBytes); };

    std::vector<CoreCoord> cores;
    std::set<CoreRange> ranges;
    for (uint32_t w = 0; w < workers; ++w) {
        cores.emplace_back(w % grid.x, w / grid.x);
        ranges.insert(CoreRange(cores.back()));
    }
    const CoreRangeSet all_cores(ranges);
    const uint32_t max_tiles = tt::div_up(tiles, scanners);
    // A list: k packed (key, column offset) words; offsets within a worker's tiles fit 16 bits.
    const uint32_t list_bytes = attrs.k * sizeof(uint32_t);
    TT_FATAL(max_tiles * tt::constants::TILE_WIDTH <= 0xFFFF, "decode_sample_candidates: too few workers for V");
    const uint32_t out_tiles = 2 * attrs.k / tt::constants::TILE_WIDTH;
    const uint32_t reader_rows_bytes = max_tiles * 2 * read_bytes(0, split);
    const uint32_t writer_rows_bytes = split < rows ? max_tiles * 2 * read_bytes(split, rows) : 0;
    add_cb(program, all_cores, kCbRows, reader_rows_bytes + writer_rows_bytes);
    add_cb(program, all_cores, kCbList, rows * tt::round_up(list_bytes, 64u));
    add_cb(program, all_cores, kCbGather, workers * tt::round_up(list_bytes, 64u));
    // The values' tiles, the indices' row, then 64 B each for inv_temp and index_offset.
    add_cb(program, all_cores, kCbOut, out_tiles * kBf16TileBytes + tt::round_up(2 * attrs.k * 4, 64u) + 128);
    const uint32_t sem = CreateSemaphore(program, all_cores, 0);
    const uint32_t done_sem = CreateSemaphore(program, all_cores, 0);

    auto ct_args = [&](uint32_t r0, uint32_t r1, uint32_t rows_offset) {
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
            tt::round_up(list_bytes, 64u),
            r0,
            r1,
            rows,
            done_sem,
            rows_offset,
            tiles,
            scanners};
        TensorAccessorArgs(logits.buffer()).append_to(ct);
        TensorAccessorArgs(args.inv_temp.buffer()).append_to(ct);
        TensorAccessorArgs(args.index_offset.buffer()).append_to(ct);
        TensorAccessorArgs(outputs[0].buffer()).append_to(ct);
        TensorAccessorArgs(outputs[1].buffer()).append_to(ct);
        return ct;
    };
    std::vector<KernelHandle> kernel_ids = {
        CreateKernel(program, kKernelPath, all_cores, ReaderDataMovementConfig(ct_args(0, split, 0)))};
    if (split < rows) {
        kernel_ids.push_back(CreateKernel(
            program, kKernelPath, all_cores, WriterDataMovementConfig(ct_args(split, rows, reader_rows_bytes))));
    }

    const CoreCoord out_core = device->worker_core_from_logical_core(cores[0]);
    for (uint32_t w = 0; w < workers; ++w) {
        const uint32_t t0 = w < scanners ? w * tiles / scanners : tiles;
        const uint32_t t1 = w < scanners ? (w + 1) * tiles / scanners : tiles;
        for (uint32_t kk = 0; kk < kernel_ids.size(); ++kk) {
            std::vector<uint32_t> rt = {
                logits.buffer()->address(),
                args.inv_temp.buffer()->address(),
                args.index_offset.buffer()->address(),
                outputs[0].buffer()->address(),
                outputs[1].buffer()->address(),
                w,
                t0,
                t1,
                static_cast<uint32_t>(out_core.x),
                static_cast<uint32_t>(out_core.y)};
            // Each of this RISC's rows' hub: worker core r.
            for (uint32_t r = kk == 0 ? 0 : split; r < (kk == 0 ? split : rows); ++r) {
                const CoreCoord hub = device->worker_core_from_logical_core(cores[r]);
                rt.push_back(hub.x);
                rt.push_back(hub.y);
            }
            SetRuntimeArgs(program, kernel_ids[kk], cores[w], rt);
        }
    }
    return cached_program_t{std::move(program), {.kernel_ids = kernel_ids, .cores = cores}};
}

void SampleCandidatesProgramFactory::override_runtime_arguments(
    cached_program_t& cached_program,
    const SampleCandidatesParams& /*attrs*/,
    const SampleCandidatesInputs& args,
    SampleCandidatesTensorReturn& outputs) {
    auto& program = cached_program.program;
    const auto& shared = cached_program.shared_variables;
    for (const auto kernel_id : shared.kernel_ids) {
        for (const auto& core : shared.cores) {
            auto& a = GetRuntimeArgs(program, kernel_id, core);
            a[0] = args.logits.buffer()->address();
            a[1] = args.inv_temp.buffer()->address();
            a[2] = args.index_offset.buffer()->address();
            a[3] = outputs[0].buffer()->address();
            a[4] = outputs[1].buffer()->address();
        }
    }
}

}  // namespace ttnn::operations::experimental::deepseek::sample_candidates
