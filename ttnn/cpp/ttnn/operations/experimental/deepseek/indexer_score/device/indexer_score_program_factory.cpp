// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include "indexer_score_program_factory.hpp"

#include <tt-metalium/host_api.hpp>
#include <tt-metalium/constants.hpp>
#include <tt-metalium/tensor_accessor_args.hpp>
#include <tt-metalium/work_split.hpp>

namespace ttnn::operations::experimental::deepseek::indexer::program {

using namespace tt;
using namespace tt::tt_metal;
using namespace tt::constants;

// Output-stationary flat deal of causal-valid output tiles (INDEXER_OP.md).
// valid(s) = min(Tt, chunk_t + s + 1) tiles per q-tile-row; V = sum valid(s)
// dealt evenly across cores in row-major order. Per-core kernels invert the
// flat index locally; fully-future tiles are never assigned. Skipped columns
// are -inf-filled by the writer (zeros are not safe: gates can be negative).
IndexerScoreProgramFactory::cached_program_t IndexerScoreProgramFactory::create(
    const operation_attributes_t& args, const tensor_args_t& tensors, tensor_return_value_t& out) {
    Program program = CreateProgram();

    const auto& q = tensors.q;
    const auto& k = tensors.k;
    const auto& w = tensors.weights;

    const uint32_t Hi = q.logical_shape()[1];
    const uint32_t Sq = q.logical_shape()[2];
    const uint32_t D = q.logical_shape()[3];
    const uint32_t T = k.logical_shape()[2];

    TT_FATAL(Sq % TILE_HEIGHT == 0 && T % TILE_WIDTH == 0 && D % TILE_WIDTH == 0, "Sq, T, D must be tile-aligned");
    TT_FATAL(args.chunk_start_idx % TILE_WIDTH == 0, "chunk_start_idx must be tile-aligned");
    TT_FATAL(args.chunk_start_idx + Sq <= T, "chunk window [{}+{}) exceeds T={}", args.chunk_start_idx, Sq, T);
    TT_FATAL(Hi % 8 == 0, "Hi={} must be a multiple of 8 (fp32 DEST passes)", Hi);
    TT_FATAL(q.logical_shape()[0] == 1, "batch 1 only");
    TT_FATAL(args.is_causal, "non-causal not implemented");

    const uint32_t Sqt = Sq / TILE_HEIGHT;
    const uint32_t Tt = T / TILE_WIDTH;
    const uint32_t Dt = D / TILE_WIDTH;
    const uint32_t chunk_t = args.chunk_start_idx / TILE_WIDTH;

    // total valid tiles
    uint64_t V = 0;
    for (uint32_t s = 0; s < Sqt; ++s) {
        V += std::min(Tt, chunk_t + s + 1);
    }

    const auto grid = q.device()->compute_with_storage_grid_size();
    const uint32_t num_cores = std::min<uint64_t>(V, (uint64_t)grid.x * grid.y);
    const auto core_ranges = num_cores_to_corerangeset(num_cores, grid, true);
    const auto cores = corerange_to_cores(core_ranges, num_cores, true);

    const uint32_t base = V / num_cores;
    const uint32_t rem = V % num_cores;

    const uint32_t bf16_tile = tile_size(DataFormat::Float16_b);
    const uint32_t fp32_tile = tile_size(DataFormat::Float32);

    auto make_cb = [&](uint32_t idx, uint32_t ntiles, DataFormat fmt, uint32_t tile_bytes) {
        CreateCircularBuffer(
            program,
            core_ranges,
            CircularBufferConfig(ntiles * tile_bytes, {{idx, fmt}}).set_page_size(idx, tile_bytes));
    };

    constexpr uint32_t cb_q = CBIndex::c_0;       // resident q row: Hi*Dt tiles
    constexpr uint32_t cb_k = CBIndex::c_1;       // k column, double buffered
    constexpr uint32_t cb_w = CBIndex::c_2;       // resident w row: Hi tiles
    constexpr uint32_t cb_mask = CBIndex::c_3;    // diagonal -inf mask, persistent
    constexpr uint32_t cb_qk = CBIndex::c_24;     // relu(q.kT) per head, fp32
    constexpr uint32_t cb_mul = CBIndex::c_25;    // relu*w per head, fp32
    constexpr uint32_t cb_acc = CBIndex::c_26;    // head accumulator, fp32
    constexpr uint32_t cb_out = CBIndex::c_16;     // untilized bf16 rows
    constexpr uint32_t cb_scratch = CBIndex::c_17;  // writer-only -inf scratch

    make_cb(cb_q, Hi * Dt, DataFormat::Float16_b, bf16_tile);
    make_cb(cb_k, 2 * Dt, DataFormat::Float16_b, bf16_tile);
    make_cb(cb_w, Hi, DataFormat::Float16_b, bf16_tile);
    make_cb(cb_mask, 1, DataFormat::Float16_b, bf16_tile);
    make_cb(cb_qk, Hi, DataFormat::Float32, fp32_tile);
    make_cb(cb_mul, 8, DataFormat::Float32, fp32_tile);
    make_cb(cb_acc, 2, DataFormat::Float32, fp32_tile);
    make_cb(cb_out, 2, DataFormat::Float16_b, bf16_tile);
    make_cb(cb_scratch, 1, DataFormat::Float16_b, bf16_tile);

    const std::vector<uint32_t> common_ct = {Hi, Sqt, Tt, Dt, chunk_t};

    std::vector<uint32_t> reader_ct = common_ct;
    TensorAccessorArgs(*q.buffer()).append_to(reader_ct);
    TensorAccessorArgs(*k.buffer()).append_to(reader_ct);
    TensorAccessorArgs(*w.buffer()).append_to(reader_ct);

    std::vector<uint32_t> writer_ct = common_ct;
    writer_ct.push_back(T * 2);  // row-major page bytes
    TensorAccessorArgs(*out.buffer()).append_to(writer_ct);

    const std::string kdir = "ttnn/cpp/ttnn/operations/experimental/deepseek/indexer_score/device/kernels/";
    auto reader_id = CreateKernel(
        program, kdir + "reader_indexer_score.cpp", core_ranges, ReaderDataMovementConfig(reader_ct));
    auto writer_id = CreateKernel(
        program, kdir + "writer_indexer_score.cpp", core_ranges, WriterDataMovementConfig(writer_ct));
    auto compute_id = CreateKernel(
        program,
        kdir + "compute_indexer_score.cpp",
        core_ranges,
        ComputeConfig{
            .math_fidelity = MathFidelity::HiFi4,
            .fp32_dest_acc_en = true,
            .dst_full_sync_en = true,
            .compile_args = common_ct});

    uint32_t flat = 0;
    for (uint32_t i = 0; i < num_cores; ++i) {
        const uint32_t count = base + (i < rem ? 1 : 0);
        SetRuntimeArgs(
            program,
            reader_id,
            cores[i],
            {q.buffer()->address(), k.buffer()->address(), w.buffer()->address(), flat, count});
        SetRuntimeArgs(program, compute_id, cores[i], {flat, count});
        SetRuntimeArgs(program, writer_id, cores[i], {out.buffer()->address(), flat, count});
        flat += count;
    }

    return {
        std::move(program),
        IndexerScoreSharedVariables{
            .kernel_handles = {reader_id, compute_id, writer_id}, .worker_cores = cores}};
}

void IndexerScoreProgramFactory::override_runtime_arguments(
    cached_program_t& cached, const operation_attributes_t&, const tensor_args_t& tensors, tensor_return_value_t& out) {
    auto& shared = cached.shared_variables;
    auto& reader_args = GetRuntimeArgs(cached.program, shared.kernel_handles[0]);
    auto& writer_args = GetRuntimeArgs(cached.program, shared.kernel_handles[2]);
    for (const auto& core : shared.worker_cores) {
        auto& r = reader_args[core.x][core.y];
        r[0] = tensors.q.buffer()->address();
        r[1] = tensors.k.buffer()->address();
        r[2] = tensors.weights.buffer()->address();
        writer_args[core.x][core.y][0] = out.buffer()->address();
    }
}

}  // namespace ttnn::operations::experimental::deepseek::indexer::program
