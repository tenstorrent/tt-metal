// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include "indexer_score_program_factory.hpp"

#include <tt-metalium/host_api.hpp>
#include <tt-metalium/constants.hpp>
#include <tt-metalium/tensor_accessor_args.hpp>
#include <tt-metalium/work_split.hpp>

#include "ttnn/operations/transformer/sdpa/device/sdpa_subblock_utils.hpp"
#include "kernels/indexer_score_work_split.hpp"  // shared host/device causal work-split formula

namespace ttnn::operations::experimental::deepseek::indexer::program {

using namespace tt;
using namespace tt::tt_metal;
using namespace tt::constants;

// Runtime-arg slots, shared by create() and override_runtime_arguments() (and matched
// positionally by the kernels). Reader: q,k,w addrs then flat_start,count; writer: out
// addr then flat_start,count; compute: flat_start,count.
namespace rt_arg {
constexpr uint32_t reader_q_addr = 0;
constexpr uint32_t reader_k_addr = 1;
constexpr uint32_t reader_w_addr = 2;
constexpr uint32_t writer_out_addr = 0;
}  // namespace rt_arg

// Patch one runtime-arg slot on a program-cache hit, asserting the slot exists.
inline void patch_arg(RuntimeArgsData& args, uint32_t index, uint32_t value, const char* name) {
    TT_FATAL(index < args.size(), "indexer_score override: {} index {} >= args size {}", name, index, args.size());
    args[index] = value;
}

// Output-stationary flat deal of causal-valid work units (INDEXER_OP.md).
// One unit = QC q-tile-rows x up-to-KC k-tiles; per q-row-group g,
// valid_max(g) = min(Tt, chunk_t + (g+1)*QC) and units(g) = ceil(valid_max/KC).
// Units are dealt evenly across cores in row-major order; kernels invert the
// flat index locally. Heads stream in HB-head groups; fully-future tiles
// inside a unit get the full -inf mask, row tails are -inf-filled by the
// writer (zeros are not safe: gates can be negative).
IndexerScoreProgramFactory::cached_program_t IndexerScoreProgramFactory::create(
    const operation_attributes_t& args, const tensor_args_t& tensors, tensor_return_value_t& out) {
    Program program = CreateProgram();

    const auto& q = tensors.q;
    const auto& k = tensors.k;
    const auto& w = tensors.weights;

    // Inputs and knobs are validated in IndexerScoreDeviceOperation::validate_on_program_cache_miss;
    // here we only derive tile dims and the one build-specific (subblock) constraint.
    const uint32_t Hi = q.logical_shape()[1];
    const uint32_t Sq = q.logical_shape()[2];
    const uint32_t D = q.logical_shape()[3];
    const uint32_t T = k.logical_shape()[2];

    const uint32_t Sqt = Sq / TILE_HEIGHT;
    const uint32_t Tt = T / TILE_WIDTH;
    const uint32_t Dt = D / TILE_WIDTH;
    const uint32_t chunk_t = args.chunk_start_idx / TILE_WIDTH;

    // work-unit knobs (elements -> tiles / heads)
    const auto& cfg = args.program_config;
    const uint32_t QC = cfg.q_chunk_size / TILE_HEIGHT;
    const uint32_t KC = cfg.k_chunk_size / TILE_WIDTH;
    const uint32_t HB = resolve_head_group(cfg, Hi);

    // qk matmul subblock: heads are output rows, k column is 1 tile wide (SDPA-style)
    constexpr bool fp32_dest_acc_en = false;                 // bf16 DEST: 8-head subblocks in half-sync
    constexpr uint32_t dst_size = fp32_dest_acc_en ? 4 : 8;  // half-sync, as in sdpa_program_factory
    const auto [qk_subblock_h, qk_subblock_w] = ttnn::prim::detail::determine_largest_subblock_size(HB, 1, dst_size);
    TT_FATAL(HB % qk_subblock_h == 0, "head group {} must be divisible by qk_subblock_h={}", HB, qk_subblock_h);

    // total valid work units (groups is exact: validate guarantees QC divides Sqt).
    // units_in_group is the shared formula the kernels' WorkUnitSpan inverts.
    const uint32_t groups = Sqt / QC;
    uint64_t V = 0;
    for (uint32_t g = 0; g < groups; ++g) {
        V += units_in_group(g, QC, KC, chunk_t, Tt);
    }

    const auto grid = q.device()->compute_with_storage_grid_size();
    const uint32_t num_cores = std::min<uint64_t>(V, (uint64_t)grid.x * grid.y);
    const auto core_ranges = num_cores_to_corerangeset(num_cores, grid, true);
    const auto cores = corerange_to_cores(core_ranges, num_cores, true);

    const uint32_t base = V / num_cores;
    const uint32_t rem = V % num_cores;

    const uint32_t bf16_tile = tile_size(DataFormat::Float16_b);
    const uint32_t fp32_tile = tile_size(DataFormat::Float32);

    // k is matmul srcA only; bfp8_b halves its DRAM/L1 footprint. q/w stay bf16.
    // bfp8 k carries ~fp8 mantissa, so LoFi covers it (extra HiFi passes would be wasted)
    // and is the lever that turns the halved BW into a compute win; bf16 k keeps HiFi2.
    const bool k_is_bfp8 = k.dtype() == DataType::BFLOAT8_B;
    const DataFormat k_fmt = k_is_bfp8 ? DataFormat::Bfp8_b : DataFormat::Float16_b;
    const uint32_t k_tile = tile_size(k_fmt);
    const MathFidelity math_fidelity = MathFidelity::HiFi2;

    auto make_cb = [&](uint32_t idx, uint32_t ntiles, DataFormat fmt, uint32_t tile_bytes) {
        CreateCircularBuffer(
            program,
            core_ranges,
            CircularBufferConfig(ntiles * tile_bytes, {{idx, fmt}}).set_page_size(idx, tile_bytes));
    };

    const bool stream_heads = HB < Hi;

    constexpr uint32_t cb_q = CBIndex::c_0;         // q head-group block: HB*QC*Dt tiles
    constexpr uint32_t cb_k = CBIndex::c_1;         // k chunk, double buffered
    constexpr uint32_t cb_w = CBIndex::c_2;         // resident w group: Hi*QC tiles
    constexpr uint32_t cb_mask = CBIndex::c_3;      // [diag -inf, full -inf], persistent
    constexpr uint32_t cb_qk = CBIndex::c_24;       // relu(q.kT) for a whole head group
    constexpr uint32_t cb_acc = CBIndex::c_26;      // unit accumulator ring
    constexpr uint32_t cb_out = CBIndex::c_16;      // untilized bf16 tiles
    constexpr uint32_t cb_scratch = CBIndex::c_17;  // writer-only -inf scratch

    make_cb(cb_q, (stream_heads ? 2 : 1) * HB * QC * Dt, DataFormat::Float16_b, bf16_tile);
    make_cb(cb_k, 2 * KC * Dt, k_fmt, k_tile);
    make_cb(cb_w, Hi * QC, DataFormat::Float16_b, bf16_tile);
    make_cb(cb_mask, 2, DataFormat::Float16_b, bf16_tile);
    const DataFormat acc_fmt = fp32_dest_acc_en ? DataFormat::Float32 : DataFormat::Float16_b;
    const uint32_t acc_tile = fp32_dest_acc_en ? fp32_tile : bf16_tile;
    // cb_qk buffers a batch of the group's relu(q.kT) tiles so compute runs that batch's
    // matmuls, then its mul+accumulates -- hoisting the matmul<->eltwise reinit out of the
    // per-head-pass loop. Batch = whole group when it fits a tile budget (QC=1, all heads
    // resident -> 64 tiles), else capped so QC>1 + resident configs (large cb_q/cb_w) still
    // fit L1; the kernel sub-batches the group's head passes by qk_batch_heads.
    // QC==1 (one q-tile-row per unit; the production case) has spare L1 -> batch the whole group
    // (one matmul/mul mode switch per output tile). QC>1 doubles cb_q/cb_w, so cap the batch.
    const uint32_t qk_batch_cap = (QC == 1) ? HB : 32u;
    const uint32_t qk_batch_heads = std::min<uint32_t>(HB, qk_batch_cap);  // multiple of qk_subblock_h
    make_cb(cb_qk, qk_batch_heads, acc_fmt, acc_tile);
    make_cb(cb_acc, 2 * QC * KC, acc_fmt, acc_tile);
    make_cb(cb_out, 2 * KC, DataFormat::Float16_b, bf16_tile);
    make_cb(cb_scratch, 1, DataFormat::Float16_b, bf16_tile);

    const std::vector<uint32_t> common_ct = {Hi, Sqt, Tt, Dt, chunk_t, QC, KC, HB};

    std::vector<uint32_t> reader_ct = common_ct;
    TensorAccessorArgs(*q.buffer()).append_to(reader_ct);
    TensorAccessorArgs(*k.buffer()).append_to(reader_ct);
    TensorAccessorArgs(*w.buffer()).append_to(reader_ct);

    std::vector<uint32_t> writer_ct = common_ct;
    constexpr uint32_t out_elem_bytes = 2;    // bf16 output (compute_output_specs)
    writer_ct.push_back(T * out_elem_bytes);  // row-major page = one full row of T scores
    TensorAccessorArgs(*out.buffer()).append_to(writer_ct);

    std::vector<uint32_t> compute_ct = common_ct;
    compute_ct.push_back(qk_subblock_h);
    compute_ct.push_back(qk_batch_heads);  // head tiles per matmul/mul phase chunk (cb_qk capacity)

    const std::string kdir = "ttnn/cpp/ttnn/operations/experimental/deepseek/indexer_score/device/kernels/";
    auto reader_id =
        CreateKernel(program, kdir + "reader_indexer_score.cpp", core_ranges, ReaderDataMovementConfig(reader_ct));
    auto writer_id =
        CreateKernel(program, kdir + "writer_indexer_score.cpp", core_ranges, WriterDataMovementConfig(writer_ct));
    auto compute_id = CreateKernel(
        program,
        kdir + "compute_indexer_score.cpp",
        core_ranges,
        ComputeConfig{
            .math_fidelity = math_fidelity,
            .fp32_dest_acc_en = fp32_dest_acc_en,
            .dst_full_sync_en = false,
            .compile_args = compute_ct});

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
            .reader_kernel = reader_id,
            .compute_kernel = compute_id,
            .writer_kernel = writer_id,
            .worker_cores = cores}};
}

void IndexerScoreProgramFactory::override_runtime_arguments(
    cached_program_t& cached, const operation_attributes_t&, const tensor_args_t& tensors, tensor_return_value_t& out) {
    auto& shared = cached.shared_variables;
    auto& reader_args = GetRuntimeArgs(cached.program, shared.reader_kernel);
    auto& writer_args = GetRuntimeArgs(cached.program, shared.writer_kernel);
    for (const auto& core : shared.worker_cores) {
        auto& r = reader_args[core.x][core.y];
        patch_arg(r, rt_arg::reader_q_addr, tensors.q.buffer()->address(), "reader.q_addr");
        patch_arg(r, rt_arg::reader_k_addr, tensors.k.buffer()->address(), "reader.k_addr");
        patch_arg(r, rt_arg::reader_w_addr, tensors.weights.buffer()->address(), "reader.w_addr");
        patch_arg(writer_args[core.x][core.y], rt_arg::writer_out_addr, out.buffer()->address(), "writer.out_addr");
    }
}

}  // namespace ttnn::operations::experimental::deepseek::indexer::program
