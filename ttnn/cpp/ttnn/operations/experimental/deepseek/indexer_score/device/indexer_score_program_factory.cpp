// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include "indexer_score_program_factory.hpp"

#include <cstdlib>

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

    // QC/KC/HB are taken verbatim from the program config -- the factory does NOT auto-tune them.
    // The caller owns the perf trade-off (see production_config() in the test for the tuned values:
    // QC trades a ~QC-fold cut in redundant K reads against bigger resident CBs + a higher compute
    // ceiling). An oversized config is rejected by the L1-fit check after the CBs are sized below,
    // not silently adjusted.

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

    // ---- grid-aligned multicast (decoupled Q/W along rows, K down columns) -------------------
    // When the dense deal lands exactly on the physical grid -- group g == grid row y, and the
    // grid_x cores of a row split that group's k-chunks into contiguous bands -- cores in the same
    // grid ROW share identical q/w (same q-rows) and cores in the same grid COLUMN share the
    // identical k-band. So one core per row reads q/w once and multicasts along the row, and one
    // core per column reads each k-chunk and multicasts down the column, killing the ~grid_x q/w
    // re-reads and the ~grid_y k re-reads. Each direction is independent: enabled only if that
    // direction's lines are contiguous NoC rectangles; otherwise that input falls back to per-core
    // DRAM reads (harvested grid / non-grid-aligned deal). See INDEXER_DATAMOVEMENT.md.
    const uint32_t units_per_group = groups > 0 ? (uint32_t)(V / groups) : 0;
    const bool grid_aligned = ttnn::operations::experimental::deepseek::indexer::dense_schedule && groups == grid.y &&
                              num_cores == (uint32_t)(grid.x * grid.y) && rem == 0 && units_per_group == grid.x * base;

    std::vector<CoreCoord> phys(num_cores);
    for (uint32_t i = 0; i < num_cores; ++i) {
        phys[i] = q.device()->worker_core_from_logical_core(cores[i]);
    }
    auto cidx = [&](uint32_t x, uint32_t y) { return y * grid.x + x; };

    // Per-line contiguity: K columns must be vertical NoC rects (shared x, contiguous y); Q/W rows
    // horizontal rects (shared y, contiguous x). Q/W also needs all heads resident (one q block).
    bool k_cols_ok = grid_aligned;
    bool q_rows_ok = grid_aligned && HB == Hi;
    for (uint32_t x = 0; x < grid.x && k_cols_ok; ++x) {
        const uint32_t px = phys[cidx(x, 0)].x;
        uint32_t ymin = phys[cidx(x, 0)].y, ymax = ymin;
        for (uint32_t y = 0; y < grid.y; ++y) {
            const auto& p = phys[cidx(x, y)];
            if (p.x != px) {
                k_cols_ok = false;
            }
            ymin = std::min<uint32_t>(ymin, p.y);
            ymax = std::max<uint32_t>(ymax, p.y);
        }
        if (ymax - ymin + 1 != grid.y) {
            k_cols_ok = false;
        }
    }
    for (uint32_t y = 0; y < grid.y && q_rows_ok; ++y) {
        const uint32_t py = phys[cidx(0, y)].y;
        uint32_t xmin = phys[cidx(0, y)].x, xmax = xmin;
        for (uint32_t x = 0; x < grid.x; ++x) {
            const auto& p = phys[cidx(x, y)];
            if (p.y != py) {
                q_rows_ok = false;
            }
            xmin = std::min<uint32_t>(xmin, p.x);
            xmax = std::max<uint32_t>(xmax, p.x);
        }
        if (xmax - xmin + 1 != grid.x) {
            q_rows_ok = false;
        }
    }
    if (std::getenv("INDEXER_NO_KMCAST") != nullptr) {
        k_cols_ok = false;
    }
    if (std::getenv("INDEXER_NO_QMCAST") != nullptr) {
        q_rows_ok = false;
    }
    // The compute-ceiling diagnostic disables reader DRAM reads; disable mcast too so the ceiling
    // stays pure (no NoC sharing traffic). Mcast is meaningless without the underlying reads.
    if (std::getenv("INDEXER_DMA_OFF") != nullptr || std::getenv("INDEXER_DMA_OFF_READER") != nullptr) {
        k_cols_ok = false;
        q_rows_ok = false;
    }
    const uint32_t k_mcast_on = k_cols_ok ? 1u : 0u;
    const uint32_t q_mcast_on = q_rows_ok ? 1u : 0u;

    // 3 semaphores per active direction: send (receivers signal ready), recv (sender relays valid
    // into it), valid (constant 1, the relay source). Mirrors SDPA chain_link's handshake.
    const uint32_t k_send_sem = k_mcast_on ? CreateSemaphore(program, core_ranges, 0) : 0;
    const uint32_t k_recv_sem = k_mcast_on ? CreateSemaphore(program, core_ranges, 0) : 0;
    const uint32_t k_valid_sem = k_mcast_on ? CreateSemaphore(program, core_ranges, 1) : 0;
    const uint32_t q_send_sem = q_mcast_on ? CreateSemaphore(program, core_ranges, 0) : 0;
    const uint32_t q_recv_sem = q_mcast_on ? CreateSemaphore(program, core_ranges, 0) : 0;
    const uint32_t q_valid_sem = q_mcast_on ? CreateSemaphore(program, core_ranges, 1) : 0;

    const uint32_t bf16_tile = tile_size(DataFormat::Float16_b);
    const uint32_t fp32_tile = tile_size(DataFormat::Float32);

    // k is matmul srcA only; bfp8_b halves its DRAM/L1 footprint. q/w stay bf16.
    // bfp8 k carries ~fp8 mantissa, so LoFi covers it (extra HiFi passes would be wasted)
    // and is the lever that turns the halved BW into a compute win; bf16 k keeps HiFi2.
    const bool k_is_bfp8 = k.dtype() == DataType::BFLOAT8_B;
    const DataFormat k_fmt = k_is_bfp8 ? DataFormat::Bfp8_b : DataFormat::Float16_b;
    const uint32_t k_tile = tile_size(k_fmt);
    const MathFidelity math_fidelity = MathFidelity::HiFi2;

    uint64_t l1_used = 0;
    auto make_cb = [&](uint32_t idx, uint32_t ntiles, DataFormat fmt, uint32_t tile_bytes) {
        l1_used += (uint64_t)ntiles * tile_bytes;
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
    constexpr uint32_t cb_out = CBIndex::c_16;      // untilized bf16 tiles (per-tile W=1 path)
    constexpr uint32_t cb_scratch = CBIndex::c_17;  // writer-only -inf scratch
    constexpr uint32_t cb_out_strip = CBIndex::c_18;  // full-width strip output (fast untilize, uniform KC push)
    constexpr uint32_t cb_acc_strip = CBIndex::c_27;  // full-width strip accumulator (uniform KC push)

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
    // Full-strip path batches qk_col_batch k-columns per matmul<->mul mode switch (one set_matmul_mode
    // + set_mul_mode per batch instead of per output tile). The gate w is column-independent and the
    // unit's whole k chunk is resident, so a row's columns can share one mode switch. Only when the
    // head group is a single chunk (qk_batch_heads == Hi) and the fast strip is used (KC >= 2); cb_qk
    // then holds qk_col_batch * qk_batch_heads relu(q.kT) tiles, capped to a fixed L1 tile budget.
    constexpr uint32_t qk_col_tile_cap = 128;  // cb_qk tile budget for the batched matmul outputs
    const bool single_chunk = (qk_batch_heads == Hi) && !stream_heads;
    const uint32_t qk_col_batch = (KC >= 2 && single_chunk)
                                      ? std::min<uint32_t>(KC, std::max<uint32_t>(1u, qk_col_tile_cap / qk_batch_heads))
                                      : 1u;
    make_cb(cb_qk, qk_col_batch * qk_batch_heads, acc_fmt, acc_tile);
    make_cb(cb_acc, 2 * QC * KC, acc_fmt, acc_tile);
    make_cb(cb_out, 2 * KC, DataFormat::Float16_b, bf16_tile);
    make_cb(cb_scratch, 1, DataFormat::Float16_b, bf16_tile);
    // A full-width unmasked row is one W>=2 fast-untilize strip: accumulate KC tiles into cb_acc_strip,
    // untilize them into cb_out_strip. Dedicated CBs with uniform KC push/pop keep the fast packer's
    // KC-tile reads contiguous (mixing KC and 1-tile pushes on one ring wraps it mid-strip). Only
    // meaningful for KC >= 2; for KC == 1 the strip path is compiled out (the tiny CBs go unused).
    make_cb(cb_out_strip, 2 * KC, DataFormat::Float16_b, bf16_tile);
    make_cb(cb_acc_strip, 2 * KC, acc_fmt, acc_tile);

    // Reject an oversized config up front instead of failing CB allocation cryptically. The reserve
    // covers kernel binaries/stack/semaphores and (under tracy) the per-RISC profiler L1 buffers.
    constexpr uint64_t l1_reserve = 320 * 1024;
    const uint64_t l1_budget = (uint64_t)q.device()->l1_size_per_core() - l1_reserve;
    TT_FATAL(
        l1_used <= l1_budget,
        "indexer_score CBs need {} B but only {} B fit L1 (per-core {} B minus {} B reserve); "
        "reduce q_chunk_size, k_chunk_size, or head_group_size",
        l1_used,
        l1_budget,
        q.device()->l1_size_per_core(),
        l1_reserve);

    const std::vector<uint32_t> common_ct = {Hi, Sqt, Tt, Dt, chunk_t, QC, KC, HB};

    // Compute-ceiling measurement toggle: when INDEXER_DMA_OFF is set the reader/writer skip
    // their NoC reads/writes (but still push/pop CBs) so the compute kernel runs unstarved and
    // sp7_math_util reports the pure compute ceiling. Off (0) by default -> zero runtime cost.
    // Appended LAST in each kernel's CT args so the TensorAccessor offsets above are unaffected.
    // INDEXER_DMA_OFF_READER / INDEXER_DMA_OFF_WRITER disable just one side, to isolate how much
    // of the DMA gap (full kernel vs compute ceiling) is reader-bound vs writer-bound.
    const bool dma_off = std::getenv("INDEXER_DMA_OFF") != nullptr;
    const bool reader_off = dma_off || std::getenv("INDEXER_DMA_OFF_READER") != nullptr;
    const uint32_t writer_dma_off = (dma_off || std::getenv("INDEXER_DMA_OFF_WRITER") != nullptr) ? 1u : 0u;
    // Reader flag is a bitmask: bit0=q, bit1=k, bit2=w. INDEXER_READ_{Q,K,W}_OFF skip just one
    // input's NoC reads (still push the CB) to attribute the reader's exposed time per tensor.
    uint32_t reader_dma_off = reader_off ? 0b111u : 0u;
    if (std::getenv("INDEXER_READ_Q_OFF") != nullptr) {
        reader_dma_off |= 0b001u;
    }
    if (std::getenv("INDEXER_READ_K_OFF") != nullptr) {
        reader_dma_off |= 0b010u;
    }
    if (std::getenv("INDEXER_READ_W_OFF") != nullptr) {
        reader_dma_off |= 0b100u;
    }

    std::vector<uint32_t> reader_ct = common_ct;
    TensorAccessorArgs(*q.buffer()).append_to(reader_ct);
    TensorAccessorArgs(*k.buffer()).append_to(reader_ct);
    TensorAccessorArgs(*w.buffer()).append_to(reader_ct);
    reader_ct.push_back(reader_dma_off);
    // multicast: on/off per direction + the 6 semaphore ids (read at fixed offsets after dma_off)
    reader_ct.push_back(k_mcast_on);
    reader_ct.push_back(q_mcast_on);
    reader_ct.push_back(k_send_sem);
    reader_ct.push_back(k_recv_sem);
    reader_ct.push_back(k_valid_sem);
    reader_ct.push_back(q_send_sem);
    reader_ct.push_back(q_recv_sem);
    reader_ct.push_back(q_valid_sem);

    std::vector<uint32_t> writer_ct = common_ct;
    constexpr uint32_t out_elem_bytes = 2;    // bf16 output (compute_output_specs)
    writer_ct.push_back(T * out_elem_bytes);  // row-major page = one full row of T scores
    TensorAccessorArgs(*out.buffer()).append_to(writer_ct);
    writer_ct.push_back(writer_dma_off);

    std::vector<uint32_t> compute_ct = common_ct;
    compute_ct.push_back(qk_subblock_h);
    compute_ct.push_back(qk_batch_heads);  // head tiles per matmul/mul phase chunk
    compute_ct.push_back(qk_col_batch);    // k-columns batched per mode switch in the full-strip path

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
        std::vector<uint32_t> r = {q.buffer()->address(), k.buffer()->address(), w.buffer()->address(), flat, count};
        // Per-core multicast args (16): K column (8) then Q/W row (8). role 0=none(DRAM read),
        // 1=sender, 2=receiver. rect (xs,ys,xe,ye) + sender (sx,sy) are physical NoC; ndst=#receivers.
        const auto u32 = [](auto v) { return static_cast<uint32_t>(v); };
        const auto push8 =
            [&](uint32_t role, uint32_t xs, uint32_t ys, uint32_t xe, uint32_t ye, const CoreCoord& s, uint32_t ndst) {
                r.push_back(role);
                r.push_back(xs);
                r.push_back(ys);
                r.push_back(xe);
                r.push_back(ye);
                r.push_back(u32(s.x));
                r.push_back(u32(s.y));
                r.push_back(ndst);
            };
        if (grid_aligned) {
            const uint32_t x = i % grid.x, y = i / grid.x;
            // K column x: sender (x,0), receivers (x,1..); vertical rect spanning the column.
            uint32_t kys = u32(phys[cidx(x, 0)].y), kye = kys;
            for (uint32_t yy = 0; yy < grid.y; ++yy) {
                kys = std::min<uint32_t>(kys, u32(phys[cidx(x, yy)].y));
                kye = std::max<uint32_t>(kye, u32(phys[cidx(x, yy)].y));
            }
            push8(
                k_mcast_on ? (y == 0 ? 1u : 2u) : 0u,
                u32(phys[cidx(x, 0)].x),
                kys,
                u32(phys[cidx(x, 0)].x),
                kye,
                phys[cidx(x, 0)],
                u32(grid.y) - 1);
            // Q/W row y: sender (0,y), receivers (1..,y); horizontal rect spanning the row.
            uint32_t qxs = u32(phys[cidx(0, y)].x), qxe = qxs;
            for (uint32_t xx = 0; xx < grid.x; ++xx) {
                qxs = std::min<uint32_t>(qxs, u32(phys[cidx(xx, y)].x));
                qxe = std::max<uint32_t>(qxe, u32(phys[cidx(xx, y)].x));
            }
            push8(
                q_mcast_on ? (x == 0 ? 1u : 2u) : 0u,
                qxs,
                u32(phys[cidx(0, y)].y),
                qxe,
                u32(phys[cidx(0, y)].y),
                phys[cidx(0, y)],
                u32(grid.x) - 1);
        } else {
            for (uint32_t z = 0; z < 16; ++z) {
                r.push_back(0);
            }
        }
        SetRuntimeArgs(program, reader_id, cores[i], r);
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
