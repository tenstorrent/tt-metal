// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include "indexer_score_program_factory.hpp"

#include <algorithm>
#include <utility>
#include <vector>

#include <tt-metalium/host_api.hpp>
#include <tt-metalium/constants.hpp>
#include <tt-metalium/tensor_accessor_args.hpp>
#include <tt-metalium/work_split.hpp>

#include "ttnn/operations/transformer/sdpa/device/sdpa_subblock_utils.hpp"
#include "kernels/indexer_score_cb.hpp"          // shared host/device circular-buffer indices
#include "kernels/indexer_score_work_split.hpp"  // shared host/device causal work-split formula

namespace ttnn::operations::experimental::indexer_score::program {

using namespace tt;
using namespace tt::tt_metal;
using namespace tt::constants;

// Runtime-arg slots, shared by create()/override_runtime_arguments() and matched positionally
// by the kernels. Reader: q,k,w addrs; writer: out addr.
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

// Dense deal landed exactly on the grid (q/w mcast along rows, k down columns) and each direction's
// lines form contiguous NoC rects. 0 = direction off (per-core DRAM read); see call-site block.
struct McastPlan {
    bool grid_aligned = false;
    uint32_t k_mcast_on = 0;  // K columns are vertical NoC rects
    uint32_t q_mcast_on = 0;  // Q/W rows are single horizontal NoC rects (covers q and w)
};

// Pure analysis of the physical core coords: grid alignment + per-line NoC-rect contiguity for the
// two mcast directions. phys is indexed row-major (y * grid.x + x).
inline McastPlan compute_mcast_plan(
    CoreCoord grid,
    const std::vector<CoreCoord>& phys,
    uint32_t groups,
    uint32_t num_cores,
    uint32_t base,
    uint32_t rem,
    uint64_t total_units,
    uint32_t HB,
    uint32_t Hi) {
    const auto cidx = [&](uint32_t x, uint32_t y) { return y * grid.x + x; };
    // grid-aligned iff group g == grid row y and the row's grid_x cores evenly split its k-chunks
    // (units_per_group == grid.x * base, no remainder, full grid).
    const uint32_t units_per_group = groups > 0 ? (uint32_t)(total_units / groups) : 0;
    const bool grid_aligned = ttnn::operations::experimental::indexer_score::dense_schedule && groups == grid.y &&
                              num_cores == (uint32_t)(grid.x * grid.y) && rem == 0 && units_per_group == grid.x * base;

    // K columns must be vertical NoC rects: shared x down the column, contiguous y spanning the grid.
    bool k_cols_ok = grid_aligned;
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
    // Q/W row-mcast needs every core in a logical row on ONE physical NoC row (mcast = the row's
    // horizontal bounding-box rect) and all heads resident (one q block). x-contiguity not required
    // (the NoC routes the bbox rect); only a row spanning multiple physical NoC rows disables it.
    bool q_rows_ok = grid_aligned && HB == Hi;
    for (uint32_t y = 0; y < grid.y && q_rows_ok; ++y) {
        const uint32_t py = phys[cidx(0, y)].y;
        for (uint32_t x = 0; x < grid.x; ++x) {
            if (phys[cidx(x, y)].y != py) {
                q_rows_ok = false;
            }
        }
    }
    return {grid_aligned, k_cols_ok ? 1u : 0u, q_rows_ok ? 1u : 0u};
}

// Output-stationary flat deal of causal-valid work units. One unit = QC q-tile-rows x up-to-KC
// k-tiles, dealt evenly across cores row-major (kernels invert the flat index). Heads stream in
// HB-head groups; fully-future tiles get the full -inf mask, row tails -inf-filled by the writer
// (zeros unsafe: gates can be negative).
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

    // Work-unit knobs, converted from elements to tiles / heads.
    const auto& cfg = args.program_config;
    const uint32_t QC = cfg.q_chunk_size / TILE_HEIGHT;
    const uint32_t KC = cfg.k_chunk_size / TILE_WIDTH;
    const uint32_t HB = resolve_head_group(cfg, Hi);

    // qk matmul subblock: heads are output rows, k column is 1 tile wide (SDPA-style), so only the
    // subblock height (head rows) is needed.
    constexpr bool fp32_dest_acc_en = false;                 // bf16 DEST: 8-head subblocks in half-sync
    constexpr uint32_t dst_size = fp32_dest_acc_en ? 4 : 8;  // half-sync, as in sdpa_program_factory
    const uint32_t qk_subblock_h = ttnn::prim::detail::determine_largest_subblock_size(HB, 1, dst_size).first;
    TT_FATAL(HB % qk_subblock_h == 0, "head group {} must be divisible by qk_subblock_h={}", HB, qk_subblock_h);

    // QC/KC/HB are verbatim from the config -- no auto-tune; the caller owns the perf trade-off (see
    // glx_config() in the test). Oversized configs hit the L1-fit check below, not a silent clamp.

    // total valid work units V (groups exact: validate guarantees QC divides Sqt). units_in_group is
    // the shared formula the kernels' WorkUnitSpan inverts.
    const uint32_t groups = Sqt / QC;
    uint64_t total_units = 0;
    for (uint32_t g = 0; g < groups; ++g) {
        total_units += units_in_group(g, QC, KC, chunk_t, Tt);
    }

    const auto grid = q.device()->compute_with_storage_grid_size();
    const uint32_t num_cores = std::min<uint64_t>(total_units, (uint64_t)grid.x * grid.y);
    const auto core_ranges = num_cores_to_corerangeset(num_cores, grid, true);
    const auto cores = corerange_to_cores(core_ranges, num_cores, true);

    const uint32_t base = total_units / num_cores;
    const uint32_t rem = total_units % num_cores;

    // ---- grid-aligned multicast (decoupled Q/W along rows, K down columns) -------------------
    // When the dense deal lands exactly on the grid, a grid ROW shares identical q/w and a grid COLUMN
    // shares the identical k-band: one core per row mcasts q/w along it, one per column mcasts k down
    // it, killing ~grid_x q/w re-reads + ~grid_y k re-reads. Each direction is independent, enabled only
    // if its lines are contiguous NoC rects; else that input falls back to per-core DRAM reads.
    std::vector<CoreCoord> phys(num_cores);
    for (uint32_t i = 0; i < num_cores; ++i) {
        phys[i] = q.device()->worker_core_from_logical_core(cores[i]);
    }
    const McastPlan plan = compute_mcast_plan(grid, phys, groups, num_cores, base, rem, total_units, HB, Hi);
    const bool grid_aligned = plan.grid_aligned;
    const uint32_t k_mcast_on = plan.k_mcast_on;
    const uint32_t q_mcast_on = plan.q_mcast_on;

    auto cidx = [&](uint32_t x, uint32_t y) { return y * grid.x + x; };
    // Physical NoC bounding box of one grid column / row, used to build the multicast rects below.
    auto phys_col_y_range = [&](uint32_t x) {  // [min y, max y] down grid column x
        uint32_t lo = static_cast<uint32_t>(phys[cidx(x, 0)].y), hi = lo;
        for (uint32_t y = 0; y < grid.y; ++y) {
            lo = std::min<uint32_t>(lo, static_cast<uint32_t>(phys[cidx(x, y)].y));
            hi = std::max<uint32_t>(hi, static_cast<uint32_t>(phys[cidx(x, y)].y));
        }
        return std::pair<uint32_t, uint32_t>{lo, hi};
    };
    auto phys_row_x_range = [&](uint32_t y) {  // [min x, max x] across grid row y
        uint32_t lo = static_cast<uint32_t>(phys[cidx(0, y)].x), hi = lo;
        for (uint32_t x = 0; x < grid.x; ++x) {
            lo = std::min<uint32_t>(lo, static_cast<uint32_t>(phys[cidx(x, y)].x));
            hi = std::max<uint32_t>(hi, static_cast<uint32_t>(phys[cidx(x, y)].x));
        }
        return std::pair<uint32_t, uint32_t>{lo, hi};
    };

    // 3 semaphores per active direction: send (receivers ready), recv (sender relays valid in), valid
    // (constant 1, relay source). Mirrors SDPA chain_link's handshake.
    const uint32_t k_send_sem = k_mcast_on ? CreateSemaphore(program, core_ranges, 0) : 0;
    const uint32_t k_recv_sem = k_mcast_on ? CreateSemaphore(program, core_ranges, 0) : 0;
    const uint32_t k_valid_sem = k_mcast_on ? CreateSemaphore(program, core_ranges, 1) : 0;
    const uint32_t q_send_sem = q_mcast_on ? CreateSemaphore(program, core_ranges, 0) : 0;
    const uint32_t q_recv_sem = q_mcast_on ? CreateSemaphore(program, core_ranges, 0) : 0;
    const uint32_t q_valid_sem = q_mcast_on ? CreateSemaphore(program, core_ranges, 1) : 0;

    const uint32_t bf16_tile = tile_size(DataFormat::Float16_b);
    const uint32_t fp32_tile = tile_size(DataFormat::Float32);

    // q (srcB) and k (srcA) are matmul inputs; either may be bfp8_b to halve its DRAM/L1 footprint
    // (w stays bf16). The format flows into the CB; the compute kernel needs no change.
    const bool q_is_bfp8 = q.dtype() == DataType::BFLOAT8_B;
    const bool k_is_bfp8 = k.dtype() == DataType::BFLOAT8_B;
    const DataFormat q_fmt = q_is_bfp8 ? DataFormat::Bfp8_b : DataFormat::Float16_b;
    const DataFormat k_fmt = k_is_bfp8 ? DataFormat::Bfp8_b : DataFormat::Float16_b;
    const uint32_t q_tile = tile_size(q_fmt);
    const uint32_t k_tile = tile_size(k_fmt);
    // Both matmul inputs bfp8 -> LoFi (a single fidelity phase, 2x matmul peak). Any bf16 input keeps
    // HiFi2 so the bf16 mantissa is not thrown away by a 1-phase pass.
    const MathFidelity math_fidelity = (q_is_bfp8 && k_is_bfp8) ? MathFidelity::LoFi : MathFidelity::HiFi2;

    uint64_t l1_used = 0;
    auto make_cb = [&](uint32_t idx, uint32_t ntiles, DataFormat fmt, uint32_t tile_bytes) {
        l1_used += (uint64_t)ntiles * tile_bytes;
        CreateCircularBuffer(
            program,
            core_ranges,
            CircularBufferConfig(ntiles * tile_bytes, {{idx, fmt}}).set_page_size(idx, tile_bytes));
    };

    const bool stream_heads = HB < Hi;

    // CB indices come from shared kernels/indexer_score_cb.hpp (factory and kernels can't disagree
    // on which index is which buffer); sizes are set here.
    make_cb(cb_q, (stream_heads ? 2 : 1) * HB * QC * Dt, q_fmt, q_tile);
    make_cb(cb_k, 2 * KC * Dt, k_fmt, k_tile);
    make_cb(cb_w, Hi * QC, DataFormat::Float16_b, bf16_tile);
    make_cb(cb_mask, num_mask_tiles, DataFormat::Float16_b, bf16_tile);
    const DataFormat acc_fmt = fp32_dest_acc_en ? DataFormat::Float32 : DataFormat::Float16_b;
    const uint32_t acc_tile = fp32_dest_acc_en ? fp32_tile : bf16_tile;
    // cb_qk buffers a batch of the group's relu(q.kT) tiles so compute runs that batch's matmuls then
    // its mul+accumulates, hoisting the matmul<->eltwise reinit out of the per-head-pass loop.
    // QC==1 has spare L1 -> batch the whole group; QC>1 doubles cb_q/cb_w, so cap at 32.
    const uint32_t qk_batch_cap = (QC == 1) ? HB : 32u;
    const uint32_t qk_batch_heads = std::min<uint32_t>(HB, qk_batch_cap);  // multiple of qk_subblock_h
    // The compute kernel walks HB in qk_batch_heads-sized chunks (chunk += qk_batch_heads), so HB must
    // be a whole multiple or the last chunk over-reads past the resident head group. Only reachable when
    // the 32-cap engages (HB > 32 && QC > 1); the deployed cases never hit it, but guard loudly.
    TT_FATAL(
        HB % qk_batch_heads == 0,
        "head_group {} not divisible by qk_batch_heads {} (QC>1 with HB>32); reduce head_group_size or q_chunk_size",
        HB,
        qk_batch_heads);
    // Full-strip path batches qk_col_batch k-columns per matmul<->mul mode switch (one switch per batch,
    // not per output tile): w is column-independent and the whole k chunk is resident, so a row's columns
    // share one switch. Only when the group is a single chunk (qk_batch_heads == Hi) and the fast strip
    // is used (KC >= 2); cb_qk then holds qk_col_batch * qk_batch_heads tiles, capped to an L1 budget.
    // Tile budget for cb_qk (the batched matmul outputs). This is really an L1-fit proxy: cb_qk holds
    // qk_col_batch * qk_batch_heads tiles, and 128 is ~the L1 left over once cb_q/cb_k/cb_w (large at
    // high head counts) are resident. It self-tunes to ~that ceiling: heads8
    // (8 heads, KC=16) lands at the full chunk 8*16=128; heads64 (64 heads) throttles to qk_col_batch=2
    // (2*64=128) -- without the cap it would pick KC and overflow L1. 128 tiles == 256 KB at bf16
    // acc_fmt (the only mode today); if fp32-dest acc is ever enabled this would be 512 KB, so prefer
    // deriving the byte budget from acc_tile if that path is added.
    constexpr uint32_t qk_col_tile_cap = 128;
    const bool single_chunk = (qk_batch_heads == Hi) && !stream_heads;
    const uint32_t qk_col_batch = (KC >= 2 && single_chunk)
                                      ? std::min<uint32_t>(KC, std::max<uint32_t>(1u, qk_col_tile_cap / qk_batch_heads))
                                      : 1u;
    make_cb(cb_qk, qk_col_batch * qk_batch_heads, acc_fmt, acc_tile);
    make_cb(cb_scratch, 1, DataFormat::Float16_b, bf16_tile);
    // cb_out_strip holds a unit's untilized output. Uniform KC push/pop keeps the packer's KC-tile
    // reads contiguous (a non-uniform push would wrap the ring mid-strip).
    make_cb(cb_out_strip, 2 * KC, DataFormat::Float16_b, bf16_tile);
    // cb_acc_strip accumulates a whole unit's QC*KC strip, then all QC strips untilize under ONE
    // pack_untilize bracket (per-strip cost amortizes over QC*KC, not KC). max(2*KC, .) keeps the
    // QC<=2 double buffer and a whole multiple of the QC*KC batch so a uniform push never wraps mid-unit.
    make_cb(cb_acc_strip, std::max(2u * KC, QC * KC), acc_fmt, acc_tile);

    // Reject an oversized config up front instead of a cryptic CB-alloc failure. Reserve covers kernel
    // binaries/stack/semaphores and (under tracy) the per-RISC profiler L1 buffers.
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

    std::vector<uint32_t> reader_ct = common_ct;
    TensorAccessorArgs(*q.buffer()).append_to(reader_ct);
    TensorAccessorArgs(*k.buffer()).append_to(reader_ct);
    TensorAccessorArgs(*w.buffer()).append_to(reader_ct);
    // multicast: on/off per direction (q_mcast_on covers q and w) then the 6 semaphore ids, at fixed
    // offsets right after the three TensorAccessors.
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

    std::vector<uint32_t> compute_ct = common_ct;
    compute_ct.push_back(qk_subblock_h);
    compute_ct.push_back(qk_batch_heads);  // head tiles per matmul/mul phase chunk
    compute_ct.push_back(qk_col_batch);    // k-columns batched per mode switch in the full-strip path

    const std::string kdir = "ttnn/cpp/ttnn/operations/experimental/indexer_score/device/kernels/";
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

    // Per-core multicast runtime args: K column then Q/W row, each an 8-tuple (push8 below); the two
    // together are what the non-grid-aligned fallback zero-fills.
    constexpr uint32_t mcast_args_per_dir = 8;
    constexpr uint32_t reader_mcast_args = 2 * mcast_args_per_dir;

    uint32_t flat = 0;
    for (uint32_t i = 0; i < num_cores; ++i) {
        const uint32_t count = base + (i < rem ? 1 : 0);
        std::vector<uint32_t> reader_rt = {
            q.buffer()->address(), k.buffer()->address(), w.buffer()->address(), flat, count};
        // Per-core mcast args: K column (8) then Q/W row (8). role 0=none(DRAM read), 1=sender,
        // 2=receiver. rect (xs,ys,xe,ye) + sender (sx,sy) are physical NoC; ndst=#receivers.
        const auto u32 = [](auto v) { return static_cast<uint32_t>(v); };
        const auto push8 =
            [&](uint32_t role, uint32_t xs, uint32_t ys, uint32_t xe, uint32_t ye, const CoreCoord& s, uint32_t ndst) {
                reader_rt.push_back(role);
                reader_rt.push_back(xs);
                reader_rt.push_back(ys);
                reader_rt.push_back(xe);
                reader_rt.push_back(ye);
                reader_rt.push_back(u32(s.x));
                reader_rt.push_back(u32(s.y));
                reader_rt.push_back(ndst);
            };
        if (grid_aligned) {
            const uint32_t x = i % grid.x, y = i / grid.x;
            // K column x: sender (x,0), receivers (x,1..); vertical rect spanning the column.
            const auto [kys, kye] = phys_col_y_range(x);
            push8(
                k_mcast_on ? (y == 0 ? 1u : 2u) : 0u,
                u32(phys[cidx(x, 0)].x),
                kys,
                u32(phys[cidx(x, 0)].x),
                kye,
                phys[cidx(x, 0)],
                u32(grid.y) - 1);
            // Q/W row y: ONE sender per row on the grid DIAGONAL (logical x == y), mcasting the WHOLE
            // row in one rect (its bbox [min x, max x] x py). Any core in the row can send (all share
            // q-rows + w); the diagonal fans senders across distinct columns vs stacking in column 0.
            // Receivers = rest of the row (ndst = grid.x - 1); hardware excludes the in-rect source.
            const auto [qxs, qxe] = phys_row_x_range(y);
            const uint32_t py = u32(phys[cidx(y, y)].y);  // diagonal sender's row (== every core's py)
            push8(q_mcast_on ? (x == y ? 1u : 2u) : 0u, qxs, py, qxe, py, phys[cidx(y, y)], u32(grid.x) - 1);
        } else {
            for (uint32_t z = 0; z < reader_mcast_args; ++z) {
                reader_rt.push_back(0);
            }
        }
        SetRuntimeArgs(program, reader_id, cores[i], reader_rt);
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
        auto& reader_rt = reader_args[core.x][core.y];
        patch_arg(reader_rt, rt_arg::reader_q_addr, tensors.q.buffer()->address(), "reader.q_addr");
        patch_arg(reader_rt, rt_arg::reader_k_addr, tensors.k.buffer()->address(), "reader.k_addr");
        patch_arg(reader_rt, rt_arg::reader_w_addr, tensors.weights.buffer()->address(), "reader.w_addr");
        patch_arg(writer_args[core.x][core.y], rt_arg::writer_out_addr, out.buffer()->address(), "writer.out_addr");
    }
}

}  // namespace ttnn::operations::experimental::indexer_score::program
