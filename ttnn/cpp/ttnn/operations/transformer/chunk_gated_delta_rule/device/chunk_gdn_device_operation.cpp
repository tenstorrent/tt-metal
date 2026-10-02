// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
// SPDX-License-Identifier: Apache-2.0

#include "chunk_gdn_device_operation.hpp"
#include "kernels/dataflow/chunk_gdn_fused_map.hpp"

#include <algorithm>
#include <cmath>
#include <utility>
#include <variant>

#include <tt-metalium/constants.hpp>
#include "ttnn/device_operation.hpp"
#include "ttnn/tensor/tensor.hpp"

using namespace tt::tt_metal;

namespace ttnn::prim {

namespace {
// Uniquely named (vs the phased prims' `check`) so it does not clash under unity builds.
void check_gdn_tensor(const Tensor& t, const char* name, DataType dt) {
    TT_FATAL(t.layout() == Layout::TILE, "chunk_gdn: {} must be TILE layout", name);
    TT_FATAL(t.dtype() == dt, "chunk_gdn: {} has wrong dtype", name);
    TT_FATAL(t.buffer() != nullptr, "chunk_gdn: {} must be on device", name);
}
}  // namespace

ChunkGdnDeviceOperation::program_factory_t ChunkGdnDeviceOperation::select_program_factory(
    const operation_attributes_t& attrs, const tensor_args_t&) {
    if (attrs.impl == ChunkGdnImpl::Mono) {
        return ChunkGdnMonoProgramFactory{};
    }
    return ChunkGdnFusedProgramFactory{};
}

void ChunkGdnDeviceOperation::validate_on_program_cache_miss(
    const operation_attributes_t& attrs, const tensor_args_t& in) {
    using namespace tt::constants;
    // Input contract shared by both factories: the mono kernel and the fused producers read the same
    // head-major tensors (identical to the phased PREP prim's contract, whose reader/compute the fused
    // producer runs unchanged; the flat forms are the prep reader's and therefore Fused-only).
    check_gdn_tensor(in.q, "q", DataType::BFLOAT16);
    check_gdn_tensor(in.k, "k", DataType::BFLOAT16);
    check_gdn_tensor(in.v, "v", DataType::BFLOAT16);  // flat [B,T,HV*V] when attrs.v_flat; else [BH,NC,C,V]
    if (attrs.v_flat) {
        TT_FATAL(attrs.HV > 0, "v_flat requires HV > 0");
        const auto& vs = in.v.logical_shape();
        TT_FATAL(vs.rank() == 3, "v_flat expects a flat [B,T,HV*V] v (got rank {})", vs.rank());
        TT_FATAL(vs[2] == attrs.HV * attrs.val_dim, "v_flat width {} != HV*V ({}*{})", vs[2], attrs.HV, attrs.val_dim);
    }
    if (attrs.qk_flat) {
        TT_FATAL(attrs.Hk > 0, "qk_flat requires Hk > 0");
        const auto& qsf = in.q.logical_shape();
        TT_FATAL(qsf.rank() == 3, "qk_flat expects a flat [B,T,Hk*K] q (got rank {})", qsf.rank());
        TT_FATAL(
            qsf[2] == attrs.Hk * attrs.key_dim, "qk_flat width {} != Hk*K ({}*{})", qsf[2], attrs.Hk, attrs.key_dim);
        TT_FATAL(attrs.qk_norm, "qk_flat requires qk_norm (flat q/k are unnormalized; norm is in-kernel)");
    }
    check_gdn_tensor(in.g, "g", DataType::FLOAT32);
    check_gdn_tensor(in.beta, "beta", DataType::FLOAT32);
    check_gdn_tensor(in.eye_c, "eye_c", DataType::FLOAT32);
    check_gdn_tensor(in.tril_c, "tril_c", DataType::FLOAT32);
    check_gdn_tensor(in.ones_c, "ones_c", DataType::FLOAT32);
    check_gdn_tensor(in.masks_c, "masks_c", DataType::FLOAT32);
    // Required: the mono reader and the fused receivers stream S from this buffer unconditionally (no
    // in-kernel zeroing); the public op builds a zero state when its own initial_state is omitted.
    check_gdn_tensor(in.initial_state, "initial_state", DataType::FLOAT32);
    TT_FATAL(attrs.chunk_size % TILE_HEIGHT == 0, "chunk_size must be a multiple of 32");
    TT_FATAL(attrs.key_dim % TILE_WIDTH == 0, "key_dim must be a multiple of 32");
    TT_FATAL(attrs.val_dim % TILE_WIDTH == 0, "val_dim must be a multiple of 32");
    validate_gdn_tinv(attrs.tinv, attrs.chunk_size, in.q);

    const auto grid = in.q.device()->compute_with_storage_grid_size();
    if (attrs.impl == ChunkGdnImpl::Mono) {
        // One core per head running the single-kernel program on head-major inputs.
        TT_FATAL(
            !attrs.v_flat && !attrs.qk_flat && !attrs.qk_norm,
            "chunk_gdn: the mono program takes head-major q/k/v only (no flat inputs, no in-kernel qk-norm)");
        TT_FATAL(
            attrs.BH <= grid.x * grid.y,
            "chunk_gdn: the mono program needs one core per head: BH={} exceeds the {}x{} grid",
            attrs.BH,
            grid.x,
            grid.y);
        return;
    }
    // Fused geometry: NV receivers per head and NP producers per head (placements 0/1) or a pool of NP
    // (placement 2). A head's receivers form a dense rectangle (the multicast target). Placement 0 needs
    // BH 1xNV row rectangles (BH <= (grid.x / NV) * grid.y) and puts the producers anywhere; placement 1
    // needs a row-local layout (fused_row_local_feasible), placement 2 a pool that fits
    // (fused_pool_feasible). The fields come from ChunkGdnFusedProgramConfig, or from the cost model when
    // unset (chunk_gdn below).
    TT_FATAL(attrs.np >= 1, "chunk_gdn_fused: np must be >= 1 (got {})", attrs.np);
    TT_FATAL(attrs.nv >= 1, "chunk_gdn_fused: nv must be >= 1 (got {})", attrs.nv);
    const uint32_t Vt = attrs.val_dim / TILE_WIDTH;
    TT_FATAL(Vt % attrs.nv == 0, "chunk_gdn_fused: nv ({}) must divide Vt ({})", attrs.nv, Vt);
    TT_FATAL(attrs.nv <= grid.x, "chunk_gdn_fused: nv ({}) exceeds the grid width {}", attrs.nv, grid.x);
    // Per-(head, slot) credit words live in one 4 KB tile of the u/mask CB.
    TT_FATAL(
        attrs.BH * attrs.nbuf <= 1024,
        "chunk_gdn_fused: BH * nbuf ({} * {}) credit words exceed the 1024-word credit tile",
        attrs.BH,
        attrs.nbuf);
    if (attrs.placement == 2) {
        // Producer pool: np is the pool size; it needs a home producer per head in a row-local layout.
        TT_FATAL(
            fused_pool_feasible(grid.x, grid.y, attrs.BH, attrs.nv, attrs.np),
            "chunk_gdn_fused: a producer pool of {} for BH={} NV={} does not fit the {}x{} grid (needs BH*NV + P "
            "cores and a row-local layout with a home producer per head)",
            attrs.np,
            attrs.BH,
            attrs.nv,
            grid.x,
            grid.y);
        TT_FATAL(
            attrs.np <= attrs.BH * attrs.num_chunks,
            "chunk_gdn_fused: a producer pool of {} exceeds the BH*NC = {} items",
            attrs.np,
            attrs.BH * attrs.num_chunks);
        TT_FATAL(
            attrs.pool_extra_den >= 1 && attrs.pool_extra_num <= attrs.pool_extra_den,
            "chunk_gdn_fused: the extras' share {}/{} is not a fraction in [0, 1]",
            attrs.pool_extra_num,
            attrs.pool_extra_den);
        return;
    }
    if (attrs.placement == 0) {
        const uint32_t hpr = grid.x / attrs.nv;
        TT_FATAL(
            attrs.BH <= hpr * grid.y,
            "chunk_gdn_fused: BH={} heads need 1x{} receiver rectangles; the {}x{} grid holds only {} ({} per row)",
            attrs.BH,
            attrs.nv,
            grid.x,
            grid.y,
            hpr * grid.y,
            hpr);
    } else {
        // Row-local: L = NV+NP cores per head in one row; heads beyond grid.y go to the
        // leftover W-L columns as blocks of NV receivers (rw x rh rectangle) + NP producers.
        const uint32_t L = attrs.nv + attrs.np;
        TT_FATAL(L <= grid.x, "chunk_gdn_fused: row-local placement needs NV+NP={} <= grid.x={}", L, grid.x);
        const uint32_t k_per_row = grid.x / L;
        if (attrs.BH > k_per_row * grid.y) {
            const uint32_t rem = attrs.BH - k_per_row * grid.y;
            const uint32_t wl = grid.x - k_per_row * L;
            TT_FATAL(
                wl >= 1,
                "chunk_gdn_fused: row-local placement: {} heads exceed the {} rows and no columns are left",
                attrs.BH,
                grid.y);
            const uint32_t rw = std::min<uint32_t>(attrs.nv, wl);
            TT_FATAL(
                attrs.nv % rw == 0,
                "chunk_gdn_fused: row-local placement: NV={} is not a multiple of the leftover width {}",
                attrs.nv,
                rw);
            const uint32_t block_h = attrs.nv / rw + (attrs.np + wl - 1) / wl;
            TT_FATAL(
                rem * block_h <= grid.y,
                "chunk_gdn_fused: row-local placement: {} leftover heads need {} rows of the {}-column block, grid has "
                "{}",
                rem,
                rem * block_h,
                wl,
                grid.y);
        }
    }
    TT_FATAL(
        attrs.BH * (attrs.nv + attrs.np) <= grid.x * grid.y,
        "chunk_gdn_fused needs BH*(NV+NP) = {}*({}+{}) = {} cores, grid has {}x{}={}",
        attrs.BH,
        attrs.nv,
        attrs.np,
        attrs.BH * (attrs.nv + attrs.np),
        grid.x,
        grid.y,
        grid.x * grid.y);
}

ChunkGdnDeviceOperation::spec_return_value_t ChunkGdnDeviceOperation::compute_output_specs(
    const operation_attributes_t& attrs, const tensor_args_t&) {
    // o: bf16 from the mono kernel (its cb_out format), fp32 from the fused program — EXACTLY the phased
    // scan's spec (a bf16 o degraded full-model quality there and was removed). final_state is fp32 on both.
    const DataType o_dtype = attrs.impl == ChunkGdnImpl::Mono ? DataType::BFLOAT16 : DataType::FLOAT32;
    const auto o_layout = TensorLayout(o_dtype, PageConfig(Layout::TILE), attrs.output_mem_config);
    const auto s_layout = TensorLayout(DataType::FLOAT32, PageConfig(Layout::TILE), attrs.output_mem_config);
    ttnn::Shape o_shape({attrs.BH, attrs.num_chunks, attrs.chunk_size, attrs.val_dim});
    ttnn::Shape s_shape({attrs.BH, attrs.key_dim, attrs.val_dim});
    return {tt::tt_metal::TensorSpec(o_shape, o_layout), tt::tt_metal::TensorSpec(s_shape, s_layout)};
}

ChunkGdnDeviceOperation::tensor_return_value_t ChunkGdnDeviceOperation::create_output_tensors(
    const operation_attributes_t& attrs, const tensor_args_t& in) {
    auto specs = compute_output_specs(attrs, in);
    auto* device = in.q.device();
    std::vector<Tensor> outs;
    outs.reserve(specs.size());
    for (const auto& spec : specs) {
        outs.push_back(create_device_tensor(spec, device));
    }
    return outs;
}

// ---------------------------------------------------------------------------------------------------
// Fused geometry cost model and placement
// ---------------------------------------------------------------------------------------------------
namespace {
// QB2 constants, re-measured 2026-10-02 (Tracy device time, T=2048 -> NC=64, C=32, K=V=128, medians of ops 2-5)
// on the producer-lever stack.
constexpr float kWpUs = 15.98f;          // producer item period; flat over BH and the producer count
constexpr float kChainSlopeUs = 0.012f;  // receiver chunk period growth per head (shared DRAM / NoC)
constexpr float kSkewAUs = 4.0f;         // slowest chain behind the median chain: A + B*BH
constexpr float kSkewBUs = 0.5f;
constexpr float kTailUs = 4.0f;          // last chunk consumed -> kernel end
constexpr float kFillAUs = 19.5f;        // first scan step done: A + min(BH, 24) * (B + C * producers)
constexpr float kFillBUs = 0.72f;
constexpr float kFillCUs = 1.0f / 144.0f;
constexpr float kRowMajorPaceUs = 8.4f;  // link-bound chunk period of the row-major placement
// Balance bumps: two bounds within `width` of each other expose each side's jitter to the other, up to `peak`
// when equal. Chain vs producers at Vtl <= 2 (narrower at depth >= 3); home vs extra producers of a pool.
constexpr float kBalancePeak = 0.08f;
constexpr float kBalanceWidthD2 = 0.25f;
constexpr float kBalanceWidthD3 = 0.15f;
constexpr float kPoolBalancePeak = 0.25f;
constexpr float kPoolBalanceWidth = 0.25f;
constexpr float kPoolStartUs = 20.0f;     // pool, Vtl <= 2: the home producers' first items end later, by
constexpr float kPoolStartShare = 0.25f;  // kPoolStartUs * min(1, share / kPoolStartShare)
constexpr float kPoolSkewUs = 4.0f;       // extra chain skew of a pool
constexpr float kPoolCreditD2Us = 0.025f;  // pool at depth 2, Vtl <= 2: credit round trip exposed per step, per head
constexpr uint32_t kHandoffTiles = 19;       // fp32 tiles per hand-off slot (C=32, K=V=128)
constexpr uint32_t kProducerPrepTiles = 48;  // the producer's prep CBs, in fp32-tile units
constexpr uint32_t kTileBytes = 4096;
constexpr uint32_t kL1BudgetBytes = 1400u * 1024u;  // Wormhole's 1464 KB less the system map and the small region
constexpr uint32_t kNvCandidates[] = {1, 2, 4, 8};  // receivers per head the model considers (those dividing Vt)
constexpr float fill_us(uint32_t BH, uint32_t producers) {
    return kFillAUs + std::min<uint32_t>(BH, 24) * (kFillBUs + kFillCUs * producers);
}
// Receiver step per V-slice width, before the per-head slope. Vtl=1 is round-trip bound (it computes in ~1.4 us);
// a pool's extras lengthen that round trip.
constexpr float t_step_us(uint32_t Vtl, bool pooled = false) {
    switch (Vtl) {
        case 1: return pooled ? 4.25f : 3.17f;
        case 2: return 2.69f;
        case 4: return 4.45f;
        default: return -1.0f;  // unmeasured width
    }
}
constexpr float balance_us(float a, float b, float width, float peak) {
    if (a <= 0.0f || b <= 0.0f) {
        return 0.0f;
    }
    const float m = std::max(a, b);
    return peak * std::max(0.0f, 1.0f - std::abs(a - b) / (width * m));
}
// Items of the busiest home producer and of the busiest extra under the shared item map.
struct PoolLoad {
    uint32_t n_home;
    uint32_t n_extra;
};
PoolLoad pool_load(uint32_t BH, uint32_t NC, uint32_t NPH, uint32_t NX, uint32_t num, uint32_t den) {
    const GdnFusedMap m{BH, NC, NPH, NX, num, den};
    uint32_t nh = 0;
    for (uint32_t h = 0; h < BH; h++) {
        nh = std::max(nh, gdn_fused_n_home_chunks(m, h));
    }
    const uint32_t ne = gdn_fused_n_extra_items(m);
    return {(nh + NPH - 1) / NPH, NX != 0 ? (ne + NX - 1) / NX : 0u};
}
// Device time of one geometry: the pipeline fill, then the slowest of the chain (NC-1 steps plus the head skew),
// the busiest home producer's remaining items and the busiest extra's, stretched by the balance bumps, then the
// tail. `producers` is the total producer count; `pooled` = a pool with extras, `share` their share of the items.
float t_fused_us(
    uint32_t BH,
    uint32_t NC,
    uint32_t Vtl,
    uint32_t depth,
    uint32_t placement,
    uint32_t producers,
    PoolLoad load,
    bool pooled,
    float share) {
    float pace = t_step_us(Vtl, pooled) + kChainSlopeUs * BH;
    if (pooled && Vtl <= 2 && depth <= 2) {
        pace += kPoolCreditD2Us * BH;
    }
    const float H = (load.n_home - 1) * kWpUs;
    const float X = load.n_extra ? (load.n_extra - 1) * kWpUs : 0.0f;
    // Row-major: the receiver rows' shared links bound the chain unless it is well production-bound.
    if (placement == 0 && std::max(H, X) < 2.0f * (NC - 1) * pace) {
        pace = std::max(pace, kRowMajorPaceUs);
    }
    const float C = (NC - 1) * pace + kSkewAUs + kSkewBUs * BH + (pooled ? kPoolSkewUs : 0.0f);
    float pen = 0.0f;
    float start = 0.0f;
    if (Vtl <= 2) {
        pen = balance_us(C, H, depth <= 2 ? kBalanceWidthD2 : kBalanceWidthD3, kBalancePeak);
        if (pooled) {
            // The home/extra balance matters only where the producers bound the run.
            const float m = std::max({C, H, X});
            const float g = std::max(0.0f, 1.0f - (m - std::max(H, X)) / (kPoolBalanceWidth * m));
            pen = std::max(pen, g * balance_us(H, X, kPoolBalanceWidth, kPoolBalancePeak));
            start = kPoolStartUs * std::min(1.0f, share / kPoolStartShare);
        }
    }
    return fill_us(BH, producers) + std::max({C, H + start, X}) * (1.0f + pen) + kTailUs;
}
// L1 at hand-off depth `depth` on the fuller side: the slots, the 4-tile u/credit CB and the larger of
// the receiver's scan CBs at the slice width (20*Vtl + 1 tiles) and the producer's prep CBs.
constexpr bool handoff_fits_l1(uint32_t Vtl, uint32_t depth) {
    const uint32_t tiles = kHandoffTiles * depth + 4 + std::max(20 * Vtl + 1, kProducerPrepTiles);
    return tiles * kTileBytes <= kL1BudgetBytes;
}
// Measured phased device time (prep + scan) at NC=64, interpolated linearly in BH; beyond the table scaled by BH/48.
constexpr float kPhasedBH[] = {4.0f, 8.0f, 12.0f, 16.0f, 32.0f, 48.0f};
constexpr float kPhasedUs[] = {359.8f, 461.5f, 603.0f, 769.4f, 1278.8f, 1907.9f};
constexpr float t_phased_us(uint32_t BH, uint32_t NC) {
    constexpr size_t n = sizeof(kPhasedBH) / sizeof(kPhasedBH[0]);
    const float bh = std::min<float>(BH, kPhasedBH[n - 1]);
    size_t i = 0;
    while (i + 2 < n && bh > kPhasedBH[i + 1]) {
        i++;
    }
    const float w = (bh - kPhasedBH[i]) / (kPhasedBH[i + 1] - kPhasedBH[i]);
    float t = kPhasedUs[i] + w * (kPhasedUs[i + 1] - kPhasedUs[i]);
    if (BH > kPhasedBH[n - 1]) {
        t *= BH / kPhasedBH[n - 1];
    }
    return t * NC / 64.0f;
}
}  // namespace

bool fused_row_local_feasible(uint32_t grid_x, uint32_t grid_y, uint32_t BH, uint32_t NV, uint32_t NP) {
    const uint32_t L = NV + NP;
    if (NV < 1 || NP < 1 || L > grid_x) {
        return false;
    }
    const uint32_t k = grid_x / L;
    if (BH <= k * grid_y) {
        return true;
    }
    const uint32_t rem = BH - k * grid_y;
    const uint32_t wl = grid_x - k * L;
    if (wl < 1) {
        return false;
    }
    const uint32_t rw = std::min<uint32_t>(NV, wl);
    if (NV % rw != 0) {
        return false;
    }
    return rem * (NV / rw + (NP + wl - 1) / wl) <= grid_y;
}

uint32_t fused_pool_home_producers(uint32_t grid_x, uint32_t grid_y, uint32_t BH, uint32_t NV, uint32_t P) {
    if (BH < 1 || NV < 1) {
        return 0;
    }
    for (uint32_t nph = std::min<uint32_t>(P / BH, grid_x); nph >= 1; nph--) {
        if (fused_row_local_feasible(grid_x, grid_y, BH, NV, nph)) {
            return nph;
        }
    }
    return 0;
}

bool fused_pool_feasible(uint32_t grid_x, uint32_t grid_y, uint32_t BH, uint32_t NV, uint32_t P) {
    return NV >= 1 && P >= 1 && BH * NV + P <= grid_x * grid_y &&
           fused_pool_home_producers(grid_x, grid_y, BH, NV, P) >= 1;
}

FusedPlacement fused_placement(
    uint32_t grid_x, uint32_t grid_y, uint32_t BH, uint32_t NV, uint32_t NP, uint32_t placement) {
    const uint32_t n_cores = grid_x * grid_y;
    TT_FATAL(NV <= grid_x, "chunk_gdn_fused: nv={} exceeds the grid width {}", NV, grid_x);
    const uint32_t HPR = grid_x / NV;  // heads per receiver row (placement 0)
    const uint32_t R = BH * NV;        // receiver cores
    // Placement 2: NP is the pool size, NPH of it per head are home producers, the rest extras.
    const uint32_t NPH = placement == 2 ? fused_pool_home_producers(grid_x, grid_y, BH, NV, NP) : NP;
    const uint32_t P = placement == 2 ? NP : BH * NP;  // producer cores
    TT_FATAL(R + P <= n_cores, "chunk_gdn_fused: R+P = {}+{} cores needed, grid has {}", R, P, n_cores);

    // ---- Placement (test_chunk_gdn_fused_geometry.py checks it against a Python oracle) ----
    std::vector<CoreCoord> rcv_cores(R);  // index h*NV + v
    std::vector<CoreCoord> prod_cores;    // index p = h*NPH + j, then the extras
    prod_cores.reserve(P);
    // Row-local: head h < grid_y owns row h — receivers at columns 0..NV-1, producers at NV..L-1.
    // NOC_1 routes -x then -y, so every producer's writes travel west inside the head's own row and
    // never share a link with another head. Heads h >= grid_y live in the leftover columns [L, W)
    // as vertical blocks: an rw x rh receiver rectangle on top, the producers row-major below it —
    // their traffic is confined to the block's columns (short -x legs, then -y within the block).
    auto place_row_local = [&](uint32_t np) {
        const uint32_t L = NV + np;
        TT_FATAL(L <= grid_x, "chunk_gdn_fused: row-local placement needs NV+NP={} <= grid_x={}", L, grid_x);
        // k heads per row, each in its own column segment [i*L, (i+1)*L): the segments' -x legs are
        // disjoint, so heads sharing a row still share no link.
        const uint32_t k_per_row = grid_x / L;
        const uint32_t n_row_heads = std::min<uint32_t>(BH, k_per_row * grid_y);
        for (uint32_t h = 0; h < n_row_heads; h++) {
            const uint32_t row = h / k_per_row;
            const uint32_t xs = (h % k_per_row) * L;
            for (uint32_t v = 0; v < NV; v++) {
                rcv_cores[h * NV + v] = CoreCoord{xs + v, row};
            }
            for (uint32_t j = 0; j < np; j++) {
                prod_cores.push_back(CoreCoord{xs + NV + j, row});
            }
        }
        if (BH > n_row_heads) {
            const uint32_t rem = BH - n_row_heads;
            const uint32_t wl = grid_x - k_per_row * L;
            TT_FATAL(wl >= 1, "chunk_gdn_fused: row-local placement: no leftover columns for {} extra heads", rem);
            const uint32_t rw = std::min<uint32_t>(NV, wl);
            TT_FATAL(
                NV % rw == 0,
                "chunk_gdn_fused: row-local placement: NV={} not a multiple of the leftover width {}",
                NV,
                rw);
            const uint32_t rh = NV / rw;
            const uint32_t block_h = rh + (np + wl - 1) / wl;
            TT_FATAL(
                rem * block_h <= grid_y,
                "chunk_gdn_fused: row-local placement: {} leftover heads need {} rows, grid has {}",
                rem,
                rem * block_h,
                grid_y);
            for (uint32_t kk = 0; kk < rem; kk++) {
                const uint32_t h = n_row_heads + kk;
                const uint32_t y_base = kk * block_h;
                const uint32_t xl = k_per_row * L;  // first leftover column
                for (uint32_t v = 0; v < NV; v++) {
                    rcv_cores[h * NV + v] = CoreCoord{xl + (v % rw), y_base + v / rw};
                }
                for (uint32_t j = 0; j < np; j++) {
                    prod_cores.push_back(CoreCoord{xl + (j % wl), y_base + rh + j / wl});
                }
            }
        }
    };
    if (placement == 0) {
        TT_FATAL(
            BH <= HPR * grid_y,
            "chunk_gdn_fused: BH={} 1x{} receiver rectangles do not fit a {}x{} grid ({} per row)",
            BH,
            NV,
            grid_x,
            grid_y,
            HPR);
        std::vector<bool> is_rcv(n_cores, false);
        for (uint32_t h = 0; h < BH; h++) {
            const uint32_t y0 = h / HPR;
            const uint32_t x0 = (h % HPR) * NV;
            for (uint32_t v = 0; v < NV; v++) {
                rcv_cores[h * NV + v] = CoreCoord{x0 + v, y0};
                is_rcv[y0 * grid_x + x0 + v] = true;
            }
        }
        for (uint32_t y = 0; y < grid_y && prod_cores.size() < P; y++) {
            for (uint32_t x = 0; x < grid_x && prod_cores.size() < P; x++) {
                if (!is_rcv[y * grid_x + x]) {
                    prod_cores.push_back(CoreCoord{x, y});
                }
            }
        }
    } else if (placement == 1) {
        place_row_local(NP);
    } else {
        TT_FATAL(
            NPH >= 1,
            "chunk_gdn_fused: a producer pool of {} has no row-local layout with a home producer per head for BH={} "
            "NV={} on a {}x{} grid",
            NP,
            BH,
            NV,
            grid_x,
            grid_y);
        place_row_local(NPH);
        // The extras: the remaining cores, row-major.
        std::vector<bool> used(n_cores, false);
        for (const CoreCoord& c : rcv_cores) {
            used[c.y * grid_x + c.x] = true;
        }
        for (const CoreCoord& c : prod_cores) {
            used[c.y * grid_x + c.x] = true;
        }
        for (uint32_t y = 0; y < grid_y && prod_cores.size() < P; y++) {
            for (uint32_t x = 0; x < grid_x && prod_cores.size() < P; x++) {
                if (!used[y * grid_x + x]) {
                    prod_cores.push_back(CoreCoord{x, y});
                }
            }
        }
    }
    TT_FATAL(prod_cores.size() == P, "chunk_gdn_fused: placement produced {} producers, need {}", prod_cores.size(), P);
    return {std::move(rcv_cores), std::move(prod_cores), NPH};
}

FusedGeometryChoice choose_fused_geometry(
    uint32_t grid_x,
    uint32_t grid_y,
    uint32_t BH,
    uint32_t NC,
    uint32_t Vt,
    uint32_t fixed_nv,
    uint32_t fixed_np,
    uint32_t fixed_nbuf,
    FusedCandidates candidates) {
    FusedGeometryChoice best;
    best.t_phased_us = t_phased_us(BH, NC);
    bool have = false;
    uint32_t best_cores = 0;
    // Hand-off depths the model chooses between (deeper measured no better); a pinned depth is taken as is.
    const uint32_t depths[2] = {fixed_nbuf ? fixed_nbuf : 2u, fixed_nbuf ? fixed_nbuf : 3u};
    const uint32_t n_depths = fixed_nbuf ? 1u : 2u;
    // One candidate: NP producers per head (placements 0/1) or a pool of NP serving every head (placement 2)
    // with the extras' share num/den; a pool with no extras (NP = BH*NPH) is the per-head geometry it is.
    auto consider = [&](uint32_t nv, uint32_t np, uint32_t placement, uint32_t depth, uint32_t num, uint32_t den) {
        const uint32_t Vtl = Vt / nv;
        if (!fixed_nbuf && !handoff_fits_l1(Vtl, depth)) {
            return;
        }
        const uint32_t producers = placement == 2 ? np : BH * np;
        const uint32_t cores = BH * nv + producers;
        const uint32_t nph = placement == 2 ? fused_pool_home_producers(grid_x, grid_y, BH, nv, np) : np;
        const uint32_t nx = placement == 2 ? np - BH * nph : 0;
        const PoolLoad load = nx ? pool_load(BH, NC, nph, nx, num, den) : PoolLoad{(NC + nph - 1) / nph, 0};
        const float t =
            t_fused_us(BH, NC, Vtl, depth, placement, producers, load, nx > 0, nx ? float(num) / den : 0.0f);
        // ties -> fewer cores, then smaller NV, then the shallower ring
        const bool better =
            !have || t < best.t_fused_us ||
            (t == best.t_fused_us &&
             (cores < best_cores || (cores == best_cores && (nv < best.nv || (nv == best.nv && depth < best.nbuf)))));
        if (better) {
            best.nv = nv;
            best.np = np;
            best.placement = placement;
            best.nbuf = depth;
            best.t_fused_us = t;
            best_cores = cores;
            have = true;
        }
    };
    auto consider_depths = [&](uint32_t nv, uint32_t np, uint32_t placement) {
        for (uint32_t i = 0; i < n_depths; i++) {
            consider(nv, np, placement, depths[i], 0, 1);
        }
    };
    // A pool with extras at the balanced share NX / P.
    auto consider_pool = [&](uint32_t nv, uint32_t P, uint32_t nx) {
        for (uint32_t i = 0; i < n_depths; i++) {
            consider(nv, P, 2, depths[i], nx, P);
        }
    };
    auto nv_ok = [&](uint32_t nv) {
        return Vt % nv == 0 && t_step_us(Vt / nv) >= 0.0f && (fixed_nv == 0 || nv == fixed_nv);
    };
    auto np_ok = [&](uint32_t np) { return fixed_np == 0 || np == std::min(fixed_np, NC); };
    if (candidates != FusedCandidates::Pool) {
        for (uint32_t nv : kNvCandidates) {
            if (!nv_ok(nv)) {
                continue;
            }
            for (uint32_t np = 1; np + nv <= grid_x; np++) {
                const uint32_t np_eff = std::min(np, NC);
                if (BH * (nv + np_eff) > grid_x * grid_y) {
                    break;
                }
                if (np_ok(np_eff) && fused_row_local_feasible(grid_x, grid_y, BH, nv, np_eff)) {
                    consider_depths(nv, np_eff, 1);
                }
            }
        }
        if (!have) {  // no row-local layout: the row-major fallback, penalised for its shared links
            for (uint32_t nv : kNvCandidates) {
                if (!nv_ok(nv) || nv > grid_x || BH > (grid_x / nv) * grid_y) {
                    continue;
                }
                const uint32_t free = grid_x * grid_y - BH * nv;
                const uint32_t np = fixed_np ? std::min(fixed_np, NC) : std::min(free / BH, NC);
                if (np >= 1 && BH * (nv + np) <= grid_x * grid_y) {
                    consider_depths(nv, np, 0);
                }
            }
        }
    }
    if (candidates != FusedCandidates::PerHead) {
        // Producer pool: P = every core the receivers leave (or the pinned size), at most one per item.
        for (uint32_t nv : kNvCandidates) {
            if (!nv_ok(nv) || BH * nv >= grid_x * grid_y) {
                continue;
            }
            const uint32_t P = std::min<uint32_t>(fixed_np ? fixed_np : grid_x * grid_y - BH * nv, BH * NC);
            if (!fused_pool_feasible(grid_x, grid_y, BH, nv, P)) {
                continue;
            }
            const uint32_t nph = fused_pool_home_producers(grid_x, grid_y, BH, nv, P);
            if (P == BH * nph) {
                // No extras: the per-head geometry (NV, NPH) itself, in whichever placement the candidates allow.
                if (candidates == FusedCandidates::Both) {
                    consider_depths(nv, nph, 1);
                } else {
                    consider_depths(nv, P, 2);
                }
            } else {
                consider_pool(nv, P, P - BH * nph);
            }
        }
    }
    best.fused_pays = have && best.t_fused_us < best.t_phased_us;
    return best;
}

// ---------------------------------------------------------------------------------------------------
// Launcher
// ---------------------------------------------------------------------------------------------------
std::vector<Tensor> chunk_gdn(
    const Tensor& q,
    const Tensor& k,
    const Tensor& v,
    const Tensor& g,
    const Tensor& beta,
    const Tensor& eye_c,
    const Tensor& tril_c,
    const Tensor& ones_c,
    const Tensor& masks_c,
    const Tensor& initial_state,
    uint32_t chunk_size,
    bool output_final_state,
    const tt::tt_metal::MemoryConfig& output_mem_config,
    const DeviceComputeKernelConfig& compute_kernel_config,
    const ChunkGdnDeviceProgramConfig& program_config,
    ttnn::transformer::ChunkGdnWyInverse wy_inverse,
    bool v_flat,
    uint32_t HV,
    bool qk_norm,
    float scale,
    bool qk_flat,
    uint32_t Hk) {
    const auto& q_shape = q.logical_shape();  // [BH,NC,C,K] head-major, or flat [B,T,Hk*K] when qk_flat
    const auto& v_shape = v.logical_shape();  // [BH,NC,C,V] head-major, or flat [B,T,HV*V] when v_flat
    // Dim derivation identical to chunk_gdn_prep (both factories consume prep's inputs; the flat forms
    // are Fused-only and rejected for Mono in validate).
    const uint32_t BH = qk_flat ? (q_shape[0] * HV) : q_shape[0];
    const uint32_t num_chunks = qk_flat ? (q_shape[1] / chunk_size) : q_shape[1];
    const uint32_t key_dim = qk_flat ? (q_shape[2] / Hk) : q_shape[3];
    const uint32_t val_dim = v_flat ? (v_shape[2] / HV) : v_shape[3];
    auto attrs = ChunkGdnDeviceOperation::operation_attributes_t{
        .impl = ChunkGdnImpl::Mono,
        .BH = BH,
        .num_chunks = num_chunks,
        .chunk_size = chunk_size,
        .key_dim = key_dim,
        .val_dim = val_dim,
        .v_flat = v_flat,
        .HV = HV,
        .qk_flat = qk_flat,
        .Hk = Hk,
        .qk_norm = qk_norm,
        .scale = scale,
        .output_final_state = output_final_state,
        .output_mem_config = output_mem_config,
        .compute_kernel_config = compute_kernel_config,
    };
    if (const auto* fused_cfg = std::get_if<ttnn::transformer::ChunkGdnFusedProgramConfig>(&program_config)) {
        attrs.impl = ChunkGdnImpl::Fused;
        // The WY-inverse method (hashed).
        attrs.tinv = gdn_tinv_resolve(wy_inverse, chunk_size, q);
        // Geometry: the program config's pinned fields, the calibrated cost model for the rest. Resolved HERE
        // (attrs construction), never in the factory — every field is hashed, so a different config compiles a
        // fresh program instead of silently serving a stale cached one.
        const auto grid0 = q.device()->compute_with_storage_grid_size();
        const uint32_t np_pin = fused_cfg->num_producers.value_or(0);
        const uint32_t nv_pin = fused_cfg->num_receivers.value_or(0);
        TT_FATAL(
            !fused_cfg->num_producers.has_value() || np_pin >= 1,
            "chunk_gdn_fused: num_producers must be >= 1 (got {})",
            np_pin);
        TT_FATAL(
            !fused_cfg->num_receivers.has_value() || nv_pin >= 1,
            "chunk_gdn_fused: num_receivers must be >= 1 (got {})",
            nv_pin);
        const uint32_t nbuf_pin = fused_cfg->handoff_depth.value_or(0);
        TT_FATAL(
            !fused_cfg->handoff_depth.has_value() || (nbuf_pin >= 1 && nbuf_pin <= 8),
            "chunk_gdn_fused: handoff_depth must be in [1, 8] (got {})",
            nbuf_pin);
        const auto& share = fused_cfg->pool_extra_share;
        TT_FATAL(
            !share.has_value() || (*share >= 0.0f && *share <= 1.0f),
            "chunk_gdn_fused: pool_extra_share must be in [0, 1] (got {})",
            share.value_or(0.0f));
        // The model fills whatever the config leaves free (geometry, hand-off depth, or all) so the pair fits
        // the grid.
        const bool pool = fused_cfg->producer_pool;
        const auto choice = choose_fused_geometry(
            grid0.x,
            grid0.y,
            BH,
            num_chunks,
            val_dim / tt::constants::TILE_WIDTH,
            nv_pin,
            np_pin,
            nbuf_pin,
            pool ? FusedCandidates::Pool : FusedCandidates::PerHead);
        TT_FATAL(
            choice.nv >= 1,
            "chunk_gdn_fused: no fused geometry fits BH={} on a {}x{} grid with num_receivers={} num_producers={} "
            "producer_pool={} (0 = free); the dispatch must choose phased",
            BH,
            grid0.x,
            grid0.y,
            nv_pin,
            np_pin,
            pool);
        attrs.nv = nv_pin ? nv_pin : choice.nv;
        if (pool) {
            // The pool size, at most one producer per item; its home producers per head come from the
            // layout (validated), the extras' share from the config or the balanced NX / P.
            attrs.np = np_pin ? std::min<uint32_t>(np_pin, BH * num_chunks) : choice.np;
            attrs.placement = 2;
            const uint32_t nph = fused_pool_home_producers(grid0.x, grid0.y, BH, attrs.nv, attrs.np);
            const uint32_t nx = nph >= 1 ? attrs.np - BH * nph : 0;
            if (nx >= 1) {
                attrs.pool_extra_num = share.has_value() ? static_cast<uint32_t>(std::lround(*share * attrs.np)) : nx;
                attrs.pool_extra_den = attrs.np;
            }
        } else {
            // Producers per head, clamped to num_chunks: a producer beyond NC would own no chunks (wasted
            // core, and the receiver's rotating credit c % NP would skip it anyway). Receivers per head must
            // divide Vt (validated).
            attrs.np = np_pin ? std::min<uint32_t>(np_pin, num_chunks) : choice.np;
        }
        attrs.nbuf = nbuf_pin ? nbuf_pin : choice.nbuf;
        attrs.unicast = fused_cfg->unicast;
        attrs.posted = fused_cfg->posted;
        TT_FATAL(
            !attrs.posted || attrs.unicast,
            "chunk_gdn_fused: posted writes require the unicast transport (unicast=true)");
        // Placement: the config's choice, else row-local whenever the (possibly pinned) geometry has a
        // row-local layout.
        if (!pool) {
            const bool row_local =
                fused_cfg->row_local.value_or(fused_row_local_feasible(grid0.x, grid0.y, BH, attrs.nv, attrs.np));
            attrs.placement = row_local ? 1u : 0u;
        }
    } else {
        // The mono program has no forward-substitution solve: AUTO resolves to Horner (attrs.tinv's default)
        // and an explicit request is refused rather than silently downgraded.
        TT_FATAL(
            wy_inverse != ttnn::transformer::ChunkGdnWyInverse::FORWARD_SUBSTITUTION,
            "chunk_gdn: the mono program computes the WY inverse with Horner only; "
            "wy_inverse=FORWARD_SUBSTITUTION is not available on it (use HORNER or AUTO)");
    }
    auto tensor_args = ChunkGdnDeviceOperation::tensor_args_t{
        .q = q,
        .k = k,
        .v = v,
        .g = g,
        .beta = beta,
        .eye_c = eye_c,
        .tril_c = tril_c,
        .ones_c = ones_c,
        .masks_c = masks_c,
        .initial_state = initial_state};
    return ttnn::device_operation::launch<ChunkGdnDeviceOperation>(attrs, tensor_args);
}

}  // namespace ttnn::prim
