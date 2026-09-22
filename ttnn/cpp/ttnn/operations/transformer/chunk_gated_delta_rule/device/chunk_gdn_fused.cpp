// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
// SPDX-License-Identifier: Apache-2.0

#include "chunk_gdn_fused.hpp"

#include <algorithm>
#include <cstdlib>
#include <utility>

#include <tt-metalium/constants.hpp>
#include "ttnn/device_operation.hpp"
#include "ttnn/tensor/tensor.hpp"

using namespace tt::tt_metal;

namespace ttnn::prim {

namespace {
// Uniquely named (vs the phased prim's `check`) so it does not clash under unity builds.
void check_fused(const Tensor& t, const char* name, DataType dt) {
    TT_FATAL(t.layout() == Layout::TILE, "chunk_gdn_fused: {} must be TILE layout", name);
    TT_FATAL(t.dtype() == dt, "chunk_gdn_fused: {} has wrong dtype", name);
    TT_FATAL(t.buffer() != nullptr, "chunk_gdn_fused: {} must be on device", name);
}
}  // namespace

ChunkGdnFusedOperation::program_factory_t ChunkGdnFusedOperation::select_program_factory(
    const operation_attributes_t&, const tensor_args_t&) {
    return ChunkGdnFusedProgramFactory{};
}

void ChunkGdnFusedOperation::validate_on_program_cache_miss(
    const operation_attributes_t& attrs, const tensor_args_t& in) {
    using namespace tt::constants;
    // Input-side checks: identical to the phased PREP prim (the fused producer runs the unchanged
    // prep reader/compute, so its input contract is prep's).
    check_fused(in.q, "q", DataType::BFLOAT16);
    check_fused(in.k, "k", DataType::BFLOAT16);
    check_fused(in.v, "v", DataType::BFLOAT16);  // flat [B,T,HV*V] when attrs.v_flat; else [BH,NC,C,V]
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
    check_fused(in.g, "g", DataType::FLOAT32);
    check_fused(in.beta, "beta", DataType::FLOAT32);
    check_fused(in.eye_c, "eye_c", DataType::FLOAT32);
    check_fused(in.tril_c, "tril_c", DataType::FLOAT32);
    check_fused(in.ones_c, "ones_c", DataType::FLOAT32);
    check_fused(in.masks_c, "masks_c", DataType::FLOAT32);
    if (in.initial_state.has_value()) {
        check_fused(*in.initial_state, "initial_state", DataType::FLOAT32);
    }
    TT_FATAL(attrs.chunk_size % TILE_HEIGHT == 0, "chunk_size must be a multiple of 32");
    TT_FATAL(attrs.key_dim % TILE_WIDTH == 0, "key_dim must be a multiple of 32");
    TT_FATAL(attrs.val_dim % TILE_WIDTH == 0, "val_dim must be a multiple of 32");
    // Geometry: NP producers + NV receivers per head. Receivers of a head form a 1xNV row rectangle
    // (the multicast target), so the grid must hold BH such rectangles: BH <= (grid.x / NV) * grid.y.
    // Producers have no placement constraint. NP=1 / NV=1 unless QWEN_GDN_NP / QWEN_GDN_NV opted in.
    TT_FATAL(attrs.np >= 1, "chunk_gdn_fused: np must be >= 1 (got {})", attrs.np);
    TT_FATAL(attrs.nv >= 1, "chunk_gdn_fused: nv must be >= 1 (got {})", attrs.nv);
    const uint32_t Vt = attrs.val_dim / TILE_WIDTH;
    TT_FATAL(Vt % attrs.nv == 0, "chunk_gdn_fused: nv ({}) must divide Vt ({})", attrs.nv, Vt);
    const auto grid = in.q.device()->compute_with_storage_grid_size();
    TT_FATAL(attrs.nv <= grid.x, "chunk_gdn_fused: nv ({}) exceeds the grid width {}", attrs.nv, grid.x);
    // Per-(head, slot) credit words live in one 4 KB tile of the u/mask CB.
    TT_FATAL(
        attrs.BH * attrs.nbuf <= 1024,
        "chunk_gdn_fused: BH * nbuf ({} * {}) credit words exceed the 1024-word credit tile",
        attrs.BH,
        attrs.nbuf);
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

ChunkGdnFusedOperation::spec_return_value_t ChunkGdnFusedOperation::compute_output_specs(
    const operation_attributes_t& attrs, const tensor_args_t&) {
    // EXACTLY ChunkGdnScanOperation::compute_output_specs: o and final_state are both fp32 (a bf16
    // o degraded full-model quality and was removed — see the phased scan op).
    const auto o_layout = TensorLayout(DataType::FLOAT32, PageConfig(Layout::TILE), attrs.output_mem_config);
    const auto s_layout = TensorLayout(DataType::FLOAT32, PageConfig(Layout::TILE), attrs.output_mem_config);
    ttnn::Shape o_shape({attrs.BH, attrs.num_chunks, attrs.chunk_size, attrs.val_dim});
    ttnn::Shape s_shape({attrs.BH, attrs.key_dim, attrs.val_dim});
    return {tt::tt_metal::TensorSpec(o_shape, o_layout), tt::tt_metal::TensorSpec(s_shape, s_layout)};
}

ChunkGdnFusedOperation::tensor_return_value_t ChunkGdnFusedOperation::create_output_tensors(
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

namespace {
// QB2 constants, measured 2026-09-21: producer item under load, receiver step (period) per V-slice
// width, pipeline fill, and the phased wall-op reference at NC=64.
constexpr float kFillUs = 65.0f;
// Producer item time depends on how many producers load the DRAM/NoC at once: 26 us with <= 28 of
// them (BH=4), 34 us with >= 84 (BH=12); linear in between.
float w_p_us(uint32_t producers) {
    const float f = std::min(1.0f, std::max(0.0f, (static_cast<float>(producers) - 28.0f) / 56.0f));
    return 26.0f + 8.0f * f;
}
float t_step_us(uint32_t Vtl) {
    switch (Vtl) {
        case 1: return 3.5f;    // receiver period with the scan-step DST batching: 3.43 compute
        case 2: return 4.9f;    // 4.82 compute
        case 4: return 7.5f;    // 7.44 compute
        default: return -1.0f;  // unmeasured width
    }
}
float t_phased_us(uint32_t BH, uint32_t NC) {
    // Measured wall-op at NC=64 (wall - 115 us glue): 4 -> 453, 8 -> 593, 12 -> 706, 16 -> 883,
    // 32 -> 1449, 48 -> 2475. Linear 310 + 35.8*BH to BH=32, then interpolated to the DRAM-saturated
    // BH=48 point.
    const float t32 = 310.0f + 35.8f * 32.0f;
    float t = (BH <= 32) ? (310.0f + 35.8f * BH) : (t32 + (2475.0f - t32) * (std::min<uint32_t>(BH, 48) - 32) / 16.0f);
    if (BH > 48) {
        t *= BH / 48.0f;
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

FusedPlacement fused_placement(
    uint32_t grid_x, uint32_t grid_y, uint32_t BH, uint32_t NV, uint32_t NP, uint32_t placement) {
    const uint32_t n_cores = grid_x * grid_y;
    TT_FATAL(NV <= grid_x, "chunk_gdn_fused: nv={} exceeds the grid width {}", NV, grid_x);
    const uint32_t HPR = grid_x / NV;  // heads per receiver row (placement 0)
    const uint32_t R = BH * NV;        // receiver cores
    const uint32_t P = BH * NP;        // producer cores
    TT_FATAL(R + P <= n_cores, "chunk_gdn_fused: R+P = {}+{} cores needed, grid has {}", R, P, n_cores);

    // ---- Placement (test_chunk_gdn_fused_geometry.py checks it against a Python oracle) ----
    std::vector<CoreCoord> rcv_cores(R);  // index h*NV + v
    std::vector<CoreCoord> prod_cores;    // index p = h*NP + j
    prod_cores.reserve(P);
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
    } else {
        // Row-local: head h < grid_y owns row h — receivers at columns 0..NV-1, producers at NV..L-1.
        // NOC_1 routes -x then -y, so every producer's writes travel west inside the head's own row and
        // never share a link with another head. Heads h >= grid_y live in the leftover columns [L, W)
        // as vertical blocks: an rw x rh receiver rectangle on top, the producers row-major below it —
        // their traffic is confined to the block's columns (short -x legs, then -y within the block).
        const uint32_t L = NV + NP;
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
            for (uint32_t j = 0; j < NP; j++) {
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
            const uint32_t block_h = rh + (NP + wl - 1) / wl;
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
                for (uint32_t j = 0; j < NP; j++) {
                    prod_cores.push_back(CoreCoord{xl + (j % wl), y_base + rh + j / wl});
                }
            }
        }
    }
    TT_FATAL(prod_cores.size() == P, "chunk_gdn_fused: placement produced {} producers, need {}", prod_cores.size(), P);
    return {std::move(rcv_cores), std::move(prod_cores)};
}

FusedGeometryChoice choose_fused_geometry(
    uint32_t grid_x, uint32_t grid_y, uint32_t BH, uint32_t NC, uint32_t Vt, uint32_t fixed_nv, uint32_t fixed_np) {
    FusedGeometryChoice best;
    best.t_phased_us = t_phased_us(BH, NC);
    bool have = false;
    auto consider = [&](uint32_t nv, uint32_t np, uint32_t placement) {
        const float ts = t_step_us(Vt / nv);
        if (ts < 0.0f) {
            return;
        }
        const float t = NC * std::max(w_p_us(BH * np) / np, ts) + kFillUs;
        // ties -> fewer cores, then smaller NV
        const bool better =
            !have || t < best.t_fused_us ||
            (t == best.t_fused_us && (nv + np < best.nv + best.np || (nv + np == best.nv + best.np && nv < best.nv)));
        if (better) {
            best.nv = nv;
            best.np = np;
            best.placement = placement;
            best.t_fused_us = t;
            have = true;
        }
    };
    auto nv_ok = [&](uint32_t nv) { return Vt % nv == 0 && (fixed_nv == 0 || nv == fixed_nv); };
    auto np_ok = [&](uint32_t np) { return fixed_np == 0 || np == std::min(fixed_np, NC); };
    for (uint32_t nv : {1u, 2u, 4u, 8u}) {
        if (!nv_ok(nv)) {
            continue;
        }
        for (uint32_t np = 1; np + nv <= grid_x; np++) {
            const uint32_t np_eff = std::min(np, NC);
            if (BH * (nv + np_eff) > grid_x * grid_y) {
                break;
            }
            if (np_ok(np_eff) && fused_row_local_feasible(grid_x, grid_y, BH, nv, np_eff)) {
                consider(nv, np_eff, 1);
            }
        }
    }
    if (!have) {  // no row-local layout: the row-major fallback (optimistic — ignores link sharing)
        for (uint32_t nv : {1u, 2u, 4u, 8u}) {
            if (!nv_ok(nv) || nv > grid_x || BH > (grid_x / nv) * grid_y) {
                continue;
            }
            const uint32_t free = grid_x * grid_y - BH * nv;
            const uint32_t np = fixed_np ? std::min(fixed_np, NC) : std::min(free / BH, NC);
            if (np >= 1 && BH * (nv + np) <= grid_x * grid_y) {
                consider(nv, np, 0);
            }
        }
    }
    best.fused_pays = have && best.t_fused_us < best.t_phased_us;
    return best;
}

std::vector<Tensor> chunk_gdn_fused(
    const Tensor& q,
    const Tensor& k,
    const Tensor& v,
    const Tensor& g,
    const Tensor& beta,
    const Tensor& eye_c,
    const Tensor& tril_c,
    const Tensor& ones_c,
    const Tensor& masks_c,
    const std::optional<Tensor>& initial_state,
    uint32_t chunk_size,
    bool output_final_state,
    const tt::tt_metal::MemoryConfig& output_mem_config,
    const DeviceComputeKernelConfig& compute_kernel_config,
    bool v_flat,
    uint32_t HV,
    bool qk_norm,
    float scale,
    bool qk_flat,
    uint32_t Hk) {
    const auto& q_shape = q.logical_shape();  // [BH,NC,C,K] head-major, or flat [B,T,Hk*K] when qk_flat
    const auto& v_shape = v.logical_shape();  // [BH,NC,C,V] head-major, or flat [B,T,HV*V] when v_flat
    // Dim derivation identical to chunk_gdn_prep (the fused op consumes prep's inputs).
    const uint32_t BH = qk_flat ? (q_shape[0] * HV) : q_shape[0];
    const uint32_t num_chunks = qk_flat ? (q_shape[1] / chunk_size) : q_shape[1];
    const uint32_t key_dim = qk_flat ? (q_shape[2] / Hk) : q_shape[3];
    const uint32_t val_dim = v_flat ? (v_shape[2] / HV) : v_shape[3];
    // Geometry defaults from the calibrated cost model (design D8 v0.3); every knob below overrides
    // its field. Read HERE (attrs construction), never in the factory — all of them are hashed, so a
    // toggle compiles a fresh program instead of silently serving a stale cached one.
    const auto grid0 = q.device()->compute_with_storage_grid_size();
    uint32_t np_env = 0, nv_env = 0;
    if (const char* e = std::getenv("QWEN_GDN_NP")) {
        const int v_np = std::atoi(e);
        TT_FATAL(v_np >= 1, "QWEN_GDN_NP must be a positive integer (got '{}')", e);
        np_env = static_cast<uint32_t>(v_np);
    }
    if (const char* e = std::getenv("QWEN_GDN_NV")) {
        const int v_nv = std::atoi(e);
        TT_FATAL(v_nv >= 1, "QWEN_GDN_NV must be a positive integer (got '{}')", e);
        nv_env = static_cast<uint32_t>(v_nv);
    }
    // The model fills whatever the knobs leave free (both, one, or none) so the pair fits the grid.
    const auto choice = choose_fused_geometry(grid0.x, grid0.y, BH, num_chunks, val_dim / TILE_WIDTH, nv_env, np_env);
    TT_FATAL(
        choice.nv >= 1,
        "chunk_gdn_fused: no fused geometry fits BH={} on a {}x{} grid with NV={} NP={} (0 = free); the dispatch must "
        "choose phased",
        BH,
        grid0.x,
        grid0.y,
        nv_env,
        np_env);
    // F3a producers per head, clamped to num_chunks: a producer beyond NC would own no chunks (wasted
    // core, and the receiver's rotating credit c % NP would skip it anyway). Receivers per head must
    // divide Vt (validated).
    const uint32_t np = np_env ? std::min<uint32_t>(np_env, num_chunks) : choice.np;
    const uint32_t nv = nv_env ? nv_env : choice.nv;
    uint32_t nbuf = 2;
    if (const char* e = std::getenv("QWEN_GDN_HANDOFF_NBUF")) {
        const int v_nb = std::atoi(e);
        TT_FATAL(v_nb >= 1 && v_nb <= 8, "QWEN_GDN_HANDOFF_NBUF must be in [1, 8] (got '{}')", e);
        nbuf = static_cast<uint32_t>(v_nb);
    }
    bool unicast = true;
    if (const char* e = std::getenv("QWEN_GDN_UNICAST")) {
        unicast = std::atoi(e) != 0;
    }
    bool posted = false;
    if (const char* e = std::getenv("QWEN_GDN_POSTED")) {
        posted = std::atoi(e) != 0;
    }
    TT_FATAL(
        !posted || unicast, "chunk_gdn_fused: QWEN_GDN_POSTED requires the unicast transport (QWEN_GDN_UNICAST=1)");
    // Placement: row-local whenever the (possibly overridden) geometry has a row-local layout.
    uint32_t placement = fused_row_local_feasible(grid0.x, grid0.y, BH, nv, np) ? 1 : 0;
    if (const char* e = std::getenv("QWEN_GDN_PLACEMENT")) {
        const int v_pl = std::atoi(e);
        TT_FATAL(v_pl == 0 || v_pl == 1, "QWEN_GDN_PLACEMENT must be 0 (row-major) or 1 (row-local), got '{}'", e);
        placement = static_cast<uint32_t>(v_pl);
    }
    auto attrs = ChunkGdnFusedOperation::operation_attributes_t{
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
        .np = np,
        .nv = nv,
        .nbuf = nbuf,
        .unicast = unicast,
        .posted = posted,
        .placement = placement,
        .has_initial_state = initial_state.has_value(),
        .output_final_state = output_final_state,
        .output_mem_config = output_mem_config,
        .compute_kernel_config = compute_kernel_config,
    };
    auto tensor_args = ChunkGdnFusedOperation::tensor_args_t{
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
    return ttnn::device_operation::launch<ChunkGdnFusedOperation>(attrs, tensor_args);
}

}  // namespace ttnn::prim
