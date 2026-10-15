// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
// SPDX-License-Identifier: Apache-2.0

#include "chunk_gdn_device_operation.hpp"

#include <algorithm>
#include <cmath>
#include <limits>
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
void check_gdn_shape(const Tensor& t, const char* name, const ttnn::Shape& expected) {
    TT_FATAL(
        t.logical_shape() == expected, "chunk_gdn: {} has shape {}, expected {}", name, t.logical_shape(), expected);
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
    // producer runs unchanged; the flat forms are the prep reader's and therefore Fused-only). The launcher
    // derives rows (heads in the tensors) once from the path's flags; every tensor is held to its path's shape
    // and the head range [head_offset, head_offset + BH) to that one rows.
    TT_FATAL(attrs.BH >= 1, "chunk_gdn: BH must be >= 1");
    TT_FATAL(attrs.rows >= 1, "chunk_gdn: rows must be >= 1");
    TT_FATAL(attrs.chunk_size % TILE_HEIGHT == 0, "chunk_size must be a multiple of 32");
    TT_FATAL(attrs.key_dim % TILE_WIDTH == 0, "key_dim must be a multiple of 32");
    TT_FATAL(attrs.val_dim % TILE_WIDTH == 0, "val_dim must be a multiple of 32");
    const uint32_t rows = attrs.rows, NC = attrs.num_chunks, C = attrs.chunk_size;
    const uint32_t K = attrs.key_dim, V = attrs.val_dim;
    // Flat tensors are token-major [B, T, heads * D] with B * HV == rows and T == NC * C.
    auto flat_batch = [&](const char* what) {
        TT_FATAL(
            attrs.HV > 0 && rows % attrs.HV == 0, "chunk_gdn: {} needs rows ({}) = B * HV ({})", what, rows, attrs.HV);
        return rows / attrs.HV;
    };
    check_gdn_tensor(in.q, "q", DataType::BFLOAT16);
    check_gdn_tensor(in.k, "k", DataType::BFLOAT16);
    check_gdn_tensor(in.v, "v", DataType::BFLOAT16);
    if (attrs.qk_flat) {
        TT_FATAL(
            attrs.Hk > 0 && attrs.HV % attrs.Hk == 0,
            "chunk_gdn: qk_flat needs Hk ({}) dividing HV ({})",
            attrs.Hk,
            attrs.HV);
        TT_FATAL(attrs.qk_norm, "qk_flat requires qk_norm (flat q/k are unnormalized; norm is in-kernel)");
        const uint32_t B = flat_batch("qk_flat");
        check_gdn_shape(in.q, "q", ttnn::Shape({B, NC * C, attrs.Hk * K}));
        check_gdn_shape(in.k, "k", ttnn::Shape({B, NC * C, attrs.Hk * K}));
    } else {
        check_gdn_shape(in.q, "q", ttnn::Shape({rows, NC, C, K}));
        check_gdn_shape(in.k, "k", ttnn::Shape({rows, NC, C, K}));
    }
    if (attrs.v_flat) {
        const uint32_t B = flat_batch("v_flat");
        check_gdn_shape(in.v, "v", ttnn::Shape({B, NC * C, attrs.HV * V}));
    } else {
        check_gdn_shape(in.v, "v", ttnn::Shape({rows, NC, C, V}));
    }
    check_gdn_tensor(in.g, "g", DataType::FLOAT32);
    check_gdn_tensor(in.beta, "beta", DataType::FLOAT32);
    check_gdn_shape(in.g, "g", ttnn::Shape({rows, NC, C, 1}));
    check_gdn_shape(in.beta, "beta", ttnn::Shape({rows, NC, C, 1}));
    check_gdn_tensor(in.eye_c, "eye_c", DataType::FLOAT32);
    check_gdn_tensor(in.tril_c, "tril_c", DataType::FLOAT32);
    check_gdn_tensor(in.ones_c, "ones_c", DataType::FLOAT32);
    check_gdn_tensor(in.masks_c, "masks_c", DataType::FLOAT32);
    check_gdn_shape(in.eye_c, "eye_c", ttnn::Shape({1, 1, C, C}));
    check_gdn_shape(in.tril_c, "tril_c", ttnn::Shape({1, 1, C, C}));
    check_gdn_shape(in.ones_c, "ones_c", ttnn::Shape({1, 1, C, C}));
    check_gdn_shape(in.masks_c, "masks_c", ttnn::Shape({1, 1, TILE_HEIGHT, 3 * TILE_WIDTH}));
    // Required: the mono reader and the fused receivers stream S from this buffer unconditionally (no
    // in-kernel zeroing); the public op builds a zero state when its own initial_state is omitted.
    check_gdn_tensor(in.initial_state, "initial_state", DataType::FLOAT32);
    check_gdn_shape(in.initial_state, "initial_state", ttnn::Shape({rows, K, V}));
    validate_gdn_tinv(attrs.tinv, attrs.chunk_size, in.q);
    TT_FATAL(
        attrs.head_offset + attrs.BH <= rows,
        "chunk_gdn: heads [{}, {}) exceed the {} heads of the inputs",
        attrs.head_offset,
        attrs.head_offset + attrs.BH,
        rows);
    TT_FATAL(
        in.o_out.has_value() == in.final_state_out.has_value(),
        "chunk_gdn: o_out and final_state_out must be given together");
    if (in.o_out.has_value()) {
        TT_FATAL(attrs.impl == ChunkGdnImpl::Fused, "chunk_gdn: preallocated outputs are for the fused program only");
        check_gdn_tensor(*in.o_out, "o_out", DataType::FLOAT32);
        check_gdn_tensor(*in.final_state_out, "final_state_out", DataType::FLOAT32);
        check_gdn_shape(*in.o_out, "o_out", ttnn::Shape({rows, NC, C, V}));
        check_gdn_shape(*in.final_state_out, "final_state_out", ttnn::Shape({rows, K, V}));
        TT_FATAL(
            in.o_out->memory_config() == attrs.output_mem_config &&
                in.final_state_out->memory_config() == attrs.output_mem_config,
            "chunk_gdn: preallocated outputs must use the op's output memory config");
    }

    const auto grid = in.q.device()->compute_with_storage_grid_size();
    if (attrs.impl == ChunkGdnImpl::Mono) {
        // One core per head running the single-kernel program on head-major inputs.
        TT_FATAL(
            !attrs.v_flat && !attrs.qk_flat && !attrs.qk_norm,
            "chunk_gdn: the mono program takes head-major q/k/v only (no flat inputs, no in-kernel qk-norm)");
        TT_FATAL(attrs.head_offset == 0, "chunk_gdn: the mono program computes every head (head_offset must be 0)");
        TT_FATAL(
            attrs.BH <= grid.x * grid.y,
            "chunk_gdn: the mono program needs one core per head: BH={} exceeds the {}x{} grid",
            attrs.BH,
            grid.x,
            grid.y);
        return;
    }
    // Fused geometry: NP producers + NV receivers per head. Receivers of a head form a 1xNV row
    // rectangle (the multicast target), so the grid must hold BH such rectangles: BH <= (grid.x / NV) *
    // grid.y. Producers have no placement constraint. NP / NV come from ChunkGdnFusedProgramConfig::
    // num_producers / num_receivers, or from the cost model when unset (chunk_gdn below).
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
    const operation_attributes_t& attrs, const tensor_args_t& in) {
    if (in.o_out.has_value()) {
        return {in.o_out->tensor_spec(), in.final_state_out->tensor_spec()};
    }
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
    if (in.o_out.has_value()) {
        return {*in.o_out, *in.final_state_out};
    }
    auto specs = compute_output_specs(attrs, in);
    auto* device = in.q.device();
    std::vector<Tensor> outs;
    outs.reserve(specs.size());
    for (const auto& spec : specs) {
        outs.push_back(create_device_tensor(spec, device));
    }
    return outs;
}

tt::tt_metal::operation::Hash ChunkGdnDeviceOperation::compute_program_hash(
    const operation_attributes_t& attrs, const tensor_args_t& in) {
    auto keyed = attrs;
    keyed.head_offset = 0;
    return tt::tt_metal::operation::hash_operation<ChunkGdnDeviceOperation>(keyed, in);
}

std::vector<tt::tt_metal::DynamicRuntimeArg> ChunkGdnDeviceOperation::get_dynamic_runtime_args(
    const operation_attributes_t& attrs,
    const tensor_args_t& in,
    tensor_return_value_t&,
    const std::optional<ttnn::MeshCoordinate>&) {
    std::vector<tt::tt_metal::DynamicRuntimeArg> args;
    if (attrs.impl != ChunkGdnImpl::Fused) {
        return args;  // the mono program runs every head (head_offset 0, validated)
    }
    const auto grid = in.q.device()->compute_with_storage_grid_size();
    const FusedPlacement layout = fused_placement(grid.x, grid.y, attrs.BH, attrs.nv, attrs.np, attrs.placement);
    const uint32_t h0 = attrs.head_offset;
    const uint32_t NC = attrs.num_chunks;
    const uint32_t NP = attrs.np;
    const uint32_t NV = attrs.nv;
    args.reserve(attrs.BH * (NP + 2 * NV));
    for (uint32_t h = 0; h < attrs.BH; h++) {
        for (uint32_t j = 0; j < NP; j++) {
            args.push_back(
                {.kernel_idx = fused_rt::kPrepReaderKernel,
                 .core = layout.producers[h * NP + j],
                 .arg_idx = fused_rt::kPrepReaderItemArg,
                 .value = (h0 + h) * NC + j});
        }
        for (uint32_t v = 0; v < NV; v++) {
            const CoreCoord& rc = layout.receivers[h * NV + v];
            args.push_back(
                {.kernel_idx = fused_rt::kReceiverReaderKernel,
                 .core = rc,
                 .arg_idx = fused_rt::receiver_head_offset_arg(NP),
                 .value = h0});
            args.push_back(
                {.kernel_idx = fused_rt::kScanWriterKernel,
                 .core = rc,
                 .arg_idx = fused_rt::kScanWriterHeadArg,
                 .value = h0 + h});
        }
    }
    return args;
}

// ---------------------------------------------------------------------------------------------------
// Fused geometry cost model and placement
// ---------------------------------------------------------------------------------------------------
namespace {
// QB2 constants, re-measured 2026-09-25 (Tracy device time, NC=64) after the scan folded its o and
// state adds into DST accumulation: producer item under load, receiver step (period) per V-slice
// width, pipeline fill, and the phased device-time reference.
constexpr float kFillUs = 65.0f;
// Producer item (Horner WY inverse): 31 us with <= 28 concurrent producers, 34 us with >= 84 (the
// fused producer measures 33 at BH=12); linear in between. The SFPU inverse lowers it to ~26.5.
constexpr float w_p_us(uint32_t producers) {
    const float f = std::min(1.0f, std::max(0.0f, (static_cast<float>(producers) - 28.0f) / 56.0f));
    return 31.0f + 3.0f * f;
}
constexpr float t_step_us(uint32_t Vtl) {
    switch (Vtl) {
        case 1: return 2.4f;    // compute 1.43; the depth-2 hand-off round trip (~2.4) is the period floor
        case 2: return 2.9f;    // period 2.92 chain-bound, compute 2.8
        case 4: return 5.0f;    // period 4.97 chain-bound, compute 4.83
        default: return -1.0f;  // unmeasured width
    }
}
constexpr float t_phased_us(uint32_t BH, uint32_t NC) {
    // Measured device time at NC=64 (prep + scan): 4 -> 404, 8 -> 518, 12 -> 635, 16 -> 806,
    // 32 -> 1292, 48 -> 2054. Linear 277 + 31.7*BH to BH=32, then interpolated to the DRAM-saturated
    // BH=48 point.
    const float t32 = 277.0f + 31.7f * 32.0f;
    float t = (BH <= 32) ? (277.0f + 31.7f * BH) : (t32 + (2054.0f - t32) * (std::min<uint32_t>(BH, 48) - 32) / 16.0f);
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

std::vector<uint32_t> fused_even_waves(uint32_t BH, uint32_t n) {
    std::vector<uint32_t> waves;
    if (BH == 0 || n == 0) {
        return waves;
    }
    n = std::min(n, BH);
    const uint32_t base = BH / n;
    const uint32_t rem = BH % n;
    waves.reserve(n);
    for (uint32_t i = 0; i < n; i++) {
        waves.push_back(base + (i < rem ? 1u : 0u));
    }
    return waves;
}

namespace {
// Measured per-head optimum of the fused program on QB2 at NC=64: one program up to 32 heads; beyond that,
// waves of 24.
constexpr uint32_t kMaxSingleWaveHeads = 32;
constexpr uint32_t kMaxWaveHeads = 24;
}  // namespace

FusedWavePlan plan_fused_waves(
    uint32_t grid_x,
    uint32_t grid_y,
    uint32_t BH,
    uint32_t NC,
    uint32_t Vt,
    uint32_t max_wave,
    uint32_t fixed_nv,
    uint32_t fixed_np) {
    FusedWavePlan plan;
    plan.t_phased_us = t_phased_us(BH, NC);
    if (BH == 0) {
        return plan;
    }
    constexpr float kInf = std::numeric_limits<float>::infinity();
    std::vector<float> t_of(BH + 1, -1.0f);  // modelled time of a W-head wave; infinity = no geometry fits
    auto t_wave = [&](uint32_t W) {
        if (t_of[W] < 0.0f) {
            const auto c = choose_fused_geometry(grid_x, grid_y, W, NC, Vt, fixed_nv, fixed_np);
            t_of[W] = c.nv >= 1 ? c.t_fused_us : kInf;
        }
        return t_of[W];
    };
    auto t_waves = [&](const std::vector<uint32_t>& waves) {
        float t = 0.0f;
        for (const uint32_t W : waves) {
            t += t_wave(W);
        }
        return t;
    };
    const uint32_t cap = max_wave ? max_wave : kMaxWaveHeads;
    uint32_t n = (max_wave == 0 && BH <= kMaxSingleWaveHeads) ? 1u : (BH + cap - 1) / cap;
    for (; n <= BH; n++) {
        auto waves = fused_even_waves(BH, n);
        const float t = t_waves(waves);
        if (std::isfinite(t)) {
            plan.waves = std::move(waves);
            plan.t_fused_us = t;
            plan.fused_pays = t < plan.t_phased_us;
            break;
        }
    }
    return plan;
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
    uint32_t Hk,
    uint32_t num_heads,
    uint32_t head_offset,
    const std::optional<Tensor>& o_out,
    const std::optional<Tensor>& final_state_out) {
    const auto& q_shape = q.logical_shape();  // [BH,NC,C,K] head-major, or flat [B,T,Hk*K] when qk_flat
    const auto& v_shape = v.logical_shape();  // [BH,NC,C,V] head-major, or flat [B,T,HV*V] when v_flat
    // Dim derivation identical to chunk_gdn_prep (both factories consume prep's inputs; the flat forms
    // are Fused-only and rejected for Mono in validate).
    const uint32_t rows = qk_flat ? (q_shape[0] * HV) : q_shape[0];
    const uint32_t BH = num_heads ? num_heads : rows;
    const uint32_t num_chunks = qk_flat ? (q_shape[1] / chunk_size) : q_shape[1];
    const uint32_t key_dim = qk_flat ? (q_shape[2] / Hk) : q_shape[3];
    const uint32_t val_dim = v_flat ? (v_shape[2] / HV) : v_shape[3];
    auto attrs = ChunkGdnDeviceOperation::operation_attributes_t{
        .impl = ChunkGdnImpl::Mono,
        .BH = BH,
        .head_offset = head_offset,
        .rows = rows,
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
        // The model fills whatever the config leaves free (both, one, or none) so the pair fits the grid.
        const auto choice = choose_fused_geometry(
            grid0.x, grid0.y, BH, num_chunks, val_dim / tt::constants::TILE_WIDTH, nv_pin, np_pin);
        TT_FATAL(
            choice.nv >= 1,
            "chunk_gdn_fused: no fused geometry fits BH={} on a {}x{} grid with num_receivers={} num_producers={} "
            "(0 = free); the dispatch must choose phased",
            BH,
            grid0.x,
            grid0.y,
            nv_pin,
            np_pin);
        // Producers per head, clamped to num_chunks: a producer beyond NC would own no chunks (wasted
        // core, and the receiver's rotating credit c % NP would skip it anyway). Receivers per head must
        // divide Vt (validated).
        attrs.np = np_pin ? std::min<uint32_t>(np_pin, num_chunks) : choice.np;
        attrs.nv = nv_pin ? nv_pin : choice.nv;
        attrs.nbuf = fused_cfg->handoff_depth;
        TT_FATAL(
            attrs.nbuf >= 1 && attrs.nbuf <= 8,
            "chunk_gdn_fused: handoff_depth must be in [1, 8] (got {})",
            attrs.nbuf);
        attrs.unicast = fused_cfg->unicast;
        attrs.posted = fused_cfg->posted;
        TT_FATAL(
            !attrs.posted || attrs.unicast,
            "chunk_gdn_fused: posted writes require the unicast transport (unicast=true)");
        // Placement: the config's choice, else row-local whenever the (possibly pinned) geometry has a
        // row-local layout.
        const bool row_local =
            fused_cfg->row_local.value_or(fused_row_local_feasible(grid0.x, grid0.y, BH, attrs.nv, attrs.np));
        attrs.placement = row_local ? 1u : 0u;
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
        .initial_state = initial_state,
        .o_out = o_out,
        .final_state_out = final_state_out};
    return ttnn::device_operation::launch<ChunkGdnDeviceOperation>(attrs, tensor_args);
}

}  // namespace ttnn::prim
