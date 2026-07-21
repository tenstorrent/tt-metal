// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
// SPDX-License-Identifier: Apache-2.0

#include "chunk_kda.hpp"

#include <cmath>
#include <cstdlib>
#include <map>
#include <mutex>
#include <tuple>
#include <utility>
#include <vector>

#include "device/chunk_kda_device_operation.hpp"
#include "device/chunk_kda_phased.hpp"

#include "ttnn/operations/core/core.hpp"
#include "ttnn/operations/creation/creation.hpp"
#include "ttnn/operations/data_movement/concat/concat.hpp"
#include "ttnn/operations/data_movement/permute/permute.hpp"
#include "ttnn/operations/data_movement/repeat_interleave/repeat_interleave.hpp"
#include "ttnn/operations/data_movement/reshape_view/reshape.hpp"
#include "ttnn/operations/data_movement/slice/slice.hpp"
#include "ttnn/operations/copy/typecast/typecast.hpp"
#include "ttnn/operations/eltwise/binary/binary.hpp"
#include "ttnn/device.hpp"

using namespace tt::tt_metal;

namespace ttnn::transformer {

namespace {

// [B,T,Hh,D] -> [B*Hh, T, D], TILE bf16 (head-major).
ttnn::Tensor head_split_tile(const ttnn::Tensor& x, uint32_t B, uint32_t T, uint32_t Hh, uint32_t D) {
    // TILE-native head-split (permute on TILE via the transpose engine — avoids the
    // untilize-with-unpadding on the small H tile-dim, which hangs in the full op graph).
    // GPU-style mixed precision: q/k/v are bf16 (gate/decay and state stay fp32). Cast here so
    // any caller (fp32 or bf16) feeds the kernel bf16 q/k/v, matching FLA's Triton dtypes.
    ttnn::Tensor t = x;
    if (t.dtype() != DataType::BFLOAT16) {
        t = ttnn::typecast(t, DataType::BFLOAT16);
    }
    t = ttnn::permute(t, ttnn::SmallVector<int64_t>{0, 2, 1, 3});  // [B, Hh, T, D] TILE
    t = ttnn::reshape(t, ttnn::Shape({B * Hh, T, D}));             // [BH, T, D] TILE
    return t;
}

// [B,T,Hn] -> [B*Hn, T], TILE fp32 (permute on TILE, no untilize).
ttnn::Tensor headvec_split_tile(const ttnn::Tensor& x, uint32_t B, uint32_t T, uint32_t Hn) {
    ttnn::Tensor t = x;
    if (t.dtype() != DataType::FLOAT32) {
        t = ttnn::typecast(t, DataType::FLOAT32);
    }
    t = ttnn::permute(t, ttnn::SmallVector<int64_t>{0, 2, 1});  // [B, Hn, T] TILE
    t = ttnn::reshape(t, ttnn::Shape({B * Hn, T}));             // [BH, T] TILE
    return t;
}

// Pad TILE [BH, T, D] to [BH, L, D] along the time dim with zeros.
ttnn::Tensor pad_time_tile(const ttnn::Tensor& x, uint32_t BH, uint32_t D, uint32_t pad, MeshDevice* dev) {
    if (pad == 0) {
        return x;
    }
    ttnn::Tensor z =
        ttnn::zeros(ttnn::Shape({BH, pad, D}), x.dtype(), Layout::TILE, std::ref(*dev), ttnn::DRAM_MEMORY_CONFIG);
    return ttnn::concat(std::vector<ttnn::Tensor>{x, z}, 1);
}

ttnn::Tensor kda_make_const_cc(const std::vector<float>& data, uint32_t C, MeshDevice* dev) {
    ttnn::Shape shape({1, 1, C, C});
    TensorLayout layout(DataType::FLOAT32, PageConfig(Layout::TILE), ttnn::DRAM_MEMORY_CONFIG);
    TensorSpec spec(shape, layout);
    return ttnn::Tensor::from_vector(data, spec, dev);
}

struct KdaConstTiles {
    ttnn::Tensor eye, tril, ones, masks;
};

// Three 32x32 quadrant masks packed into one [1,1,32,96] tile-row (tile 0 = top-left,
// tile 1 = bottom-right, tile 2 = bottom-left). Used by the prep kernel's 16x16 sub-blocked
// WY inverse to isolate the four 16-quadrants of each 32x32 diagonal block.
ttnn::Tensor kda_make_quadrant_masks(MeshDevice* dev) {
    std::vector<float> m(32 * 96, 0.0f);
    for (uint32_t i = 0; i < 32; i++) {
        for (uint32_t j = 0; j < 32; j++) {
            const bool lo_i = i < 16, lo_j = j < 16;
            m[i * 96 + 0 * 32 + j] = (lo_i && lo_j) ? 1.0f : 0.0f;    // Qtl
            m[i * 96 + 1 * 32 + j] = (!lo_i && !lo_j) ? 1.0f : 0.0f;  // Qbr
            m[i * 96 + 2 * 32 + j] = (!lo_i && lo_j) ? 1.0f : 0.0f;   // Q10
        }
    }
    ttnn::Shape shape({1, 1, 32, 96});
    TensorLayout layout(DataType::FLOAT32, PageConfig(Layout::TILE), ttnn::DRAM_MEMORY_CONFIG);
    return ttnn::Tensor::from_vector(m, TensorSpec(shape, layout), dev);
}

// eye/tril/ones depend only on the chunk size, and the zero initial-state only on shape — none
// depend on runtime data, and all must be device-resident before trace capture (host<->device
// transfers are illegal under trace). The op therefore takes these as optional arguments so the
// CALLER owns them (built once, e.g. on the model/layer object) and their lifetime is tied to the
// device — not to a process-lifetime C++ static, which would deallocate at exit AFTER the device is
// gone and SIGSEGV. These builders are the eager-only fallback for callers that don't supply them
// (a build here does a host upload and so is NOT valid under trace capture — pass the tensors in).
KdaConstTiles kda_build_const_tiles(uint32_t C, MeshDevice* dev) {
    std::vector<float> eye_data(static_cast<size_t>(C) * C, 0.0f);
    std::vector<float> tril_data(static_cast<size_t>(C) * C, 0.0f);
    for (uint32_t i = 0; i < C; i++) {
        eye_data[i * C + i] = 1.0f;
        for (uint32_t j = 0; j <= i; j++) {
            tril_data[i * C + j] = 1.0f;
        }
    }
    std::vector<float> ones_data(static_cast<size_t>(C) * C, 1.0f);
    return KdaConstTiles{
        kda_make_const_cc(eye_data, C, dev),
        kda_make_const_cc(tril_data, C, dev),
        kda_make_const_cc(ones_data, C, dev),
        kda_make_quadrant_masks(dev)};
}

ttnn::Tensor kda_build_zero_state(uint32_t BH, uint32_t K, uint32_t V, MeshDevice* dev) {
    return ttnn::zeros(
        ttnn::Shape({BH, K, V}), DataType::FLOAT32, Layout::TILE, std::ref(*dev), ttnn::DRAM_MEMORY_CONFIG);
}

}  // namespace

std::tuple<ttnn::Tensor, std::optional<ttnn::Tensor>> chunk_kda(
    const ttnn::Tensor& q_in,
    const ttnn::Tensor& k_in,
    const ttnn::Tensor& v_in,
    const ttnn::Tensor& g_in,
    const ttnn::Tensor& beta_in,
    std::optional<float> scale_opt,
    const std::optional<ttnn::Tensor>& initial_state,
    bool output_final_state,
    uint32_t chunk_size,
    bool use_qk_l2norm,
    bool output_head_major,
    const std::optional<ttnn::MemoryConfig>& memory_config,
    const std::optional<ttnn::DeviceComputeKernelConfig>& compute_kernel_config,
    const std::optional<ttnn::Tensor>& eye,
    const std::optional<ttnn::Tensor>& tril,
    const std::optional<ttnn::Tensor>& ones,
    const std::optional<ttnn::Tensor>& masks) {
    TT_FATAL(!use_qk_l2norm, "chunk_kda: use_qk_l2norm not yet supported; pre-normalize q/k on host");

    auto* dev = q_in.device();
    const auto& qs = q_in.logical_shape();  // [B,T,H,K]   (or flat [B,T,H*K] under OPT-A)
    const auto& vs = v_in.logical_shape();  // [B,T,HV,V]  (or flat [B,T,HV*V] under OPT-A)
    const uint32_t B = qs[0];
    const uint32_t T = qs[1];
    // OPT-A (QWEN_GDN_FLAT_QKV): rank-3 q/k/v are FLAT token-major tensors — the adapter skipped the
    // head-split relayout. Head counts can't be read off a flat width, so: HV comes from beta [B,T,HV];
    // for the flat q/k path we assume per-head K==V (true for GDN: linear_key_head_dim==value_head_dim),
    // so K=V and H = q_flat_width / K. The prep reader tile-addresses q/k/v out of the flat grids.
    const bool flat_v = (vs.rank() == 3);
    const bool flat_qk = (qs.rank() == 3);
    const uint32_t HV = flat_v ? beta_in.logical_shape()[2] : vs[2];
    const uint32_t V = flat_v ? (vs[2] / HV) : vs[3];
    const uint32_t K = flat_qk ? V : qs[3];
    const uint32_t H = flat_qk ? (qs[2] / K) : qs[2];
    if (flat_qk) {
        TT_FATAL(qs[2] == H * K, "flat q width {} != H*K ({}*{}); flat q/k path assumes K==V", qs[2], H, K);
    }
    TT_FATAL(HV % H == 0, "HV ({}) must be divisible by H ({})", HV, H);
    const uint32_t G = HV / H;
    const uint32_t BH = B * HV;

    const float scale = scale_opt.has_value() ? *scale_opt : (1.0f / std::sqrt(static_cast<float>(K)));

    const uint32_t C = chunk_size;
    const uint32_t pad = (C - (T % C)) % C;
    const uint32_t L = T + pad;
    const uint32_t NC = L / C;

    // Head-split (row-major fp32). OPT-A: flat q/k stay token-major [B,T,H*K] (just bf16-cast); the
    // prep reader tile-addresses head hk's chunk c and does the GQA head-map itself.
    auto as_bf16 = [&](const ttnn::Tensor& t) {
        return t.dtype() != DataType::BFLOAT16 ? ttnn::typecast(t, DataType::BFLOAT16) : t;
    };
    ttnn::Tensor q = flat_qk ? as_bf16(q_in) : head_split_tile(q_in, B, T, H, K);
    ttnn::Tensor k = flat_qk ? as_bf16(k_in) : head_split_tile(k_in, B, T, H, K);
    // OPT-A: flat v stays token-major [B,T,HV*V] (just bf16-cast); the prep reader addresses it.
    // Otherwise head-split to [BH,T,V] as usual.
    ttnn::Tensor v = flat_v ? (v_in.dtype() != DataType::BFLOAT16 ? ttnn::typecast(v_in, DataType::BFLOAT16) : v_in)
                            : head_split_tile(v_in, B, T, HV, V);
    ttnn::Tensor g = headvec_split_tile(g_in, B, T, HV);        // [B*HV, T] TILE
    ttnn::Tensor beta = headvec_split_tile(beta_in, B, T, HV);  // [B*HV, T] TILE

    // GQA expand q,k from H heads to HV heads (repeat_interleave along head-major dim 0).
    // OPT-A: for flat q/k the reader maps value-head hv -> key-head hk=hv/G at read time, so no expand.
    if (G > 1 && !flat_qk) {
        q = ttnn::repeat_interleave(q, G, 0);
        k = ttnn::repeat_interleave(k, G, 0);
    }

    // OPT-B: flat q/k arrive raw (a flat tensor can't be L2-normed over D on host), so the prep kernel
    // L2-normalizes q/k over K and folds q's `scale` into that norm. Thus qk_norm is exactly "q/k came
    // in flat" (Ct==1 only; the in-kernel norm uses cb_supd/cb_stmp, free only at chunk_size==32). When
    // NOT flat, q/k are already host-normalized, so we fold scale into q here as before.
    const bool qk_norm = flat_qk && (C == 32);
    if (!qk_norm) {
        q = ttnn::multiply(q, scale);
    }

    // Pad time to a multiple of C (q/k/v are TILE; g/beta are RM). Flat tensors require pad==0
    // (asserted below), so no padding for the flat q/k/v.
    if (!flat_qk) {
        q = pad_time_tile(q, BH, K, pad, dev);
        k = pad_time_tile(k, BH, K, pad, dev);
    }
    if (!flat_v) {
        v = pad_time_tile(v, BH, V, pad, dev);
    }
    // g, beta are [BH, T] TILE; pad along dim 1.
    if (pad > 0) {
        ttnn::Tensor zc = ttnn::zeros(
            ttnn::Shape({BH, pad}), DataType::FLOAT32, Layout::TILE, std::ref(*dev), ttnn::DRAM_MEMORY_CONFIG);
        g = ttnn::concat(std::vector<ttnn::Tensor>{g, zc}, 1);
        beta = ttnn::concat(std::vector<ttnn::Tensor>{beta, zc}, 1);
    }

    // q/k/v already TILE -> just reshape to per-chunk [BH, NC, C, D].
    auto to_chunks_tile = [&](const ttnn::Tensor& t, uint32_t D) {
        return ttnn::reshape(t, ttnn::Shape({BH, NC, C, D}));
    };
    // OPT-A: flat q/k/v are passed straight to the prep prim (reader does the per-chunk addressing).
    ttnn::Tensor q_c = flat_qk ? q : to_chunks_tile(q, K);
    ttnn::Tensor k_c = flat_qk ? k : to_chunks_tile(k, K);
    ttnn::Tensor v_c = flat_v ? v : to_chunks_tile(v, V);
    // g, beta -> [BH, NC, C, 1] TILE (already TILE; reshape only).
    ttnn::Tensor g_c = ttnn::reshape(g, ttnn::Shape({BH, NC, C, 1}));
    ttnn::Tensor beta_c = ttnn::reshape(beta, ttnn::Shape({BH, NC, C, 1}));

    // Constant tiles eye_C, tril_C, ones_C [1,1,C,C], masks [1,1,32,96]. Caller-supplied (built once
    // on the model/layer and passed in) so they're device-resident before trace capture and their
    // lifetime is device-scoped. If a caller omits them we build here (eager only — a build does a
    // host upload, which is illegal under trace capture, so traced callers MUST pass them in).
    const bool has_const_tiles = eye.has_value() && tril.has_value() && ones.has_value() && masks.has_value();
    KdaConstTiles ct_fallback;
    if (!has_const_tiles) {
        ct_fallback = kda_build_const_tiles(C, dev);
    }
    const ttnn::Tensor& eye_c = has_const_tiles ? *eye : ct_fallback.eye;
    const ttnn::Tensor& tril_c = has_const_tiles ? *tril : ct_fallback.tril;
    const ttnn::Tensor& ones_c = has_const_tiles ? *ones : ct_fallback.ones;
    const ttnn::Tensor& masks_c = has_const_tiles ? *masks : ct_fallback.masks;

    // Initial state [B,HV,K,V] -> [BH,K,V] fp32 TILE. Always provide (zeros if absent) so the reader
    // always reads S (no in-kernel zeroing). Traced callers pass a persistent state buffer (never
    // absent); the zeros() fallback here is eager-only (device-side fill, uncached).
    std::optional<ttnn::Tensor> s0;
    if (initial_state.has_value()) {
        ttnn::Tensor s = *initial_state;
        if (s.dtype() != DataType::FLOAT32) {
            s = ttnn::typecast(s, DataType::FLOAT32);
        }
        s0 = ttnn::reshape(s, ttnn::Shape({BH, K, V}));
    } else {
        s0 = kda_build_zero_state(BH, K, V, dev);
    }

    const auto out_mem = memory_config.value_or(ttnn::DRAM_MEMORY_CONFIG);
    const auto kernel_cfg = init_device_compute_kernel_config(
        dev->arch(),
        compute_kernel_config,
        MathFidelity::HiFi4,
        /*default_approx_mode=*/false,
        /*default_fp32_acc=*/true,
        /*default_l1_acc=*/false);

    // Phase-split path: prep -> (DRAM hand-off) -> scan. The prep phase does all state-independent
    // per-chunk work (incl. the WY inverse) fanned across the grid; the scan phase carries the
    // recurrent state. Same math as the monolithic op. This is the DEFAULT; QWEN_GDN_PHASED=0 falls
    // back to the single-kernel monolithic op (benchmark/debug only). Read fresh (not static) so a
    // caller toggling it between calls is honored.
    const bool phased = [] {
        const char* e = std::getenv("QWEN_GDN_PHASED");
        return e == nullptr || e[0] != '0';
    }();

    ttnn::Tensor o_c;          // [BH, NC, C, V]
    ttnn::Tensor final_state;  // [BH, K, V]
    TT_FATAL(!flat_v || phased, "OPT-A flat v is only supported on the phased path (set QWEN_GDN_PHASED=1)");
    TT_FATAL(!flat_v || pad == 0, "OPT-A flat v requires T ({}) to be a multiple of chunk_size ({})", T, C);
    TT_FATAL(!flat_qk || (phased && qk_norm), "OPT-A flat q/k needs the phased path + in-kernel norm (Ct==1)");
    TT_FATAL(!flat_qk || pad == 0, "OPT-A flat q/k requires T ({}) to be a multiple of chunk_size ({})", T, C);
    if (phased) {
        auto prep = ttnn::prim::chunk_kda_prep(
            q_c,
            k_c,
            v_c,
            g_c,
            beta_c,
            eye_c,
            tril_c,
            ones_c,
            masks_c,
            C,
            out_mem,
            kernel_cfg,
            flat_v,
            HV,
            qk_norm,
            scale,
            flat_qk,
            H);
        // prep = {v_beta, kd, q_decay, intra, k_dec_t, dl, t_inv}
        auto scan = ttnn::prim::chunk_kda_scan(
            prep[0],
            prep[1],
            prep[2],
            prep[3],
            prep[4],
            prep[5],
            prep[6],
            s0,
            C,
            output_final_state,
            out_mem,
            kernel_cfg);
        o_c = scan[0];
        final_state = scan[1];
        // DEBUG: QWEN_GDN_DUMP=<idx> routes prep[idx] out through the o path (idx 2 = q_decay,
        // shape [BH,NC,C,K]; only valid to view via o when K==V). Isolates prep-write/scan-read bugs.
        static const char* dumpenv = std::getenv("QWEN_GDN_DUMP");
        if (dumpenv != nullptr && dumpenv[0] != '\0') {
            o_c = prep[static_cast<size_t>(std::atoi(dumpenv))];
        }
    } else {
        auto results = ttnn::prim::chunk_kda(
            q_c, k_c, v_c, g_c, beta_c, eye_c, tril_c, ones_c, s0, C, output_final_state, out_mem, kernel_cfg);
        o_c = results[0];
        final_state = results[1];
    }

    std::optional<ttnn::Tensor> final_opt;
    if (output_final_state) {
        final_opt = ttnn::reshape(final_state, ttnn::Shape({B, HV, K, V}));
    }

    // Head-major output [BH,T,V] TILE: the kernel already produced o head-major, so avoid the
    // token<->head permute round-trip (the default path permutes to [B,T,HV,V] and the GDN
    // adapter permutes right back). C and V are tile-aligned, so when there is no time padding
    // the fold NC,C -> T is a pure metadata reshape (zero relayout).
    if (output_head_major) {
        ttnn::Tensor o;
        if (pad == 0) {
            o = ttnn::reshape(o_c, ttnn::Shape({BH, L, V}));  // [BH,T,V] TILE, metadata-only
        } else {
            ttnn::Tensor t = ttnn::to_layout(o_c, Layout::ROW_MAJOR);
            t = ttnn::reshape(t, ttnn::Shape({BH, L, V}));
            t = ttnn::slice(
                t,
                ttnn::SmallVector<int32_t>{0, 0, 0},
                ttnn::SmallVector<int32_t>{static_cast<int32_t>(BH), static_cast<int32_t>(T), static_cast<int32_t>(V)},
                ttnn::SmallVector<int32_t>{1, 1, 1});
            o = ttnn::to_layout(t, Layout::TILE);  // [BH,T,V] TILE
        }
        return {o, final_opt};
    }

    // Default: token-major o [BH,NC,C,V] -> [B,T,HV,V] (ROW_MAJOR).
    ttnn::Tensor o = ttnn::to_layout(o_c, Layout::ROW_MAJOR);
    o = ttnn::reshape(o, ttnn::Shape({BH, L, V}));
    if (pad > 0) {
        o = ttnn::slice(
            o,
            ttnn::SmallVector<int32_t>{0, 0, 0},
            ttnn::SmallVector<int32_t>{static_cast<int32_t>(BH), static_cast<int32_t>(T), static_cast<int32_t>(V)},
            ttnn::SmallVector<int32_t>{1, 1, 1});
    }
    o = ttnn::reshape(o, ttnn::Shape({B, HV, T, V}));
    o = ttnn::permute(o, ttnn::SmallVector<int64_t>{0, 2, 1, 3});  // [B,T,HV,V] (ROW_MAJOR)
    // NOTE: returned in ROW_MAJOR. Tilizing [B,T,HV,V] with HV in the tile dim is avoided
    // here (a TILE round-trip on the small HV tile-dim was problematic); callers can tilize.
    return {o, final_opt};
}

}  // namespace ttnn::transformer
