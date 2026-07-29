// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
// SPDX-License-Identifier: Apache-2.0

#include "chunk_gated_delta_rule.hpp"

#include <cmath>
#include <cstdlib>
#include <map>
#include <mutex>
#include <tuple>
#include <utility>
#include <vector>

#include "device/chunk_gated_delta_rule_device_operation.hpp"
#include "device/chunk_gdn_phased.hpp"

#include "ttnn/operations/ccl/broadcast/broadcast.hpp"
#include "ttnn/operations/core/core.hpp"
#include "ttnn/operations/creation/creation.hpp"
#include "ttnn/operations/data_movement/common/common.hpp"
#include "ttnn/operations/data_movement/clone/clone.hpp"
#include "ttnn/operations/data_movement/concat/concat.hpp"
#include "ttnn/operations/data_movement/permute/permute.hpp"
#include "ttnn/operations/data_movement/repeat_interleave/repeat_interleave.hpp"
#include "ttnn/operations/data_movement/reshape_view/reshape.hpp"
#include "ttnn/operations/data_movement/slice/slice.hpp"
#include "ttnn/operations/copy/typecast/typecast.hpp"
#include "ttnn/operations/eltwise/binary/binary.hpp"
#include "ttnn/operations/matmul/matmul.hpp"
#include "ttnn/operations/point_to_point/point_to_point.hpp"
#include "ttnn/device.hpp"
#include <tt-metalium/allocator.hpp>
#include <tt-metalium/work_split.hpp>

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

// [B,T,H,D] -> [B*H,T,D], TILE fp32.
ttnn::Tensor head_split_float_tile(const ttnn::Tensor& x, uint32_t B, uint32_t T, uint32_t H, uint32_t D) {
    ttnn::Tensor t = x.dtype() == DataType::FLOAT32 ? x : ttnn::typecast(x, DataType::FLOAT32);
    t = ttnn::permute(t, ttnn::SmallVector<int64_t>{0, 2, 1, 3});
    return ttnn::reshape(t, ttnn::Shape({B * H, T, D}));
}
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

ttnn::Tensor make_const_cc(const std::vector<float>& data, uint32_t C, MeshDevice* dev) {
    ttnn::Shape shape({1, 1, C, C});
    TensorLayout layout(DataType::FLOAT32, PageConfig(Layout::TILE), ttnn::DRAM_MEMORY_CONFIG);
    tt::tt_metal::TensorSpec spec(shape, layout);
    return ttnn::Tensor::from_vector(data, spec, dev);
}

struct ConstTiles {
    ttnn::Tensor eye, tril, ones, masks;
};

size_t chunk_gdn_prep_l1_bytes_per_bank(
    uint32_t BH,
    uint32_t NC,
    uint32_t C,
    uint32_t K,
    uint32_t V,
    bool vector_gate,
    uint32_t output_bf16_mask,
    MeshDevice* device) {
    const auto spec = [&](const ttnn::Shape& shape, uint32_t output_index) {
        const auto dtype = (output_bf16_mask & (1u << output_index)) ? DataType::BFLOAT16 : DataType::FLOAT32;
        return tt::tt_metal::TensorSpec(shape, TensorLayout(dtype, PageConfig(Layout::TILE), ttnn::L1_MEMORY_CONFIG));
    };
    const std::vector<tt::tt_metal::TensorSpec> specs = {
        spec(ttnn::Shape({BH, NC, C, V}), 0),
        spec(ttnn::Shape({BH, NC, C, K}), 1),
        spec(ttnn::Shape({BH, NC, C, K}), 2),
        spec(ttnn::Shape({BH, NC, C, C}), 3),
        spec(ttnn::Shape({BH, NC, K, C}), 4),
        spec(ttnn::Shape({BH, NC, vector_gate ? K : 1, 1}), 5),
        spec(ttnn::Shape({BH, NC, C, C}), 6),
    };
    const auto num_banks = device->allocator()->get_num_banks(BufferType::L1);
    const auto alignment = device->allocator()->get_alignment(BufferType::L1);
    size_t bytes_per_bank = 0;
    for (const auto& output_spec : specs) {
        bytes_per_bank += tt::tt_metal::detail::calculate_bank_size_spread(
            output_spec.compute_packed_buffer_size_bytes(),
            output_spec.compute_page_size_bytes(),
            num_banks,
            alignment);
    }
    return bytes_per_bank;
}

ttnn::Tensor slice_group_axis(
    const ttnn::Tensor& tensor, uint32_t start, uint32_t end, const tt::tt_metal::MemoryConfig& memory_config) {
    const auto& shape = tensor.logical_shape();
    TT_FATAL(shape.rank() == 4, "group-axis slice expects a rank-4 tensor");
    return ttnn::slice(
        tensor,
        ttnn::SmallVector<int32_t>{0, static_cast<int32_t>(start), 0, 0},
        ttnn::SmallVector<int32_t>{
            static_cast<int32_t>(shape[0]),
            static_cast<int32_t>(end),
            static_cast<int32_t>(shape[2]),
            static_cast<int32_t>(shape[3])},
        ttnn::SmallVector<int32_t>{1, 1, 1, 1},
        memory_config);
}

std::pair<ttnn::Tensor, ttnn::Tensor> inclusive_affine_prefix(
    ttnn::Tensor transform_a,
    ttnn::Tensor transform_b,
    uint32_t groups_per_head,
    const tt::tt_metal::MemoryConfig& memory_config,
    const DeviceComputeKernelConfig& compute_kernel_config) {
    // Hillis-Steele scan: at each power-of-two distance, T[i] becomes
    // T[i] composed after the prefix ending at i-distance.
    for (uint32_t distance = 1; distance < groups_per_head; distance *= 2) {
        auto leading_a = slice_group_axis(transform_a, 0, distance, memory_config);
        auto leading_b = slice_group_axis(transform_b, 0, distance, memory_config);
        auto after_a = slice_group_axis(transform_a, distance, groups_per_head, memory_config);
        auto after_b = slice_group_axis(transform_b, distance, groups_per_head, memory_config);
        auto before_a = slice_group_axis(transform_a, 0, groups_per_head - distance, memory_config);
        auto before_b = slice_group_axis(transform_b, 0, groups_per_head - distance, memory_config);
        auto composed_a = ttnn::matmul(
            after_a,
            before_a,
            false,
            false,
            memory_config,
            DataType::FLOAT32,
            std::nullopt,
            std::nullopt,
            compute_kernel_config);
        auto composed_b = ttnn::matmul(
            after_a,
            before_b,
            false,
            false,
            memory_config,
            DataType::FLOAT32,
            std::nullopt,
            std::nullopt,
            compute_kernel_config);
        composed_b = ttnn::add(composed_b, after_b, std::nullopt, memory_config);
        transform_a = ttnn::concat({leading_a, composed_a}, 1, memory_config);
        transform_b = ttnn::concat({leading_b, composed_b}, 1, memory_config);
    }
    return {transform_a, transform_b};
}

// Three 32x32 quadrant masks packed into one [1,1,32,96] tile-row (tile 0 = top-left,
// tile 1 = bottom-right, tile 2 = bottom-left). Used by the prep kernel's 16x16 sub-blocked
// WY inverse to isolate the four 16-quadrants of each 32x32 diagonal block.
ttnn::Tensor make_quadrant_masks(MeshDevice* dev) {
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
    return ttnn::Tensor::from_vector(m, tt::tt_metal::TensorSpec(shape, layout), dev);
}

// eye/tril/ones depend only on the chunk size, and the zero initial-state only on shape — none
// depend on runtime data, and all must be device-resident before trace capture (host<->device
// transfers are illegal under trace). The op therefore takes these as optional arguments so the
// CALLER owns them (built once, e.g. on the model/layer object) and their lifetime is tied to the
// device — not to a process-lifetime C++ static, which would deallocate at exit AFTER the device is
// gone and SIGSEGV. These builders are the eager-only fallback for callers that don't supply them
// (a build here does a host upload and so is NOT valid under trace capture — pass the tensors in).
ConstTiles build_const_tiles(uint32_t C, MeshDevice* dev) {
    std::vector<float> eye_data(static_cast<size_t>(C) * C, 0.0f);
    std::vector<float> tril_data(static_cast<size_t>(C) * C, 0.0f);
    for (uint32_t i = 0; i < C; i++) {
        eye_data[i * C + i] = 1.0f;
        for (uint32_t j = 0; j <= i; j++) {
            tril_data[i * C + j] = 1.0f;
        }
    }
    std::vector<float> ones_data(static_cast<size_t>(C) * C, 1.0f);
    return ConstTiles{
        make_const_cc(eye_data, C, dev),
        make_const_cc(tril_data, C, dev),
        make_const_cc(ones_data, C, dev),
        make_quadrant_masks(dev)};
}

ttnn::Tensor build_zero_state(uint32_t BH, uint32_t K, uint32_t V, MeshDevice* dev) {
    return ttnn::zeros(
        ttnn::Shape({BH, K, V}), DataType::FLOAT32, Layout::TILE, std::ref(*dev), ttnn::DRAM_MEMORY_CONFIG);
}

}  // namespace

std::tuple<ttnn::Tensor, std::optional<ttnn::Tensor>> chunk_gated_delta_rule(
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
    TT_FATAL(!use_qk_l2norm, "chunk_gated_delta_rule: use_qk_l2norm not yet supported; pre-normalize q/k on host");

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
        TT_FATAL(!flat_v || pad == 0, "chunk_kda flat v requires T to be divisible by chunk_size");
        if (!flat_v) {
            v = pad_time_tile(v, BH, V, pad, dev);
        }
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
    ConstTiles ct_fallback;
    if (!has_const_tiles) {
        ct_fallback = build_const_tiles(C, dev);
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
        s0 = build_zero_state(BH, K, V, dev);
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
        auto prep = ttnn::prim::chunk_gdn_prep(
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
        auto scan = ttnn::prim::chunk_gdn_scan(
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
        auto results = ttnn::prim::chunk_gated_delta_rule(
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

std::tuple<ttnn::Tensor, std::optional<ttnn::Tensor>> chunk_kda(
    const ttnn::Tensor& q_in,
    const ttnn::Tensor& k_in,
    const ttnn::Tensor& v_in,
    const ttnn::Tensor& g_in,
    const ttnn::Tensor& beta_in,
    std::optional<float> scale_opt,
    const std::optional<ttnn::Tensor>& initial_state,
    bool output_final_state,
    bool output_head_major,
    uint32_t chunk_size,
    const std::optional<ttnn::MemoryConfig>& memory_config,
    const std::optional<ttnn::DeviceComputeKernelConfig>& compute_kernel_config,
    const std::optional<ttnn::Tensor>& eye,
    const std::optional<ttnn::Tensor>& tril,
    const std::optional<ttnn::Tensor>& ones,
    const std::optional<ttnn::Tensor>& masks,
    const std::optional<ttnn::Tensor>& rms_gate,
    const std::optional<ttnn::Tensor>& rms_weight,
    float rms_epsilon,
    uint32_t summary_group_chunks,
    const std::optional<uint32_t>& sequence_parallel_axis,
    const std::optional<ttnn::Tensor>& affine_identity,
    const std::optional<ttnn::Tensor>& affine_zero) {
    const auto& qs = q_in.logical_shape();
    const auto& vs = v_in.logical_shape();
    const auto& gs = g_in.logical_shape();
    const auto& bs = beta_in.logical_shape();
    const bool flat_v = vs.rank() == 3;
    const bool flat_qk = qs.rank() == 3;
    const bool flat_g = gs.rank() == 3;
    TT_FATAL(flat_g || gs.rank() == 4, "chunk_kda expects rank-3 or rank-4 g");
    TT_FATAL(bs.rank() == 3, "chunk_kda beta must be [B,T,H]");
    const uint32_t B = bs[0], T = bs[1], H = bs[2];
    TT_FATAL(!flat_g || gs[2] % H == 0, "chunk_kda flat g width {} must be divisible by H={}", gs[2], H);
    const uint32_t K = flat_g ? (gs[2] / H) : gs[3];
    TT_FATAL(flat_qk || qs.rank() == 4, "chunk_kda expects rank-3 or rank-4 q/k");
    TT_FATAL(flat_v || vs.rank() == 4, "chunk_kda expects rank-3 or rank-4 v");
    TT_FATAL(!flat_qk || qs[2] == H * K, "chunk_kda flat q/k width {} must equal H*K={}*{}", qs[2], H, K);
    TT_FATAL(!flat_v || vs[2] % H == 0, "chunk_kda flat v width {} must be divisible by H={}", vs[2], H);
    const uint32_t V = flat_v ? (vs[2] / H) : vs[3];
    const bool distributed_prefix = sequence_parallel_axis.has_value();
    TT_FATAL(
        distributed_prefix == affine_identity.has_value() && distributed_prefix == affine_zero.has_value(),
        "sequence_parallel_axis, affine_identity, and affine_zero must be provided together");
    TT_FATAL(!distributed_prefix || *sequence_parallel_axis < 2, "sequence_parallel_axis must be 0 or 1");
    TT_FATAL(chunk_size == 32, "chunk_kda currently requires chunk_size=32, got {}", chunk_size);
    TT_FATAL(
        k_in.logical_shape() == qs && qs[0] == B && qs[1] == T &&
            (flat_qk ? qs[2] == H * K : (qs[2] == H && qs[3] == K)) && vs[0] == B && vs[1] == T &&
            (flat_v ? vs[2] == H * V : (vs[2] == H && vs[3] == V)),
        "chunk_kda q/k/v shapes are inconsistent");
    TT_FATAL(
        gs[0] == B && gs[1] == T && (flat_g ? gs[2] == H * K : (gs[2] == H && gs[3] == K)),
        "chunk_kda g must be [B,T,H,K] or flat [B,T,H*K]");
    TT_FATAL(bs[0] == B && bs[1] == T && bs[2] == H, "chunk_kda beta must be [B,T,H]");

    auto* dev = q_in.device();
    const uint32_t BH = B * H;
    const uint32_t C = chunk_size;
    const uint32_t pad = (C - (T % C)) % C;
    const uint32_t L = T + pad;
    const uint32_t NC = L / C;
    const float scale = scale_opt.value_or(1.0f / std::sqrt(static_cast<float>(K)));

    auto as_bf16 = [](const ttnn::Tensor& tensor) {
        return tensor.dtype() == DataType::BFLOAT16 ? tensor : ttnn::typecast(tensor, DataType::BFLOAT16);
    };
    ttnn::Tensor q = flat_qk ? as_bf16(q_in) : ttnn::multiply(head_split_tile(q_in, B, T, H, K), scale);
    ttnn::Tensor k = flat_qk ? as_bf16(k_in) : head_split_tile(k_in, B, T, H, K);
    ttnn::Tensor v = flat_v ? (v_in.dtype() == DataType::BFLOAT16 ? v_in : ttnn::typecast(v_in, DataType::BFLOAT16))
                            : head_split_tile(v_in, B, T, H, V);
    ttnn::Tensor g = flat_g ? g_in : head_split_float_tile(g_in, B, T, H, K);
    ttnn::Tensor beta = headvec_split_tile(beta_in, B, T, H);
    TT_FATAL(!flat_qk || pad == 0, "chunk_kda flat q/k requires T to be divisible by chunk_size");
    TT_FATAL(!flat_v || pad == 0, "chunk_kda flat v requires T to be divisible by chunk_size");
    TT_FATAL(!flat_g || pad == 0, "chunk_kda flat g requires T to be divisible by chunk_size");
    if (!flat_qk) {
        q = pad_time_tile(q, BH, K, pad, dev);
        k = pad_time_tile(k, BH, K, pad, dev);
    }
    if (!flat_v) {
        v = pad_time_tile(v, BH, V, pad, dev);
    }
    if (!flat_g) {
        g = pad_time_tile(g, BH, K, pad, dev);
    }
    if (pad > 0) {
        auto zeros = ttnn::zeros(
            ttnn::Shape({BH, pad}), DataType::FLOAT32, Layout::TILE, std::ref(*dev), ttnn::DRAM_MEMORY_CONFIG);
        beta = ttnn::concat(std::vector<ttnn::Tensor>{beta, zeros}, 1);
    }
    if (!flat_qk) {
        q = ttnn::reshape(q, ttnn::Shape({BH, NC, C, K}));
        k = ttnn::reshape(k, ttnn::Shape({BH, NC, C, K}));
    }
    if (!flat_v) {
        v = ttnn::reshape(v, ttnn::Shape({BH, NC, C, V}));
    }
    if (!flat_g) {
        g = ttnn::reshape(g, ttnn::Shape({BH, NC, C, K}));
    }
    beta = ttnn::reshape(beta, ttnn::Shape({BH, NC, C, 1}));

    const bool has_const_tiles = eye.has_value() && tril.has_value() && ones.has_value() && masks.has_value();
    ConstTiles fallback;
    if (!has_const_tiles) {
        fallback = build_const_tiles(C, dev);
    }
    const auto& eye_c = has_const_tiles ? *eye : fallback.eye;
    const auto& tril_c = has_const_tiles ? *tril : fallback.tril;
    const auto& ones_c = has_const_tiles ? *ones : fallback.ones;
    const auto& masks_c = has_const_tiles ? *masks : fallback.masks;

    std::optional<ttnn::Tensor> s0;
    if (initial_state.has_value()) {
        auto state = initial_state->dtype() == DataType::FLOAT32 ? *initial_state
                                                                 : ttnn::typecast(*initial_state, DataType::FLOAT32);
        s0 = ttnn::reshape(state, ttnn::Shape({BH, K, V}));
    } else {
        s0 = build_zero_state(BH, K, V, dev);
    }

    const auto out_mem = memory_config.value_or(ttnn::DRAM_MEMORY_CONFIG);
    const auto kernel_cfg = init_device_compute_kernel_config(
        dev->arch(),
        compute_kernel_config,
        MathFidelity::HiFi4,
        /*default_approx_mode=*/false,
        /*default_fp32_acc=*/true,
        /*default_l1_acc=*/false);
    // Selected by the storage sweep; the private env override preserves FP32 and A/B replay.
    uint32_t prep_bf16_mask = 0x26;
    if (const char* env = std::getenv("QWEN_KDA_PREP_BF16_MASK")) {
        char* end = nullptr;
        const auto parsed = std::strtoul(env, &end, 0);
        TT_FATAL(
            end != env && *end == '\0' && parsed <= 0xFFFFFFFFUL, "QWEN_KDA_PREP_BF16_MASK must be a uint32 integer");
        prep_bf16_mask = static_cast<uint32_t>(parsed);
    }
    TT_FATAL((prep_bf16_mask & ~0x37u) == 0, "unsupported KDA prep BF16 mask 0x{:x}", prep_bf16_mask);
    const auto prep_cb_bytes = ttnn::prim::chunk_gdn_prep_cb_size_bytes(C, K, V, true, g.dtype(), prep_bf16_mask);
    const auto prep_output_bytes_per_bank =
        chunk_gdn_prep_l1_bytes_per_bank(BH, NC, C, K, V, true, prep_bf16_mask, dev);
    const auto l1_largest_free_block = dev->allocator()->get_statistics(BufferType::L1).largest_free_block_bytes;
    const bool prep_fits_l1 = prep_cb_bytes + prep_output_bytes_per_bank <= l1_largest_free_block;
    // L1 is faster for the retained small geometry, but prep tensors and the program's static CBs
    // share each worker bank. Fall back to DRAM when their exact combined footprint cannot fit.
    const bool force_prep_dram = std::getenv("QWEN_KDA_PREP_DRAM") != nullptr;
    const auto prep_mem =
        distributed_prefix || force_prep_dram || !prep_fits_l1 ? ttnn::DRAM_MEMORY_CONFIG : ttnn::L1_MEMORY_CONFIG;
    auto prep = ttnn::prim::chunk_gdn_prep(
        q,
        k,
        v,
        g,
        beta,
        eye_c,
        tril_c,
        ones_c,
        masks_c,
        C,
        prep_mem,
        kernel_cfg,
        flat_v,
        H,
        flat_qk,
        scale,
        flat_qk,
        H,
        flat_g,
        true,
        prep_bf16_mask);
    std::optional<std::vector<ttnn::Tensor>> grouped_scan;
    std::optional<ttnn::Tensor> distributed_final_state;
    // Configurable affine-summary construction: contiguous chunks become one
    // independent pseudo-head. Running the proven recurrence from zero gives B; running
    // from I gives A+B. State-only mode drains token outputs without materializing them.
    const bool use_persistent_group_prefix = NC >= 160 && std::getenv("QWEN_KDA_SERIAL_SCAN") == nullptr;
    const bool build_group_summaries = distributed_prefix || std::getenv("QWEN_KDA_GROUP_SUMMARY") != nullptr ||
                                       std::getenv("QWEN_KDA_GROUP_PREFIX") != nullptr || use_persistent_group_prefix;
    if (build_group_summaries) {
        TT_FATAL(summary_group_chunks > 0, "summary_group_chunks must be positive");
        TT_FATAL(
            NC % summary_group_chunks == 0,
            "local chunk count {} must be divisible by summary_group_chunks {}",
            NC,
            summary_group_chunks);
        TT_FATAL(K == V, "grouped KDA affine prefix currently requires K == V, got K={} and V={}", K, V);
        const uint32_t groups_per_head = NC / summary_group_chunks;
        const uint32_t group_heads = BH * groups_per_head;
        const auto worker_grid = dev->compute_with_storage_grid_size();
        TT_FATAL(
            group_heads <= worker_grid.x * worker_grid.y,
            "grouped KDA needs {} summary owners (B*local_heads*local_groups), but only {} worker cores are available",
            group_heads,
            worker_grid.x * worker_grid.y);
        auto grouped = prep;
        grouped[0] = ttnn::reshape(grouped[0], ttnn::Shape({group_heads, summary_group_chunks, C, V}));
        grouped[1] = ttnn::reshape(grouped[1], ttnn::Shape({group_heads, summary_group_chunks, C, K}));
        grouped[2] = ttnn::reshape(grouped[2], ttnn::Shape({group_heads, summary_group_chunks, C, K}));
        grouped[3] = ttnn::reshape(grouped[3], ttnn::Shape({group_heads, summary_group_chunks, C, C}));
        grouped[4] = ttnn::reshape(grouped[4], ttnn::Shape({group_heads, summary_group_chunks, K, C}));
        grouped[5] = ttnn::reshape(grouped[5], ttnn::Shape({group_heads, summary_group_chunks, K, 1}));
        grouped[6] = ttnn::reshape(grouped[6], ttnn::Shape({group_heads, summary_group_chunks, C, C}));
        auto summary_mem = prep_mem;
        if (std::getenv("QWEN_KDA_SUMMARY_INTERLEAVED") == nullptr) {
            const auto summary_cores =
                tt::tt_metal::num_cores_to_corerangeset(group_heads, dev->compute_with_storage_grid_size(), true);
            summary_mem = ttnn::operations::data_movement::create_sharded_memory_config(
                ttnn::Shape({group_heads, K, K}),
                summary_cores,
                ttnn::operations::data_movement::ShardStrategy::HEIGHT,
                ShardOrientation::ROW_MAJOR,
                std::array<uint32_t, 2>{K, K},
                Layout::TILE);
        }
        auto summaries = ttnn::prim::chunk_gdn_scan(
            grouped[0],
            grouped[1],
            grouped[2],
            grouped[3],
            grouped[4],
            grouped[5],
            grouped[6],
            std::nullopt,
            C,
            true,
            summary_mem,
            kernel_cfg,
            true,
            true,
            eye_c,
            std::nullopt,
            std::nullopt,
            0,
            1e-5f,
            true);
        auto summary_a = summaries[0];
        auto summary_b = summaries[1];
        if (distributed_prefix || std::getenv("QWEN_KDA_GROUP_PREFIX") != nullptr || use_persistent_group_prefix) {
            TT_FATAL(s0.has_value(), "group-prefix scan requires initial state");
            const auto prefix_mem = distributed_prefix
                                        ? out_mem
                                        : (std::getenv("QWEN_KDA_PREFIX_DRAM") == nullptr ? ttnn::L1_MEMORY_CONFIG
                                                                                          : ttnn::DRAM_MEMORY_CONFIG);
            if (distributed_prefix) {
                auto [partition_a, partition_b] =
                    ttnn::prim::kda_affine_compose(summary_a, summary_b, groups_per_head, prefix_mem, kernel_cfg);
                auto identity = ttnn::reshape(*affine_identity, ttnn::Shape({BH, K, K}));
                auto zero = ttnn::reshape(*affine_zero, ttnn::Shape({BH, K, V}));
                auto [partition_entry_state, final_state] = kda_distributed_affine_prefix(
                    partition_a, partition_b, *s0, identity, zero, *sequence_parallel_axis, out_mem, kernel_cfg);
                distributed_final_state = final_state;
                auto group_initial_states = ttnn::prim::kda_affine_prefix(
                    summary_a, summary_b, partition_entry_state, groups_per_head, prefix_mem, kernel_cfg);
                grouped_scan = ttnn::prim::chunk_gdn_scan(
                    grouped[0],
                    grouped[1],
                    grouped[2],
                    grouped[3],
                    grouped[4],
                    grouped[5],
                    grouped[6],
                    group_initial_states,
                    C,
                    true,
                    out_mem,
                    kernel_cfg,
                    true);
            } else if (use_persistent_group_prefix) {
                auto group_initial_states =
                    ttnn::prim::kda_affine_prefix(summary_a, summary_b, *s0, groups_per_head, prefix_mem, kernel_cfg);
                grouped_scan = ttnn::prim::chunk_gdn_scan(
                    grouped[0],
                    grouped[1],
                    grouped[2],
                    grouped[3],
                    grouped[4],
                    grouped[5],
                    grouped[6],
                    group_initial_states,
                    C,
                    true,
                    out_mem,
                    kernel_cfg,
                    true);
            } else {
                summary_a = ttnn::reshape(summary_a, ttnn::Shape({BH, groups_per_head, K, K}));
                auto summary_b_grouped = ttnn::reshape(summary_b, ttnn::Shape({BH, groups_per_head, K, V}));
                auto [prefix_a, prefix_b] =
                    inclusive_affine_prefix(summary_a, summary_b_grouped, groups_per_head, prefix_mem, kernel_cfg);
                auto initial = ttnn::reshape(*s0, ttnn::Shape({BH, 1, K, V}));
                auto repeated_initial = ttnn::repeat_interleave(initial, groups_per_head, 1, prefix_mem);
                auto group_end_states = ttnn::matmul(
                    prefix_a,
                    repeated_initial,
                    false,
                    false,
                    prefix_mem,
                    DataType::FLOAT32,
                    std::nullopt,
                    std::nullopt,
                    kernel_cfg);
                group_end_states = ttnn::add(group_end_states, prefix_b, std::nullopt, prefix_mem);
                auto group_initial_states = initial;
                if (groups_per_head > 1) {
                    group_initial_states = ttnn::concat(
                        {initial, slice_group_axis(group_end_states, 0, groups_per_head - 1, prefix_mem)},
                        1,
                        prefix_mem);
                }
                group_initial_states = ttnn::reshape(group_initial_states, ttnn::Shape({group_heads, K, V}));
                grouped_scan = ttnn::prim::chunk_gdn_scan(
                    grouped[0],
                    grouped[1],
                    grouped[2],
                    grouped[3],
                    grouped[4],
                    grouped[5],
                    grouped[6],
                    group_initial_states,
                    C,
                    true,
                    out_mem,
                    kernel_cfg,
                    true);
            }
            (*grouped_scan)[0] = ttnn::reshape((*grouped_scan)[0], ttnn::Shape({BH, NC, C, V}));
            auto all_final_states = ttnn::reshape((*grouped_scan)[1], ttnn::Shape({BH, groups_per_head, K, V}));
            (*grouped_scan)[1] = ttnn::reshape(
                slice_group_axis(all_final_states, groups_per_head - 1, groups_per_head, prefix_mem),
                ttnn::Shape({BH, K, V}));
            if (distributed_final_state.has_value()) {
                (*grouped_scan)[1] = *distributed_final_state;
            }
        }
    }
    std::vector<ttnn::Tensor> scan;
    if (grouped_scan.has_value()) {
        scan = *grouped_scan;
    } else {
        scan = ttnn::prim::chunk_gdn_scan(
            prep[0],
            prep[1],
            prep[2],
            prep[3],
            prep[4],
            prep[5],
            prep[6],
            s0,
            C,
            true,
            out_mem,
            kernel_cfg,
            true,
            false,
            std::nullopt,
            rms_gate,
            rms_weight,
            H,
            rms_epsilon);
    }

    std::optional<ttnn::Tensor> final_state;
    if (output_final_state) {
        final_state = ttnn::reshape(scan[1], ttnn::Shape({B, H, K, V}));
    }
    if (output_head_major && pad == 0) {
        if (rms_gate.has_value()) {
            return {scan[2], final_state};
        }
        return {ttnn::reshape(scan[0], ttnn::Shape({BH, T, V})), final_state};
    }

    ttnn::Tensor output = ttnn::to_layout(scan[0], Layout::ROW_MAJOR);
    output = ttnn::reshape(output, ttnn::Shape({BH, L, V}));
    if (pad > 0) {
        output = ttnn::slice(
            output,
            ttnn::SmallVector<int32_t>{0, 0, 0},
            ttnn::SmallVector<int32_t>{static_cast<int32_t>(BH), static_cast<int32_t>(T), static_cast<int32_t>(V)},
            ttnn::SmallVector<int32_t>{1, 1, 1});
    }
    output = ttnn::reshape(output, ttnn::Shape({B, H, T, V}));
    output = ttnn::permute(output, ttnn::SmallVector<int64_t>{0, 2, 1, 3});
    return {output, final_state};
}

std::tuple<ttnn::Tensor, ttnn::Tensor> kda_distributed_affine_prefix(
    const ttnn::Tensor& transform_a,
    const ttnn::Tensor& transform_b,
    const ttnn::Tensor& initial_state,
    const ttnn::Tensor& identity_a,
    const ttnn::Tensor& zero_b,
    uint32_t sequence_parallel_axis,
    const std::optional<ttnn::MemoryConfig>& memory_config,
    const std::optional<ttnn::DeviceComputeKernelConfig>& compute_kernel_config) {
    TT_FATAL(sequence_parallel_axis < 2, "sequence_parallel_axis must be 0 or 1");
    TT_FATAL(
        transform_a.logical_shape() == transform_b.logical_shape() &&
            transform_a.logical_shape() == initial_state.logical_shape() &&
            transform_a.logical_shape() == identity_a.logical_shape() &&
            transform_a.logical_shape() == zero_b.logical_shape(),
        "distributed KDA affine prefix requires equal batched [K,K] tensor shapes");
    TT_FATAL(transform_a.logical_shape().rank() >= 3, "distributed KDA affine prefix expects batched matrices");
    TT_FATAL(
        transform_a.logical_shape()[-2] == transform_a.logical_shape()[-1],
        "distributed KDA affine prefix currently requires K == V");
    for (const auto* tensor : {&transform_a, &transform_b, &initial_state, &identity_a, &zero_b}) {
        TT_FATAL(tensor->dtype() == DataType::FLOAT32, "distributed KDA affine prefix requires FP32 tensors");
        TT_FATAL(tensor->layout() == Layout::TILE, "distributed KDA affine prefix requires TILE tensors");
        TT_FATAL(
            tensor->memory_config().memory_layout() == TensorMemoryLayout::INTERLEAVED,
            "distributed KDA affine prefix requires interleaved tensors");
    }

    auto* mesh_device = transform_a.device();
    TT_FATAL(mesh_device != nullptr, "distributed KDA affine prefix requires a mesh device");
    const auto mesh_shape = mesh_device->shape();
    TT_FATAL(mesh_shape.dims() == 2, "distributed KDA affine prefix requires a 2D mesh");
    const uint32_t sp_size = mesh_shape[sequence_parallel_axis];
    const uint32_t tp_size = mesh_shape[1 - sequence_parallel_axis];
    TT_FATAL(sp_size > 1, "distributed KDA affine prefix requires SP > 1");
    const auto out_mem = memory_config.value_or(ttnn::DRAM_MEMORY_CONFIG);
    auto coordinate = [sequence_parallel_axis](uint32_t sp_rank, uint32_t tp_rank) {
        return sequence_parallel_axis == 0 ? MeshCoordinate(sp_rank, tp_rank) : MeshCoordinate(tp_rank, sp_rank);
    };

    // The API returns states rather than prefix transforms, so propagate the state directly for every SP mesh.
    // Each TP line follows the same schedule.
    const auto transform_a_compute = ttnn::typecast(transform_a, DataType::BFLOAT16, ttnn::L1_MEMORY_CONFIG);
    const auto transform_b_compute = ttnn::typecast(transform_b, DataType::BFLOAT16, ttnn::L1_MEMORY_CONFIG);
    auto entry_state = ttnn::clone(initial_state, std::nullopt, out_mem, compute_kernel_config);
    auto entry_state_transport = ttnn::typecast(entry_state, DataType::BFLOAT16, ttnn::L1_MEMORY_CONFIG);
    auto matmul_bf16 = [&](const ttnn::Tensor& lhs, const ttnn::Tensor& rhs) {
        return ttnn::matmul(
            lhs,
            rhs,
            false,
            false,
            ttnn::L1_MEMORY_CONFIG,
            DataType::BFLOAT16,
            std::nullopt,
            std::nullopt,
            compute_kernel_config);
    };
    auto carry = matmul_bf16(transform_a_compute, entry_state_transport);
    carry = ttnn::add(carry, transform_b_compute, std::nullopt, ttnn::L1_MEMORY_CONFIG);
    for (uint32_t destination = 1; destination < sp_size; ++destination) {
        for (uint32_t tp_rank = 0; tp_rank < tp_size; ++tp_rank) {
            entry_state_transport = ttnn::point_to_point(
                carry,
                coordinate(destination, tp_rank),
                coordinate(destination - 1, tp_rank),
                ttnn::ccl::Topology::Linear,
                entry_state_transport,
                std::nullopt);
        }
        entry_state = ttnn::typecast(entry_state_transport, DataType::FLOAT32, out_mem);
        carry = matmul_bf16(transform_a_compute, entry_state_transport);
        carry = ttnn::add(carry, transform_b_compute, std::nullopt, ttnn::L1_MEMORY_CONFIG);
    }

    auto final_state_transport = ttnn::broadcast(
        carry,
        coordinate(sp_size - 1, 0),
        std::nullopt,
        ttnn::L1_MEMORY_CONFIG,
        ttnn::ccl::Topology::Linear,
        sequence_parallel_axis);
    auto final_state = ttnn::typecast(final_state_transport, DataType::FLOAT32, out_mem);
    return {entry_state, final_state};
}

std::tuple<ttnn::Tensor, ttnn::Tensor> kda_convolution_halo(
    const ttnn::Tensor& projected_qkv,
    const ttnn::Tensor& initial_carry,
    uint32_t sequence_parallel_axis,
    const std::optional<ttnn::MemoryConfig>& memory_config,
    const std::optional<ttnn::DeviceComputeKernelConfig>& compute_kernel_config) {
    TT_FATAL(sequence_parallel_axis < 2, "sequence_parallel_axis must be 0 or 1");
    const auto qkv_shape = projected_qkv.logical_shape();
    const auto carry_shape = initial_carry.logical_shape();
    TT_FATAL(qkv_shape.rank() == 3 && carry_shape.rank() == 3, "KDA convolution halo expects rank-3 tensors");
    TT_FATAL(
        qkv_shape[0] == carry_shape[0] && qkv_shape[2] == carry_shape[2],
        "KDA convolution halo requires matching batch and channel dimensions");
    const uint32_t history = carry_shape[1];
    TT_FATAL(history > 0 && qkv_shape[1] >= history, "KDA convolution halo requires 0 < history <= local T");
    TT_FATAL(projected_qkv.dtype() == initial_carry.dtype(), "KDA convolution halo requires matching dtypes");
    TT_FATAL(projected_qkv.layout() == initial_carry.layout(), "KDA convolution halo requires matching layouts");
    for (const auto* tensor : {&projected_qkv, &initial_carry}) {
        TT_FATAL(
            tensor->memory_config().memory_layout() == TensorMemoryLayout::INTERLEAVED,
            "KDA convolution halo requires interleaved tensors");
    }

    auto* mesh_device = projected_qkv.device();
    TT_FATAL(mesh_device != nullptr, "KDA convolution halo requires a mesh device");
    const auto mesh_shape = mesh_device->shape();
    TT_FATAL(mesh_shape.dims() == 2, "KDA convolution halo requires a 2D mesh");
    const uint32_t sp_size = mesh_shape[sequence_parallel_axis];
    const uint32_t tp_size = mesh_shape[1 - sequence_parallel_axis];
    TT_FATAL(sp_size > 1, "KDA convolution halo requires SP > 1");
    const auto out_mem = memory_config.value_or(ttnn::DRAM_MEMORY_CONFIG);
    auto coordinate = [sequence_parallel_axis](uint32_t sp_rank, uint32_t tp_rank) {
        return sequence_parallel_axis == 0 ? MeshCoordinate(sp_rank, tp_rank) : MeshCoordinate(tp_rank, sp_rank);
    };

    auto local_final_carry = ttnn::slice(
        projected_qkv,
        ttnn::SmallVector<int32_t>{0, static_cast<int32_t>(qkv_shape[1] - history), 0},
        ttnn::SmallVector<int32_t>{
            static_cast<int32_t>(qkv_shape[0]), static_cast<int32_t>(qkv_shape[1]), static_cast<int32_t>(qkv_shape[2])},
        ttnn::SmallVector<int32_t>{1, 1, 1},
        out_mem);

    auto partition_carry = ttnn::clone(initial_carry, std::nullopt, out_mem, compute_kernel_config);
    for (uint32_t tp_rank = 0; tp_rank < tp_size; ++tp_rank) {
        for (uint32_t destination = 1; destination < sp_size; ++destination) {
            partition_carry = ttnn::point_to_point(
                local_final_carry,
                coordinate(destination, tp_rank),
                coordinate(destination - 1, tp_rank),
                ttnn::ccl::Topology::Linear,
                partition_carry,
                std::nullopt);
        }
    }

    auto final_carry = ttnn::clone(initial_carry, std::nullopt, out_mem, compute_kernel_config);
    for (uint32_t tp_rank = 0; tp_rank < tp_size; ++tp_rank) {
        const auto sender_coord = coordinate(sp_size - 1, tp_rank);
        for (uint32_t destination = 0; destination < sp_size; ++destination) {
            final_carry = ttnn::point_to_point(
                local_final_carry,
                coordinate(destination, tp_rank),
                sender_coord,
                ttnn::ccl::Topology::Linear,
                final_carry,
                std::nullopt);
        }
    }
    return {partition_carry, final_carry};
}

ttnn::Tensor kda_gated_rms_norm(
    const ttnn::Tensor& input,
    const ttnn::Tensor& gate,
    const ttnn::Tensor& weight,
    uint32_t num_heads,
    float epsilon,
    const std::optional<ttnn::MemoryConfig>& memory_config,
    const std::optional<ttnn::DeviceComputeKernelConfig>& compute_kernel_config) {
    const auto output_memory_config = memory_config.value_or(ttnn::DRAM_MEMORY_CONFIG);
    const auto kernel_config = init_device_compute_kernel_config(
        input.device()->arch(),
        compute_kernel_config,
        MathFidelity::HiFi4,
        /*default_approx_mode=*/true,
        /*default_fp32_acc=*/true,
        /*default_l1_acc=*/true);
    return ttnn::prim::kda_gated_rms_norm(input, gate, weight, num_heads, epsilon, output_memory_config, kernel_config);
}

std::tuple<ttnn::Tensor, ttnn::Tensor, ttnn::Tensor> kda_causal_conv1d_split(
    const ttnn::Tensor& input,
    const ttnn::Tensor& state,
    const ttnn::Tensor& tap0,
    const ttnn::Tensor& tap1,
    const ttnn::Tensor& tap2,
    const ttnn::Tensor& tap3,
    uint32_t q_width,
    uint32_t k_width,
    uint32_t v_width,
    const std::optional<ttnn::MemoryConfig>& memory_config,
    const std::optional<ttnn::DeviceComputeKernelConfig>& compute_kernel_config) {
    const auto out_mem = memory_config.value_or(ttnn::DRAM_MEMORY_CONFIG);
    const auto kernel_config = init_device_compute_kernel_config(
        input.device()->arch(),
        compute_kernel_config,
        MathFidelity::HiFi4,
        /*default_approx_mode=*/true,
        /*default_fp32_acc=*/false,
        /*default_l1_acc=*/false);
    auto outputs = ttnn::prim::kda_causal_conv1d_split(
        input, state, tap0, tap1, tap2, tap3, q_width, k_width, v_width, out_mem, kernel_config);
    return {outputs[0], outputs[1], outputs[2]};
}

}  // namespace ttnn::transformer
