// SPDX-FileCopyrightText: © 2024 Tenstorrent Inc.
// SPDX-License-Identifier: Apache-2.0
//
// stockham_host.hpp — Tensor cache + grid helpers shared by the FFT
//                     ProgramSpec factories (single_tile / batched
//                     Stockham, fft_radix_pass).
//
// Provides:
//   * BatchFFTPlan: cached per-(device, sub_N) Tensor pair holding the
//                   batched single-tile Stockham twiddle table.  Built
//                   once on first use of any FFT size by
//                   get_cached_batch_plan() and reused for every
//                   subsequent dispatch to the same length.
//   * pick_batch_grid / max_cores_for_grid / batch_logical_core:
//                   grid-mapping helpers used by every factory to lay work
//                   out across the (grid_x, grid_y) Tensix rectangle.
//   * clear_batch_plan_cache(): registered with
//                   fft_device_cache_clear.hpp so
//                   ttnn.experimental.clear_fft_device_caches() releases
//                   the cached tensors before close_device().
//
// This header does NOT build a Program, own input/output buffers, or hold
// host scratch — the runtime path goes exclusively through the
// ProgramSpec factories in device/*_factory.cpp, which build their
// own program per (dtype, N, B) tuple and bind cached twiddle tensors
// through named TensorParameters.

#pragma once

#include "tt-metalium/host_api.hpp"
#include "tt-metalium/distributed.hpp"
#include "tt-metalium/mesh_device.hpp"
#include "fft_inner_host.hpp"  // fft_example::{log2u, kTileElems}
#include "ttnn/operations/experimental/fft/device/leak_static_cache.hpp"
#include "ttnn/tensor/tensor.hpp"
#include "ttnn/tensor/types.hpp"
#include "ttnn/types.hpp"

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cassert>
#include <memory>
#include <unordered_map>
#include <utility>
#include <vector>

namespace fft_stockham {

using fft_example::kTileElems;
using fft_example::log2u;
using tt::tt_metal::distributed::MeshDevice;

// Power-of-two check.
inline bool is_pow2(uint32_t n) { return n != 0 && (n & (n - 1)) == 0; }

// ─── BatchFFTPlan ───────────────────────────────────────────────────────
// Twiddle-only Tensor pair for the batched single-tile Stockham
// radix-2 kernel.  Cached per (device, sub_N) — a single dispatch shape
// re-uses the same twiddle table across every subsequent call.
struct BatchFFTPlan {
    ttnn::Tensor tw_r;
    ttnn::Tensor tw_i;
    std::weak_ptr<MeshDevice> device_weak;
};

struct ZeroImagPlan {
    ttnn::Tensor zero;
    std::weak_ptr<MeshDevice> device_weak;
};

inline std::pair<uint32_t, uint32_t> pick_batch_grid(uint32_t num_cores, uint32_t grid_x) {
    // Search downward from grid_x for the largest divisor of num_cores
    // that is <= grid_x. Guaranteed to terminate at cols=1 in the worst case.
    uint32_t cols = (num_cores < grid_x) ? num_cores : grid_x;
    while (cols > 1 && num_cores % cols != 0) {
        --cols;
    }
    return {cols, num_cores / cols};
}

inline uint32_t max_cores_for_grid(uint32_t grid_x, uint32_t grid_y) {
    uint32_t best = 1;
    for (uint32_t p = 2; p <= grid_x * grid_y; p *= 2) {
        // Is there cols <= grid_x dividing p with p/cols <= grid_y?
        bool ok = false;
        for (uint32_t c = std::min(p, grid_x); c >= 1; --c) {
            if (p % c == 0 && p / c <= grid_y) {
                ok = true;
                break;
            }
        }
        if (ok) {
            best = p;
        }
    }
    return best;
}

inline tt::tt_metal::CoreCoord batch_logical_core(uint32_t c, uint32_t grid_cols) {
    return tt::tt_metal::CoreCoord{c % grid_cols, c / grid_cols};
}

// LOG2_SUB_N tiles per side; tile s holds the stage-s twiddles for a
// single-tile (P=1) radix-2 sub-FFT of length sub_N. Identical to the
// inner kernel's local-stage twiddle layout.
inline std::pair<std::vector<float>, std::vector<float>> batch_twiddles(uint32_t sub_N, uint32_t log2_sub_N) {
    const size_t total = static_cast<size_t>(log2_sub_N) * kTileElems;
    std::vector<float> r(total, 0.0f), i(total, 0.0f);
    const uint32_t num_pairs = sub_N / 2u;

    for (uint32_t s = 0; s < log2_sub_N; ++s) {
        const double M = static_cast<double>(1u << (s + 1));
        const uint32_t stride_mask = (1u << s) - 1u;
        float* tile_r = r.data() + static_cast<size_t>(s) * kTileElems;
        float* tile_i = i.data() + static_cast<size_t>(s) * kTileElems;
        for (uint32_t p = 0; p < num_pairs; ++p) {
            const uint32_t k = p & stride_mask;
            const double angle = -2.0 * M_PI * static_cast<double>(k) / M;
            tile_r[p] = static_cast<float>(std::cos(angle));
            tile_i[p] = static_cast<float>(std::sin(angle));
        }
    }
    return {std::move(r), std::move(i)};
}

// Upload the twiddle table for `sub_N` to two DRAM Tensors on `md`.
// Fp32 internally; the reader kernel expands to bf16 downstream when the
// caller's input is bf16.
inline std::shared_ptr<BatchFFTPlan> make_batch_plan(std::shared_ptr<MeshDevice> md, uint32_t sub_N) {
    using namespace tt::tt_metal;
    assert(sub_N <= kTileElems && "batch path requires sub_N <= 1024 (single tile per sub-FFT)");
    assert(is_pow2(sub_N) && sub_N >= 2);

    auto bp = std::make_shared<BatchFFTPlan>();
    bp->device_weak = md;
    const uint32_t log2_sub_N = log2u(sub_N);

    auto [tw_r_data, tw_i_data] = batch_twiddles(sub_N, log2_sub_N);
    const ttnn::Shape shape{ttnn::SmallVector<uint32_t>{log2_sub_N, kTileElems}};
    const TensorSpec spec(shape, TensorLayout(DataType::FLOAT32, PageConfig(Layout::ROW_MAJOR), MemoryConfig{}));
    bp->tw_r = ttnn::Tensor::from_vector(std::move(tw_r_data), spec, md.get());
    bp->tw_i = ttnn::Tensor::from_vector(std::move(tw_i_data), spec, md.get());
    return bp;
}

namespace detail {
inline std::unordered_map<uint64_t, std::shared_ptr<BatchFFTPlan>>& batch_plan_cache() {
    return ttnn::experimental::prim::fft_cache::leak_static_map<
        std::unordered_map<uint64_t, std::shared_ptr<BatchFFTPlan>>>();
}
inline std::unordered_map<uint64_t, std::shared_ptr<ZeroImagPlan>>& zero_imag_cache() {
    return ttnn::experimental::prim::fft_cache::leak_static_map<
        std::unordered_map<uint64_t, std::shared_ptr<ZeroImagPlan>>>();
}
// Twiddle table is a function of sub_N only.  Batch size does not affect
// the cache identity (it never did — the old key hashed batch but every
// caller passed batch=1).
inline uint64_t batch_plan_key(MeshDevice* md, uint32_t sub_N) {
    return reinterpret_cast<uint64_t>(md) ^ (uint64_t{sub_N} * 0x9E3779B97F4A7C15ull);
}
inline uint64_t zero_imag_key(MeshDevice* md, tt::tt_metal::DataType dtype, uint32_t batch) {
    return reinterpret_cast<uint64_t>(md) ^ (static_cast<uint64_t>(dtype) * 0x9E3779B97F4A7C15ull) ^
           (static_cast<uint64_t>(batch) * 0xBF58476D1CE4E5B9ull);
}
}  // namespace detail

// Public API preserved for call-site compatibility.  `batch` is accepted
// for source-level compatibility with older factory code but is not part
// of the cache key (the twiddle table only depends on sub_N).
inline std::shared_ptr<BatchFFTPlan> get_cached_batch_plan(
    std::shared_ptr<MeshDevice> md, uint32_t sub_N, uint32_t /*batch*/ = 1u) {
    const uint64_t key = detail::batch_plan_key(md.get(), sub_N);
    auto& cache = detail::batch_plan_cache();
    auto it = cache.find(key);
    if (it != cache.end()) {
        if (it->second->device_weak.lock()) {
            return it->second;
        }
        cache.erase(it);
    }
    auto bp = make_batch_plan(md, sub_N);
    cache.emplace(key, bp);
    return bp;
}

inline std::shared_ptr<ZeroImagPlan> get_cached_zero_imag(
    std::shared_ptr<MeshDevice> md, tt::tt_metal::DataType dtype, uint32_t batch) {
    using namespace tt::tt_metal;
    const uint64_t key = detail::zero_imag_key(md.get(), dtype, batch);
    auto& cache = detail::zero_imag_cache();
    auto it = cache.find(key);
    if (it != cache.end()) {
        if (it->second->device_weak.lock()) {
            return it->second;
        }
        cache.erase(it);
    }

    auto plan = std::make_shared<ZeroImagPlan>();
    plan->device_weak = md;
    const ttnn::Shape shape{ttnn::SmallVector<uint32_t>{batch, kTileElems}};
    const TensorSpec spec(shape, TensorLayout(dtype, PageConfig(Layout::ROW_MAJOR), MemoryConfig{}));
    if (dtype == DataType::BFLOAT16) {
        plan->zero = ttnn::Tensor::from_vector(
            std::vector<bfloat16>(static_cast<size_t>(batch) * kTileElems, bfloat16(0.0f)), spec, md.get());
    } else {
        plan->zero = ttnn::Tensor::from_vector(std::vector<float>(static_cast<size_t>(batch) * kTileElems, 0.0f), spec, md.get());
    }
    cache.emplace(key, plan);
    return plan;
}

inline void clear_batch_plan_cache() {
    detail::batch_plan_cache().clear();
    detail::zero_imag_cache().clear();
}

}  // namespace fft_stockham
