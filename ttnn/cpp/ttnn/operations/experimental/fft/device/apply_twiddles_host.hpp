// SPDX-FileCopyrightText: © 2026 Tenstorrent Inc.
// SPDX-License-Identifier: Apache-2.0
//
// apply_twiddles_host.hpp — host-side twiddle-table generation + per-device
// cache for ttnn::prim::apply_twiddles.
//
// The twiddle table T[n2, k1] = exp(-2*pi*i*n2*k1/N) (N == N1*N2) is a
// pure function of (N1, N2).  We upload it to DRAM ONCE per (device, N1,
// N2) tuple and reuse the MeshBuffer for every subsequent op call.  The
// hot path (apply_twiddles_factory::create_descriptor) only does a cache
// lookup — no CPU FFT-math, no host→device transfer.
//
// Layout matches stockham_host::pass2_twiddle_table:
//   - N2 tiles per buffer (one tile per twiddle row)
//   - Each tile holds N1 valid floats in slots [0, N1) and zeros in
//     [N1, kTileElems).  The zero-padding is what lets the SFPU complex-
//     multiply propagate garbage in compute slots [N1, kTileElems)
//     without corrupting the valid output slots.

#pragma once

#include <cmath>
#include <cstdint>
#include <memory>
#include <unordered_map>
#include <vector>

#include "tt-metalium/distributed.hpp"
#include "tt-metalium/mesh_buffer.hpp"
#include "tt-metalium/mesh_command_queue.hpp"

#include "fft_inner_host.hpp"   // fft_example::make_mesh_buf, kTileElems, ...

namespace ttnn::experimental::prim::apply_twiddles_host {

constexpr uint32_t kTileHW        = 32u;
constexpr uint32_t kTileElems_at  = kTileHW * kTileHW;          // 1024
constexpr uint32_t kTileBytesFp32_at = kTileElems_at * 4u;      // 4096

struct TwiddlePlan {
    uint32_t N1 = 0;
    uint32_t N2 = 0;
    std::shared_ptr<tt::tt_metal::distributed::MeshBuffer> tw_r_buf;
    std::shared_ptr<tt::tt_metal::distributed::MeshBuffer> tw_i_buf;
    // Weak reference to the owning device.  lock() returns nullptr once the
    // device is fully destroyed, correctly detecting stale entries even when
    // the heap allocator reuses the same raw MeshDevice* address.
    std::weak_ptr<tt::tt_metal::distributed::MeshDevice> device_weak;
};

// Build T[n2, k1] = (cos, sin) for n2 ∈ [0, N2), k1 ∈ [0, N1).
// One tile per twiddle row, kTileElems floats wide.  Slots [N1, kTileElems)
// are zero so SFPU mul against garbage compute lanes stays zero.
inline std::pair<std::vector<float>, std::vector<float>>
build_twiddle_table(uint32_t N1, uint32_t N2) {
    const uint32_t N = N1 * N2;
    const size_t total = static_cast<size_t>(N2) * kTileElems_at;
    const double tau_over_N = -2.0 * M_PI / static_cast<double>(N);

    std::vector<float> r(total, 0.0f), i(total, 0.0f);
    for (uint32_t n2 = 0; n2 < N2; ++n2) {
        float* tile_r = r.data() + static_cast<size_t>(n2) * kTileElems_at;
        float* tile_i = i.data() + static_cast<size_t>(n2) * kTileElems_at;
        for (uint32_t k1 = 0; k1 < N1; ++k1) {
            const double angle = tau_over_N *
                                 static_cast<double>(n2) *
                                 static_cast<double>(k1);
            tile_r[k1] = static_cast<float>(std::cos(angle));
            tile_i[k1] = static_cast<float>(std::sin(angle));
        }
    }
    return {std::move(r), std::move(i)};
}

inline std::unordered_map<uint64_t, std::shared_ptr<TwiddlePlan>>& cache() {
    static std::unordered_map<uint64_t, std::shared_ptr<TwiddlePlan>> c;
    return c;
}

// Cache key combines (device ptr, N1, N2).  Twiddle math is dtype-
// independent (always fp32) so dtype is NOT part of the key.
inline uint64_t make_key(
    tt::tt_metal::distributed::MeshDevice* md, uint32_t N1, uint32_t N2)
{
    return reinterpret_cast<uint64_t>(md)
         ^ (static_cast<uint64_t>(N1) * 0x9E3779B97F4A7C15ull)
         ^ (static_cast<uint64_t>(N2) * 0xBF58476D1CE4E5B9ull);
}

inline std::shared_ptr<TwiddlePlan> get_or_create(
    std::shared_ptr<tt::tt_metal::distributed::MeshDevice> md,
    uint32_t N1, uint32_t N2)
{
    using namespace tt::tt_metal::distributed;

    const uint64_t key = make_key(md.get(), N1, N2);
    auto& c = cache();
    auto it = c.find(key);
    if (it != c.end()) {
        if (it->second->device_weak.lock()) return it->second;
        c.erase(it);   // stale: device was destroyed (and ptr may be reused)
    }

    auto plan = std::make_shared<TwiddlePlan>();
    plan->N1 = N1;
    plan->N2 = N2;
    plan->device_weak = md;

    const uint32_t bytes = N2 * kTileBytesFp32_at;
    plan->tw_r_buf = fft_example::make_mesh_buf(md, bytes, kTileBytesFp32_at);
    plan->tw_i_buf = fft_example::make_mesh_buf(md, bytes, kTileBytesFp32_at);

    auto [r, i] = build_twiddle_table(N1, N2);
    MeshCommandQueue& cq = md->mesh_command_queue();
    WriteShard(cq, plan->tw_r_buf, r, MeshCoordinate(0, 0), /*blocking=*/true);
    WriteShard(cq, plan->tw_i_buf, i, MeshCoordinate(0, 0), /*blocking=*/true);

    c.emplace(key, plan);
    return plan;
}

}  // namespace ttnn::experimental::prim::apply_twiddles_host
