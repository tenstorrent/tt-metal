// SPDX-FileCopyrightText: © 2026 Tenstorrent Inc.
// SPDX-License-Identifier: Apache-2.0
//
// apply_twiddles_xl_host.hpp — host-side delta-table generation + per-device
// cache for ttnn::prim::apply_twiddles_xl.
//
// The "delta" table is the per-row phase increment used by the on-the-fly
// twiddle recurrence:
//
//     delta[i] = exp(-2πi · i / full_N)        for i ∈ [0, big_modulus)
//
// At runtime the kernel looks up delta[row % big_modulus] once per row
// (a single 8-byte DRAM read) and builds the length-P twiddle row in L1
// via the recurrence tw[k] = tw[k-1] · delta, with tw[0] = (1, 0).
//
// Cache key combines (device ptr, big_modulus, full_N).  Twiddle math
// is dtype-independent (always fp32) so dtype is NOT part of the key.

#pragma once

#include <cmath>
#include <cstdint>
#include <memory>
#include <unordered_map>
#include <vector>

#include "tt-metalium/distributed.hpp"
#include "tt-metalium/mesh_buffer.hpp"
#include "tt-metalium/mesh_command_queue.hpp"

#include "fft_inner_host.hpp"   // fft_example::make_mesh_buf

namespace ttnn::experimental::prim::apply_twiddles_xl_host {

constexpr uint32_t kTileHW           = 32u;
constexpr uint32_t kTileElems        = kTileHW * kTileHW;          // 1024
constexpr uint32_t kTileBytesFp32    = kTileElems * 4u;            // 4096

struct DeltaPlan {
    uint32_t big_modulus = 0;
    uint32_t full_N      = 0;
    // Tile-padded fp32 buffers, big_modulus entries each.  Stored as two
    // separate buffers (real, imag) so the reader can do two independent
    // 4-byte DRAM reads per row.
    std::shared_ptr<tt::tt_metal::distributed::MeshBuffer> dr_buf;
    std::shared_ptr<tt::tt_metal::distributed::MeshBuffer> di_buf;
    // Total bytes of EACH of dr_buf/di_buf — tile-rounded.
    uint32_t bytes_per_buf = 0;
};

inline std::pair<std::vector<float>, std::vector<float>>
build_delta_table(uint32_t big_modulus, uint32_t full_N) {
    // Pad up to a multiple of kTileElems so the buffer is tile-aligned —
    // the addrgen the reader uses (InterleavedAddrGenFast<true>) bills
    // a full tile per page lookup.
    const uint32_t padded = ((big_modulus + kTileElems - 1u) / kTileElems) * kTileElems;
    std::vector<float> r(padded, 0.0f), i(padded, 0.0f);

    const double tau_over_N = -2.0 * M_PI / static_cast<double>(full_N);
    for (uint32_t k = 0; k < big_modulus; ++k) {
        const double angle = tau_over_N * static_cast<double>(k);
        r[k] = static_cast<float>(std::cos(angle));
        i[k] = static_cast<float>(std::sin(angle));
    }
    // Pad slots [big_modulus, padded) stay zero — they are never accessed
    // (kernel masks with `% big_modulus`) but stay zero for safety.
    return {std::move(r), std::move(i)};
}

inline std::unordered_map<uint64_t, std::shared_ptr<DeltaPlan>>& cache() {
    static std::unordered_map<uint64_t, std::shared_ptr<DeltaPlan>> c;
    return c;
}

inline uint64_t make_key(
    tt::tt_metal::distributed::MeshDevice* md,
    uint32_t big_modulus,
    uint32_t full_N)
{
    return reinterpret_cast<uint64_t>(md)
         ^ (static_cast<uint64_t>(big_modulus) * 0x9E3779B97F4A7C15ull)
         ^ (static_cast<uint64_t>(full_N)      * 0xBF58476D1CE4E5B9ull);
}

inline std::shared_ptr<DeltaPlan> get_or_create(
    std::shared_ptr<tt::tt_metal::distributed::MeshDevice> md,
    uint32_t big_modulus,
    uint32_t full_N)
{
    using namespace tt::tt_metal::distributed;

    const uint64_t key = make_key(md.get(), big_modulus, full_N);
    auto& c = cache();
    auto it = c.find(key);
    if (it != c.end()) return it->second;

    auto plan = std::make_shared<DeltaPlan>();
    plan->big_modulus = big_modulus;
    plan->full_N      = full_N;

    const uint32_t padded = ((big_modulus + kTileElems - 1u) / kTileElems) * kTileElems;
    const uint32_t bytes  = padded * 4u;
    plan->bytes_per_buf   = bytes;
    plan->dr_buf = fft_example::make_mesh_buf(md, bytes, kTileBytesFp32);
    plan->di_buf = fft_example::make_mesh_buf(md, bytes, kTileBytesFp32);

    auto [r, i] = build_delta_table(big_modulus, full_N);
    // Vectors are padded to `padded` already.
    MeshCommandQueue& cq = md->mesh_command_queue();
    WriteShard(cq, plan->dr_buf, r, MeshCoordinate(0, 0), /*blocking=*/true);
    WriteShard(cq, plan->di_buf, i, MeshCoordinate(0, 0), /*blocking=*/true);

    c.emplace(key, plan);
    return plan;
}

}  // namespace ttnn::experimental::prim::apply_twiddles_xl_host
