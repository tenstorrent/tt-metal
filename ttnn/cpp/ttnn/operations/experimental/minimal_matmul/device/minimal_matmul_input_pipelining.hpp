// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <cstdint>
#include <cstdlib>
#include <optional>

#include <tt-metalium/allocator.hpp>
#include <tt-metalium/device.hpp>
#include <tt-metalium/hal_types.hpp>

namespace ttnn::experimental::prim::input_pipelining {

// How the in0 / in1 data-movement kernels of minimal_matmul and all_gather_minimal_matmul_async move input blocks:
// the depth of the input buffers and how a block travels down the store-and-forward chain (see
// move_block_cut_through and MM_INJECTOR_PREFETCH in minimal_matmul/device/kernels/input_pipelining.hpp).
//
// Every choice here is a pure function of the block geometry, the program's static L1 footprint and the device, all
// of which the program cache keys on (through the op's attributes, tensor specs and device), so it is part of the
// program hash without a separate field. Nothing reads live L1 occupancy.
//
// Experiment overrides, for A/B runs only and unset in production: TT_MM_IN_BUFFERS=<blocks>,
// TT_MM_CUT_THROUGH=<chunks, 0 = off>, TT_MM_INJ_PREFETCH=<0|1>. They bypass the L1 check and are not part of the
// program hash, so a process must not change them between runs of one shape.

inline std::optional<uint32_t> env_override(const char* name) {
    if (const char* v = std::getenv(name); v != nullptr && v[0] != '\0') {
        return static_cast<uint32_t>(std::strtoul(v, nullptr, 10));
    }
    return std::nullopt;
}

// L1 left free, beyond the program's own buffers, before the input buffers may take a third block. Covers what the
// static estimate cannot see: L1-resident tensors and the L1_SMALL region.
inline constexpr uint64_t kL1Reserve = 128 * 1024;

// Cut-through pays a flush and a semaphore increment per chunk and per hop. It is a win when the data-movement
// kernels have slack to spend on that, i.e. when a K block carries enough math per tile moved; below that the movers
// are already the bottleneck and the per-chunk syncs land on the critical path. The measure is the per-block
// arithmetic intensity M_b * N_b / (M_b + N_b), in output tiles per input tile, of the block a core actually computes
// (block sizes clamped to the core's share of M and N).
inline constexpr uint32_t kCutThroughMinIntensity = 2;
inline constexpr uint32_t kCutThroughChunks = 4;

// Row chunks per forwarded block (0 = store-and-forward whole blocks) for a core block of M_b x N_b output tiles.
inline uint32_t cut_through_chunks(uint32_t M_b, uint32_t N_b) {
    const uint32_t chunks = (M_b * N_b >= kCutThroughMinIntensity * (M_b + N_b)) ? kCutThroughChunks : 0;
    return env_override("TT_MM_CUT_THROUGH").value_or(chunks);
}

// Whether injectors prefetch their next K block. Used where cut-through is off; the two are exclusive.
inline bool injector_prefetch(uint32_t cut_through) {
    if (cut_through > 0) {
        return false;
    }
    return env_override("TT_MM_INJ_PREFETCH").value_or(1) != 0;
}

// in0 / in1 buffer depth in blocks: 3 when every per-core buffer, with the third block (`third_block_bytes`, an in0
// block plus an in1 block), fits the static L1 space with kL1Reserve to spare; else 2. `buffer_bytes` is every buffer
// the program allocates per core with the inputs double-buffered.
inline uint32_t input_buffer_blocks(tt::tt_metal::IDevice* device, uint64_t buffer_bytes, uint64_t third_block_bytes) {
    const uint64_t base = device->allocator()->get_base_allocator_addr(tt::tt_metal::HalMemType::L1);
    const uint64_t ceiling = device->l1_size_per_core();
    const bool fits = ceiling > base + kL1Reserve && buffer_bytes + third_block_bytes <= ceiling - base - kL1Reserve;
    const uint32_t blocks = env_override("TT_MM_IN_BUFFERS").value_or(fits ? 3u : 2u);
    return blocks < 2 ? 2 : blocks;
}

}  // namespace ttnn::experimental::prim::input_pipelining
