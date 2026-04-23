// SPDX-FileCopyrightText: © 2025 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <cstdint>
#include "internal/risc_attribs.h"

/**
 * Convert linear flat index to zigzag flat index for per-head load balancing.
 *
 * In causal attention with is_balanced=true, Q chunks in the first half of each head
 * (positions 0 to num_q_chunks/2-1) process fewer KV chunks than Q chunks in the
 * second half. This creates work imbalance when cores process consecutive Q chunks.
 *
 * Per-head zigzag interleaves light and heavy work within each core:
 *   - Even positions (0, 2, 4, ...) map to forward indices: 0, 1, 2, 3, ...
 *   - Odd positions (1, 3, 5, ...) map to backward indices: N-1, N-2, N-3, ...
 *
 * Example with num_q_chunks=20:
 *   pos 0 -> q_chunk 0  (light)
 *   pos 1 -> q_chunk 19 (heavy)
 *   pos 2 -> q_chunk 1  (light)
 *   pos 3 -> q_chunk 18 (heavy)
 *   ...
 *
 * This works for ANY division of work across cores (no exact division constraint).
 */
FORCE_INLINE uint32_t linear_to_zigzag(uint32_t linear_flat, uint32_t num_q_chunks) {
    const uint32_t head_idx = linear_flat / num_q_chunks;
    const uint32_t pos_in_head = linear_flat % num_q_chunks;

    uint32_t q_chunk;
    if (pos_in_head % 2 == 0) {
        // Even positions: forward from start
        q_chunk = pos_in_head / 2;
    } else {
        // Odd positions: backward from end
        q_chunk = num_q_chunks - 1 - (pos_in_head / 2);
    }

    return head_idx * num_q_chunks + q_chunk;
}

/**
 * Remap Q index for work balancing.
 *
 * When use_zigzag is true, applies zigzag remapping to balance light/heavy work.
 * When false, returns the linear index unchanged.
 *
 * With -O3, the compiler eliminates the dead branch when use_zigzag is constexpr.
 */
FORCE_INLINE uint32_t remap_q_index(uint32_t linear_index, uint32_t num_q_chunks, bool use_zigzag) {
    if (use_zigzag) {
        return linear_to_zigzag(linear_index, num_q_chunks);
    }
    return linear_index;
}

/**
 * Result of decomposing a flat q-chunk index into (batch, head, q_chunk) coordinates.
 */
struct FlatQIndex {
    uint32_t nb;
    uint32_t nq;
    uint32_t q_chunk;
};

/**
 * Decompose a linear flat index in [0, B*NQH*num_q_chunks) into (nb, nq, q_chunk).
 *
 * Applies the remap (linear or zigzag) first, then splits the remapped flat index as:
 *   nb      = flat / (NQH * num_q_chunks)
 *   nq      = (flat / num_q_chunks) % NQH
 *   q_chunk = flat % num_q_chunks
 *
 * Used by SDPA reader/writer kernels under flat work distribution to fetch the right
 * Q/K/V and write outputs for each (batch, head, q_chunk) triple a core is assigned.
 */
FORCE_INLINE FlatQIndex
decompose_flat_q_index(uint32_t linear_index, uint32_t num_q_chunks, uint32_t NQH, bool use_zigzag) {
    const uint32_t flat = remap_q_index(linear_index, num_q_chunks, use_zigzag);
    return {
        /*nb=*/flat / (NQH * num_q_chunks),
        /*nq=*/(flat / num_q_chunks) % NQH,
        /*q_chunk=*/flat % num_q_chunks,
    };
}

/**
 * Single-chip ring-iter proxy case (kernel-side; values mirror
 * ttnn::operations::transformer::RingProxyCase).
 *
 * - None: regular SDPA (full Q × full K).
 * - Up:   K-loop capped at k_num_chunks/2 (reader/compute); full Q range assigned.
 * - Down: only the heavy Q half assigned; decomposition runs against q_num_chunks/2 and
 *         shifts q_chunk up by q_num_chunks/2.
 */
enum class RingProxyMode : uint8_t { None = 0, Up = 1, Down = 2 };

/**
 * Effective Q chunk count + offset for a given proxy mode.
 *
 * DOWN halves the assigned Q chunk space and points at the upper half. UP and None keep the
 * full range at offset 0 (UP's K-loop cap is applied separately at the K-iteration site).
 */
struct ProxyQRange {
    uint32_t q_num_effective;
    uint32_t q_chunk_offset;
};

FORCE_INLINE constexpr ProxyQRange proxy_q_range(uint32_t q_num_chunks, RingProxyMode proxy) {
    if (proxy == RingProxyMode::Down) {
        return {q_num_chunks / 2, q_num_chunks / 2};
    }
    return {q_num_chunks, 0};
}

/**
 * Decompose a flat index into (nb, nq, q_chunk) honoring the proxy mode. The returned
 * q_chunk is already shifted into the range the actual Q/K/V tensors live in, so callers
 * can feed it straight into tile-index math.
 */
FORCE_INLINE FlatQIndex decompose_flat_q_index_with_proxy(
    uint32_t linear_index, uint32_t q_num_chunks, uint32_t NQH, bool use_zigzag, RingProxyMode proxy) {
    const ProxyQRange range = proxy_q_range(q_num_chunks, proxy);
    const FlatQIndex flat = decompose_flat_q_index(linear_index, range.q_num_effective, NQH, use_zigzag);
    return {flat.nb, flat.nq, flat.q_chunk + range.q_chunk_offset};
}

/**
 * Compute-side flat q_chunk lookup. Equivalent to the q_chunk field of
 * decompose_flat_q_index_with_proxy, but skips the (nb, nq) split — compute only needs
 * q_chunk for causal masking; reader/writer are the ones that fetch tensor tiles.
 */
FORCE_INLINE uint32_t proxy_q_chunk(uint32_t q_iter, uint32_t q_num_chunks, bool use_zigzag, RingProxyMode proxy) {
    const ProxyQRange range = proxy_q_range(q_num_chunks, proxy);
    return remap_q_index(q_iter, range.q_num_effective, use_zigzag) % range.q_num_effective + range.q_chunk_offset;
}

// Compile-time proxy mode gated by SDPA_RING_PROXY_{UP,DOWN} defines the host emits when
// program_config.ring_proxy_case is Up/Down. Defined once here so all three kernels
// (reader/writer/compute) share one source of truth.
#if defined(SDPA_RING_PROXY_DOWN)
constexpr RingProxyMode sdpa_proxy_mode = RingProxyMode::Down;
#elif defined(SDPA_RING_PROXY_UP)
constexpr RingProxyMode sdpa_proxy_mode = RingProxyMode::Up;
#else
constexpr RingProxyMode sdpa_proxy_mode = RingProxyMode::None;
#endif
