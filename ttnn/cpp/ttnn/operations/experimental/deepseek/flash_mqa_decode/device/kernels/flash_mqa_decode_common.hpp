// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <cstdint>

// Chunk plan shared by the reader, writer and compute kernels. Every kernel derives it from the
// same runtime lengths, so all three agree on which chunks a core attends and on the tree shape.
namespace flash_mqa_decode {

struct Lengths {
    std::uint32_t ring;
    std::uint32_t selected;
};

struct Chunk {
    // 0: kv (ring), 1: selected_kv.
    std::uint32_t segment;
    // First row of the chunk within its segment.
    std::uint32_t first_row;
    std::uint32_t valid_rows;
};

inline std::uint32_t div_up(std::uint32_t a, std::uint32_t b) { return (a + b - 1) / b; }
inline std::uint32_t min_u32(std::uint32_t a, std::uint32_t b) { return a < b ? a : b; }

inline Lengths clamp_lengths(
    std::uint32_t ring, std::uint32_t selected, std::uint32_t ring_capacity, std::uint32_t selected_capacity) {
    return {min_u32(ring, ring_capacity), min_u32(selected, selected_capacity)};
}

inline std::uint32_t num_chunks(const Lengths& lengths, std::uint32_t k_chunk_size) {
    return div_up(lengths.ring, k_chunk_size) + div_up(lengths.selected, k_chunk_size);
}

// Chunks are numbered ring first, then selected; S-block b attends chunks b, b + num_s_blocks, ...
inline Chunk chunk_at(std::uint32_t v, const Lengths& lengths, std::uint32_t k_chunk_size) {
    const std::uint32_t ring_chunks = div_up(lengths.ring, k_chunk_size);
    if (v < ring_chunks) {
        const std::uint32_t first_row = v * k_chunk_size;
        return {0, first_row, min_u32(k_chunk_size, lengths.ring - first_row)};
    }
    const std::uint32_t first_row = (v - ring_chunks) * k_chunk_size;
    return {1, first_row, min_u32(k_chunk_size, lengths.selected - first_row)};
}

inline std::uint32_t num_block_chunks(std::uint32_t s_block, std::uint32_t total_chunks, std::uint32_t num_s_blocks) {
    return s_block < total_chunks ? div_up(total_chunks - s_block, num_s_blocks) : 0;
}

// Binomial reduction tree over the active S-blocks [0, num_active). At step s, block b receives from
// b + 2^s when b is a multiple of 2^(s+1); otherwise b sends to b - 2^s and leaves the tree. Active
// blocks are a prefix, so a receiver's partners are always steps 0, 1, ..., num_received - 1.
inline std::uint32_t num_received(std::uint32_t s_block, std::uint32_t num_active, std::uint32_t num_steps) {
    std::uint32_t received = 0;
    for (std::uint32_t step = 0; step < num_steps; ++step) {
        const std::uint32_t span = 1u << step;
        if ((s_block & ((span << 1) - 1)) != 0) {
            break;
        }
        if (s_block + span < num_active) {
            ++received;
        }
    }
    return received;
}

// Step at which block b (> 0) sends its partial to b - 2^step.
inline std::uint32_t send_step(std::uint32_t s_block) {
    std::uint32_t step = 0;
    while (((s_block >> step) & 1u) == 0) {
        ++step;
    }
    return step;
}

}  // namespace flash_mqa_decode
