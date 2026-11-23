// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <algorithm>
#include <cstdint>
#include <utility>

// Compiled by both host and device: the kernel API's div_up on device, tt-metalium's on the host.
#if defined(KERNEL_BUILD) || defined(FW_BUILD)
#include "api/alignment.h"
#else
#include <tt-metalium/math.hpp>
#endif

// Per-core work split from one runtime length word. Host factories and device kernels derive every
// length-dependent count from these, so a sender and its receiver can never disagree on the split.
//
// Three splits, by what the cores need to agree on:
// - core_start / core_end: a clamped uniform split for cores that work independently; a core past the end at a
//   short length gets the empty range and returns early.
// - uniform_split: the unclamped uniform split in whole blocks that the matmul kernels share; every core walks the
//   same block count so the in0 / in1 forwarding handshakes line up, and a trailing core clips its own reads and
//   writes against the total.
// - balanced_start / balanced_end: split_work_to_cores' order, for ops whose default (host-split) factory the
//   runtime-rows path mirrors.
// chunked_range pairs core_start / core_end (per_core = div_up(total, num_cores)) and balanced_range pairs
// balanced_start / balanced_end, each as one core's [start, end).
namespace ttnn::runtime_length {

// Ceiling division (`b` non-zero), pinned to the uint32_t the splits use: the kernel API's div_up on device,
// tt::div_up on the host, both the quotient-plus-carry form that does not wrap where `a + b - 1` would. Explicit
// template arguments, so a kernel's own global div_up declared earlier in the translation unit is never picked up.
constexpr uint32_t div_up(uint32_t a, uint32_t b) {
#if defined(KERNEL_BUILD) || defined(FW_BUILD)
    return ::div_up<uint32_t, uint32_t>(a, b);
#else
    return tt::div_up(a, b);
#endif
}

// First unit of core `core_idx` under a split of `per_core` units; equals `total` (empty range) for cores past
// the end at short lengths.
constexpr uint32_t core_start(uint32_t core_idx, uint32_t per_core, uint32_t total) {
    return std::min(core_idx * per_core, total);
}

constexpr uint32_t core_end(uint32_t core_idx, uint32_t per_core, uint32_t total) {
    return std::min((core_idx + 1) * per_core, total);
}

struct UniformSplit {
    uint32_t per_core;         // units every core walks: div_up(total, num_cores)
    uint32_t blocks_per_core;  // whole blocks of `block` units per core
    uint32_t padded_total;     // per_core * num_cores: the extent the uniform split spans
    uint32_t start;            // this core's first unit (unclamped)
    uint32_t end;              // one past this core's last unit (unclamped)
};

constexpr UniformSplit uniform_split(uint32_t total, uint32_t num_cores, uint32_t core_idx, uint32_t block) {
    const uint32_t per_core = div_up(total, num_cores);
    return {per_core, div_up(per_core, block), per_core * num_cores, core_idx * per_core, (core_idx + 1) * per_core};
}

// Balanced split in split_work_to_cores order: the first `total % num_cores` cores take one extra unit.
constexpr uint32_t balanced_start(uint32_t core_idx, uint32_t total, uint32_t num_cores) {
    return core_idx * (total / num_cores) + std::min(core_idx, total % num_cores);
}

constexpr uint32_t balanced_end(uint32_t core_idx, uint32_t total, uint32_t num_cores) {
    return balanced_start(core_idx + 1, total, num_cores);
}

// This core's [start, end) under the clamped uniform split of `total` units over `num_cores` (the
// split_blocks_for_tilize order); empty for a core past the end.
constexpr std::pair<uint32_t, uint32_t> chunked_range(uint32_t core_idx, uint32_t total, uint32_t num_cores) {
    const uint32_t per_core = div_up(total, num_cores);
    return {core_start(core_idx, per_core, total), core_end(core_idx, per_core, total)};
}

// This core's [start, end) under the balanced split.
constexpr std::pair<uint32_t, uint32_t> balanced_range(uint32_t core_idx, uint32_t total, uint32_t num_cores) {
    return {balanced_start(core_idx, total, num_cores), balanced_end(core_idx, total, num_cores)};
}

}  // namespace ttnn::runtime_length
