// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <algorithm>
#include <cstdint>
#include <span>
#include <type_traits>

#include "ttnn/operations/matmul/device/config/registry/matmul_registry_descriptor.hpp"

namespace ttnn::operations::matmul::registry::compact {

// Output tiles a subblock may occupy in the DST register file, mirroring
// ttnn::get_dest_reg_count (compute_kernel_config.cpp). DST holds
// 64 * 16 rows of 16 datums = 16384 datums, i.e. 16 tiles at the 32x32 tile
// this registry keys; double-buffered DST (dst_full_sync_en == false) halves
// that to 8, and FP32 accumulation halves it again to 4. Budgeting at the
// double-buffered value -- the smaller of the two dst_full_sync_en cases --
// means an admitted subblock is legal at either setting of that knob, which
// matters now that it is a key axis and entries measured at one value are
// looked up beside entries measured at the other.
inline constexpr std::uint32_t kDestSubblockTileBudget = 8;
inline constexpr std::uint32_t kFp32DestSubblockTileBudget = 4;

// math_approx_mode is normalized to this on every lookup key and every emitted
// entry. It is TTNN's own default and the only value a call that supplies no
// compute-kernel config can present. Normalizing rather than matching is sound
// only while the flag is inert: it gates nothing but the SFPU APPROX define,
// and an admitted matmul kernel runs no SFPU op because has_activation is
// rejected outright. entries_permit_math_approx_normalization turns that
// precondition into a build failure the moment a table stops honouring it.
inline constexpr bool kMathApproxModeIsInertAt = false;

// The checked runtime registry contains complete native recipes only. Model
// training and prediction happen offline before an exact entry is promoted.
struct ProgramConfigDescriptor {
    ProgramFamily family{};
    std::uint16_t compute_grid_x{};
    std::uint16_t compute_grid_y{};
    std::uint32_t in0_block_w{};
    std::uint32_t out_subblock_h{};
    std::uint32_t out_subblock_w{};
    std::uint32_t per_core_m{};
    std::uint32_t per_core_n{};
    std::uint32_t out_block_h{};
    std::uint32_t out_block_w{};
    std::uint32_t num_global_cb_receivers{};
    bool allowed_worker_cores_present{};
    bool fuse_batch{};
    bool mcast_in0{};
    bool transpose_mcast{};
    bool fused_activation_present{};
    bool gather_in0{};
    bool hop_cores_present{};
    bool untilize_out{};
    bool stream_in1{};

    auto operator<=>(const ProgramConfigDescriptor&) const = default;
};

struct ProgramConfigCandidate {
    ProgramConfigDescriptor program_config{};
    ComputeKernelDescriptor compute_kernel_config{};
};

struct ProgramConfigExactEntry {
    RegistryEntryId entry_id{};
    KeyDescriptor key{};
    ProgramConfigDescriptor program_config{};
    ComputeKernelDescriptor compute_kernel_config{};

    auto operator<=>(const ProgramConfigExactEntry&) const = default;
};

static_assert(std::is_trivially_copyable_v<ProgramConfigExactEntry>);
static_assert(std::is_standard_layout_v<ProgramConfigExactEntry>);

inline constexpr const ProgramConfigExactEntry* lookup_program_config_exact(
    const KeyDescriptor& key, const std::span<const ProgramConfigExactEntry> entries) noexcept {
    const auto candidate = std::lower_bound(
        entries.begin(),
        entries.end(),
        key,
        [](const ProgramConfigExactEntry& entry, const KeyDescriptor& requested_key) {
            return entry.key < requested_key;
        });
    return candidate != entries.end() && candidate->key == key ? &*candidate : nullptr;
}

// The key carries the compute-kernel knobs, so the caller's own knob vector has
// to be spelled the way the table spells it. Only math_approx_mode is
// normalized (see kMathApproxModeIsInertAt); the other five knobs each change
// the arithmetic or the scheduling and are matched exactly.
constexpr KeyDescriptor normalize_key_compute_kernel(KeyDescriptor key) noexcept {
    key.compute_kernel.math_approx_mode = kMathApproxModeIsInertAt;
    return key;
}

// Every entry's recipe must be spelled with exactly the compute-kernel knobs
// its key binds. A measurement is evidence only for the knobs it ran at, so an
// entry whose value disagreed with its key would answer a caller with numerics
// they never asked for.
constexpr bool entries_bind_key_compute_kernel(const std::span<const ProgramConfigExactEntry> entries) noexcept {
    for (const auto& entry : entries) {
        if (entry.key.compute_kernel != entry.compute_kernel_config) {
            return false;
        }
    }
    return true;
}

// The companion precondition for normalize_key_compute_kernel: normalizing
// math_approx_mode out of the key is sound only while no admitted entry can
// run an SFPU op, so every entry must be activation-free on both sides of the
// key/value pair and carry the normalized spelling.
constexpr bool entries_permit_math_approx_normalization(
    const std::span<const ProgramConfigExactEntry> entries) noexcept {
    for (const auto& entry : entries) {
        if (entry.key.has_activation || entry.program_config.fused_activation_present ||
            entry.key.compute_kernel.math_approx_mode != kMathApproxModeIsInertAt ||
            entry.compute_kernel_config.math_approx_mode != kMathApproxModeIsInertAt) {
            return false;
        }
    }
    return true;
}

// Bank evidence is portable across board identities, but not harvested worker
// grids: distinct 11x10, 12x10, and 13x10 winners remain distinct exact keys.
constexpr KeyDescriptor direct_bank_key(KeyDescriptor key) noexcept {
    key.board_capability_class = 0;
    key.topology_sha256 = {};
    return key;
}

constexpr bool legal_program_config_candidate(
    const KeyDescriptor& key, const ProgramConfigCandidate& candidate) noexcept {
    const auto& program = candidate.program_config;
    if (program.fused_activation_present || program.gather_in0 || program.hop_cores_present || program.untilize_out ||
        program.stream_in1) {
        return false;
    }
    if (program.compute_grid_x == 0 || program.compute_grid_y == 0 || program.compute_grid_x > key.compute_grid_x ||
        program.compute_grid_y > key.compute_grid_y || program.in0_block_w == 0 || program.out_subblock_h == 0 ||
        program.out_subblock_w == 0 || program.per_core_m == 0 || program.per_core_n == 0 ||
        program.allowed_worker_cores_present || key.input_a.tile_height == 0 || key.input_a.tile_width == 0 ||
        key.input_b.tile_height == 0 || key.input_b.tile_width == 0 || key.padded_m % key.input_a.tile_height != 0 ||
        key.padded_k % key.input_a.tile_width != 0 || key.padded_k % key.input_b.tile_height != 0 ||
        key.padded_n % key.input_b.tile_width != 0) {
        return false;
    }
    const auto m_tiles = key.padded_m / key.input_a.tile_height;
    const auto a_k_tiles = key.padded_k / key.input_a.tile_width;
    const auto b_k_tiles = key.padded_k / key.input_b.tile_height;
    const auto n_tiles = key.padded_n / key.input_b.tile_width;
    if (a_k_tiles != b_k_tiles || a_k_tiles % program.in0_block_w != 0 ||
        program.per_core_m % program.out_subblock_h != 0 || program.per_core_n % program.out_subblock_w != 0 ||
        static_cast<std::uint64_t>(program.out_subblock_h) * program.out_subblock_w >
            (candidate.compute_kernel_config.fp32_dest_acc_en ? kFp32DestSubblockTileBudget
                                                              : kDestSubblockTileBudget)) {
        return false;
    }
    switch (program.family) {
        case ProgramFamily::MultiCoreReuse:
            return program.out_block_h == 0 && program.out_block_w == 0 && program.num_global_cb_receivers == 0 &&
                   !program.fuse_batch && !program.mcast_in0 && !program.transpose_mcast &&
                   m_tiles % program.per_core_m == 0 && n_tiles == program.per_core_n;
        case ProgramFamily::MultiCast1D: {
            if (!program.fuse_batch || program.transpose_mcast || program.per_core_n > 64 ||
                program.out_block_h != program.per_core_m || program.out_block_w != program.per_core_n ||
                program.num_global_cb_receivers != 1) {
                return false;
            }
            const auto m_blocks = m_tiles / program.per_core_m + (m_tiles % program.per_core_m != 0);
            const auto n_blocks = n_tiles / program.per_core_n + (n_tiles % program.per_core_n != 0);
            const auto core_count = static_cast<std::uint64_t>(program.compute_grid_x) * program.compute_grid_y;
            const bool complete_axis =
                program.mcast_in0 ? program.per_core_m == m_tiles : program.per_core_n == n_tiles;
            return complete_axis && n_blocks != 0 && m_blocks <= core_count / n_blocks;
        }
        case ProgramFamily::MultiCast2D: {
            if (!program.fuse_batch || program.mcast_in0 || program.out_block_h != program.per_core_m ||
                program.out_block_w != program.per_core_n || program.num_global_cb_receivers != 0) {
                return false;
            }
            const auto m_blocks = m_tiles / program.per_core_m + (m_tiles % program.per_core_m != 0);
            const auto n_blocks = n_tiles / program.per_core_n + (n_tiles % program.per_core_n != 0);
            return program.transpose_mcast ? m_blocks <= program.compute_grid_x && n_blocks <= program.compute_grid_y
                                           : m_blocks <= program.compute_grid_y && n_blocks <= program.compute_grid_x;
        }
    }
    return false;
}

}  // namespace ttnn::operations::matmul::registry::compact
