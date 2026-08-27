// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <algorithm>
#include <array>
#include <compare>
#include <cstddef>
#include <cstdint>
#include <span>
#include <type_traits>

namespace ttnn::operations::matmul::registry::compact {

// Shared by the compact-lock validator, emitted entries, runtime key
// construction, and compatibility attestation. Changing it requires a new
// exporter/runtime contract rather than a local literal update.
inline constexpr std::uint16_t kCodegenRecipeAbi = 1;

enum class Domain : std::uint8_t { DenseMatmul = 0, DenseLinear = 1, DenseAddmm = 2 };
// Enumerator values are stable selector ABI. emit_cpp.py explicitly reorders
// canonically reviewed lock entries into this POD's defaulted runtime order.
enum class DataType : std::uint8_t { BFloat16 = 0, BFloat8B = 1, Float32 = 2 };
enum class Layout : std::uint8_t { RowMajor = 0, Tile = 1 };
enum class MemoryLayout : std::uint8_t { Interleaved = 0 };
enum class BufferType : std::uint8_t { Dram = 0, L1 = 1 };
enum class ProgramFamily : std::uint8_t { MultiCoreReuse = 0, MultiCast1D = 1, MultiCast2D = 2 };
enum class MathFidelity : std::uint8_t { LoFi = 0, HiFi2 = 1, HiFi3 = 2, HiFi4 = 3 };
enum class ThrottleLevel : std::uint8_t { NoThrottle = 0, Throttle1 = 1, Throttle2 = 2, Throttle3 = 3 };

using Sha256 = std::array<std::uint8_t, 32>;
using RegistryEntryId = Sha256;

struct TensorDescriptor {
    BufferType buffer_type{};
    DataType dtype{};
    Layout layout{};
    MemoryLayout memory_layout{};
    std::uint16_t tile_height{};
    std::uint16_t tile_width{};

    auto operator<=>(const TensorDescriptor&) const = default;
};

struct KeyDescriptor {
    std::uint32_t architecture{};
    bool bcast_batch_present{};
    bool bcast_batch{};
    std::uint32_t board_capability_class{};
    std::uint16_t codegen_recipe_abi{};
    std::uint16_t compute_grid_x{};
    std::uint16_t compute_grid_y{};
    std::uint16_t device_count{};
    bool has_activation{};
    bool has_bias{};
    TensorDescriptor input_a{};
    TensorDescriptor input_b{};
    std::uint64_t logical_k{};
    std::uint64_t logical_m{};
    std::uint64_t logical_n{};
    std::uint16_t mesh_cols{};
    std::uint16_t mesh_rows{};
    TensorDescriptor output{};
    std::uint64_t padded_k{};
    std::uint64_t padded_m{};
    std::uint64_t padded_n{};
    bool run_batched{};
    std::uint16_t schema_version{};
    Sha256 topology_sha256{};
    bool transpose_a{};
    bool transpose_b{};
    bool untilize_out{};
    // The three public-operation domains are disjoint lookup axes. V1 admits
    // only the exact no-bias/no-activation subset; addmm additionally binds
    // exact IEEE-754 alpha/beta spellings and admits beta +/-0 only, so the
    // otherwise unkeyed additive input is provably unused.
    Domain domain{};
    std::uint32_t alpha_f32_bits{};
    std::uint32_t beta_f32_bits{};

    auto operator<=>(const KeyDescriptor&) const = default;
};

struct MultiCoreReuseDescriptor {
    std::uint16_t compute_grid_x{};
    std::uint16_t compute_grid_y{};
    std::uint32_t in0_block_w{};
    std::uint32_t out_subblock_h{};
    std::uint32_t out_subblock_w{};
    std::uint32_t per_core_m{};
    std::uint32_t per_core_n{};
    // The first supported family admits only the exact native null state.
    bool allowed_worker_cores_present{};

    auto operator<=>(const MultiCoreReuseDescriptor&) const = default;
};

struct ComputeKernelDescriptor {
    MathFidelity math_fidelity{};
    ThrottleLevel throttle_level{};
    bool math_approx_mode{};
    bool fp32_dest_acc_en{};
    bool packer_l1_acc{};
    bool dst_full_sync_en{};

    auto operator<=>(const ComputeKernelDescriptor&) const = default;
};

struct CallStateDescriptor {
    TensorDescriptor output{};
    bool untilize_out{};
    // These bits record exact admitted defaults, rather than synthesizing them.
    bool bcast_batch_is_null{};
    bool user_core_coord_is_null{};
    bool user_fused_activation_is_null{};
    bool user_run_batched_is_false{};
    bool transpose_a_is_false{};
    bool transpose_b_is_false{};
    bool output_tile_is_null{};
    bool global_cb_is_null{};
    bool sub_device_id_is_null{};

    auto operator<=>(const CallStateDescriptor&) const = default;
};

struct ReplayDescriptor {
    std::uint16_t schema_version{};
    ProgramFamily family{};
    MultiCoreReuseDescriptor program_config{};
    ComputeKernelDescriptor compute_kernel_config{};
    CallStateDescriptor call_state{};

    auto operator<=>(const ReplayDescriptor&) const = default;
};

struct EntryDescriptor {
    RegistryEntryId entry_id{};
    KeyDescriptor key{};
    ReplayDescriptor replay{};

    auto operator<=>(const EntryDescriptor&) const = default;
};

struct TableMetadata {
    std::uint16_t lock_schema_version{};
    std::uint16_t key_schema_version{};
    std::uint16_t replay_schema_version{};
    // Zero disables typed exact selection. Schema 1 is the legacy attested
    // form; schema 2 is deterministic direct-bank evidence, which binds the
    // semantic/build contract without inventing device-session provenance.
    std::uint16_t program_config_only_evidence_schema_version{};
    // Enabled online models independently require the same bound evidence
    // schema even when the lock contains no exact entries.
    std::uint16_t online_program_config_model_evidence_schema_version{};
    Sha256 content_sha256{};
    Sha256 semantic_source_sha256{};
    Sha256 build_identity_sha256{};
    Sha256 runtime_capability_sha256{};
    // Zero for exact-only locks. Active online models bind this independently
    // reconstructed digest in both table metadata and model metadata.
    Sha256 online_model_bundle_binding_sha256{};
};

static_assert(std::is_trivially_copyable_v<EntryDescriptor>);
static_assert(std::is_standard_layout_v<EntryDescriptor>);
static_assert(sizeof(EntryDescriptor) <= 512);

// Exact lower-bound lookup over immutable sorted storage. No hashing, allocation,
// file access, or native recipe construction occurs here.
inline constexpr const EntryDescriptor* lookup_exact(
    const KeyDescriptor& key, const std::span<const EntryDescriptor> entries) noexcept {
    const auto candidate = std::lower_bound(
        entries.begin(), entries.end(), key, [](const EntryDescriptor& entry, const KeyDescriptor& requested_key) {
            return entry.key < requested_key;
        });
    return candidate != entries.end() && candidate->key == key ? &*candidate : nullptr;
}

class ExactIndex {
public:
    constexpr explicit ExactIndex(const std::span<const EntryDescriptor> entries) noexcept : entries_(entries) {}

    constexpr const EntryDescriptor* lookup(const KeyDescriptor& key) const noexcept {
        return lookup_exact(key, entries_);
    }

    constexpr std::size_t size() const noexcept { return entries_.size(); }

private:
    std::span<const EntryDescriptor> entries_;
};

static_assert(std::is_trivially_copyable_v<ExactIndex>);

}  // namespace ttnn::operations::matmul::registry::compact
