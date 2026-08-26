// SPDX-FileCopyrightText: 2026 Tenstorrent USA, Inc.
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

namespace ttnn::experimental::all_gather_minimal_matmul_registry::compact {

inline constexpr std::uint16_t kKeySchemaVersion = 1;
inline constexpr std::uint16_t kReplaySchemaVersion = 1;
inline constexpr std::uint16_t kCodegenRecipeAbi = 1;
inline constexpr std::uint16_t kTableLockSchemaVersion = 1;
inline constexpr std::size_t kMaxTensorRank = 8;
inline constexpr std::size_t kMaxChunkSizes = 64;
inline constexpr std::size_t kMaxActivationParameters = 8;

using Sha256 = std::array<std::uint8_t, 32>;
using RegistryEntryId = Sha256;

struct TensorDescriptor {
    std::uint8_t rank{};
    std::array<std::uint64_t, kMaxTensorRank> logical_shape{};
    std::array<std::uint64_t, kMaxTensorRank> padded_shape{};
    std::uint32_t dtype{};
    std::uint32_t layout{};
    std::uint32_t memory_layout{};
    std::uint32_t buffer_type{};
    std::uint16_t tile_height{};
    std::uint16_t tile_width{};
    bool tile_transpose_of_faces{};
    bool tile_transpose_within_face{};
    Sha256 memory_config_sha256{};
    Sha256 tensor_topology_sha256{};

    auto operator<=>(const TensorDescriptor&) const = default;
};

struct OptionalTensorDescriptor {
    bool present{};
    TensorDescriptor tensor{};

    auto operator<=>(const OptionalTensorDescriptor&) const = default;
};

struct DeviceDescriptor {
    std::uint32_t architecture{};
    std::uint32_t board_capability_class{};
    std::uint16_t device_count{};
    std::uint16_t mesh_rows{};
    std::uint16_t mesh_cols{};
    std::uint16_t compute_grid_x{};
    std::uint16_t compute_grid_y{};
    Sha256 ordered_mesh_sha256{};
    Sha256 topology_sha256{};
    Sha256 runtime_capability_sha256{};

    auto operator<=>(const DeviceDescriptor&) const = default;
};

struct WorkloadDescriptor {
    std::uint64_t logical_m{};
    std::uint64_t logical_k{};
    std::uint64_t logical_n{};
    std::uint64_t padded_m{};
    std::uint64_t padded_k{};
    std::uint64_t padded_n{};
    std::uint64_t batch{};

    auto operator<=>(const WorkloadDescriptor&) const = default;
};

struct OperationDescriptor {
    std::uint32_t topology{};
    std::uint32_t fsdp_topology{};
    std::uint32_t num_links{};
    std::uint32_t ring_size{};
    bool cluster_axis_present{};
    std::uint32_t cluster_axis{};
    bool fsdp_cluster_axis_present{};
    std::uint32_t fsdp_cluster_axis{};
    std::uint32_t fsdp_ring_size{};
    std::uint32_t semaphore_count{};
    std::uint32_t fsdp_semaphore_count{};
    bool barrier_semaphore_present{};
    bool persistent_output_present{};
    bool persistent_weight_present{};
    bool force_transpose{};
    std::uint32_t num_workers_per_link{};
    std::uint32_t num_buffers_per_channel{};
    bool scalar_present{};
    std::uint32_t scalar_f32_bits{};
    std::int32_t chunks{};
    std::int32_t dim{};
    std::uint8_t chunk_size_count{};
    std::array<std::uint32_t, kMaxChunkSizes> chunk_sizes{};
    bool fuse_swiglu{};
    bool activation_present{};
    std::uint32_t activation_op{};
    std::uint8_t activation_parameter_count{};
    std::array<std::uint32_t, kMaxActivationParameters> activation_parameter_f32_bits{};
    bool output_dtype_present{};
    std::uint32_t output_dtype{};
    bool output_memory_config_present{};
    std::uint32_t output_memory_layout{};
    std::uint32_t output_buffer_type{};
    std::uint32_t output_layout{};
    std::uint16_t output_tile_height{};
    std::uint16_t output_tile_width{};
    bool output_tile_transpose_of_faces{};
    bool output_tile_transpose_within_face{};
    Sha256 output_memory_config_sha256{};
    Sha256 output_tensor_topology_sha256{};

    auto operator<=>(const OperationDescriptor&) const = default;
};

struct KeyDescriptor {
    std::uint16_t schema_version{kKeySchemaVersion};
    std::uint16_t codegen_recipe_abi{kCodegenRecipeAbi};
    DeviceDescriptor device{};
    WorkloadDescriptor workload{};
    OperationDescriptor operation{};
    TensorDescriptor input{};
    TensorDescriptor weight{};
    OptionalTensorDescriptor bias{};
    OptionalTensorDescriptor ternary_input_a{};
    OptionalTensorDescriptor ternary_input_b{};
    OptionalTensorDescriptor persistent_output{};
    OptionalTensorDescriptor persistent_weight{};

    auto operator<=>(const KeyDescriptor&) const = default;
};

struct MinimalMatmulConfigDescriptor {
    std::uint32_t m_block_size{};
    std::uint32_t k_block_size{};
    std::uint32_t n_block_size{};
    std::uint32_t subblock_h{};
    std::uint32_t subblock_w{};
    std::uint16_t compute_grid_x{};
    std::uint16_t compute_grid_y{};

    auto operator<=>(const MinimalMatmulConfigDescriptor&) const = default;
};

struct ComputeKernelDescriptor {
    std::uint32_t math_fidelity{};
    bool math_approx_mode{};
    bool fp32_dest_acc_en{};
    bool packer_l1_acc{};
    bool dst_full_sync_en{};
    std::uint32_t throttle_level{};

    auto operator<=>(const ComputeKernelDescriptor&) const = default;
};

struct ReplayDescriptor {
    std::uint16_t schema_version{kReplaySchemaVersion};
    MinimalMatmulConfigDescriptor config{};
    ComputeKernelDescriptor compute_kernel_config{};

    auto operator<=>(const ReplayDescriptor&) const = default;
};

struct EntryDescriptor {
    RegistryEntryId entry_id{};
    KeyDescriptor key{};
    ReplayDescriptor replay{};

    auto operator<=>(const EntryDescriptor&) const = default;
};

struct TableMetadata {
    std::uint16_t key_schema_version{kKeySchemaVersion};
    std::uint16_t replay_schema_version{kReplaySchemaVersion};
    Sha256 content_sha256{};
    Sha256 semantic_source_sha256{};
    Sha256 build_identity_sha256{};
    Sha256 runtime_capability_sha256{};

    auto operator<=>(const TableMetadata&) const = default;
};

// This is the complete native hand-off from the offline predictor/exporter to
// TT-metal.  The lock and entries are generated as C++ constants and compiled
// into the operation; the runtime never parses a model, JSON, or a sidecar.
// Provenance digests are deliberately separate so changing evidence, predictor,
// or exporter cannot silently preserve the same certified table identity.
struct TableLock {
    std::uint16_t schema_version{kTableLockSchemaVersion};
    std::uint16_t codegen_recipe_abi{kCodegenRecipeAbi};
    std::uint64_t entry_count{};
    TableMetadata metadata{};
    Sha256 evidence_manifest_sha256{};
    Sha256 predictor_sha256{};
    Sha256 exporter_sha256{};

    auto operator<=>(const TableLock&) const = default;
};

enum class TableValidationStatus : std::uint8_t {
    Valid,
    Empty,
    LockSchemaMismatch,
    EntryCountMismatch,
    MissingLockDigest,
    EntrySchemaMismatch,
    MissingEntryId,
    RuntimeCapabilityMismatch,
    EntriesNotStrictlySorted,
};

inline constexpr bool sha256_is_zero(const Sha256& digest) noexcept {
    for (const auto byte : digest) {
        if (byte != 0) {
            return false;
        }
    }
    return true;
}

// This constexpr structural validator is intentionally usable from a
// static_assert in generated data.  Cryptographic digests are produced and
// independently checked by codegen; native code binds their exact values and
// additionally rejects missing provenance, ABI drift, wrong capability tables,
// duplicate keys, or a table whose order would invalidate binary search.
inline constexpr TableValidationStatus validate_table_lock(
    const TableLock& lock, const std::span<const EntryDescriptor> entries) noexcept {
    if (lock.schema_version != kTableLockSchemaVersion ||
        lock.metadata.key_schema_version != kKeySchemaVersion ||
        lock.metadata.replay_schema_version != kReplaySchemaVersion || lock.codegen_recipe_abi != kCodegenRecipeAbi) {
        return TableValidationStatus::LockSchemaMismatch;
    }
    if (lock.entry_count != entries.size()) {
        return TableValidationStatus::EntryCountMismatch;
    }
    if (entries.empty()) {
        return TableValidationStatus::Empty;
    }
    if (sha256_is_zero(lock.metadata.content_sha256) || sha256_is_zero(lock.metadata.semantic_source_sha256) ||
        sha256_is_zero(lock.metadata.build_identity_sha256) ||
        sha256_is_zero(lock.metadata.runtime_capability_sha256) || sha256_is_zero(lock.evidence_manifest_sha256) ||
        sha256_is_zero(lock.predictor_sha256) || sha256_is_zero(lock.exporter_sha256)) {
        return TableValidationStatus::MissingLockDigest;
    }
    for (std::size_t index = 0; index < entries.size(); ++index) {
        const auto& entry = entries[index];
        if (entry.key.schema_version != kKeySchemaVersion || entry.key.codegen_recipe_abi != kCodegenRecipeAbi ||
            entry.replay.schema_version != kReplaySchemaVersion) {
            return TableValidationStatus::EntrySchemaMismatch;
        }
        if (sha256_is_zero(entry.entry_id)) {
            return TableValidationStatus::MissingEntryId;
        }
        if (entry.key.device.runtime_capability_sha256 != lock.metadata.runtime_capability_sha256) {
            return TableValidationStatus::RuntimeCapabilityMismatch;
        }
        if (index != 0 && !(entries[index - 1].key < entry.key)) {
            return TableValidationStatus::EntriesNotStrictlySorted;
        }
    }
    return TableValidationStatus::Valid;
}

inline constexpr const EntryDescriptor* lookup_exact(
    const KeyDescriptor& key, const std::span<const EntryDescriptor> entries) noexcept {
    const auto candidate = std::lower_bound(
        entries.begin(), entries.end(), key, [](const EntryDescriptor& entry, const KeyDescriptor& requested) {
            return entry.key < requested;
        });
    return candidate != entries.end() && candidate->key == key ? &*candidate : nullptr;
}

static_assert(std::is_trivially_copyable_v<EntryDescriptor>);
static_assert(std::is_standard_layout_v<EntryDescriptor>);
static_assert(sizeof(EntryDescriptor) <= 4096);
static_assert(std::is_trivially_copyable_v<TableLock>);
static_assert(std::is_standard_layout_v<TableLock>);
static_assert(sizeof(TableLock) <= 512);

}  // namespace ttnn::experimental::all_gather_minimal_matmul_registry::compact
