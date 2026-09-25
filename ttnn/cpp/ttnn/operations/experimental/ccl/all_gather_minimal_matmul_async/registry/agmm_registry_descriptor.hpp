// SPDX-FileCopyrightText: 2026 Tenstorrent USA, Inc.
// SPDX-License-Identifier: Apache-2.0
#pragma once

#include <algorithm>
#include <array>
#include <compare>
#include <cstddef>
#include <cstdint>
#include <span>

namespace ttnn::experimental::all_gather_minimal_matmul_registry::compact {
inline constexpr std::uint16_t kKeySchemaVersion = 1;
inline constexpr std::uint16_t kReplaySchemaVersion = 1;
inline constexpr std::uint16_t kCodegenRecipeAbi = 2;
inline constexpr std::uint32_t kBlackholeArchitecture = 3;
inline constexpr std::size_t kMaxTensorRank = 8;
inline constexpr std::size_t kMaxChunkSizes = 64;
inline constexpr std::size_t kMaxActivationParameters = 8;

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
    auto operator<=>(const TensorDescriptor&) const = default;
};
struct OptionalTensorDescriptor {
    bool present{};
    TensorDescriptor tensor{};
    auto operator<=>(const OptionalTensorDescriptor&) const = default;
};
struct DeviceDescriptor {
    std::uint32_t architecture{};
    std::uint16_t device_count{};
    std::uint16_t mesh_rows{};
    std::uint16_t mesh_cols{};
    std::uint16_t compute_grid_x{};
    std::uint16_t compute_grid_y{};
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
    // Generated cohorts are sorted by the complete exact key and queried with
    // lower_bound. Ordering therefore has a concrete lookup contract here;
    // nested key descriptors provide lexicographic ordering for this operator.
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
    bool operator==(const MinimalMatmulConfigDescriptor&) const = default;
};
struct ComputeKernelDescriptor {
    std::uint32_t math_fidelity{};
    bool math_approx_mode{};
    bool fp32_dest_acc_en{};
    bool packer_l1_acc{};
    bool dst_full_sync_en{};
    std::uint32_t throttle_level{};
    bool operator==(const ComputeKernelDescriptor&) const = default;
};
struct ReplayDescriptor {
    std::uint16_t schema_version{kReplaySchemaVersion};
    MinimalMatmulConfigDescriptor config{};
    ComputeKernelDescriptor compute_kernel_config{};
    bool operator==(const ReplayDescriptor&) const = default;
};
struct EntryDescriptor {
    KeyDescriptor key{};
    ReplayDescriptor replay{};
    bool operator==(const EntryDescriptor&) const = default;
};
struct CohortDescriptor {
    DeviceDescriptor device{};
    std::span<const EntryDescriptor> entries{};
};

inline constexpr bool is_supported_device(const DeviceDescriptor& device) noexcept {
    if (device.architecture != kBlackholeArchitecture || device.compute_grid_x == 0 || device.compute_grid_y == 0 ||
        static_cast<std::uint32_t>(device.mesh_rows) * device.mesh_cols != device.device_count) {
        return false;
    }
    constexpr std::array<std::uint16_t, 2> supported_device_counts{8, 32};
    return std::find(supported_device_counts.begin(), supported_device_counts.end(), device.device_count) !=
           supported_device_counts.end();
}
inline constexpr bool validate_entries(
    const DeviceDescriptor& certified_device, std::span<const EntryDescriptor> entries) noexcept {
    if (!is_supported_device(certified_device) || entries.empty()) {
        return false;
    }
    for (std::size_t index = 0; index < entries.size(); ++index) {
        const auto& entry = entries[index];
        if (entry.key.schema_version != kKeySchemaVersion || entry.key.codegen_recipe_abi != kCodegenRecipeAbi ||
            entry.replay.schema_version != kReplaySchemaVersion || entry.key.device != certified_device ||
            (index != 0 && !(entries[index - 1].key < entry.key))) {
            return false;
        }
    }
    return true;
}
inline constexpr const EntryDescriptor* lookup_exact(
    const KeyDescriptor& key, std::span<const EntryDescriptor> entries) noexcept {
    const auto candidate = std::lower_bound(
        entries.begin(), entries.end(), key, [](const EntryDescriptor& entry, const KeyDescriptor& requested) {
            return entry.key < requested;
        });
    return candidate != entries.end() && candidate->key == key ? &*candidate : nullptr;
}
}  // namespace ttnn::experimental::all_gather_minimal_matmul_registry::compact
