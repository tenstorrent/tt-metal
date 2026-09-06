// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include "ttnn/operations/matmul/device/config/registry/matmul_registry_compatibility.hpp"

#include <array>
#include <bit>
#include <cstddef>
#include <string_view>

namespace ttnn::operations::matmul::registry {
namespace {

constexpr std::array<std::uint32_t, 64> kSha256RoundConstants{
    0x428a2f98, 0x71374491, 0xb5c0fbcf, 0xe9b5dba5, 0x3956c25b, 0x59f111f1, 0x923f82a4, 0xab1c5ed5,
    0xd807aa98, 0x12835b01, 0x243185be, 0x550c7dc3, 0x72be5d74, 0x80deb1fe, 0x9bdc06a7, 0xc19bf174,
    0xe49b69c1, 0xefbe4786, 0x0fc19dc6, 0x240ca1cc, 0x2de92c6f, 0x4a7484aa, 0x5cb0a9dc, 0x76f988da,
    0x983e5152, 0xa831c66d, 0xb00327c8, 0xbf597fc7, 0xc6e00bf3, 0xd5a79147, 0x06ca6351, 0x14292967,
    0x27b70a85, 0x2e1b2138, 0x4d2c6dfc, 0x53380d13, 0x650a7354, 0x766a0abb, 0x81c2c92e, 0x92722c85,
    0xa2bfe8a1, 0xa81a664b, 0xc24b8b70, 0xc76c51a3, 0xd192e819, 0xd6990624, 0xf40e3585, 0x106aa070,
    0x19a4c116, 0x1e376c08, 0x2748774c, 0x34b0bcb5, 0x391c0cb3, 0x4ed8aa4a, 0x5b9cca4f, 0x682e6ff3,
    0x748f82ee, 0x78a5636f, 0x84c87814, 0x8cc70208, 0x90befffa, 0xa4506ceb, 0xbef9a3f7, 0xc67178f2};

constexpr std::uint32_t rotate_right(const std::uint32_t value, const unsigned shift) noexcept {
    return (value >> shift) | (value << (32 - shift));
}

void sha256_compress(std::array<std::uint32_t, 8>& state, const std::uint8_t* block) noexcept {
    std::array<std::uint32_t, 64> schedule{};
    for (std::size_t index = 0; index < 16; ++index) {
        schedule[index] = (static_cast<std::uint32_t>(block[index * 4]) << 24) |
                          (static_cast<std::uint32_t>(block[index * 4 + 1]) << 16) |
                          (static_cast<std::uint32_t>(block[index * 4 + 2]) << 8) |
                          static_cast<std::uint32_t>(block[index * 4 + 3]);
    }
    for (std::size_t index = 16; index < schedule.size(); ++index) {
        const auto s0 = rotate_right(schedule[index - 15], 7) ^ rotate_right(schedule[index - 15], 18) ^
                        (schedule[index - 15] >> 3);
        const auto s1 =
            rotate_right(schedule[index - 2], 17) ^ rotate_right(schedule[index - 2], 19) ^ (schedule[index - 2] >> 10);
        schedule[index] = schedule[index - 16] + s0 + schedule[index - 7] + s1;
    }

    auto a = state[0];
    auto b = state[1];
    auto c = state[2];
    auto d = state[3];
    auto e = state[4];
    auto f = state[5];
    auto g = state[6];
    auto h = state[7];
    for (std::size_t index = 0; index < schedule.size(); ++index) {
        const auto sum1 = rotate_right(e, 6) ^ rotate_right(e, 11) ^ rotate_right(e, 25);
        const auto choice = (e & f) ^ (~e & g);
        const auto temporary1 = h + sum1 + choice + kSha256RoundConstants[index] + schedule[index];
        const auto sum0 = rotate_right(a, 2) ^ rotate_right(a, 13) ^ rotate_right(a, 22);
        const auto majority = (a & b) ^ (a & c) ^ (b & c);
        const auto temporary2 = sum0 + majority;
        h = g;
        g = f;
        f = e;
        e = d + temporary1;
        d = c;
        c = b;
        b = a;
        a = temporary1 + temporary2;
    }
    state[0] += a;
    state[1] += b;
    state[2] += c;
    state[3] += d;
    state[4] += e;
    state[5] += f;
    state[6] += g;
    state[7] += h;
}

template <std::size_t Capacity>
class FixedPreimage {
public:
    void append(const std::string_view value) noexcept {
        for (const auto character : value) {
            bytes_[size_++] = static_cast<std::uint8_t>(character);
        }
    }

    void append_u32(const std::uint32_t value) noexcept {
        for (unsigned shift = 0; shift < 32; shift += 8) {
            bytes_[size_++] = static_cast<std::uint8_t>(value >> shift);
        }
    }

    void append_u64(const std::uint64_t value) noexcept {
        append_u32(static_cast<std::uint32_t>(value));
        append_u32(static_cast<std::uint32_t>(value >> 32));
    }

    void append(const compact::Sha256& digest) noexcept {
        for (const auto byte : digest) {
            bytes_[size_++] = byte;
        }
    }

    std::span<const std::uint8_t> bytes() const noexcept { return {bytes_.data(), size_}; }

private:
    std::array<std::uint8_t, Capacity> bytes_{};
    std::size_t size_ = 0;
};

bool board_and_cluster_agree(const DeviceCompatibilityFacts& facts) noexcept {
    switch (facts.board_class) {
        case AttestationBoardClass::BlackholeP100: return facts.cluster_class == AttestationClusterClass::BlackholeP100;
        case AttestationBoardClass::BlackholeP150:
            return facts.cluster_class == AttestationClusterClass::BlackholeP150 ||
                   facts.cluster_class == AttestationClusterClass::BlackholeP150X2 ||
                   facts.cluster_class == AttestationClusterClass::BlackholeP150X4 ||
                   facts.cluster_class == AttestationClusterClass::BlackholeP150X8;
        case AttestationBoardClass::BlackholeGalaxy:
            return facts.cluster_class == AttestationClusterClass::BlackholeGalaxy;
    }
    return false;
}

bool is_nonzero(const compact::Sha256& digest) noexcept {
    for (const auto byte : digest) {
        if (byte != 0) {
            return true;
        }
    }
    return false;
}

}  // namespace

compact::Sha256 registry_sha256(const std::span<const std::uint8_t> bytes) noexcept {
    std::array<std::uint32_t, 8> state{
        0x6a09e667, 0xbb67ae85, 0x3c6ef372, 0xa54ff53a, 0x510e527f, 0x9b05688c, 0x1f83d9ab, 0x5be0cd19};
    std::array<std::uint8_t, 64> block{};
    std::size_t offset = 0;
    while (bytes.size() - offset >= block.size()) {
        sha256_compress(state, bytes.data() + offset);
        offset += block.size();
    }
    const auto remainder = bytes.size() - offset;
    for (std::size_t index = 0; index < remainder; ++index) {
        block[index] = bytes[offset + index];
    }
    block[remainder] = 0x80;
    if (remainder >= 56) {
        sha256_compress(state, block.data());
        block.fill(0);
    }
    const auto bit_count = static_cast<std::uint64_t>(bytes.size()) * 8;
    for (std::size_t index = 0; index < 8; ++index) {
        block[63 - index] = static_cast<std::uint8_t>(bit_count >> (index * 8));
    }
    sha256_compress(state, block.data());

    compact::Sha256 result{};
    for (std::size_t index = 0; index < state.size(); ++index) {
        result[index * 4] = static_cast<std::uint8_t>(state[index] >> 24);
        result[index * 4 + 1] = static_cast<std::uint8_t>(state[index] >> 16);
        result[index * 4 + 2] = static_cast<std::uint8_t>(state[index] >> 8);
        result[index * 4 + 3] = static_cast<std::uint8_t>(state[index]);
    }
    return result;
}

DeviceCompatibilityResult derive_device_compatibility(const DeviceCompatibilityFacts& facts) noexcept {
    if (!facts.device_initialized) {
        return {.status = DeviceCompatibilityStatus::DeviceUninitialized};
    }
    if (facts.remote_only) {
        return {.status = DeviceCompatibilityStatus::RemoteDevice};
    }
    if (facts.device_count != 1 || facts.mesh_rows != 1 || facts.mesh_cols != 1) {
        return {.status = DeviceCompatibilityStatus::NotOneChip};
    }
    if (!facts.active_sub_device_manager_is_default) {
        return {.status = DeviceCompatibilityStatus::ActiveSubDeviceManager};
    }
    if (facts.architecture != 3) {
        return {.status = DeviceCompatibilityStatus::UnsupportedArchitecture};
    }
    if (!board_and_cluster_agree(facts)) {
        return {.status = DeviceCompatibilityStatus::BoardClusterMismatch};
    }
    if (!facts.firmware_bundle_present || !facts.ethernet_firmware_present) {
        return {.status = DeviceCompatibilityStatus::FirmwareUnavailable};
    }
    const auto harvested_columns = std::popcount(facts.tensix_harvesting_mask);
    if (harvested_columns > 2 || facts.compute_grid_x != 13U - harvested_columns || facts.compute_grid_y != 10 ||
        facts.physical_grid_x == 0 || facts.physical_grid_y == 0 || facts.logical_grid_x == 0 ||
        facts.logical_grid_y == 0 || facts.dram_grid_x == 0 || facts.dram_grid_y == 0 || facts.num_hw_cqs == 0 ||
        facts.num_dram_channels == 0 || facts.l1_size_per_core == 0 || facts.dram_size_per_channel == 0) {
        return {.status = DeviceCompatibilityStatus::InvalidCapability};
    }

    FixedPreimage<160> topology;
    topology.append("ttnn.matmul.registry.topology.v1");
    topology.append_u32(facts.architecture);
    topology.append_u32(static_cast<std::uint32_t>(facts.board_class));
    topology.append_u32(facts.device_count);
    topology.append_u32(facts.mesh_rows);
    topology.append_u32(facts.mesh_cols);
    const auto topology_sha256 = registry_sha256(topology.bytes());

    FixedPreimage<160> capability;
    capability.append("ttnn.matmul.registry.runtime-capability.v1");
    capability.append(topology_sha256);
    capability.append_u32(static_cast<std::uint32_t>(facts.cluster_class));
    capability.append_u32(facts.num_hw_cqs);
    capability.append_u32(facts.num_dram_channels);
    capability.append_u32(facts.l1_size_per_core);
    capability.append_u64(facts.dram_size_per_channel);
    capability.append_u32(facts.firmware_bundle_major);
    capability.append_u32(facts.firmware_bundle_minor);
    capability.append_u32(facts.firmware_bundle_patch);
    capability.append_u32(facts.ethernet_firmware_major);
    capability.append_u32(facts.ethernet_firmware_minor);
    capability.append_u32(facts.ethernet_firmware_patch);
    return {
        .status = DeviceCompatibilityStatus::Success, .runtime_capability_sha256 = registry_sha256(capability.bytes())};
}

CompatibilityStatus validate_registry_compatibility(
    const compact::TableMetadata& expected,
    const BuildCompatibilityDigests& actual_build,
    const DeviceCompatibilityResult& actual_device) noexcept {
    if (expected.compatibility_schema_version != kRegistryCompatibilitySchemaVersion ||
        !is_nonzero(expected.semantic_source_sha256) || !is_nonzero(expected.build_identity_sha256) ||
        !is_nonzero(expected.runtime_capability_sha256) || !is_nonzero(actual_build.semantic_source_sha256) ||
        !is_nonzero(actual_build.build_identity_sha256)) {
        return CompatibilityStatus::CompatibilityUnavailable;
    }
    switch (actual_device.status) {
        case DeviceCompatibilityStatus::Success: break;
        case DeviceCompatibilityStatus::QueryFailed: return CompatibilityStatus::DeviceQueryFailed;
        case DeviceCompatibilityStatus::DeviceUninitialized: return CompatibilityStatus::DeviceUninitialized;
        case DeviceCompatibilityStatus::RemoteDevice: return CompatibilityStatus::RemoteDevice;
        case DeviceCompatibilityStatus::NotOneChip: return CompatibilityStatus::NotOneChipDevice;
        case DeviceCompatibilityStatus::ActiveSubDeviceManager: return CompatibilityStatus::ActiveSubDeviceManager;
        case DeviceCompatibilityStatus::UnsupportedArchitecture: return CompatibilityStatus::UnsupportedArchitecture;
        case DeviceCompatibilityStatus::UnsupportedBoard: return CompatibilityStatus::UnsupportedBoard;
        case DeviceCompatibilityStatus::UnsupportedCluster: return CompatibilityStatus::UnsupportedCluster;
        case DeviceCompatibilityStatus::BoardClusterMismatch: return CompatibilityStatus::BoardClusterMismatch;
        case DeviceCompatibilityStatus::FirmwareUnavailable: return CompatibilityStatus::FirmwareUnavailable;
        case DeviceCompatibilityStatus::InvalidCapability: return CompatibilityStatus::InvalidDeviceCapability;
    }
    if (!is_nonzero(actual_device.runtime_capability_sha256)) {
        return CompatibilityStatus::DeviceAttestationUnavailable;
    }
    if (expected.semantic_source_sha256 != actual_build.semantic_source_sha256) {
        return CompatibilityStatus::SemanticSourceMismatch;
    }
    if (expected.build_identity_sha256 != actual_build.build_identity_sha256) {
        return CompatibilityStatus::BuildIdentityMismatch;
    }
    if (expected.runtime_capability_sha256 != actual_device.runtime_capability_sha256) {
        return CompatibilityStatus::RuntimeCapabilityMismatch;
    }
    return CompatibilityStatus::Compatible;
}

}  // namespace ttnn::operations::matmul::registry
