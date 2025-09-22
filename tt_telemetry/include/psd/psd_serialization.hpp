// SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <string>
#include <memory>
#include <optional>
#include <vector>

#include <umd/device/types/arch.hpp>

namespace tt::umd {
class Cluster;
}

namespace tt::tt_metal {

class PSD;
// Emit PSD to a text proto file
void emit_physical_system_descriptor_to_text_proto(const PSD& descriptor, const std::optional<std::string>& file_path);
// Serialize PSD to protobuf binary format (byte vector)
std::vector<uint8_t> serialize_physical_system_descriptor_to_bytes(const PSD& descriptor);

// Deserialize from protobuf binary format to PSD (byte vector)
PSD deserialize_physical_system_descriptor_from_bytes(
    const std::unique_ptr<tt::umd::Cluster>& cluster,
    tt::ARCH arch,
    const std::vector<uint8_t>& data,
    bool using_mock_cluster_desc = false);

}  // namespace tt::tt_metal
