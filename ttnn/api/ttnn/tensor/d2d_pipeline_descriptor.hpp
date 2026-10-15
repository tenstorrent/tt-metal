// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

// Where every D2DStreamService receiver in a multi-rank pipeline lives, gathered onto one
// rank and written to a file another process can read without a MeshDevice. Written by
// D2DStreamService::export_pipeline_descriptor.

#pragma once

#include <cstdint>
#include <optional>
#include <string>
#include <vector>

#include <tt-metalium/experimental/fabric/fabric_types.hpp>
#include <tt-metalium/experimental/sockets/pcie_core_location.hpp>
#include <tt-metalium/hal_types.hpp>

#include "ttnn/tensor/d2d_stream_service.hpp"

namespace ttnn {

// One coord of a D2D receiver.
struct D2DReceiverCoordDescriptor {
    tt::tt_metal::distributed::PCIeCoreLocation service_core;
    tt::tt_fabric::FabricNodeId fabric_node_id{tt::tt_fabric::MeshId{0}, 0};
    tt::tt_fabric::FabricNodeId sender_fabric_node_id{tt::tt_fabric::MeshId{0}, 0};
    tt::tt_metal::DeviceAddr backing_tensor_addr = 0;
    tt::tt_metal::DeviceAddr stage_gate_base_addr = 0;  // gate i is the uint32 at + i * L1 alignment; 0 without gates

    bool operator==(const D2DReceiverCoordDescriptor&) const = default;
};

// The receiving end of one D2D socket.
struct D2DReceiverDescriptor {
    int sender_rank = 0;
    uint32_t metadata_size_bytes = 0;
    D2DStageGateConfig stage_gate;  // num_gates == 0 when gates are off
    std::vector<D2DReceiverCoordDescriptor> coords;

    bool operator==(const D2DReceiverDescriptor&) const = default;
};

struct D2DRankDescriptor {
    int rank = 0;
    std::string host_name;
    std::optional<D2DReceiverDescriptor> inbound;  // empty on a rank with no inbound socket

    bool operator==(const D2DRankDescriptor&) const = default;
};

struct D2DPipelineDescriptor {
    static constexpr uint32_t kVersion = 1;

    std::vector<D2DRankDescriptor> ranks;  // ordered by rank

    bool operator==(const D2DPipelineDescriptor&) const = default;

    std::vector<uint8_t> serialize() const;
    static D2DPipelineDescriptor deserialize(const std::vector<uint8_t>& bytes);

    // Atomic write (tmp + rename).
    void write_to_file(const std::string& path) const;

    // Wait for the file to appear, then read it. Throws on timeout or version mismatch.
    static D2DPipelineDescriptor wait_and_read(const std::string& path, uint32_t timeout_ms = 10000);
};

// /dev/shm/tt_d2d_pipeline_<service_id>.bin
std::string d2d_pipeline_descriptor_path(const std::string& service_id);

}  // namespace ttnn
