// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include "ttnn/tensor/d2d_pipeline_descriptor.hpp"

#include <fmt/format.h>
#include <tt_stl/assert.hpp>
#include <internal/descriptor_file.hpp>

#include "d2d_pipeline_descriptor_generated.h"

namespace ttnn {

namespace fb = d2d_flatbuffer;
namespace dist_fb = tt::tt_metal::distributed::flatbuffer;
using tt::tt_metal::distributed::PCIeCoreLocation;

namespace {

flatbuffers::Offset<fb::PCIeCoreLocation> to_flatbuffer(
    flatbuffers::FlatBufferBuilder& builder, const PCIeCoreLocation& loc) {
    const auto& c = loc.core.device_coord.coords();
    const auto mesh_coord =
        dist_fb::CreateMeshCoordinate(builder, builder.CreateVector(std::vector<uint32_t>(c.begin(), c.end())));
    return fb::CreatePCIeCoreLocation(
        builder,
        loc.pcie_device_id,
        mesh_coord,
        static_cast<uint32_t>(loc.core.core_coord.x),
        static_cast<uint32_t>(loc.core.core_coord.y),
        static_cast<uint32_t>(loc.virtual_core.x),
        static_cast<uint32_t>(loc.virtual_core.y));
}

PCIeCoreLocation from_flatbuffer(const fb::PCIeCoreLocation& f) {
    TT_FATAL(
        f.mesh_coord() != nullptr && f.mesh_coord()->values() != nullptr,
        "D2DPipelineDescriptor: service core missing mesh coord");
    const auto* values = f.mesh_coord()->values();
    return PCIeCoreLocation{
        .pcie_device_id = f.pcie_device_id(),
        .core = tt::tt_metal::distributed::MeshCoreCoord(
            tt::tt_metal::distributed::MeshCoordinate(ttsl::Span<const uint32_t>(values->data(), values->size())),
            tt::tt_metal::CoreCoord(f.core_x(), f.core_y())),
        .virtual_core = tt::tt_metal::CoreCoord(f.virtual_core_x(), f.virtual_core_y()),
    };
}

flatbuffers::Offset<fb::D2DReceiver> to_flatbuffer(
    flatbuffers::FlatBufferBuilder& builder, const D2DReceiverDescriptor& rx) {
    std::vector<flatbuffers::Offset<fb::D2DReceiverCoord>> coords;
    for (const auto& c : rx.coords) {
        coords.push_back(fb::CreateD2DReceiverCoord(
            builder,
            to_flatbuffer(builder, c.service_core),
            *c.fabric_node_id.mesh_id,
            c.fabric_node_id.chip_id,
            *c.sender_fabric_node_id.mesh_id,
            c.sender_fabric_node_id.chip_id,
            c.backing_tensor_addr,
            c.stage_gate_base_addr));
    }
    return fb::CreateD2DReceiver(
        builder,
        rx.sender_rank,
        rx.metadata_size_bytes,
        rx.stage_gate.num_gates,
        rx.stage_gate.slot_id_offset_bytes,
        rx.stage_gate.gate_mode_offset_bytes,
        builder.CreateVector(coords));
}

D2DReceiverDescriptor from_flatbuffer(const fb::D2DReceiver& f) {
    D2DReceiverDescriptor rx{
        .sender_rank = f.sender_rank(),
        .metadata_size_bytes = f.metadata_size_bytes(),
        .stage_gate =
            D2DStageGateConfig{
                .num_gates = f.num_stage_gates(),
                .slot_id_offset_bytes = f.stage_gate_slot_offset_bytes(),
                .gate_mode_offset_bytes = f.stage_gate_mode_offset_bytes()},
    };
    if (f.coords() != nullptr) {
        for (const auto* c : *f.coords()) {
            TT_FATAL(c->service_core() != nullptr, "D2DPipelineDescriptor: coord entry missing service core");
            rx.coords.push_back(D2DReceiverCoordDescriptor{
                .service_core = from_flatbuffer(*c->service_core()),
                .fabric_node_id = tt::tt_fabric::FabricNodeId(tt::tt_fabric::MeshId{c->mesh_id()}, c->chip_id()),
                .sender_fabric_node_id =
                    tt::tt_fabric::FabricNodeId(tt::tt_fabric::MeshId{c->sender_mesh_id()}, c->sender_chip_id()),
                .backing_tensor_addr = c->backing_tensor_addr(),
                .stage_gate_base_addr = c->stage_gate_base_addr()});
        }
    }
    return rx;
}

}  // namespace

std::vector<uint8_t> D2DPipelineDescriptor::serialize() const {
    flatbuffers::FlatBufferBuilder builder(1024);
    std::vector<flatbuffers::Offset<fb::D2DRank>> rank_offsets;
    for (const auto& r : ranks) {
        const auto host_name = builder.CreateString(r.host_name);
        const auto inbound = r.inbound ? to_flatbuffer(builder, *r.inbound) : flatbuffers::Offset<fb::D2DReceiver>();
        rank_offsets.push_back(fb::CreateD2DRank(builder, r.rank, host_name, inbound));
    }
    builder.Finish(fb::CreateD2DPipelineDescriptor(builder, kVersion, builder.CreateVector(rank_offsets)));
    return {builder.GetBufferPointer(), builder.GetBufferPointer() + builder.GetSize()};
}

D2DPipelineDescriptor D2DPipelineDescriptor::deserialize(const std::vector<uint8_t>& bytes) {
    flatbuffers::Verifier verifier(bytes.data(), bytes.size());
    TT_FATAL(fb::VerifyD2DPipelineDescriptorBuffer(verifier), "D2DPipelineDescriptor: malformed buffer");
    const auto* root = fb::GetD2DPipelineDescriptor(bytes.data());
    TT_FATAL(
        root->version() == kVersion,
        "D2DPipelineDescriptor version mismatch: got {}, expected {}",
        root->version(),
        kVersion);

    D2DPipelineDescriptor desc;
    if (root->ranks() == nullptr) {
        return desc;
    }
    for (const auto* fr : *root->ranks()) {
        D2DRankDescriptor r{.rank = fr->rank(), .host_name = fr->host_name() ? fr->host_name()->str() : ""};
        if (fr->inbound() != nullptr) {
            r.inbound = from_flatbuffer(*fr->inbound());
        }
        desc.ranks.push_back(std::move(r));
    }
    return desc;
}

void D2DPipelineDescriptor::write_to_file(const std::string& path) const {
    const auto bytes = serialize();
    tt::tt_metal::internal::write_file_atomically(path, ttsl::Span<const uint8_t>(bytes.data(), bytes.size()));
}

D2DPipelineDescriptor D2DPipelineDescriptor::wait_and_read(const std::string& path, uint32_t timeout_ms) {
    return deserialize(tt::tt_metal::internal::wait_and_read_file(path, timeout_ms));
}

std::string d2d_pipeline_descriptor_path(const std::string& service_id) {
    return fmt::format("/dev/shm/tt_d2d_pipeline_{}.bin", service_id);
}

}  // namespace ttnn
