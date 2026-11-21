// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include "ttnn/tensor/d2d_pipeline_descriptor.hpp"

#include <sys/stat.h>

#include <algorithm>
#include <cctype>
#include <cerrno>
#include <chrono>
#include <cstdio>
#include <cstring>
#include <filesystem>
#include <fstream>
#include <thread>

#include <fmt/format.h>
#include <tt-metalium/experimental/sockets/shm_resource_tracker.hpp>
#include <tt_stl/assert.hpp>

#include "d2d_pipeline_descriptor_generated.h"

namespace ttnn {

namespace fb = d2d_flatbuffer;
namespace dist_fb = tt::tt_metal::distributed::flatbuffer;

namespace {

// Writes to `path + ".tmp"`, then renames it to `path`, so a reader never sees a partial file.
void write_file_atomically(const std::string& path, const std::vector<uint8_t>& bytes) {
    const std::string tmp_path = path + ".tmp";
    std::ofstream ofs(tmp_path, std::ios::binary);
    TT_FATAL(ofs.is_open(), "Failed to open descriptor file for writing: {}", tmp_path);
    ofs.write(reinterpret_cast<const char*>(bytes.data()), static_cast<std::streamsize>(bytes.size()));
    ofs.close();
    TT_FATAL(!ofs.fail(), "Failed to write descriptor file: {}", tmp_path);
    TT_FATAL(
        std::rename(tmp_path.c_str(), path.c_str()) == 0,
        "Failed to rename descriptor file from {} to {}: {}",
        tmp_path,
        path,
        std::strerror(errno));
}

// Waits up to `timeout_ms` for `path` to exist, then returns its contents.
std::vector<uint8_t> wait_and_read_file(const std::string& path, uint32_t timeout_ms) {
    const auto start = std::chrono::steady_clock::now();
    while (!std::filesystem::exists(path)) {
        if (std::chrono::steady_clock::now() - start > std::chrono::milliseconds(timeout_ms)) {
            TT_THROW("Timeout waiting for descriptor file: {}", path);
        }
        std::this_thread::sleep_for(std::chrono::milliseconds(1));
    }
    std::ifstream ifs(path, std::ios::binary | std::ios::ate);
    TT_FATAL(ifs.is_open(), "Failed to open descriptor file for reading: {}", path);
    const auto pos = ifs.tellg();
    TT_FATAL(pos > 0, "Descriptor file is empty or unreadable: {}", path);
    std::vector<uint8_t> bytes(static_cast<std::size_t>(pos));
    ifs.seekg(0, std::ios::beg);
    ifs.read(reinterpret_cast<char*>(bytes.data()), static_cast<std::streamsize>(bytes.size()));
    TT_FATAL(!ifs.fail(), "Failed to read descriptor file: {}", path);
    return bytes;
}

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
        const auto service_core = to_flatbuffer(builder, c.service_core);
        const auto metadata_core =
            c.metadata_core ? to_flatbuffer(builder, *c.metadata_core) : flatbuffers::Offset<fb::PCIeCoreLocation>();
        coords.push_back(fb::CreateD2DReceiverCoord(
            builder,
            service_core,
            *c.fabric_node_id.mesh_id,
            c.fabric_node_id.chip_id,
            *c.sender_fabric_node_id.mesh_id,
            c.sender_fabric_node_id.chip_id,
            c.backing_tensor_addr,
            c.stage_gate_base_addr,
            metadata_core));
    }
    return fb::CreateD2DReceiver(
        builder,
        rx.sender_rank,
        rx.metadata_size_bytes,
        rx.stage_gate.num_gates,
        rx.stage_gate.slot_id_offset_bytes,
        rx.stage_gate.gate_mode_offset_bytes,
        builder.CreateVector(coords),
        rx.metadata_addr,
        rx.stage_gate_stride_bytes);
}

D2DReceiverDescriptor from_flatbuffer(const fb::D2DReceiver& f) {
    D2DReceiverDescriptor rx{
        .sender_rank = f.sender_rank(),
        .metadata_size_bytes = f.metadata_size_bytes(),
        .metadata_addr = f.metadata_addr(),
        .stage_gate =
            D2DStageGateConfig{
                .num_gates = f.num_stage_gates(),
                .slot_id_offset_bytes = f.stage_gate_slot_offset_bytes(),
                .gate_mode_offset_bytes = f.stage_gate_mode_offset_bytes()},
        .stage_gate_stride_bytes = f.stage_gate_stride_bytes(),
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
                .stage_gate_base_addr = c->stage_gate_base_addr(),
                .metadata_core = c->metadata_core() != nullptr
                                     ? std::optional<PCIeCoreLocation>(from_flatbuffer(*c->metadata_core()))
                                     : std::nullopt});
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
        std::vector<flatbuffers::Offset<fb::D2DBoundary>> boundaries;
        for (const auto& b : r.boundaries) {
            std::vector<flatbuffers::Offset<fb::D2DReceiver>> receivers;
            receivers.reserve(b.receivers.size());
            for (const auto& rx : b.receivers) {
                receivers.push_back(to_flatbuffer(builder, rx));
            }
            boundaries.push_back(fb::CreateD2DBoundary(builder, builder.CreateVector(receivers)));
        }
        rank_offsets.push_back(fb::CreateD2DRank(builder, r.rank, host_name, builder.CreateVector(boundaries)));
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
        if (fr->boundaries() != nullptr) {
            for (const auto* fb_boundary : *fr->boundaries()) {
                D2DBoundaryDescriptor b;
                if (fb_boundary->receivers() != nullptr) {
                    for (const auto* rx : *fb_boundary->receivers()) {
                        b.receivers.push_back(from_flatbuffer(*rx));
                    }
                }
                r.boundaries.push_back(std::move(b));
            }
        }
        desc.ranks.push_back(std::move(r));
    }
    return desc;
}

void D2DPipelineDescriptor::write_to_file(const std::string& path) const { write_file_atomically(path, serialize()); }

D2DPipelineDescriptor D2DPipelineDescriptor::wait_and_read(const std::string& path, uint32_t timeout_ms) {
    return deserialize(wait_and_read_file(path, timeout_ms));
}

std::string d2d_pipeline_descriptor_path(const std::string& service_id) {
    // The id goes into a file name that gets written and removed, so keep it to one plain name.
    const bool plain = !service_id.empty() && std::all_of(service_id.begin(), service_id.end(), [](char c) {
        return std::isalnum(static_cast<unsigned char>(c)) || c == '_' || c == '-' || c == '.';
    });
    TT_FATAL(
        plain && service_id.find("..") == std::string::npos,
        "d2d_pipeline_descriptor_path: service_id '{}' must be letters, digits, '_', '-' or '.', with no '..'",
        service_id);
    return fmt::format("/dev/shm/tt_d2d_pipeline_{}.bin", service_id);
}

D2DPipelineDescriptorFile::D2DPipelineDescriptorFile(std::string path, const D2DPipelineDescriptor& pipeline) :
    path_(std::move(path)) {
    // Track before writing. The tracker's first use removes files left by dead processes, and
    // one of those could be this path.
    auto& tracker = tt::tt_metal::distributed::ShmResourceTracker::instance();
    tracker.track_file(path_);  // removed at process exit too, in case this object never gets destroyed
    try {
        pipeline.write_to_file(path_);
        struct stat st{};
        TT_FATAL(::stat(path_.c_str(), &st) == 0, "D2DPipelineDescriptorFile: can't stat {}", path_);
        device_ = st.st_dev;
        inode_ = st.st_ino;
    } catch (...) {
        std::remove(path_.c_str());
        tracker.untrack_file(path_);
        throw;
    }
}

D2DPipelineDescriptorFile::~D2DPipelineDescriptorFile() {
    // Each write makes a new file (tmp + rename), so a different inode means a newer export owns
    // the path now. Leave its file and its tracking alone.
    struct stat st{};
    if (::stat(path_.c_str(), &st) == 0 && (st.st_dev != device_ || st.st_ino != inode_)) {
        return;
    }
    if (std::remove(path_.c_str()) == 0 || errno == ENOENT) {
        tt::tt_metal::distributed::ShmResourceTracker::instance().untrack_file(path_);
    }
}

}  // namespace ttnn
