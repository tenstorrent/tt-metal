// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

// Program pieces of the sparse KV push shared by its two hosts: the standalone push (HighBwSparsePushFactory) and the
// push fused into sparse_sdpa (SparseSDPAPushFusedFactory). Both run the same kernels on the same core layout:
// one fabric channel per (link, direction), each with kSendersPerChannel sender cores (table reader + writer) feeding
// one fabric mux core that owns the router connection.

#pragma once

#include <algorithm>
#include <cstdint>
#include <map>
#include <set>
#include <string>
#include <vector>

#include <tt-metalium/experimental/fabric/fabric.hpp>
#include <tt-metalium/tensor_accessor_args.hpp>
#include <tt-metalium/work_split.hpp>
#include "ttnn/operations/ccl/ccl_common.hpp"
#include "ttnn/operations/ccl/common/host/moe_utils.hpp"
#include "ttnn/tensor/tensor.hpp"
#include "sparse_push_plan_device_operation.hpp"

namespace ttnn::operations::experimental::high_bw_all_gather::sparse_push {

constexpr uint32_t kDirections = 4;  // E, W, N, S (the eth_chan_directions order get_neighbors returns)
constexpr uint32_t kSendersPerChannel = 2;
constexpr uint8_t kMuxBuffersPerChannel = 2;
constexpr uint32_t kPacketSlots = 8;
constexpr uint32_t kMetaEntryBytes = 16;
constexpr uint32_t kScalarCbBytes = 64;
// Requester's table header and per-owner counts (512 B) plus a window of 16 bitmap blocks (128 B each).
constexpr uint32_t kTableCbBytes = 4096;
constexpr uint32_t kHeaderSlots = 1 + 4;  // semaphore increments + one per packet in flight

enum Cb : uint32_t { kData = 0, kMeta = 1, kTable = 2, kScalar = 3, kHeaders = 4 };

constexpr const char* kKernelDir = "ttnn/cpp/ttnn/operations/experimental/high_bw_all_gather/device/kernels/";

struct CircularBuffer {
    uint32_t id;
    uint32_t page_bytes;
    uint32_t pages;
    tt::DataFormat format;
};

inline std::vector<CircularBuffer> circular_buffers(uint32_t packet_bytes) {
    return {
        {kData, packet_bytes, kPacketSlots, tt::DataFormat::RawUInt8},
        {kMeta, kMetaEntryBytes, kPacketSlots, tt::DataFormat::RawUInt32},
        {kTable, kTableCbBytes, 1, tt::DataFormat::RawUInt32},
        {kScalar, kScalarCbBytes, 1, tt::DataFormat::RawUInt32},
        {kHeaders,
         static_cast<uint32_t>(tt::tt_fabric::get_tt_fabric_packet_header_size_bytes()),
         kHeaderSlots,
         tt::DataFormat::RawUInt32},
    };
}

inline uint32_t pages_per_packet(uint32_t page_bytes) {
    return std::max<uint32_t>(
        1u, static_cast<uint32_t>(tt::tt_fabric::get_tt_fabric_max_payload_size_bytes()) / page_bytes);
}

inline CoreRangeSet to_range_set(const std::vector<CoreCoord>& cores) {
    std::set<CoreRange> ranges;
    for (const auto& c : cores) {
        ranges.insert(CoreRange(c, c));
    }
    return CoreRangeSet(ranges);
}

// Sender cores first (sender w belongs to channel w / kSendersPerChannel, channel c = link * kDirections + dir), then
// one mux core per channel.
struct CoreLayout {
    uint32_t links = 0;
    std::vector<CoreCoord> senders;
    std::vector<CoreCoord> muxes;
    CoreRangeSet sender_range;
    CoreRangeSet mux_range;

    uint32_t channels() const { return links * kDirections; }
    uint32_t num_senders() const { return static_cast<uint32_t>(senders.size()); }
};

inline uint32_t cores_needed(uint32_t links) { return links * kDirections * (kSendersPerChannel + 1); }

inline CoreLayout core_layout(const CoreRangeSet& grid, uint32_t links) {
    TT_FATAL(links >= 1 && links <= 2, "the sparse KV push supports 1 or 2 links, got {}", links);
    const auto cores = corerange_to_cores(grid, std::nullopt, true);
    const uint32_t senders = links * kDirections * kSendersPerChannel;
    TT_FATAL(
        cores.size() >= cores_needed(links),
        "the sparse KV push needs {} sender + {} mux cores, the grid has {}",
        senders,
        links * kDirections,
        cores.size());
    CoreLayout layout{.links = links};
    layout.senders.assign(cores.begin(), cores.begin() + senders);
    layout.muxes.assign(cores.begin() + senders, cores.begin() + cores_needed(links));
    layout.sender_range = to_range_set(layout.senders);
    layout.mux_range = to_range_set(layout.muxes);
    return layout;
}

inline tt::tt_fabric::FabricMuxConfig mux_config(tt::tt_metal::distributed::MeshDevice* mesh_device) {
    return tt::tt_fabric::FabricMuxConfig(
        /*num_full_size_channels=*/kSendersPerChannel,
        /*num_header_only_channels=*/0,
        /*num_buffers_full_size_channel=*/kMuxBuffersPerChannel,
        /*num_buffers_header_only_channel=*/0,
        /*buffer_size_bytes_full_size_channel=*/tt::tt_fabric::get_tt_fabric_channel_buffer_size_bytes(),
        /*base_l1_address=*/mesh_device->allocator()->get_base_allocator_addr(tt::tt_metal::HalMemType::L1));
}

// Destination fabric ids (row-major ranks) and mesh shape: both kernels look up every destination's first hop.
inline std::map<std::string, std::string> kernel_defines(tt::tt_metal::distributed::MeshDevice* mesh_device) {
    const auto mesh_shape = mesh_device->get_view().shape();
    std::vector<uint32_t> chip_ids, mesh_ids;
    for (const auto& coord : tt::tt_metal::distributed::MeshCoordinateRange(mesh_shape)) {
        const auto node = mesh_device->get_fabric_node_id(coord);
        chip_ids.push_back(static_cast<uint32_t>(node.chip_id));
        mesh_ids.push_back(static_cast<uint32_t>(*node.mesh_id));
    }
    return {
        {"DEST_CHIP_ID", ttnn::operations::ccl::common::stringify(chip_ids)},
        {"DEST_MESH_ID", ttnn::operations::ccl::common::stringify(mesh_ids)},
        {"FABRIC_2D", "1"},
        {"MESH_ROWS", std::to_string(mesh_shape[0])},
        {"MESH_COLS", std::to_string(mesh_shape[1])},
    };
}

// The chip's four torus neighbours, in channel direction order.
inline std::vector<tt::tt_metal::distributed::MeshCoordinate> torus_neighbors(
    const tt::tt_metal::distributed::MeshDeviceView& mesh_view,
    const tt::tt_metal::distributed::MeshCoordinate& coord) {
    const auto [neighbors, directions] =
        ttnn::operations::ccl::common::get_neighbors(mesh_view, coord, tt::tt_fabric::Topology::Torus, std::nullopt);
    TT_FATAL(
        neighbors.size() == kDirections && std::all_of(directions.begin(), directions.end(), [](bool d) { return d; }),
        "the sparse KV push needs all four torus neighbours");
    return neighbors;
}

// What the table reader pushes from the KVPE cache `input` (slot from batch_index_tensor[0] when given).
struct ReaderConfig {
    const Tensor& input;
    const Tensor& table;
    const std::optional<Tensor>& batch_index_tensor;
    uint32_t num_devices;
    uint32_t stripe_chunk;
    uint32_t max_rows;
    uint32_t push_part;  // 0 all, 1 prefix, 2 remainder
};

inline std::vector<uint32_t> reader_compile_args(const ReaderConfig& cfg, uint32_t num_dram_banks) {
    const uint32_t page_bytes = cfg.input.buffer()->aligned_page_size();
    const auto layout = SparsePushPlanLayout::make(cfg.max_rows, cfg.stripe_chunk);
    TT_FATAL(
        cfg.table.padded_shape()[2] == layout.pages * cfg.num_devices,
        "push table has {} pages, expected {} per device",
        cfg.table.padded_shape()[2],
        layout.pages);
    std::vector<uint32_t> ct = {
        page_bytes,
        pages_per_packet(page_bytes),
        num_dram_banks,
        cfg.input.buffer()->num_pages() / cfg.input.padded_shape()[0],  // pages per cache slot
        cfg.batch_index_tensor.has_value() ? 1u : 0u,
        cfg.num_devices,
        cfg.stripe_chunk,
        layout.num_blocks,
        layout.pages,
        SparsePushPlanLayout::kOwnerOff,
        SparsePushPlanLayout::kBitmapOff,
        SparsePushPlanLayout::kOwnerPrefixOff,
        cfg.push_part,
    };
    tt::tt_metal::TensorAccessorArgs(cfg.input.buffer()).append_to(ct);
    tt::tt_metal::TensorAccessorArgs(cfg.table.buffer()).append_to(ct);
    if (cfg.batch_index_tensor.has_value()) {
        tt::tt_metal::TensorAccessorArgs(cfg.batch_index_tensor->buffer()).append_to(ct);
    }
    return ct;
}

// Reader common runtime args: {input, table, batch index tensor, num layers, layer index, scalar slot}.
inline std::vector<uint32_t> reader_common_args(
    const ReaderConfig& cfg,
    uint32_t batch_slot_num_layers,
    uint32_t batch_slot_layer_idx,
    uint32_t input_batch_index) {
    return {
        cfg.input.buffer()->address(),
        cfg.table.buffer()->address(),
        cfg.batch_index_tensor.has_value() ? cfg.batch_index_tensor->buffer()->address() : 0u,
        batch_slot_num_layers,
        batch_slot_layer_idx,
        input_batch_index};
}

inline std::vector<uint32_t> reader_runtime_args(const CoreLayout& layout, uint32_t sender, uint32_t my_rank) {
    const uint32_t channel = sender / kSendersPerChannel;
    return {
        channel / kDirections,  // link
        channel % kDirections,  // direction
        my_rank,
        sender % kSendersPerChannel,
        kSendersPerChannel,
        layout.links};
}

inline std::vector<uint32_t> writer_compile_args(
    const Tensor& output,
    uint32_t num_devices,
    uint32_t my_rank,
    const CoreLayout& layout,
    const tt::tt_fabric::FabricMuxConfig& mux) {
    const uint32_t page_bytes = output.buffer()->aligned_page_size();
    std::vector<uint32_t> ct = {
        page_bytes,
        num_devices,
        my_rank,
        layout.num_senders(),
        pages_per_packet(page_bytes) * page_bytes,
        layout.links};
    tt::tt_metal::TensorAccessorArgs(output.buffer()).append_to(ct);
    ttnn::ccl::fabric_mux_connection_ct_args(
        kSendersPerChannel, tt::tt_fabric::FabricMuxChannelType::FULL_SIZE_CHANNEL, mux, ct);
    return ct;
}

// Sender runtime args: {sender, direction, link, sub, sync core x/y, every sender's NoC x/y, mux client ABI}.
// Works for a Program or a ProgramDescriptor (the mux client semaphores are allocated in it).
template <typename ProgramOrDescriptor>
std::vector<uint32_t> writer_runtime_args(
    tt::tt_metal::distributed::MeshDevice* mesh_device,
    const CoreLayout& layout,
    uint32_t sender,
    const tt::tt_fabric::FabricMuxConfig& mux,
    ProgramOrDescriptor& program) {
    const uint32_t channel = sender / kSendersPerChannel, sub = sender % kSendersPerChannel;
    const auto sync_core = mesh_device->worker_core_from_logical_core(layout.senders[0]);
    std::vector<uint32_t> rt = {
        sender,
        channel % kDirections,
        channel / kDirections,
        sub,
        static_cast<uint32_t>(sync_core.x),
        static_cast<uint32_t>(sync_core.y)};
    for (const auto& core : layout.senders) {
        const auto phys = mesh_device->worker_core_from_logical_core(core);
        rt.push_back(static_cast<uint32_t>(phys.x));
        rt.push_back(static_cast<uint32_t>(phys.y));
    }
    ttnn::ccl::fabric_mux_connection_rt_args(
        /*mux_connection_valid=*/true,
        /*is_termination_master=*/sub == 0,
        tt::tt_fabric::FabricMuxChannelType::FULL_SIZE_CHANNEL,
        mesh_device->worker_core_from_logical_core(layout.muxes[channel]),
        /*worker_id=*/sub,
        layout.senders[sender],
        mux,
        program,
        mesh_device->worker_core_from_logical_core(layout.senders[channel * kSendersPerChannel]),
        rt);
    return rt;
}

// Mux runtime args of channel c: its connection to the neighbour in that channel's direction.
template <typename ProgramOrDescriptor>
std::vector<uint32_t> mux_runtime_args(
    tt::tt_metal::distributed::MeshDevice* mesh_device,
    const tt::tt_metal::distributed::MeshCoordinate& coord,
    const std::vector<tt::tt_metal::distributed::MeshCoordinate>& neighbors,
    const CoreLayout& layout,
    uint32_t channel,
    const tt::tt_fabric::FabricMuxConfig& mux,
    ProgramOrDescriptor& program) {
    return mux.get_fabric_mux_run_time_args(
        mesh_device->get_fabric_node_id(coord),
        mesh_device->get_fabric_node_id(neighbors[channel % kDirections]),
        channel / kDirections,
        program,
        layout.muxes[channel]);
}

}  // namespace ttnn::operations::experimental::high_bw_all_gather::sparse_push
