// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

// Gemma4 sliding ring SDPA's copy of the ring-attention neighbor halo exchange
// (ring_attention_all_gather_async_multi_core_with_workers_program_factory), with its own halo kernels, so
// Gemma4-only changes leave the shared exchange (GPT-OSS sliding) untouched.

#include <cstdint>
#include <optional>
#include <vector>

#include <tt-metalium/core_coord.hpp>
#include <tt-metalium/program_descriptors.hpp>
#include "ttnn/device_operation.hpp"
#include "ttnn/operations/ccl/ccl_host_types.hpp"
#include "ttnn/operations/experimental/ccl/ring_attention_all_gather_async/device/ring_attention_all_gather_async_device_operation_types.hpp"

namespace ttnn::experimental::gemma4 {

// Sliding attention exchanges one or two predecessor tails into compact output slots.
struct SlidingHaloConfig {
    uint32_t send_to_next_start_Ht;
    uint32_t send_to_next_count_Ht;
    uint32_t send_second_start_Ht = 0;
    // A linear topology has no physical wrap link, so a hop whose destination wraps the ring travels
    // backward instead. distance is the fabric hop count to the (nearest) receiver.
    bool send_backward = false;
    uint32_t distance = 1;
    // Which cyclic predecessor this exchange ships from: 1 = the immediate neighbour. A halo wider than
    // one Q slab is covered by several exchanges, each shipping one slab tail (tail_tile_rows) per Q
    // segment into its own block of the compact buffer, starting at tile row dest_row_base (first
    // segment) or second_dest_row_base (second).
    uint32_t hop = 1;
    uint32_t tail_tile_rows = 0;  // 0 = halo_tile_rows (one-hop halo)
    uint32_t dest_row_base = 0;
    uint32_t second_dest_row_base = 0;
    // Non-empty for a line multicast over hops [hop, hop + hop_origin_rows.size()): the source tile row
    // each hop ships.
    std::vector<uint32_t> hop_origin_rows;
    // Fabric link index this exchange starts from. A one-hop halo spreads over every link; with
    // more hops, each hop uses one link and hops beyond the link count time-share (see below).
    uint32_t link_base = 0;
    // Every exchange delivers its per-link ready-increment to the hop-1 exchange's workers
    // (rendezvous_noc[link]), and only that exchange waits and signals the SDPA: one signaller per
    // semaphore, since concurrent Semaphore::up from two cores can lose an update. Each collecting worker
    // expects one arrival per remote predecessor. Empty rendezvous_noc: each worker's own core.
    uint32_t arrivals_expected = 1;
    std::vector<CoreCoord> rendezvous_noc;
    // A halo with more hops than fabric links time-shares a link. An ERISC exposes one worker
    // sender channel per direction, and concurrent workers on it stall each other, but SEQUENTIAL
    // reuse is the fabric's own protocol: close() persists the producer cursor and the next open()
    // adopts it (edm_fabric_worker_adapters.hpp). So a later hop waits on this local semaphore,
    // which its predecessor on the same link increments after closing its connection.
    bool waits_for_predecessor = false;
    bool signals_successor = false;
    uint32_t chain_semaphore_id = 0;
    uint32_t successor_noc_x = 0;
    uint32_t successor_noc_y = 0;

    bool multicast() const { return !hop_origin_rows.empty(); }
    bool collects_arrivals() const { return hop == 1; }
    bool has_rendezvous() const { return !rendezvous_noc.empty(); }
    uint32_t tail_rows() const { return tail_tile_rows != 0 ? tail_tile_rows : halo_tile_rows; }

    // Metadata supplies the source tails and cache slot during trace replay.
    const ttnn::Tensor* slot_id = nullptr;
    const ttnn::Tensor* kv_actual_isl = nullptr;
    uint32_t kv_cache_num_layers = 1;
    uint32_t kv_cache_layer_idx = 0;
    uint32_t q_local_tile_rows = 0;
    uint32_t halo_tile_rows = 0;
    uint32_t source_device = 0;

    bool derives_cache_batch_on_device() const { return slot_id != nullptr; }
    bool derives_start_on_device() const { return kv_actual_isl != nullptr; }
};

// Halo reader/writer runtime-arg layout (sliding_halo_{reader,writer}.cpp).
namespace halo_rt {
constexpr uint32_t kNeighborReaderRuntimeArgHeaderCount = 2;
// The halo GlobalSemaphore's address: not in the program hash, so re-patched on every dispatch.
constexpr uint32_t kNeighborReaderSemaphoreArg = 0;
constexpr uint32_t kNeighborWriterSemaphoreArg = 2;
constexpr uint32_t kNeighborReaderTensorDescriptorFieldCount = 8;
constexpr uint32_t kNeighborReaderMetadataTensorDescriptorFieldCount = kNeighborReaderTensorDescriptorFieldCount + 1;
constexpr uint32_t kNeighborReaderInputTileStartFieldOffset = 2;
constexpr uint32_t kNeighborReaderInputTileEndFieldOffset = 3;
constexpr uint32_t kNeighborReaderInputBatchBaseFieldOffset = 4;
constexpr uint32_t kNeighborReaderFirstOriginFieldOffset = 5;
constexpr uint32_t kNeighborReaderSecondOriginFieldOffset = 6;
constexpr uint32_t kNeighborReaderHaloPagesFieldOffset = 7;
constexpr uint32_t kNeighborWriterRuntimeArgHeaderCount = 4;
constexpr uint32_t kNeighborWriterTensorDescriptorFieldCount = 7;
constexpr uint32_t kNeighborWriterInputTileStartFieldOffset = 2;
constexpr uint32_t kNeighborWriterInputTileEndFieldOffset = 3;
constexpr uint32_t kNeighborWriterOutputOriginPageFieldOffset = 4;
constexpr uint32_t kNeighborMulticastBlockOriginsOffset = 1;
}  // namespace halo_rt

void sliding_halo_exchange_helper(
    tt::tt_metal::ProgramDescriptor& desc,
    const std::vector<Tensor>& input_tensors,
    const MeshCoordinate& target_device_coord,
    const MeshCoordinate& transport_device_coord,
    const MeshCoordinate& unicast_destination_coord,
    std::vector<Tensor>& output_tensors,
    uint32_t num_links,
    uint32_t ring_size,
    uint32_t ring_index,
    ttnn::ccl::Topology topology,
    const std::vector<GlobalSemaphore>& semaphores,
    const std::optional<tt::tt_metal::SubDeviceId>& sub_device_id,
    const ttnn::experimental::ccl::AllGatherFusedOpSignaler& fused_op_signaler,
    CoreCoord core_grid_offset,
    ttnn::ccl::CoreAllocationStrategy core_allocation_strategy,
    std::optional<uint32_t> input_batch_slice_idx,
    std::optional<uint32_t> gather_valid_Ht,
    const SlidingHaloConfig& halo);

}  // namespace ttnn::experimental::gemma4
