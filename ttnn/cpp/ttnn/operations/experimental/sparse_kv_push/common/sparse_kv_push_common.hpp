// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <cstdint>
#include <vector>

#include <tt-metalium/experimental/distributed_tensor/topology/tensor_topology.hpp>
#include <tt-metalium/experimental/distributed_tensor/topology/distributed_tensor_configs.hpp>
#include "ttnn/distributed/types.hpp"
#include "ttnn/tensor/tensor.hpp"
#include "kernels/sparse_push_constants.hpp"

namespace ttnn::operations::experimental::sparse_kv_push {

// One fabric channel per (link, direction the chip has a fabric neighbour in), each with kSendersPerChannel sender
// cores (reader + writer) feeding one mux core that owns the router connection.
constexpr uint32_t kMaxDirections = 4;  // E, W, N, S
constexpr uint32_t kSendersPerChannel = 2;

constexpr uint32_t cores_needed(uint32_t links, uint32_t directions = kMaxDirections) {
    return links * directions * (kSendersPerChannel + 1);
}

// A direction a chip sends in: the fabric's eth_chan_directions value for it, the neighbour that way, and the fabric
// link indices that forward to it (get_forwarding_link_indices); the push's link l uses links[l].
struct FabricDirection {
    uint32_t eth_direction;
    tt::tt_metal::distributed::MeshCoordinate neighbor;
    std::vector<uint32_t> links;
};

// Whether the push can route on this mesh: a 2D fabric (FABRIC_2D or a torus on either or both axes) on a 2D mesh of at
// least 2 chips. Which links exist is discovered per chip (fabric_directions), so no axis needs to wrap.
bool fabric_supported(tt::tt_metal::distributed::MeshDevice* mesh_device);

// The directions the chip at `coord` of `tensor`'s mesh has fabric neighbours in, in E, W, N, S order: both ways along
// an axis the fabric closes into a ring, only the existing ones along a line (none at the far end), none along an axis
// of one chip.
std::vector<FabricDirection> fabric_directions(
    const Tensor& tensor, const tt::tt_metal::distributed::MeshCoordinate& coord);

// The most directions any chip of `tensor`'s mesh sends in; the push's core grid is sized for it.
uint32_t max_fabric_directions(const Tensor& tensor);

// Rows travel in packets of pages_per_packet rows, dealt round robin over the DRAM banks. The plan lays out the
// compact slots and the push writes them, so both take the shape from here.
struct PacketShape {
    uint32_t pages_per_packet;
    uint32_t banks;
    uint32_t unit() const { return pages_per_packet * banks; }  // owner runs are padded to whole units
};
PacketShape packet_shape(tt::tt_metal::distributed::MeshDevice* mesh_device, uint32_t page_bytes);

// Sizes of a plan table for a cache of max_rows rows in blocks of stripe_chunk (word layout:
// ::sparse_push::plan_table).
struct PlanTableLayout {
    static constexpr uint32_t kBitmapOff = ::sparse_push::plan_table::kBitmapOff;
    static constexpr uint32_t kPageWords = ::sparse_push::plan_table::kPageWords;

    uint32_t bitmap_words = 0;
    uint32_t num_blocks = 0;
    uint32_t pages = 0;        // per chip
    uint32_t image_words = 0;  // the plan op's L1 image of the table: whole pages

    static PlanTableLayout make(uint32_t max_rows, uint32_t stripe_chunk) {
        PlanTableLayout l;
        l.bitmap_words = max_rows / ::sparse_push::plan_table::kRowsPerWord;
        l.num_blocks = max_rows / stripe_chunk;
        l.pages = (kBitmapOff + l.bitmap_words + kPageWords - 1) / kPageWords;
        l.image_words = l.pages * kPageWords;
        return l;
    }
};

// Shared by the plan and the push, which must be built with the same stripe_chunk and max_rows.
void validate_stripe_chunk(uint32_t stripe_chunk, uint32_t max_rows);

// The plan splits the indices over its cores in multiples of this, so each core's slice starts 64 B aligned.
constexpr uint32_t kPlanIdxAlign = 16;

// The plan spans the whole worker grid.
uint32_t plan_cores(tt::tt_metal::distributed::MeshDevice* mesh_device);

// The plan reduces the bitmap in kChunkRows-row chunks, at most kMaxChunksPerCore of them on each core.
bool plan_fits(tt::tt_metal::distributed::MeshDevice* mesh_device, uint32_t max_rows);

// The label of a tensor whose rows (dim 2) differ per chip, chips in row-major order: the plan's tables and the push's
// compact buffers.
tt::tt_metal::TensorTopology rows_over_every_chip(const tt::tt_metal::distributed::MeshShape& mesh_shape);

// Compact-buffer rows for up to distinct_rows needed rows of row pages of page_bytes: every distinct row plus at most
// one unit of padding per owner, or every owner's run full (an owner holds at most its share of the cache's blocks).
uint32_t output_rows(
    tt::tt_metal::distributed::MeshDevice* mesh_device,
    uint32_t distinct_rows,
    uint32_t max_rows,
    uint32_t stripe_chunk,
    uint32_t page_bytes);

}  // namespace ttnn::operations::experimental::sparse_kv_push
