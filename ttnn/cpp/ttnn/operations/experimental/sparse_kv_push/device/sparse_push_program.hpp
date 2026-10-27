// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

// One fabric channel per (link, direction), each with kSendersPerChannel sender cores (reader + writer) feeding one mux
// core that owns the router connection.

#pragma once

#include <cstdint>
#include <optional>

#include <tt-metalium/core_coord.hpp>
#include <tt-metalium/experimental/fabric/fabric_types.hpp>
#include <tt-metalium/kernel_types.hpp>
#include <tt-metalium/program.hpp>
#include <tt-metalium/program_descriptors.hpp>
#include <tt_stl/reflection.hpp>
#include "ttnn/distributed/types.hpp"
#include "ttnn/global_semaphore.hpp"
#include "ttnn/tensor/tensor.hpp"

namespace ttnn::operations::experimental::sparse_kv_push::sparse_push {

constexpr uint32_t kDirections = 4;  // E, W, N, S (the eth_chan_directions order get_neighbors returns)
constexpr uint32_t kSendersPerChannel = 2;

constexpr uint32_t cores_needed(uint32_t links) { return links * kDirections * (kSendersPerChannel + 1); }

// Rows travel in packets of pages_per_packet rows, dealt round robin over the DRAM banks. The plan lays out the
// compact slots and the push writes them, so both take the shape from here.
struct PacketShape {
    uint32_t pages_per_packet;
    uint32_t banks;
    uint32_t unit() const { return pages_per_packet * banks; }  // owner runs are padded to whole units
};
PacketShape packet_shape(tt::tt_metal::distributed::MeshDevice* mesh_device, uint32_t page_bytes);

struct PushConfig {
    const Tensor& input;
    const Tensor& output;
    const Tensor& table;
    const std::optional<Tensor>& batch_index_tensor;
    uint32_t stripe_chunk;
    uint32_t max_rows;
    uint32_t part;  // 0 all, 1 prefix, 2 remainder
    uint32_t num_links;
    const CoreRangeSet& core_grid;
    tt::tt_fabric::FabricConfig fabric_config;
    const std::optional<GlobalSemaphore>& ready_semaphore;
    const std::optional<GlobalSemaphore>& data_valid_semaphore;
    uint32_t batch_index;
    uint32_t batch_slot_num_layers;
    uint32_t batch_slot_layer_idx;
};

// Shared by the plan and the push, which must be built with the same stripe_chunk and max_rows.
void validate_stripe_chunk(uint32_t stripe_chunk, uint32_t max_rows);

// Everything the kernels rely on, on a program cache miss.
void validate_config(const PushConfig& cfg);
// What the program hash leaves out (the cache slot and the semaphores), on every dispatch.
void validate_runtime(const PushConfig& cfg);
ttsl::hash::hash_t program_hash(const PushConfig& cfg);

// Kernel handles of the push in the program built from the descriptor.
struct Kernels {
    tt::tt_metal::KernelHandle reader;
    tt::tt_metal::KernelHandle writer;
};

// Appends the push's circular buffers, semaphores and kernels for the chip at coord.
Kernels append(tt::tt_metal::ProgramDescriptor& desc, const PushConfig& cfg, const MeshCoordinate& coord);

// Re-applies everything the program hash leaves out: buffer and semaphore addresses and the cache slot.
void override_runtime_arguments(tt::tt_metal::Program& program, const Kernels& kernels, const PushConfig& cfg);

}  // namespace ttnn::operations::experimental::sparse_kv_push::sparse_push
