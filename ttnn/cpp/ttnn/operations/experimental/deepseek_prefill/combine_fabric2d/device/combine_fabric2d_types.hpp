// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <optional>

#include "ttnn/tensor/tensor.hpp"
#include "ttnn/types.hpp"
#include "ttnn/distributed/types.hpp"
#include <tt-metalium/core_coord.hpp>
#include <tt-metalium/experimental/fabric/fabric.hpp>

namespace ttnn::operations::experimental::deepseek_prefill::combine_fabric2d {

// Isolated fabric-transfer experiment op. No input tensors — the op reserves its own L1 scratch.
//
// Every fabric eth core of every device (num_links toward each axis neighbor) gets ONE worker core in
// its physical column, running a producer (writer RISC) and a receiver (reader RISC). Each link is
// full duplex for payload: the producer sends `num_tokens` chunks of `chunk_size_bytes` to the peer
// worker across the cable, while the receiver consumes the peer's chunks. The producer owns the eth
// channel's single fabric connection and also forwards the receiver's credit returns.
//
// `device` is FIRST so the framework's get_first_object_of_type<MeshDevice*>() over
// attribute_values() (tuple element 0) finds the mesh.
struct CombineFabric2dParams {
    ttnn::MeshDevice* device = nullptr;
    uint32_t num_links = 2;
    uint32_t num_tokens = 100;
    uint32_t chunk_size_bytes = 14336;  // 7168 bf16 elements = one token
    uint32_t num_slots = 32;            // ring depth; also the producer's initial write_up_to credit
    uint32_t axis = 0;                  // mesh axis along which the neighbors are chosen
    tt::tt_fabric::Topology topology = tt::tt_fabric::Topology::Mesh;

    static constexpr auto attribute_names =
        std::forward_as_tuple("device", "num_links", "num_tokens", "chunk_size_bytes", "num_slots", "axis", "topology");
    auto attribute_values() const {
        return std::forward_as_tuple(device, num_links, num_tokens, chunk_size_bytes, num_slots, axis, topology);
    }
};

struct CombineFabric2dInputs {};

}  // namespace ttnn::operations::experimental::deepseek_prefill::combine_fabric2d
