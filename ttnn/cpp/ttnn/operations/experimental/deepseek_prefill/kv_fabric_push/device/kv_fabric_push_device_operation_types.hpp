// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <cstdint>
#include <vector>

#include "ttnn/tensor/tensor.hpp"

namespace ttnn::experimental::prim {

// One launch of the push: every chip of the mesh runs one core that takes its work for this launch (a header page,
// the destination table, then the runs) from its H2D socket and writes the runs' pages into the destination mesh's
// DRAM over TT-Fabric. Nothing a launch moves is in these attributes, so every launch of a request (and every request
// to the same destination chips) hits the program cache.
struct KvFabricPushParams {
    uint32_t dst_mesh_id = 0;
    std::vector<uint32_t> dst_chips;  // fabric chip id of each destination; the connection's route is to the first
    uint32_t dst_banks = 0;           // DRAM banks of a destination chip
    uint32_t max_dst_bytes = 0;       // L1 for the destination table (a multiple of socket_page_size)
    uint32_t socket_page_size = 0;
    uint32_t mesh_num_cols = 0;
    std::vector<uint32_t> socket_config_addrs;  // the socket's config buffer on each chip (row-major coordinate)
    std::vector<uint32_t> tilize;  // per source: 1 = ROW_MAJOR rows tilized into 32x32 bf16 tiles on the way
};

struct KvFabricPushInputs {
    // Every buffer a run reads, DRAM (interleaved or ND-sharded); a run names its source by index.
    std::vector<Tensor> sources;
};

}  // namespace ttnn::experimental::prim
