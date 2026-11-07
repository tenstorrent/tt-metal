// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <cstdint>

#include "ttnn/tensor/tensor.hpp"

namespace ttnn::prim::qsr {

// How the gathered slices travel between ranks. Both arms run on one chip over the NoC; a rank is the
// cluster that holds one shard of the input (plus optional extra worker clusters).
//   DIRECT: every rank pulls every slice straight from its owner's input shard.
//   RING:   unidirectional ring: each rank sends its own slice to the next rank's output shard, then
//           store-and-forwards the slices it receives (the all_gather_async protocol with the fabric
//           transport replaced by NoC writes + per-thread ready semaphores).
enum class AllGatherTopology : uint8_t { DIRECT = 0, RING = 1 };

struct AllGatherOperationAttributes {
    AllGatherTopology topology = AllGatherTopology::DIRECT;
    uint32_t num_workers_per_rank = 1;  // clusters per rank: the shard's home cluster + (K-1) helpers
    uint32_t reader_threads = 2;        // DM threads of the reader kernel (== writer_threads)
    uint32_t writer_threads = 2;        // DM threads of the writer kernel
    uint32_t tiles_per_packet = 4;      // tiles moved per reserve/push batch
    uint32_t chunks_per_sync = 1;       // RING: packets per ready-semaphore credit
};

struct AllGatherTensorArgs {
    Tensor input_tensor;
};

}  // namespace ttnn::prim::qsr
