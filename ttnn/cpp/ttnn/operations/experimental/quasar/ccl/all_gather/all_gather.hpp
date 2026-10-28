// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <cstdint>
#include <string>

#include "ttnn/types.hpp"

namespace ttnn::operations::experimental::quasar {

// Intra-chip all-gather of a height- or width-sharded TILE tensor: rank r = the cluster holding shard r.
// Returns [num_ranks, ...input shape...] height-sharded over the same clusters, so every output shard is
// the whole gathered tensor. topology: "direct" (one kernel of reader_threads <= 6 DM threads per rank pulls
// every slice straight into the local shard; writer_threads and num_workers_per_rank are unused) or "ring"
// (unidirectional store-and-forward with per-thread ready credits; reader_threads == writer_threads).
Tensor all_gather(
    const Tensor& input_tensor,
    const std::string& topology = "direct",
    uint32_t num_workers_per_rank = 1,
    uint32_t reader_threads = 2,
    uint32_t writer_threads = 2,
    uint32_t tiles_per_packet = 4,
    uint32_t chunks_per_sync = 1);

}  // namespace ttnn::operations::experimental::quasar
