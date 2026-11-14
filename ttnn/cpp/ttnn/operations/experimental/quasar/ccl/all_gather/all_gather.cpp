// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include "all_gather.hpp"

#include "device/all_gather_device_operation.hpp"

namespace ttnn::operations::experimental::quasar {

Tensor all_gather(
    const Tensor& input_tensor,
    const std::string& topology,
    uint32_t num_workers_per_rank,
    uint32_t reader_threads,
    uint32_t writer_threads,
    uint32_t tiles_per_packet,
    uint32_t chunks_per_sync) {
    ttnn::prim::qsr::AllGatherTopology topo;
    if (topology == "direct") {
        topo = ttnn::prim::qsr::AllGatherTopology::DIRECT;
    } else if (topology == "ring") {
        topo = ttnn::prim::qsr::AllGatherTopology::RING;
    } else {
        TT_THROW("quasar.all_gather: topology must be 'direct' or 'ring', got '{}'", topology);
    }
    return ttnn::prim::qsr::all_gather(
        input_tensor,
        ttnn::prim::qsr::AllGatherOperationAttributes{
            .topology = topo,
            .num_workers_per_rank = num_workers_per_rank,
            .reader_threads = reader_threads,
            .writer_threads = writer_threads,
            .tiles_per_packet = tiles_per_packet,
            .chunks_per_sync = chunks_per_sync,
        });
}

}  // namespace ttnn::operations::experimental::quasar
