// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <cstdint>
#include <vector>

#include "combine_fabric2d_placement.hpp"
#include "kernels/dataflow/combine_fabric2d_chunk.hpp"
#include "kernels/dataflow/combine_fabric2d_kernel_interface.hpp"
#include "ttnn/operations/experimental/deepseek_prefill/fabric2d_common/fabric2d_assignments.hpp"

namespace ttnn::operations::experimental::deepseek_prefill::combine_fabric2d {

// Who sends what to whom, on which stream and through which channel, is the same ring schedule
// dispatch_fabric2d runs; see fabric2d_assignments.hpp. Here a run is one local expert's output for one
// dispatch group member, and the schedule runs once per local expert, so the slack the own phase buys before
// the forwards is a pass, not a run -- interleaving forwards earlier would shrink it to a chunk or two.
using fabric2d::Assignment;
using fabric2d::forwarding_classes;
using fabric2d::generate_assignments;
using fabric2d::num_vcs;
using fabric2d::vc_after_hop;
using fabric2d::vc_for_own;
static_assert(::cmbf2d::VC_NONE == fabric2d::VC_NONE, "the host writes fabric2d::VC_NONE into the channel tables");

// In the order the upstream chip emits them into its region, which is how both sides agree on positions. The
// kernel sizes its expected final writes from ::cmbf2d::stream_chunk, the device twin of fabric2d::stream_chunk.
inline std::vector<::cmbf2d::ChunkDescriptor> forwarding_chunks(
    StreamId stream, uint32_t my_dg_index, uint32_t ring_extent, uint32_t num_links) {
    std::vector<::cmbf2d::ChunkDescriptor> chunks;
    for (const auto& c : fabric2d::forwarding_chunks(stream, my_dg_index, ring_extent, num_links)) {
        chunks.push_back(::cmbf2d::ChunkDescriptor{
            .origin_dg_index = c.origin_pos,
            .dst_dg_index = c.dst_pos,
            .split_idx = c.split_idx,
            .split_count = c.split_count});
    }
    return chunks;
}

}  // namespace ttnn::operations::experimental::deepseek_prefill::combine_fabric2d
