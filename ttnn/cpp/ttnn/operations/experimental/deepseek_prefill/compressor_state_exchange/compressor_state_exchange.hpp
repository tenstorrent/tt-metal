// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <cstdint>
#include <tuple>

#include "ttnn/operations/ccl/ccl_host_types.hpp"
#include "ttnn/tensor/tensor.hpp"

namespace ttnn::operations::experimental::deepseek_prefill::compressor_state_exchange {

// Compressor state exchange for CSA prefill.
//
// The CSA compressor pools a ratio-4 window whose first tokens overlap the tail of the previous
// window. When the sequence is sequence-parallel (SP) sharded, each rank's first window needs the
// boundary ("overlap") state produced by the rank before it. This op moves that state one hop along
// the SP axis (cluster_axis): rank 0 receives the caller-injected temporal/initial state, and every
// other rank receives its predecessor's local [1, 1, 64, head_dim] Blaze-compatible KV/score state.
// The other mesh axis is independent TP lanes, replicated. The local inputs are never modified --
// they remain the outgoing states consumed by the next SP rank, and on the final active rank the
// local state is byte-compatible with Blaze decode migration.
//
// Two fabric backends, selected at runtime from the active FabricConfig:
//   * FABRIC_1D: a direct point_to_point send from each sender coordinate to its successor, per TP
//     lane. Simple one-hop fan-out; the natural primitive when 1D fabric routing is available.
//   * FABRIC_2D: point_to_point is not used on the 2D-fabric path, so instead we all_gather the
//     per-rank states along the row dim (dim=2) across cluster_axis and run a small device-side
//     select kernel that copies each rank's predecessor slab out of the gathered buffer (rank 0
//     takes the injected state). The gathered layout is why states must be [1, 1, 64, head_dim]:
//     the selector reads a rank's slab as one contiguous tile block, which only holds for a single
//     leading [B, C] slice.
//
// Shift Blaze-compatible compressor states by one device along cluster_axis.
// Rank zero receives the injected temporal state; every other rank receives its
// predecessor's local state. The local inputs remain the outgoing states used
// by the next SP rank and, on the final active rank, by decode migration.
std::tuple<ttnn::Tensor, ttnn::Tensor> compressor_state_exchange(
    const ttnn::Tensor& local_kv_state,
    const ttnn::Tensor& local_score_state,
    const ttnn::Tensor& initial_kv_state,
    const ttnn::Tensor& initial_score_state,
    uint32_t cluster_axis = 0,
    ::ttnn::ccl::Topology topology = ::ttnn::ccl::Topology::Linear);

// Preserve active ranks' outgoing states and copy the last active rank's state
// to every trailing rank whose local slab contains no valid tokens.
ttnn::Tensor propagate_compressor_state(
    const ttnn::Tensor& local_state,
    uint32_t seq_len_actual,
    uint32_t local_seq_len,
    uint32_t cluster_axis = 0,
    ::ttnn::ccl::Topology topology = ::ttnn::ccl::Topology::Linear);

}  // namespace ttnn::operations::experimental::deepseek_prefill::compressor_state_exchange

namespace ttnn {
using operations::experimental::deepseek_prefill::compressor_state_exchange::compressor_state_exchange;
}  // namespace ttnn
