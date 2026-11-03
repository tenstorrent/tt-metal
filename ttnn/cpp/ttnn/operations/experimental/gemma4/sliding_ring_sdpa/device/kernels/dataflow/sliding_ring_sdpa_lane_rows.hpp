// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <cstdint>

#include "ttnn/operations/transformer/sdpa/device/kernels/dataflow/dataflow_common.hpp"

namespace gemma4_sliding {

// Lanes: Q and the output stack the lanes' rows, [1, NH, num_lanes * q_local_padded_Nt, width_t].
// Lane nb's Q chunk q_chunk of head nq, and the end of that lane's rows.
struct LaneQChunkRows {
    Slice slice;
    uint32_t end_seq_tile;
};
inline LaneQChunkRows lane_q_chunk_rows(
    uint32_t nb, uint32_t nq, uint32_t q_chunk, uint32_t Sq_chunk_t, uint32_t q_local_padded_Nt, uint32_t width_t) {
    const uint32_t row = nb * q_local_padded_Nt + q_chunk * Sq_chunk_t;
    return {Slice(0, nq, row, row + Sq_chunk_t, 0, width_t), (nb + 1) * q_local_padded_Nt};
}

}  // namespace gemma4_sliding
