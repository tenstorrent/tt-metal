// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

// The shared expert's matmul-core kernels' helpers (shared_expert_compute, shared_expert_row_dataflow,
// shared_expert_column_dataflow - every engine kernel takes the SHARED_EXPERT_* defines: head_engine_program.cpp):
//   * global_row_tile: a chunk-local row tile's row tile of the call;
//   * (compute) matmul_block_init_at<fidelity> / matmul_block_at<fidelity>: every matmul of the head's compute kernels
//     names its fidelity at the call site.
#pragma once

#include <cstdint>
#if !defined(COMPILE_FOR_TRISC)
#include "api/dataflow/dataflow_api.h"
#endif

// The global row tile of chunk `chunk`'s chunk-local row tile local_row_tile. With row chunks over TP shards
// (row_chunks > 1 and shards > 1) a chunk is a block of every owner, so the owners' rows stay contiguous: the host
// defines SHARED_EXPERT_OWNER_CHUNK_ROW_TILES = chunk row tiles / shards and SHARED_EXPERT_OWNER_ROW_TILES =
// rows / 32 / shards; else chunk-major (SHARED_EXPERT_CHUNK_ROW_TILES, row tiles a chunk).
FORCE_INLINE uint32_t global_row_tile(uint32_t chunk, uint32_t local_row_tile) {
#ifdef SHARED_EXPERT_OWNER_CHUNK_ROW_TILES
    return (local_row_tile / SHARED_EXPERT_OWNER_CHUNK_ROW_TILES) * SHARED_EXPERT_OWNER_ROW_TILES +
           chunk * SHARED_EXPERT_OWNER_CHUNK_ROW_TILES + local_row_tile % SHARED_EXPERT_OWNER_CHUNK_ROW_TILES;
#else
    return chunk * SHARED_EXPERT_CHUNK_ROW_TILES + local_row_tile;
#endif
}

#ifdef COMPILE_FOR_TRISC
// (the shared expert's configured fidelity: MoEDtypes::shared_fidelity)
#include "ttnn/cpp/ttnn/operations/experimental/moe_block/common/kernels/matmul_fidelity.hpp"
#endif
