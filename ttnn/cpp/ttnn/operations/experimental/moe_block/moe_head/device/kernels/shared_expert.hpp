// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

// The shared expert's matmul-core kernels' helpers (shared_expert_compute, shared_expert_row_dataflow,
// shared_expert_column_dataflow - every engine kernel takes the shared_expert_* named args: head_engine_program.cpp):
//   * global_row_tile: a chunk-local row tile's row tile of the call;
//   * (compute) matmul_block_init_at<fidelity> / matmul_block_at<fidelity>: every matmul of the head's compute kernels
//     names its fidelity at the call site.
#pragma once

#include <cstdint>
#include "api/compile_time_args.h"
#if !defined(COMPILE_FOR_TRISC)
#include "api/dataflow/dataflow_api.h"
#endif
#include "ttnn/cpp/ttnn/operations/experimental/moe_block/moe_head/device/kernels/head_named_args.hpp"

// The global row tile of chunk `chunk`'s chunk-local row tile local_row_tile. With row chunks over TP shards
// (row_chunks > 1 and shards > 1) a chunk is a block of every owner, so the owners' rows stay contiguous: the host
// names shared_expert_owner_chunk_row_tiles = chunk row tiles / shards and shared_expert_owner_row_tiles =
// rows / 32 / shards; else (shared_expert_owner_chunk_row_tiles 0) chunk-major (shared_expert_chunk_row_tiles, row
// tiles a chunk).
FORCE_INLINE uint32_t global_row_tile(uint32_t chunk, uint32_t local_row_tile) {
    namespace names = moe_head_named_args;
    constexpr uint32_t owner_chunk_row_tiles = get_named_compile_time_arg_val(names::kSharedExpertOwnerChunkRowTiles);
    constexpr uint32_t owner_row_tiles = get_named_compile_time_arg_val(names::kSharedExpertOwnerRowTiles);
    constexpr uint32_t chunk_row_tiles = get_named_compile_time_arg_val(names::kSharedExpertChunkRowTiles);
    constexpr uint32_t owner_chunk_divisor = owner_chunk_row_tiles ? owner_chunk_row_tiles : 1;  // (never divided by 0)
    if constexpr (owner_chunk_row_tiles > 0) {
        return (local_row_tile / owner_chunk_divisor) * owner_row_tiles + chunk * owner_chunk_row_tiles +
               local_row_tile % owner_chunk_divisor;
    } else {
        return chunk * chunk_row_tiles + local_row_tile;
    }
}

#ifdef COMPILE_FOR_TRISC
// (the shared expert's configured fidelity: MoEDtypes::shared_fidelity)
#include "ttnn/cpp/ttnn/operations/experimental/moe_block/common/kernels/matmul_fidelity.hpp"
#endif
