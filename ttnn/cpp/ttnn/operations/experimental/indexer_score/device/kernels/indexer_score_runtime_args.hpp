// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
// SPDX-License-Identifier: Apache-2.0
#pragma once

#include <cstdint>

// Common arguments shared by the single-chip and fused-ring factories and kernels.
namespace indexer_common {
namespace reader {
enum : uint32_t {
    Q,
    K,
    W,
    KLocal,
    BatchOffset,
    KvLength,
    LocalBatchOffset,
    ChunkMetadata,
    DeviceIndex,
    TpIndex,
    SlotMetadata,
    NumLayers,
    LayerIndex,
    // DRAM address of the 1-element real-token-end tensor (0 when uncapped). Last named slot, so the
    // shard-order / fused-ring tails that derive from Count shift with it automatically.
    ValidEnd,
    Count
};
}
namespace writer {
enum : uint32_t { Output, KvLength, ChunkStart, StraddleQ, StraddleJump, Count };
}
namespace compute {
enum : uint32_t { KvLength, ChunkStart, StraddleQ, StraddleJump, Count };
}
}  // namespace indexer_common

// The only per-core argument is the row-major core identity. Reader common
// arguments append ring metadata (fused only), then physical X/Y multicast axes.
namespace indexer_rt::reader {
constexpr uint32_t FusedRingWidth = 9;
}  // namespace indexer_rt::reader
