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

// Per-core arguments: the shared schedule, reader multicast tuples, then fused-only data.
namespace indexer_rt {
namespace schedule {
enum : uint32_t { RowGroup, GroupStride, NumGroups, BandStart, NumBands, MaxBands, Count };
}
namespace reader {
constexpr uint32_t McastWidth = 8;
constexpr uint32_t KMcast = schedule::Count;
constexpr uint32_t QWMcast = KMcast + McastWidth;
constexpr uint32_t FusedRing = QWMcast + McastWidth;
constexpr uint32_t FusedRingWidth = 9;
constexpr uint32_t BandPermutation = FusedRing + FusedRingWidth;
}  // namespace reader
namespace compute {
constexpr uint32_t BandPermutation = schedule::Count;
}
namespace writer {
constexpr uint32_t BandPermutation = schedule::Count;
}
}  // namespace indexer_rt
