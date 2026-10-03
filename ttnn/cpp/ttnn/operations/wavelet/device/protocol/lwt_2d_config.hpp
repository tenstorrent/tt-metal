// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <cstdint>

namespace ttnn::operations::wavelet::device_protocol {

inline constexpr uint32_t kLwt2DTileHeight = 32;
inline constexpr uint32_t kLwt2DTileWidth = 32;
inline constexpr uint32_t kLwt2DFullTileElements = kLwt2DTileHeight * kLwt2DTileWidth;
inline constexpr uint32_t kLwt2DFullTileBytes = kLwt2DFullTileElements * sizeof(float);
inline constexpr uint32_t kLwt2DInitialPlaneCount = 4;
inline constexpr uint32_t kLwt2DPlaneCount = 5;
inline constexpr uint32_t kLwt2DBandCount = 4;
inline constexpr uint32_t kLwt2DWorkspaceCb = 10;
inline constexpr uint32_t kLwt2DSplitScratchTileRows = 5;
inline constexpr uint32_t kLwt2DSplitScratchTileColumns = 5;
inline constexpr uint32_t kLwt2DSplitScratchTileCount = kLwt2DSplitScratchTileRows * kLwt2DSplitScratchTileColumns;
inline constexpr uint32_t kLwt2DSplitScratchBytes = kLwt2DSplitScratchTileCount * kLwt2DFullTileBytes;
inline constexpr uint32_t kLwt2DSymmetricSplitScratchTileRows = 3;
inline constexpr uint32_t kLwt2DSymmetricSplitScratchTileColumns = 3;
inline constexpr uint32_t kLwt2DSymmetricSplitScratchTileCount =
    kLwt2DSymmetricSplitScratchTileRows * kLwt2DSymmetricSplitScratchTileColumns;
inline constexpr uint32_t kLwt2DInitialEe = 8;
inline constexpr uint32_t kLwt2DInitialEo = 12;
inline constexpr uint32_t kLwt2DInitialOe = 16;
inline constexpr uint32_t kLwt2DInitialOo = 20;
inline constexpr uint32_t kLwt2DRectYBegin = 0;
inline constexpr uint32_t kLwt2DRectYLength = 1;
inline constexpr uint32_t kLwt2DRectXBegin = 2;
inline constexpr uint32_t kLwt2DRectXLength = 3;
inline constexpr uint32_t kLwt2DChunkConfigWordCount = 32;
inline constexpr uint32_t kLwt2DChunkConfigPageBytes = kLwt2DChunkConfigWordCount * sizeof(uint32_t);

inline constexpr uint32_t kLwt2DRouteAxis = 0;
inline constexpr uint32_t kLwt2DRouteSourceSlot = 2;
inline constexpr uint32_t kLwt2DRouteBaseSlot = 3;
inline constexpr uint32_t kLwt2DRouteOutputSlot = 4;
inline constexpr uint32_t kLwt2DRouteSourceRect = 5;
inline constexpr uint32_t kLwt2DRouteBaseRect = 9;
inline constexpr uint32_t kLwt2DRouteOutputRect = 13;
inline constexpr uint32_t kLwt2DRouteFlags = 17;
inline constexpr uint32_t kLwt2DRouteConfigWordCount = 32;
inline constexpr uint32_t kLwt2DRouteConfigPageBytes = kLwt2DRouteConfigWordCount * sizeof(uint32_t);

inline constexpr uint32_t kLwt2DRouteFlagMetadataOnly = 1U << 0;
inline constexpr uint32_t kLwt2DRouteFlagScale = 1U << 1;

inline constexpr uint32_t kLwt2DBandFinalYBegin = 0;
inline constexpr uint32_t kLwt2DBandFinalYLength = 1;
inline constexpr uint32_t kLwt2DBandFinalXBegin = 2;
inline constexpr uint32_t kLwt2DBandFinalXLength = 3;
inline constexpr uint32_t kLwt2DBandLl = 4;
inline constexpr uint32_t kLwt2DBandLh = 9;
inline constexpr uint32_t kLwt2DBandHl = 14;
inline constexpr uint32_t kLwt2DBandHh = 19;
inline constexpr uint32_t kLwt2DBandSourceSlot = 0;
inline constexpr uint32_t kLwt2DBandSourceRect = 1;
inline constexpr uint32_t kLwt2DBandConfigWordCount = 32;
inline constexpr uint32_t kLwt2DBandConfigPageBytes = kLwt2DBandConfigWordCount * sizeof(uint32_t);

static_assert(kLwt2DChunkConfigPageBytes == 128);
static_assert(kLwt2DRouteConfigPageBytes == 128);
static_assert(kLwt2DBandConfigPageBytes == 128);
static_assert(kLwt2DChunkConfigPageBytes % 64 == 0);
static_assert(kLwt2DRouteConfigPageBytes % 64 == 0);
static_assert(kLwt2DBandConfigPageBytes % 64 == 0);
static_assert(kLwt2DInitialPlaneCount + 1 == kLwt2DPlaneCount);

}  // namespace ttnn::operations::wavelet::device_protocol
