// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <cstdint>

#include "ttnn/operations/wavelet/common/storage_contract.hpp"

namespace ttnn::operations::wavelet::device_protocol {

constexpr uint32_t kStepCoeffCapacity = 17;
constexpr uint32_t kStickBytes = kStickPageBytes;
constexpr uint32_t kLwtCacheStickCount = 4;

constexpr uint32_t kLwtRowsPerGroup = 32;
constexpr uint32_t kLwtOutputBlocksPerRow = 3;
constexpr uint32_t kLwtHalfStickElements = 16;
constexpr uint32_t kLwtHalfStickBytes = kLwtHalfStickElements * sizeof(float);
constexpr uint32_t kLwtNarrowTileElements = kLwtRowsPerGroup * kLwtHalfStickElements;
constexpr uint32_t kLwtNarrowTileBytes = kLwtNarrowTileElements * sizeof(float);
constexpr uint32_t kLwtGroupOutputElements = kLwtRowsPerGroup * kLwtOutputBlocksPerRow * kLwtHalfStickElements;
constexpr uint32_t kIlwtGroupOutputElements = 2 * kLwtGroupOutputElements;

constexpr uint32_t kRouteConfigWordCount = 16;
constexpr uint32_t kRouteConfigPageBytes = kRouteConfigWordCount * sizeof(uint32_t);

enum RouteConfigWord : uint32_t {
    kRouteType = 0,
    kRouteSourceAddr = 1,
    kRouteSourceLength = 2,
    kRouteBaseAddr = 3,
    kRouteBaseLength = 4,
    kRouteOutputAddr = 5,
    kRouteOutputLength = 6,
    kRouteSourceOffset = 7,
    kRouteBaseOffset = 8,
    kRouteSourceLeftPad = 9,
    kRouteOutputOffset = 10,
    kRouteGroupCount = 11,
    kRouteFlags = 12,
};

constexpr uint32_t kRouteFlagFinalDram = 1U << 0;
constexpr uint32_t kRouteFlagIlwtFinalInterleave = 1U << 1;
constexpr uint32_t kRouteFlagFinalEven = 1U << 2;
constexpr uint32_t kRouteFlagFinalOdd = 1U << 3;
constexpr uint32_t kRouteFlagSourceTileMirror = 1U << 4;
constexpr uint32_t kRouteFlagBaseTileMirror = 1U << 5;
constexpr uint32_t kRouteFlagOutputTileMirror = 1U << 6;

constexpr uint32_t kLwtChunkConfigWordCount = 16;
constexpr uint32_t kLwtChunkConfigPageBytes = kLwtChunkConfigWordCount * sizeof(uint32_t);
static_assert(
    kLwtChunkConfigPageBytes == kRouteConfigPageBytes,
    "The shared 1D config-page loader requires identical chunk and route page sizes");

enum LwtChunkConfigWord : uint32_t {
    kLwtInitialEvenBegin = 0,
    kLwtInitialEvenLength = 1,
    kLwtInitialOddBegin = 2,
    kLwtInitialOddLength = 3,
    kIlwtApproximationBegin = 0,
    kIlwtApproximationLength = 1,
    kIlwtDetailBegin = 2,
    kIlwtDetailLength = 3,
    kIlwtFinalEvenAddr = 4,
    kIlwtFinalEvenStorageLength = 5,
    kIlwtFinalEvenOffset = 6,
    kIlwtFinalEvenBegin = 7,
    kIlwtFinalOddAddr = 8,
    kIlwtFinalOddStorageLength = 9,
    kIlwtFinalOddOffset = 10,
    kIlwtFinalOddBegin = 11,
    kIlwtOutputBegin = 12,
    kIlwtOutputLength = 13,
};

}  // namespace ttnn::operations::wavelet::device_protocol
