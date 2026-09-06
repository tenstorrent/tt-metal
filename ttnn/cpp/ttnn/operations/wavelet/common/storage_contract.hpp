// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <cstdint>
#include <limits>

namespace ttnn::operations::wavelet {

inline constexpr uint32_t kStickWidth = 32;
inline constexpr uint32_t kStickPageBytes = kStickWidth * sizeof(float);
inline constexpr uint32_t kMaxSignedDeviceIndex = static_cast<uint32_t>(std::numeric_limits<int32_t>::max());
inline constexpr uint32_t kMax2DLogicalExtent = kMaxSignedDeviceIndex / 2;

static_assert(kStickPageBytes == 128, "The 1D wavelet path requires 128-byte FP32 sticks");

}  // namespace ttnn::operations::wavelet
