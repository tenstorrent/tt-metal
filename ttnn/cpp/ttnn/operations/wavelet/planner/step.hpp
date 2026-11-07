// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <cstdint>

namespace ttnn::operations::wavelet {

enum class StepType : uint8_t {
    kPredict = 0,
    kUpdate = 1,
    kScaleEven = 2,
    kScaleOdd = 3,
    kSwap = 4,
};

}  // namespace ttnn::operations::wavelet
