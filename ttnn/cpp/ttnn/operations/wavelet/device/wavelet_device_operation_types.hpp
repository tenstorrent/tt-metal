// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include "ttnn/operations/wavelet/wavelet_types.hpp"
#include "ttnn/tensor/tensor.hpp"

#include <array>
#include <cstdint>
#include <optional>
#include <tuple>

namespace ttnn::prim {

enum class WaveletBoundaryMode : uint8_t {
    Zero,
    Constant,
    Symmetric,
    Reflect,
    Periodic,
    Smooth,
    Antisymmetric,
    Antireflect,
};

enum class WaveletTransform : uint8_t { Lwt1D, Ilwt1D, Lwt2D, Ilwt2D };

}  // namespace ttnn::prim
