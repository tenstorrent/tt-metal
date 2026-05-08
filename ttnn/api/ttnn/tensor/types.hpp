// SPDX-FileCopyrightText: © 2023 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <tt-metalium/experimental/tensor/tensor_types.hpp>
#include "ttnn/tensor/shape/shape.hpp"

namespace ttnn {

enum class PyDType {
    FLOAT32,
    FLOAT64,
    FLOAT16,
    BFLOAT16,
    INT8,
    INT16,
    INT32,
    INT64,
    UINT8,
    UINT16,
    UINT32,
    UINT64,
    BOOL
};

enum class StorageType {
    HOST = 0,
    DEVICE = 1,
};

static constexpr std::size_t MAX_NUM_DIMENSIONS = 8;

using Array1D = std::array<uint32_t, 1>;
using Array2D = std::array<uint32_t, 2>;
using Array3D = std::array<uint32_t, 3>;
using Array4D = std::array<uint32_t, 4>;
using Array5D = std::array<uint32_t, 5>;
using Array6D = std::array<uint32_t, 6>;
using Array7D = std::array<uint32_t, 7>;
using Array8D = std::array<uint32_t, 8>;

}  // namespace ttnn

// Compatibility aliases - ttnn tensor infrastructure has moved to the ttnn namespace.
namespace tt::tt_metal {

[[deprecated("use ttnn::MAX_NUM_DIMENSIONS instead. This alias may be removed after Jun 2026.")]]
static constexpr std::size_t MAX_NUM_DIMENSIONS = ttnn::MAX_NUM_DIMENSIONS;

using StorageType [[deprecated("use ttnn::StorageType instead. This alias may be removed after Jun 2026.")]] =
    ttnn::StorageType;
using Array1D [[deprecated("use ttnn::Array1D instead. This alias may be removed after Jun 2026.")]] = ttnn::Array1D;
using Array2D [[deprecated("use ttnn::Array2D instead. This alias may be removed after Jun 2026.")]] = ttnn::Array2D;
using Array3D [[deprecated("use ttnn::Array3D instead. This alias may be removed after Jun 2026.")]] = ttnn::Array3D;
using Array4D [[deprecated("use ttnn::Array4D instead. This alias may be removed after Jun 2026.")]] = ttnn::Array4D;
using Array5D [[deprecated("use ttnn::Array5D instead. This alias may be removed after Jun 2026.")]] = ttnn::Array5D;
using Array6D [[deprecated("use ttnn::Array6D instead. This alias may be removed after Jun 2026.")]] = ttnn::Array6D;
using Array7D [[deprecated("use ttnn::Array7D instead. This alias may be removed after Jun 2026.")]] = ttnn::Array7D;
using Array8D [[deprecated("use ttnn::Array8D instead. This alias may be removed after Jun 2026.")]] = ttnn::Array8D;

}  // namespace tt::tt_metal
