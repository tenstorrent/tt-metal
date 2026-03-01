// SPDX-FileCopyrightText: © 2023 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

#pragma once
#include <cstdint>
#include <optional>

#include <tt-metalium/mesh_device.hpp>
#include <tt-metalium/tilize_utils.hpp>
#include <tt-metalium/experimental/tensor/tensor_apis.hpp>

#include "ttnn/tensor/tensor.hpp"
#include "ttnn/tensor/tensor_utils.hpp"
#include "ttnn/tensor/types.hpp"
#include "ttnn/tensor/layout/layout.hpp"

namespace tt::tt_metal::tensor_impl {

// ===============================================================================================================================================
// The following declarations have been moved to tt-metalium/experimental/tensor:
// - bfloat4_b, bfloat8_b structs (details/tensor_impl.hpp)
// - convert_layout_tile_to_row_major, encode_tensor_data, decode_tensor_data (details/tensor_impl.hpp)
// - allocate_device_buffer, allocate_host_buffer (details/tensor_impl.hpp)
// - dispatch template (details/tensor_impl.hpp)
// - to_host, copy_to_host, to_device, copy_to_device, to_layout (tensor_apis.hpp)
// - pad, unpad, pad_to_tile, unpad_from_tile, to_dtype (tensor_apis.hpp)
// ===============================================================================================================================================

// view() functions have been moved to tt-metalium/experimental/tensor/details/legacy_view.hpp

// ======================================================================================
//                                 Runtime Tensor Creation Functions
// ======================================================================================

// allocate_tensor_on_device is now MeshTensor::allocate_on_device in tt-metalium/experimental/tensor/mesh_tensor.hpp

// HostTensor factory functions (from_vector, from_span, from_borrowed_data, to_vector)
// are now static member functions of HostTensor defined in tt-metalium/experimental/tensor/host_tensor.hpp

// ======================================================================================
//                                  TTNN Tensor Only Functions
// ======================================================================================

// ======================================================================================
//                                         Print
// ======================================================================================

std::ostream& operator<<(std::ostream& os, const DataType& dtype);

enum class TensorPrintProfile {
    Empty,
    Short,
    Full,
};

enum class SciMode {
    Enable,
    Disable,
    Default,
};

struct PrintOptions {
    TensorPrintProfile profile = TensorPrintProfile::Short;
    SciMode sci_mode = SciMode::Default;
    int precision = 4;
};

extern PrintOptions TTNN_PRINT_OPTIONS;

std::string to_string(const Tensor& tensor);

Tensor extract_shard(const Tensor& tensor, const uint32_t& core_id);

}  // namespace tt::tt_metal::tensor_impl
