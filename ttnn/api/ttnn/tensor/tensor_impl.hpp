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

// ======================================================================================
//                                  .view()
//        These maybe replaced by dedicated view types, See: #38093
// ======================================================================================
HostTensor view(
    const HostTensor& tensor,
    const tt::tt_metal::Shape& new_logical_shape,
    const tt::tt_metal::Shape& new_padded_shape);

MeshTensor view(
    const MeshTensor& tensor,
    const tt::tt_metal::Shape& new_logical_shape,
    const tt::tt_metal::Shape& new_padded_shape);

// ======================================================================================
//                                 Runtime Tensor Creation Functions
// ======================================================================================

// Creations, these should be static factory functions of HostTensor and MeshTensor

tt::tt_metal::MeshTensor allocate_tensor_on_device(const TensorSpec& tensor_spec, distributed::MeshDevice* mesh_device);

// ======================================================================================
//                                  HostTensor Factory Functions
// ======================================================================================

namespace host_tensor {

// ======================================================================================
//                                  HostTensor from_xxx
// ======================================================================================

template <typename T>
HostTensor from_vector(std::vector<T>&& buffer, const TensorSpec& spec, T pad_value = 0);

template <typename T>
HostTensor from_span(ttsl::Span<const T> buffer, const TensorSpec& spec, T pad_value = 0);

template <typename T>
HostTensor from_borrowed_data(
    ttsl::Span<T> buffer, const Shape& shape, MemoryPin buffer_pin, const std::optional<Tile>& tile = std::nullopt);

// ======================================================================================
//                                  HostTensor to_vector()
// ======================================================================================

// Converts a HostTensor to a std::vector<T>.
// Elements in the vector will be stored in row-major order. The type of the requested vector has to match that of
// the Tensor; block float formats such as BFLOAT8_B and BFLOAT4_B require T equal float.
template <typename T>
std::vector<T> to_vector(const HostTensor& tensor);

}  // namespace host_tensor

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
