// SPDX-FileCopyrightText: © 2023 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#pragma once
#include <cstdint>
#include <optional>
#include <span>

#include <tt-metalium/mesh_device.hpp>
#include <tt-metalium/tilize_utils.hpp>
#include <tt-metalium/experimental/tensor/impl/tensor_impl.hpp>

#include "ttnn/tensor/tensor.hpp"
#include "ttnn/tensor/tensor_utils.hpp"
#include "ttnn/tensor/types.hpp"
#include "ttnn/tensor/layout/tensor_layout.hpp"

namespace tt::tt_metal::tensor_impl {

// ===============================================================================================================================================
//                                                              High Level APIs
// ===============================================================================================================================================

// ======================================================================================
//                                  .to_layout()
// ======================================================================================

HostTensor to_layout(const HostTensor& tensor, Layout target_layout);

// ======================================================================================
//                                  .view()
// ======================================================================================

HostTensor view(
    const HostTensor& tensor,
    const tt::tt_metal::Shape& new_logical_shape,
    const tt::tt_metal::Shape& new_padded_shape);

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

HostTensor to_dtype(const HostTensor& input_tensor, DataType dtype);

// ======================================================================================
//                                  HostTensor Factory Functions
// ======================================================================================
// These functions create HostTensor objects without device involvement.
// They are the underlying implementation for Tensor::from_xxx methods.

namespace host_tensor {

template <typename T>
HostTensor from_span(ttsl::Span<const T> buffer, const TensorSpec& spec, T pad_value = 0);

template <typename T>
HostTensor from_borrowed_data(
    ttsl::Span<T> buffer, const Shape& shape, MemoryPin pin, const std::optional<Tile>& tile = std::nullopt);

template <typename T>
HostTensor from_vector(const std::vector<T>& buffer, const TensorSpec& spec, T pad_value = 0);

template <typename T>
HostTensor from_vector(std::vector<T>&& buffer, const TensorSpec& spec, T pad_value = 0);

template <typename T>
std::vector<T> to_vector(const HostTensor& tensor);

}  // namespace host_tensor

}  // namespace tt::tt_metal::tensor_impl
