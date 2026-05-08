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

namespace ttnn::tensor_impl {

// ===============================================================================================================================================
//                                                              High Level APIs
// ===============================================================================================================================================

// ======================================================================================
//                                  .view()
// ======================================================================================

tt::tt_metal::HostTensor view(
    const tt::tt_metal::HostTensor& tensor,
    const tt::tt_metal::Shape& new_logical_shape,
    const tt::tt_metal::Shape& new_padded_shape);

// ======================================================================================
//                                         Print
// ======================================================================================

std::ostream& operator<<(std::ostream& os, const tt::tt_metal::DataType& dtype);

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

std::string to_string(const ttnn::Tensor& tensor);

ttnn::Tensor extract_shard(const ttnn::Tensor& tensor, const uint32_t& core_id);

}  // namespace ttnn::tensor_impl

// Compatibility bridges - ttnn tensor infrastructure has moved to the ttnn namespace.
namespace tt::tt_metal::tensor_impl {

using TensorPrintProfile
    [[deprecated("use ttnn::tensor_impl::TensorPrintProfile instead. This alias may be removed after Jun 2026.")]] =
        ttnn::tensor_impl::TensorPrintProfile;
using SciMode [[deprecated("use ttnn::tensor_impl::SciMode instead. This alias may be removed after Jun 2026.")]] =
    ttnn::tensor_impl::SciMode;
using PrintOptions
    [[deprecated("use ttnn::tensor_impl::PrintOptions instead. This alias may be removed after Jun 2026.")]] =
        ttnn::tensor_impl::PrintOptions;

[[deprecated("use ttnn::tensor_impl::TTNN_PRINT_OPTIONS instead. This alias may be removed after Jun 2026.")]]
inline ttnn::tensor_impl::PrintOptions& TTNN_PRINT_OPTIONS = ttnn::tensor_impl::TTNN_PRINT_OPTIONS;

template <int = 0>
[[deprecated("use ttnn::tensor_impl::view instead. This alias may be removed after Jun 2026.")]]
inline tt::tt_metal::HostTensor view(
    const tt::tt_metal::HostTensor& tensor,
    const tt::tt_metal::Shape& new_logical_shape,
    const tt::tt_metal::Shape& new_padded_shape) {
    return ttnn::tensor_impl::view(tensor, new_logical_shape, new_padded_shape);
}

template <int = 0>
[[deprecated("use ttnn::tensor_impl::operator<< instead. This alias may be removed after Jun 2026.")]]
inline std::ostream& operator<<(std::ostream& os, const tt::tt_metal::DataType& dtype) {
    return ttnn::tensor_impl::operator<<(os, dtype);
}

template <int = 0>
[[deprecated("use ttnn::tensor_impl::to_string instead. This alias may be removed after Jun 2026.")]]
inline std::string to_string(const ttnn::Tensor& tensor) {
    return ttnn::tensor_impl::to_string(tensor);
}

template <int = 0>
[[deprecated("use ttnn::tensor_impl::extract_shard instead. This alias may be removed after Jun 2026.")]]
inline ttnn::Tensor extract_shard(const ttnn::Tensor& tensor, const uint32_t& core_id) {
    return ttnn::tensor_impl::extract_shard(tensor, core_id);
}

}  // namespace tt::tt_metal::tensor_impl
