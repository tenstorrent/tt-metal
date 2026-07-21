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

std::string to_string(const Tensor& tensor);

Tensor extract_shard(const Tensor& tensor, const uint32_t& core_id);

}  // namespace ttnn::tensor_impl

namespace tt::tt_metal::tensor_impl {

using ttnn::tensor_impl::extract_shard;
using ttnn::tensor_impl::PrintOptions;
using ttnn::tensor_impl::SciMode;
using ttnn::tensor_impl::TensorPrintProfile;
using ttnn::tensor_impl::TTNN_PRINT_OPTIONS;
using ttnn::tensor_impl::operator<<;
using ttnn::tensor_impl::to_string;
using ttnn::tensor_impl::view;

}  // namespace tt::tt_metal::tensor_impl

namespace tt::tt_metal {

using ttnn::tensor_impl::to_string;

}  // namespace tt::tt_metal
