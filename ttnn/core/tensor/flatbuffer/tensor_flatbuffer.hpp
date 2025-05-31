// SPDX-FileCopyrightText: © 2025 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <tt_stl/span.hpp>

#include "tensor_generated.h"

#include "ttnn/tensor/types.hpp"
#include "ttnn/tensor/tensor_spec.hpp"
#include "ttnn/tensor/tensor.hpp"

namespace ttnn {

// Converts FlatBuffer tensor to Tensor object, using inline file storage to offset into `tensor_data`.
// Only inline file storage (data stored in same file) is currently supported.
Tensor from_flatbuffer(const ttnn::flatbuffer::Tensor* fb_tensor, tt::stl::Span<std::byte> tensor_data);

// Converts Tensor object to FlatBuffer representation.
// Only inline file storage (data stored in the same file) is currently supported.
flatbuffers::Offset<ttnn::flatbuffer::Tensor> to_flatbuffer(
    const Tensor& tensor, flatbuffers::FlatBufferBuilder& builder);

}  // namespace ttnn
