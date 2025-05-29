// SPDX-FileCopyrightText: © 2025 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include "tensor_generated.h"

#include "ttnn/tensor/types.hpp"
#include "ttnn/tensor/tensor_spec.hpp"
#include "ttnn/tensor/tensor.hpp"

namespace ttnn {

// Converts FlatBuffer tensor to Tensor object, using `data_region` as the source of tensor data.
// Only inline file storage (data stored in same file) is currently supported.
Tensor from_flatbuffer(const ttnn::flatbuffer::Tensor* fb_tensor, const std::byte* data_region);

// Converts Tensor object to FlatBuffer representation.
// Only inline file storage (data stored in same file) is currently supported.
flatbuffers::Offset<ttnn::flatbuffer::Tensor> to_flatbuffer(
    const Tensor& tensor, flatbuffers::FlatBufferBuilder& builder);

}  // namespace ttnn
