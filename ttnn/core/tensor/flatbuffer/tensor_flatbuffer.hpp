
// SPDX-FileCopyrightText: © 2025 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include "tensor_generated.h"

#include "ttnn/tensor/types.hpp"
#include "ttnn/tensor/tensor_spec.hpp"
#include "ttnn/tensor/tensor.hpp"

namespace ttnn {

Tensor flatbuffer_to_tensor(const ttnn::flatbuffer::Tensor* fb_tensor, const std::byte* data_region);

flatbuffers::Offset<ttnn::flatbuffer::Tensor> tensor_to_flatbuffer(
    const Tensor& tensor, flatbuffers::FlatBufferBuilder& builder);

}  // namespace ttnn
