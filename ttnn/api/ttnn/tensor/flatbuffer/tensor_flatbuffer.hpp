// SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <tt_stl/span.hpp>

#include <tt-metalium/tensor/tensor_generated.h>
#include <tt-metalium/tensor/types.hpp>
#include <tt-metalium/tensor/tensor_spec.hpp>
#include <tt-metalium/tensor/tensor.hpp>

namespace tt::tt_metal {

// Converts FlatBuffer tensor to Tensor object, using inline file storage to offset into `tensor_data`.
// The data is provided in `tensor_data` and `memory_pin` to load the tensor data lazily.
//
// Only inline file storage (data stored in same file) is currently supported.
Tensor from_flatbuffer(
    const flatbuffer::Tensor* fb_tensor,
    tt::stl::Span<std::byte> tensor_data,
    const tt::tt_metal::MemoryPin& memory_pin);

// Converts Tensor object to FlatBuffer representation, writing the serialized flatbuffer object to `builder` and
// recording tensor buffers that need to be serialized in-order to `buffers` vector. Replicated buffers are
// deduplicated, so that the number of copies that need to be written out from `buffers` is minimized.
//
// Only inline file storage (data stored in the same file) is currently supported.
flatbuffers::Offset<flatbuffer::Tensor> to_flatbuffer(
    const Tensor& tensor, flatbuffers::FlatBufferBuilder& builder, std::vector<tt::tt_metal::HostBuffer>& buffers);

}  // namespace tt::tt_metal
