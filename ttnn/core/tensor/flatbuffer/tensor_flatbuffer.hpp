// SPDX-FileCopyrightText: © 2025 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <cstdint>
#include <vector>

#include <tt_stl/span.hpp>
#include <tt-metalium/mesh_coord.hpp>
#include <tt-metalium/experimental/distributed_tensor/topology/tensor_topology.hpp>

#include "tensor_generated.h"

#include "tensor/flatbuffer/tensor_file_layout.hpp"

#include "ttnn/tensor/types.hpp"
#include "ttnn/tensor/tensor_spec.hpp"
#include "ttnn/tensor/tensor.hpp"

namespace ttnn {

// Where one shard of a serialized tensor lives, relative to the start of the file's tensor data region.
struct TensorFileShard {
    tt::tt_metal::distributed::MeshCoordinate coord;
    uint64_t offset = 0;
    uint64_t size = 0;
};

// Everything a FlatBuffer tensor header says, without touching tensor data. Replicated shards may share an offset.
struct TensorFileDescription {
    tt::tt_metal::TensorSpec spec;
    tt::tt_metal::distributed::MeshShape mesh_shape;
    tt::tt_metal::TensorTopology topology;
    std::vector<TensorFileShard> shards;
};

TensorFileDescription describe_tensor_file(const ttnn::flatbuffer::Tensor* fb_tensor);

// Converts FlatBuffer tensor to Tensor object, using inline file storage to offset into `tensor_data`.
// The data is provided in `tensor_data` and `memory_pin` to load the tensor data lazily.
//
// Only inline file storage (data stored in same file) is currently supported.
Tensor from_flatbuffer(
    const ttnn::flatbuffer::Tensor* fb_tensor,
    ttsl::Span<std::byte> tensor_data,
    const tt::tt_metal::MemoryPin& memory_pin);

// Converts Tensor object to FlatBuffer representation, writing the serialized flatbuffer object to `builder` and
// recording tensor buffers that need to be serialized in-order to `buffers` vector, each with the offset it must
// be written at. Replicated buffers are deduplicated, so that the number of copies that need to be written out
// from `buffers` is minimized.
//
// Only inline file storage (data stored in the same file) is currently supported.
flatbuffers::Offset<ttnn::flatbuffer::Tensor> to_flatbuffer(
    const Tensor& tensor, flatbuffers::FlatBufferBuilder& builder, std::vector<SerializedTensorBuffer>& buffers);

}  // namespace ttnn
