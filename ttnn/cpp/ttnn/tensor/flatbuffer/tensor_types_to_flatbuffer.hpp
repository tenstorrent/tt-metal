// SPDX-FileCopyrightText: © 2025 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include "tensor_types_generated.h"

#include "ttnn/tensor/types.hpp"
#include "ttnn/tensor/tensor_spec.hpp"

namespace ttnn {

flatbuffers::Offset<flatbuffer::CoreCoord> to_flatbuffer(
    flatbuffers::FlatBufferBuilder& builder, const CoreCoord& core_coord);
flatbuffers::Offset<flatbuffer::CoreRange> to_flatbuffer(
    flatbuffers::FlatBufferBuilder& builder, const CoreRange& core_range);
flatbuffers::Offset<flatbuffer::CoreRangeSet> to_flatbuffer(
    flatbuffers::FlatBufferBuilder& builder, const CoreRangeSet& core_range_set);

flatbuffer::ShardOrientation to_flatbuffer(tt::tt_metal::ShardOrientation orientation);
flatbuffer::ShardMode to_flatbuffer(tt::tt_metal::ShardMode shard_mode);
flatbuffers::Offset<flatbuffer::ShardSpec> to_flatbuffer(
    const tt::tt_metal::ShardSpec& spec, flatbuffers::FlatBufferBuilder& builder);

flatbuffer::TensorMemoryLayout to_flatbuffer(tt::tt_metal::TensorMemoryLayout layout);
flatbuffer::BufferType to_flatbuffer(tt::tt_metal::BufferType type);
flatbuffer::DataType to_flatbuffer(tt::tt_metal::DataType type);

flatbuffers::Offset<flatbuffer::MemoryConfig> to_flatbuffer(
    const tt::tt_metal::MemoryConfig& config, flatbuffers::FlatBufferBuilder& builder);
flatbuffers::Offset<flatbuffer::TensorLayout> to_flatbuffer(
    const tt::tt_metal::TensorLayout& layout, flatbuffers::FlatBufferBuilder& builder);
flatbuffers::Offset<flatbuffer::TensorSpec> to_flatbuffer(
    const tt::tt_metal::TensorSpec& spec, flatbuffers::FlatBufferBuilder& builder);

}  // namespace ttnn
