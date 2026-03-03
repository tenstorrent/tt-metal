// SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include "tensor_spec_generated.h"

#include "ttnn/tensor/types.hpp"
#include "ttnn/tensor/tensor_spec.hpp"

namespace ttnn {

flatbuffers::Offset<flatbuffer::CoreRangeSet> to_flatbuffer(
    flatbuffers::FlatBufferBuilder& builder, const CoreRangeSet& core_range_set);
CoreRangeSet from_flatbuffer(const flatbuffer::CoreRangeSet* core_range_set);

flatbuffer::DataType to_flatbuffer(tt::tt_metal::DataType type);
tt::tt_metal::DataType from_flatbuffer(flatbuffer::DataType type);

flatbuffers::Offset<flatbuffer::MemoryConfig> to_flatbuffer(
    const tt::tt_metal::MemoryConfig& config, flatbuffers::FlatBufferBuilder& builder);
flatbuffers::Offset<flatbuffer::TensorSpec> to_flatbuffer(
    const tt::tt_metal::TensorSpec& spec, flatbuffers::FlatBufferBuilder& builder);

tt::tt_metal::MemoryConfig from_flatbuffer(const flatbuffer::MemoryConfig* config);
tt::tt_metal::TensorSpec from_flatbuffer(const flatbuffer::TensorSpec* spec);

}  // namespace ttnn
