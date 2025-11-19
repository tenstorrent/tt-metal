// SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <tt-metalium/tensor/tensor_spec_generated.h>
#include <tt-metalium/tensor/types.hpp>
#include <tt-metalium/tensor/tensor_spec.hpp>

namespace tt::tt_metal {

flatbuffers::Offset<flatbuffer::MemoryConfig> to_flatbuffer(
    const tt::tt_metal::MemoryConfig& config, flatbuffers::FlatBufferBuilder& builder);
flatbuffers::Offset<flatbuffer::TensorSpec> to_flatbuffer(
    const tt::tt_metal::TensorSpec& spec, flatbuffers::FlatBufferBuilder& builder);

tt::tt_metal::MemoryConfig from_flatbuffer(const flatbuffer::MemoryConfig* config);
tt::tt_metal::TensorSpec from_flatbuffer(const flatbuffer::TensorSpec* spec);

}  // namespace tt::tt_metal
