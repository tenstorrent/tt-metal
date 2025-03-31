// SPDX-FileCopyrightText: © 2025 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include "tensor_types_generated.h"
#include "ttnn/tensor/types.hpp"
#include "ttnn/tensor/tensor_spec.hpp"

namespace ttnn {

tt::tt_metal::BufferType from_flatbuffer(flatbuffer::BufferType type);
tt::tt_metal::TensorMemoryLayout from_flatbuffer(flatbuffer::TensorMemoryLayout layout);
tt::tt_metal::DataType from_flatbuffer(flatbuffer::DataType type);
tt::tt_metal::ShardOrientation from_flatbuffer(flatbuffer::ShardOrientation orientation);
tt::tt_metal::ShardMode from_flatbuffer(flatbuffer::ShardMode mode);
CoreCoord from_flatbuffer(const flatbuffer::CoreCoord* fb_coord);
CoreRange from_flatbuffer(const flatbuffer::CoreRange* fb_coord);
CoreRangeSet from_flatbuffer(const flatbuffer::CoreRangeSet* fb_coord);
tt::tt_metal::ShardSpec from_flatbuffer(const flatbuffer::ShardSpec* spec);
tt::tt_metal::MemoryConfig from_flatbuffer(const flatbuffer::MemoryConfig* config);
tt::tt_metal::TensorLayout from_flatbuffer(const flatbuffer::TensorLayout* layout);
tt::tt_metal::TensorSpec from_flatbuffer(const flatbuffer::TensorSpec* spec);

}  // namespace ttnn
