// SPDX-FileCopyrightText: © 2025 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include "tensor_types_generated.h"
#include "ttnn/tensor/types.hpp"
#include "ttnn/tensor/tensor_spec.hpp"

namespace ttnn {

BufferType from_flatbuffer(flatbuffer::BufferType type);
TensorMemoryLayout from_flatbuffer(flatbuffer::TensorMemoryLayout layout);
DataType from_flatbuffer(flatbuffer::DataType type);
ShardOrientation from_flatbuffer(flatbuffer::ShardOrientation orientation);
ShardMode from_flatbuffer(flatbuffer::ShardMode mode);
CoreCoord from_flatbuffer(const flatbuffer::CoreCoord* fb_coord);
CoreRange from_flatbuffer(const flatbuffer::CoreRange* fb_coord);
CoreRangeSet from_flatbuffer(const flatbuffer::CoreRangeSet* fb_coord);
ShardSpec from_flatbuffer(const flatbuffer::ShardSpec* spec);
MemoryConfig from_flatbuffer(const flatbuffer::MemoryConfig* config);
TensorLayout from_flatbuffer(const flatbuffer::TensorLayout* layout);
TensorSpec from_flatbuffer(const flatbuffer::TensorSpec* spec);

}  // namespace ttnn
