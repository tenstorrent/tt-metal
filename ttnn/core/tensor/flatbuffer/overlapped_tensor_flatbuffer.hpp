// SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <tt_stl/span.hpp>

#include "overlapped_tensor_generated.h"

#include "ttnn/tensor/overlapped_tensor.hpp"

namespace ttnn {

flatbuffers::Offset<flatbuffer::OverlappedTensors> overlapped_tensors_to_flatbuffer(
    const std::vector<tt::tt_metal::OverlappedTensorView>& views,
    flatbuffers::FlatBufferBuilder& builder,
    std::vector<tt::tt_metal::HostBuffer>& buffers);

std::vector<tt::tt_metal::OverlappedTensorView> overlapped_tensors_from_flatbuffer(
    const flatbuffer::OverlappedTensors* fb,
    tt::stl::Span<std::byte> tensor_data,
    const tt::tt_metal::MemoryPin& memory_pin);

}  // namespace ttnn
