// SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#include "tensor/flatbuffer/overlapped_tensor_flatbuffer.hpp"
#include "tensor/flatbuffer/tensor_flatbuffer.hpp"
#include "tensor/flatbuffer/tensor_spec_flatbuffer.hpp"

#include <flatbuffers/flatbuffers.h>

#include "overlapped_tensor_generated.h"

#include <vector>
#include <cstdint>

namespace ttnn {

flatbuffers::Offset<flatbuffer::OverlappedTensors> overlapped_tensors_to_flatbuffer(
    const std::vector<tt::tt_metal::OverlappedTensorView>& views,
    flatbuffers::FlatBufferBuilder& builder,
    std::vector<tt::tt_metal::HostBuffer>& buffers) {
    TT_FATAL(!views.empty(), "Need at least one OverlappedTensorView to serialize");

    auto fused_offset = ttnn::to_flatbuffer(views[0].fused_tensor, builder, buffers);

    std::vector<flatbuffers::Offset<flatbuffer::OverlappedView>> view_offsets;
    view_offsets.reserve(views.size());
    for (const auto& v : views) {
        auto crs_offset = ttnn::to_flatbuffer(builder, v.core_range_set);
        view_offsets.push_back(flatbuffer::CreateOverlappedView(
            builder,
            v.tensor_shape[0],
            v.tensor_shape[1],
            v.shard_shape[0],
            v.shard_shape[1],
            crs_offset,
            ttnn::to_flatbuffer(v.dtype),
            v.tile_shape[0],
            v.tile_shape[1],
            v.byte_offset));
    }

    auto views_vec = builder.CreateVector(view_offsets);
    return flatbuffer::CreateOverlappedTensors(builder, fused_offset, views_vec);
}

std::vector<tt::tt_metal::OverlappedTensorView> overlapped_tensors_from_flatbuffer(
    const flatbuffer::OverlappedTensors* fb,
    tt::stl::Span<std::byte> tensor_data,
    const tt::tt_metal::MemoryPin& memory_pin) {
    TT_FATAL(fb != nullptr, "OverlappedTensors flatbuffer pointer must not be null");
    TT_FATAL(fb->fused_tensor() != nullptr, "fused_tensor is required");
    TT_FATAL(fb->views() != nullptr && fb->views()->size() > 0, "At least one view is required");

    auto fused = ttnn::from_flatbuffer(fb->fused_tensor(), tensor_data, memory_pin);

    std::vector<tt::tt_metal::OverlappedTensorView> result;
    result.reserve(fb->views()->size());
    for (const auto* fb_view : *fb->views()) {
        result.push_back(tt::tt_metal::OverlappedTensorView{
            .fused_tensor = fused,
            .tensor_shape = {fb_view->tensor_shape_h(), fb_view->tensor_shape_w()},
            .shard_shape = {fb_view->shard_shape_h(), fb_view->shard_shape_w()},
            .core_range_set = ttnn::from_flatbuffer(fb_view->core_range_set()),
            .dtype = ttnn::from_flatbuffer(fb_view->dtype()),
            .tile_shape = {fb_view->tile_h(), fb_view->tile_w()},
            .byte_offset = fb_view->byte_offset(),
        });
    }
    return result;
}

}  // namespace ttnn
