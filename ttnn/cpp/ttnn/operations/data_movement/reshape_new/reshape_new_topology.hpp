// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <tt-metalium/experimental/distributed_tensor/topology/tensor_topology.hpp>
#include <tt-metalium/shape.hpp>

namespace ttnn::operations::data_movement::reshape_new {

// Relabels a tensor's placements across a per-device reshape from `old_shape` to `new_shape` (#64322).
//
// A reshape keeps every element's row-major position within a device, so the device-to-device split on dim d
// survives exactly when some new dim d' starts at the same row-major offset: prod(old[:d]) == prod(new[:d']).
// Each Shard(d) maps to the first such d' whose extent is above 1, keeping a negative spelling negative. When
// there is none, the split no longer corresponds to any dim and the input label is kept unchanged.
tt::tt_metal::TensorTopology remap_topology(
    const tt::tt_metal::TensorTopology& topology,
    const tt::tt_metal::Shape& old_shape,
    const tt::tt_metal::Shape& new_shape);

}  // namespace ttnn::operations::data_movement::reshape_new
