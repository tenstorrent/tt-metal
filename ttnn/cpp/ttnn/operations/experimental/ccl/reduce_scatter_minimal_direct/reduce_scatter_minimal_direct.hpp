// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <optional>
#include <vector>

#include "ttnn/tensor/tensor.hpp"
#include "ttnn/types.hpp"
#include <tt-metalium/sub_device_types.hpp>
#include <tt-metalium/core_coord.hpp>

namespace ttnn::experimental {

// Direct (one-shot) reduce-scatter: good for latency bound, small shapes.
//
// persistent_buffers, when provided, must be exactly {output, staging} as produced by
// reduce_scatter_minimal_direct_create_persistent_buffers.
ttnn::Tensor reduce_scatter_minimal_direct(
    const ttnn::Tensor& input_tensor,
    int32_t dim,
    std::optional<uint32_t> cluster_axis = std::nullopt,
    std::optional<uint32_t> num_links = std::nullopt,
    const std::optional<ttnn::MemoryConfig>& memory_config = std::nullopt,
    const std::optional<std::vector<ttnn::Tensor>>& persistent_buffers = std::nullopt,
    std::optional<tt::tt_metal::SubDeviceId> sub_device_id = std::nullopt,
    const std::optional<CoreRangeSet>& sub_core_grid = std::nullopt);

// Whether this op can run a given case AT ALL -- structural constraints only (TILE layout, a scatter dim
// that splits into whole pages, and a single WRAPPING axis: a Ring on a 1D fabric or a Torus axis on a 2D
// one. On a mesh with both extents > 1 the wrapping axis must be named explicitly via cluster_axis).
bool reduce_scatter_minimal_direct_is_applicable(
    const ttnn::Tensor& input_tensor, int32_t dim, std::optional<uint32_t> cluster_axis = std::nullopt);

// Allocate the persistent buffer set {output, staging} sized to match a given input.
std::vector<ttnn::Tensor> reduce_scatter_minimal_direct_create_persistent_buffers(
    const ttnn::Tensor& input_tensor, int32_t dim, std::optional<uint32_t> cluster_axis = std::nullopt);

// Allocate ONLY the staging buffer (element [1] of the set above), for callers that own their output
// tensor already and just want the op's staging half. staging is the part that
// cannot be reproduced by hand -- it is an opaque chunk-paged UINT8 tensor whose page size follows the
// op's chunk granularity and whose placement (L1 height-sharded over the whole compute grid / L1
// interleaved / DRAM) is chosen from the shape. Pass the result through as
// persistent_buffers = {your_output, this}. `dim` and `cluster_axis` must match the op call.
ttnn::Tensor reduce_scatter_minimal_direct_create_staging_buffer(
    const ttnn::Tensor& input_tensor, int32_t dim, std::optional<uint32_t> cluster_axis = std::nullopt);

}  // namespace ttnn::experimental
