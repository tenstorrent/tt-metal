// SPDX-FileCopyrightText: © 2025 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include "mesh_partition.hpp"
#include "device/mesh_partition_device_operation.hpp"
#include "ttnn/operation.hpp"
#include "ttnn/operations/ccl/ccl_host_types.hpp"
#include "ttnn/operations/copy/typecast/typecast.hpp"
#include "ttnn/operations/core/core.hpp"
#include <tt-metalium/sub_device.hpp>
#include <tt-metalium/tensor/tensor_types.hpp>

namespace ttnn {

namespace {

// Whether tiled `input_tensor` must be split along `dim` in row-major order. The device operation's tiled slice, like
// ttnn::slice's, cuts only whole tiles, and only of tensors of rank 2 or more.
bool needs_row_major(const ttnn::Tensor& input_tensor, uint32_t dim, uint32_t cluster_size) {
    const auto& shape = input_tensor.logical_shape();
    const uint32_t rank = shape.rank();
    // An out-of-range or indivisible dim is left to the device operation's validation.
    if (input_tensor.layout() != Layout::TILE || dim >= rank || shape[dim] % cluster_size != 0) {
        return false;
    }
    if (rank == 1) {
        return true;
    }
    // Dims before the last two always hold whole tiles.
    if (dim + 2 < rank) {
        return false;
    }
    const auto tile = input_tensor.tensor_spec().tile();
    const uint32_t tile_extent = dim + 1 == rank ? tile.get_width() : tile.get_height();
    return (shape[dim] / cluster_size) % tile_extent != 0;
}

}  // namespace

ttnn::Tensor mesh_partition(
    const ttnn::Tensor& input_tensor,
    int32_t dim,
    std::optional<uint32_t> cluster_axis,
    const std::optional<ttnn::MemoryConfig>& memory_config) {
    const uint32_t cluster_size = ttnn::operations::ccl::detail::get_cluster_axis_size(input_tensor, cluster_axis);
    if (cluster_size == 1) {
        return input_tensor;
    }

    const auto output_memory_config = memory_config.value_or(input_tensor.memory_config());
    const uint32_t rank = input_tensor.logical_shape().rank();
    const uint32_t normalized_dim = dim < 0 ? static_cast<uint32_t>(rank + dim) : static_cast<uint32_t>(dim);
    if (!needs_row_major(input_tensor, normalized_dim, cluster_size)) {
        return ttnn::prim::mesh_partition(input_tensor, dim, cluster_axis, output_memory_config);
    }

    // Split in row-major order and tilize again, as ttnn::slice does. Block floats exist only in tiles, so they make
    // the trip as bfloat16.
    const auto dtype = input_tensor.dtype();
    const bool block_float = tt::tt_metal::is_block_float(dtype);
    auto tensor = block_float ? ttnn::typecast(input_tensor, DataType::BFLOAT16) : input_tensor;
    tensor =
        ttnn::prim::mesh_partition(ttnn::to_layout(tensor, Layout::ROW_MAJOR), dim, cluster_axis, output_memory_config);
    tensor = ttnn::to_layout(tensor, Layout::TILE, std::nullopt, output_memory_config);
    return block_float ? ttnn::typecast(tensor, dtype, output_memory_config) : tensor;
}

}  // namespace ttnn
