// SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#include "reduce_scatter_validate_utils.hpp"

#include <tt-metalium/buffer.hpp>
#include <tt-metalium/constants.hpp>
#include "ttnn/operations/experimental/ccl/composite_common.hpp"

using namespace tt::tt_metal;

namespace ttnn::experimental::ccl {

void reduce_scatter_common_validates(
    const ttnn::Tensor& input_tensor,
    ttnn::ccl::Topology topology,
    uint32_t dim,
    uint32_t num_links,
    uint32_t ring_size,
    const ttnn::MemoryConfig& memory_config,
    const std::optional<ttnn::Tensor>& optional_output_tensor) {
    const auto page_size = input_tensor.buffer()->page_size();
    TT_FATAL(
        topology == ttnn::ccl::Topology::Ring || topology == ttnn::ccl::Topology::Linear,
        "Topology must be either Ring or Linear");
    TT_FATAL(page_size % input_tensor.buffer()->alignment() == 0, "reduce_scatter currently requires aligned pages");

    TT_FATAL(input_tensor.storage_type() == StorageType::DEVICE, "Input tensor must be on device");
    TT_FATAL(input_tensor.buffer() != nullptr, "Input tensor must be allocated in buffers on device");
    TT_FATAL(num_links > 0, "num_links must be greater than 0");

    const auto& rank = input_tensor.logical_shape().rank();
    TT_FATAL(rank > 1, "reduce_scatter currently supports rank 2 tensors at minimum");
    TT_FATAL(dim < rank, "Invalid scatter dim {} for rank {} tensor", dim, rank);

    const uint32_t normalized_dim = std::get<0>(composite_common::normalize_dim_4d(dim, rank));
    const auto& input_shape = input_tensor.padded_shape();

    if (normalized_dim == 2 || normalized_dim == 3) {
        uint32_t tile_size = normalized_dim == 2 ? tt::constants::TILE_HEIGHT : tt::constants::TILE_WIDTH;
        TT_FATAL(
            (input_shape[dim] / tile_size) % ring_size == 0,
            "Number of tiles at dimension {} must be divisible by ring_size",
            dim);
    } else {
        TT_FATAL(input_shape[dim] % ring_size == 0, "Dimension {} must be divisible by ring_size", dim);
    }

    TT_FATAL(
        input_tensor.memory_config().memory_layout() == TensorMemoryLayout::INTERLEAVED ||
            input_tensor.memory_config().memory_layout() == TensorMemoryLayout::WIDTH_SHARDED ||
            input_tensor.memory_config().memory_layout() == TensorMemoryLayout::BLOCK_SHARDED ||
            input_tensor.memory_config().memory_layout() == TensorMemoryLayout::HEIGHT_SHARDED,
        "Unsupported input tensor memory layout");

    if (input_tensor.memory_config().memory_layout() == TensorMemoryLayout::BLOCK_SHARDED) {
        TT_FATAL(
            input_tensor.memory_config().buffer_type() == BufferType::L1,
            "DRAM block sharding not supported for input");
    }

    if (optional_output_tensor.has_value()) {
        const auto& output_tensor = optional_output_tensor.value();

        TT_FATAL(
            output_tensor.memory_config().memory_layout() == TensorMemoryLayout::INTERLEAVED ||
                output_tensor.memory_config().memory_layout() == TensorMemoryLayout::WIDTH_SHARDED ||
                output_tensor.memory_config().memory_layout() == TensorMemoryLayout::BLOCK_SHARDED ||
                output_tensor.memory_config().memory_layout() == TensorMemoryLayout::HEIGHT_SHARDED,
            "Unsupported output tensor memory layout");

        TT_FATAL(output_tensor.storage_type() == StorageType::DEVICE, "Output tensor must be on device");
        TT_FATAL(
            output_tensor.layout() == input_tensor.layout(), "Output tensor layout must match input tensor layout");
        TT_FATAL(output_tensor.dtype() == input_tensor.dtype(), "Output tensor dtype must match input tensor dtype");
        TT_FATAL(
            output_tensor.tensor_spec().page_config() == input_tensor.tensor_spec().page_config(),
            "Output tensor page config must match input tensor page config");
        TT_FATAL(
            output_tensor.memory_config() == memory_config,
            "Output tensor memory config must match provided memory_config");

        auto output_shape = output_tensor.padded_shape();
        TT_FATAL(
            output_shape.size() == input_shape.size(),
            "Output tensor must have same number of dimensions as input tensor");

        for (size_t i = 0; i < input_shape.size(); ++i) {
            if (i == dim) {
                TT_FATAL(
                    output_shape[i] == input_shape[i] / ring_size,
                    "Output tensor dimension {} must be input dimension / ring_size",
                    i);
            } else {
                TT_FATAL(output_shape[i] == input_shape[i], "Output tensor dimension {} must match input dimension", i);
            }
        }

        if (output_tensor.memory_config().memory_layout() == TensorMemoryLayout::BLOCK_SHARDED) {
            TT_FATAL(
                output_tensor.memory_config().buffer_type() == BufferType::L1,
                "DRAM block sharding not supported for output");
        }
    }
}

}  // namespace ttnn::experimental::ccl
