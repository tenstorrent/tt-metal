// SPDX-FileCopyrightText: © 2024 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#include "reduce_scatter_minimal_async_op.hpp"
#include "ttnn/operations/experimental/ccl/composite_common.hpp"
#include "ttnn/operations/functions.hpp"
#include "ttnn/operations/math.hpp"
#include "ttnn/global_semaphore.hpp"

#include "ttnn/tensor/tensor_utils.hpp"

namespace ttnn {

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
        topology == ::ttnn::ccl::Topology::Ring || topology == ::ttnn::ccl::Topology::Linear,
        "topology must be Ring or Linear");
    TT_FATAL(
        page_size % input_tensor.buffer()->alignment() == 0,
        "reduce_scatter_minimal_async currently requires aligned pages");

    TT_FATAL(input_tensor.storage_type() == StorageType::DEVICE, "Operands to all_gather need to be on device!");
    TT_FATAL(
        input_tensor.buffer() != nullptr,
        "Operands to reduce_scatter_minimal_async need to be allocated in buffers on device!");
    TT_FATAL(num_links > 0, "Error, num_links should be more than 0 but has {}", num_links);

    TT_FATAL(input_tensor.storage_type() == StorageType::DEVICE, "input_tensor must be on device");
    TT_FATAL(input_tensor.buffer() != nullptr, "input_tensor must have a buffer");

    const auto& rank = input_tensor.logical_shape().rank();

    TT_FATAL(rank > 1, "reduce_scatter currently supports rank 2 tensors at minimum");
    TT_FATAL(dim < rank, "Invalid scatter dim {} for rank {} tensor", dim, rank);

    const uint32_t normalized_dim = std::get<0>(composite_common::normalize_dim_4d(dim, rank));
    const auto& input_shape = input_tensor.padded_shape();
    if (normalized_dim == 2 || normalized_dim == 3) {
        uint32_t tile_size = normalized_dim == 2 ? tt::constants::TILE_HEIGHT : tt::constants::TILE_WIDTH;
        TT_FATAL(
            (input_shape[dim] / tile_size) % ring_size == 0,
            "Error, The number of tiles at input tensor dimension {} should be divisible by ring_size but the number "
            "of tiles is {} and the ring_size is {}",
            dim,
            input_shape[dim] / tile_size,
            ring_size);
    } else {
        TT_FATAL(
            input_shape[dim] % ring_size == 0,
            "Error, input tensor dimension {} should be divisible by ring_size but is {} and the ring_size is {}",
            dim,
            input_shape[dim],
            ring_size);
    }

    TT_FATAL(
        input_tensor.memory_config().memory_layout() == TensorMemoryLayout::INTERLEAVED ||
            input_tensor.memory_config().memory_layout() == TensorMemoryLayout::WIDTH_SHARDED ||
            input_tensor.memory_config().memory_layout() == TensorMemoryLayout::BLOCK_SHARDED ||
            input_tensor.memory_config().memory_layout() == TensorMemoryLayout::HEIGHT_SHARDED,
        "Unsupported input tensor memory layout {}.",
        input_tensor.memory_config().memory_layout());

    if (input_tensor.memory_config().memory_layout() == TensorMemoryLayout::BLOCK_SHARDED) {
        TT_FATAL(
            input_tensor.memory_config().buffer_type() == BufferType::L1, "We don't support input DRAM block sharding");
    }

    if (optional_output_tensor.has_value()) {
        const auto& output_tensor = optional_output_tensor.value();

        TT_FATAL(
            output_tensor.memory_config().memory_layout() == TensorMemoryLayout::INTERLEAVED ||
                output_tensor.memory_config().memory_layout() == TensorMemoryLayout::WIDTH_SHARDED ||
                output_tensor.memory_config().memory_layout() == TensorMemoryLayout::BLOCK_SHARDED ||
                output_tensor.memory_config().memory_layout() == TensorMemoryLayout::HEIGHT_SHARDED,
            "Unsupported output tensor memory layout {}.",
            output_tensor.memory_config().memory_layout());

        TT_FATAL(
            output_tensor.storage_type() == StorageType::DEVICE,
            "Operands to reduce_scatter_minimal_async need to be on device!");
        TT_FATAL(
            output_tensor.layout() == input_tensor.layout(),
            "Error, Output tensor layout should be same as input tensor layout but has {}",
            output_tensor.layout());
        TT_FATAL(
            output_tensor.dtype() == input_tensor.dtype(),
            "Error, Output tensor dtype should be same as input tensor dtype but has {}",
            output_tensor.dtype());
        TT_FATAL(
            output_tensor.tensor_spec().page_config() == input_tensor.tensor_spec().page_config(),
            "Error, Output tensor page config should be same as input tensor page config but has {}",
            output_tensor.tensor_spec().page_config());
        TT_FATAL(
            output_tensor.memory_config() == memory_config,
            "Error, Output tensor memory config {} should be same as output_mem_config {}",
            output_tensor.memory_config(),
            memory_config);

        // check the output tensor size
        auto output_shape = output_tensor.padded_shape();
        TT_FATAL(
            output_shape.size() == input_shape.size(),
            "Error, Output tensor shape should have same number of dimensions as input tensor but has {}",
            output_shape.size());
        for (size_t i = 0; i < input_shape.size(); ++i) {
            if (i == dim) {
                TT_FATAL(
                    output_shape[i] == input_shape[i] / ring_size,
                    "Error, Output tensor shape at dimension {} should be {} but has {}",
                    i,
                    input_shape[i] / ring_size,
                    output_shape[i]);
            } else {
                TT_FATAL(
                    output_shape[i] == input_shape[i],
                    "Error, Output tensor shape at dimension {} should be {} but has {}",
                    i,
                    input_shape[i],
                    output_shape[i]);
            }
        }

        // Don't support DRAM block sharding
        if (output_tensor.memory_config().memory_layout() == TensorMemoryLayout::BLOCK_SHARDED) {
            TT_FATAL(
                output_tensor.memory_config().buffer_type() == BufferType::L1,
                "We don't support output DRAM block sharding");
        }
    }
}

}  // namespace ttnn
