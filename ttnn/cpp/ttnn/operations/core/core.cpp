// SPDX-FileCopyrightText: © 2023 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include "ttnn/operations/core/core.hpp"

#include <utility>

#include <tt-metalium/command_queue.hpp>
#include "ttnn/operations/data_movement/move/move.hpp"
#include "ttnn/operations/data_movement/reshape_on_device/reshape.hpp"
#include "ttnn/operations/data_movement/reshape_view/reshape.hpp"
#include "ttnn/distributed/types.hpp"
#include "ttnn/operations/data_movement/sharded/sharded_to_interleaved/sharded_to_interleaved.hpp"
#include "ttnn/operations/data_movement/sharded/interleaved_to_sharded/interleaved_to_sharded.hpp"
#include "ttnn/tensor/tensor.hpp"

namespace ttnn::operations::core {

ttnn::Tensor unsqueeze_to_4D(const ttnn::Tensor& tensor) {
    const auto rank = tensor.logical_shape().rank();
    if (rank == 4) {
        return tensor;
    }
    if (rank > 4) {
        TT_THROW("Tensor rank is greater than 4");
    }

    return ttnn::reshape(tensor, tensor.logical_shape().to_rank(4), tensor.padded_shape().to_rank(4));
}

ttnn::Tensor squeeze_from_4D(const ttnn::Tensor& tensor, const int rank) {
    if (tensor.logical_shape().rank() != 4) {
        TT_THROW("Tensor has to be of rank 4!");
    }
    if (rank < 1 or rank > 4) {
        TT_THROW("Cannot use squeeze_from_4D to set the tensor to the rank of {}!", rank);
    }

    if (rank == 4) {
        return tensor;
    }
    return ttnn::reshape(tensor, tensor.logical_shape().to_rank(rank), tensor.padded_shape().to_rank(rank));
}

ttnn::Tensor to_device(
    const ttnn::Tensor& tensor, IDevice* device, const std::optional<MemoryConfig>& memory_config, QueueId cq_id) {
    return tensor.to_device(device, memory_config.value_or(ttnn::DRAM_MEMORY_CONFIG), cq_id);
}

ttnn::Tensor to_device(
    const ttnn::Tensor& tensor,
    MeshDevice* mesh_device,
    const std::optional<MemoryConfig>& memory_config,
    QueueId cq_id) {
    auto mem_config = memory_config.value_or(ttnn::DRAM_MEMORY_CONFIG);
    return tensor.to_device(mesh_device, mem_config, cq_id);
}

ttnn::Tensor from_device(const ttnn::Tensor& tensor, bool blocking, QueueId cq_id) {
    return tensor.cpu(blocking, cq_id);
}

void deallocate(Tensor& tensor, bool force) { tensor.deallocate(force); }

Tensor reallocate(const Tensor& input_tensor, const std::optional<MemoryConfig>& memory_config) {
    return ttnn::move(input_tensor, memory_config);
}

}  // namespace ttnn::operations::core
