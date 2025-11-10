// SPDX-FileCopyrightText: © 2023 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include "ttnn/operations/core/core.hpp"

#include <utility>

#include "ttnn/operations/data_movement/move/move.hpp"
#include "ttnn/operations/data_movement/reshape_on_device/reshape.hpp"
#include "ttnn/operations/data_movement/reshape_view/reshape.hpp"
#include "ttnn/distributed/types.hpp"
#include "ttnn/operations/data_movement/sharded/sharded_to_interleaved/sharded_to_interleaved.hpp"
#include "ttnn/operations/data_movement/sharded/interleaved_to_sharded/interleaved_to_sharded.hpp"
#include "ttnn/tensor/tensor.hpp"
#include "ttnn/experimental/lazy/lazy_device_operation.hpp"

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
    const ttnn::Tensor& tensor,
    MeshDevice* mesh_device,
    const std::optional<MemoryConfig>& memory_config,
    std::optional<QueueId> queue_id) {
    auto mem_config = memory_config.value_or(ttnn::DRAM_MEMORY_CONFIG);

    auto to_device_operation = std::make_shared<ToDeviceOperation>(mesh_device, mem_config, queue_id);
    auto lazy_inputs = ttnn::experimental::lazy::make_lazy_device_operation_inputs<ToDeviceOperation>(tensor);
    return tt::tt_metal::Tensor::make_lazy_tensor(lazy_inputs, to_device_operation, tensor.tensor_spec());
}

ttnn::Tensor from_device(const ttnn::Tensor& tensor, bool blocking, std::optional<QueueId> queue_id) {
    return tensor.cpu(blocking, queue_id);
}

void deallocate(Tensor& tensor, bool force) { tensor.deallocate(force); }

Tensor reallocate(const Tensor& input_tensor, const std::optional<MemoryConfig>& memory_config) {
    return ttnn::move(input_tensor, memory_config);
}

}  // namespace ttnn::operations::core
