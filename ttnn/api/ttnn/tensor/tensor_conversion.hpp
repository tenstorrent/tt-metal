#pragma once

#include <ttnn/tensor/tensor.hpp>
#include <ttnn/distributed/distributed_tensor.hpp>

namespace tt {

namespace tt_metal {

Tensor create_device_tensor_from_host_data(
    const TensorSpec& tensor_spec,
    const HostBuffer& host_buffer,
    ttnn::distributed::MeshDevice* device,
    ttnn::QueueId cq_id,
    float pad_value,
    const ttnn::distributed::TensorToMesh* mesh_mapper);
}
}  // namespace tt
