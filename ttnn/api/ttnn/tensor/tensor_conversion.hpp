#pragma once

#include <ttnn/tensor/tensor.hpp>
#include <ttnn/distributed/distributed_tensor.hpp>

namespace tt {

namespace tt_metal {

enum class host_buffer_data_type {
    FLOAT32,
    FLOAT64,
    FLOAT16,
    BFLOAT16,
    INT8,
    INT16,
    INT32,
    INT64,
    UINT8,
    UINT16,
    UINT32,
    UINT64,
    BOOL
};

Tensor create_device_tensor_from_host_data(
    const TensorSpec& tensor_spec,
    const host_buffer_data_type& host_data_type,
    std::function<HostBuffer(DataType)> get_host_data,
    ttnn::distributed::MeshDevice* device,
    std::optional<ttnn::QueueId> cq_id,
    float pad_value,
    const ttnn::distributed::TensorToMesh* mesh_mapper);
}
}  // namespace tt
