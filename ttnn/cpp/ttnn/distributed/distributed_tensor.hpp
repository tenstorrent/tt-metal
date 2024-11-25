// SPDX-FileCopyrightText: © 2024 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include "ttnn/tensor/tensor.hpp"
#include "ttnn/distributed/types.hpp"

namespace ttnn::distributed::api {

namespace tensor_composers {

class TensorToMesh {
public:
    virtual ~TensorToMesh() = default;
    virtual std::vector<Tensor> map(const Tensor& tensor) = 0;
    virtual DistributedTensorConfig config() const = 0;
};

class MeshToTensor {
public:
    virtual ~MeshToTensor() = default;
    virtual Tensor compose(std::vector<Tensor>& tensors) = 0;
};

}  // namespace tensor_composers

// Distributes a host tensor onto multi-device configuration according to the distributed_tensor_config.
Tensor distribute_tensor(const Tensor& tensor, MeshDevice& mesh_device, tensor_composers::TensorToMesh& mapper);

// Aggregates a multi-device tensor into a host tensor.
Tensor aggregate_tensor(const Tensor& tensor, tensor_composers::MeshToTensor& composer);

}  // namespace ttnn::distributed::api
