// SPDX-FileCopyrightText: © 2024 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include <algorithm>

#include "ttnn/distributed/api.hpp"
#include "ttnn/distributed/distributed_tensor.hpp"
#include "common/assert.hpp"
#include "ttnn/distributed/distributed_tensor_config.hpp"

namespace ttnn::distributed::api {
namespace {

class ReplicateTensorToMesh : public TensorToMesh {
public:
    ReplicateTensorToMesh(MeshDevice& mesh_device) : mesh_device_(mesh_device) {}

    std::vector<Tensor> map(const Tensor& tensor) override {
        std::vector<Tensor> tensors;
        tensors.reserve(mesh_device_.num_devices());
        std::fill_n(std::back_inserter(tensors), mesh_device_.num_devices(), tensor);
        return tensors;
    }

    DistributedTensorConfig config() const override {
        return DistributedTensorConfig{ReplicateTensor{mesh_device_.num_devices()}};
    }

private:
    MeshDevice& mesh_device_;
};

class ShardTensorToMesh : public TensorToMesh {
public:
    ShardTensorToMesh(MeshDevice& mesh_device, int shard_dim) : mesh_device_(mesh_device), shard_dim_(shard_dim) {}

    std::vector<Tensor> map(const Tensor& tensor) override {
        // TODO: implement this
        return {};
    }

    DistributedTensorConfig config() const override { return DistributedTensorConfig{ShardTensor{shard_dim_}}; }

private:
    MeshDevice& mesh_device_;
    int shard_dim_;
};

class Shard2dTensorToMesh : public TensorToMesh {
public:
    Shard2dTensorToMesh(MeshDevice& mesh_device, const MeshShape& mesh_shape, const Shard2dConfig& config) :
        mesh_device_(mesh_device), mesh_shape_(mesh_shape), config_(config) {}

    std::vector<Tensor> map(const Tensor& tensor) override {
        // TODO: implement this
        return {};
    }

    DistributedTensorConfig config() const override {
        return DistributedTensorConfig{ShardTensor2D(ShardMesh{
            .y = mesh_shape_.first,
            .x = mesh_shape_.second,
        })};
    }

private:
    MeshDevice& mesh_device_;
    MeshShape mesh_shape_;
    Shard2dConfig config_;
};

class ConcatMeshToTensor : public MeshToTensor {
public:
    ConcatMeshToTensor(int concat_dim) : concat_dim_(concat_dim) {}

    Tensor compose(const std::vector<Tensor>& tensors) override {
        // TODO: implement this
        return Tensor();
    }

private:
    int concat_dim_ = -1;
};

class ConcatMesh2dToTensor : public MeshToTensor {
public:
    ConcatMesh2dToTensor(const Concat2dConfig& config) : config_(config) {}

    Tensor compose(const std::vector<Tensor>& tensors) override {
        // TODO: implement this
        return Tensor();
    }

private:
    Concat2dConfig config_;
};

}  // namespace

std::unique_ptr<TensorToMesh> replicate_tensor_to_mesh_mapper(MeshDevice& mesh_device) {
    return std::make_unique<ReplicateTensorToMesh>(mesh_device);
}

std::unique_ptr<TensorToMesh> shard_tensor_to_mesh_mapper(MeshDevice& mesh_device, int shard_dim) {
    return std::make_unique<ShardTensorToMesh>(mesh_device, shard_dim);
}

std::unique_ptr<TensorToMesh> shard_tensor_2d_to_mesh_mapper(
    MeshDevice& mesh_device, const MeshShape& mesh_shape, const Shard2dConfig& config) {
    return std::make_unique<Shard2dTensorToMesh>(mesh_device, mesh_shape, config);
}

std::unique_ptr<MeshToTensor> concat_mesh_to_tensor_composer(int concat_dim) {
    return std::make_unique<ConcatMeshToTensor>(concat_dim);
}

std::unique_ptr<MeshToTensor> concat_mesh_2d_to_tensor_composer(const Concat2dConfig& config) {
    return std::make_unique<ConcatMesh2dToTensor>(config);
}

Tensor distribute_tensor(const Tensor& tensor, MeshDevice& mesh_device, TensorToMesh& mapper) {
    TT_ASSERT(tensor.storage_type() == StorageType::OWNED, "TensorToMesh only supports owned tensors");
    std::vector<Tensor> tensors = mapper.map(tensor);
    Tensor output = aggregate_as_tensor(tensors, mapper.config());
    return output.to(&mesh_device);
}

Tensor aggregate_tensor(const Tensor& tensor, MeshToTensor& composer) {
    TT_ASSERT(tensor.storage_type() == StorageType::MULTI_DEVICE, "MeshToTensor only supports multi device tensors");
    return composer.compose(get_tensors_from_multi_device_storage(tensor));
}
}  // namespace ttnn::distributed::api
