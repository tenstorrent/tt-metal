// SPDX-FileCopyrightText: © 2024 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include "ttnn/distributed/api.hpp"
#include "ttnn/distributed/distributed_tensor.hpp"
#include <tt-metalium/assert.hpp>
#include "ttnn/distributed/distributed_tensor_config.hpp"
#include "ttnn/distributed/types.hpp"
#include "ttnn/tensor/xtensor/partition.hpp"

namespace ttnn::distributed {
namespace {

class ReplicateTensorToMesh : public TensorToMesh {
public:
    ReplicateTensorToMesh(size_t num_devices) : num_devices_(num_devices) {}

    std::vector<Tensor> map(const Tensor& tensor) const override {
        std::vector<Tensor> tensors;
        tensors.reserve(num_devices_);
        std::fill_n(std::back_inserter(tensors), num_devices_, tensor);
        return tensors;
    }

    tt::tt_metal::DistributedTensorConfig config() const override {
        return tt::tt_metal::DistributedTensorConfig{tt::tt_metal::ReplicateTensor{num_devices_}};
    }

private:
    size_t num_devices_ = 0;
};

class ShardTensorToMesh : public TensorToMesh {
public:
    ShardTensorToMesh(size_t num_devices, int dim) : num_devices_(num_devices), shard_dim_(dim) {}

    std::vector<Tensor> map(const Tensor& tensor) const override {
        return experimental::xtensor::chunk(tensor, num_devices_, shard_dim_);
    }

    tt::tt_metal::DistributedTensorConfig config() const override {
        return tt::tt_metal::DistributedTensorConfig{tt::tt_metal::ShardTensor{shard_dim_}};
    }

private:
    size_t num_devices_ = 0;
    int shard_dim_ = -1;
};

class ShardTensorTo2dMesh : public TensorToMesh {
public:
    ShardTensorTo2dMesh(size_t mesh_rows, size_t mesh_cols, const Shard2dConfig& config) :
        mesh_rows_(mesh_rows), mesh_cols_(mesh_cols), config_(config) {}

    std::vector<Tensor> map(const Tensor& tensor) const override {
        const auto [row_dim, col_dim] = config_;

        std::vector<Tensor> row_tensors;

        // Shard along rows
        if (!row_dim.has_value()) {
            row_tensors.reserve(mesh_rows_);
            for (int i = 0; i < mesh_rows_; ++i) {
                row_tensors.push_back(tensor);
            }
        } else {
            row_tensors = experimental::xtensor::chunk(tensor, mesh_rows_, *row_dim);
        }

        std::vector<Tensor> tensor_shards;
        tensor_shards.reserve(mesh_rows_ * mesh_cols_);
        // Shard along columns
        if (!col_dim.has_value()) {
            for (const auto& t : row_tensors) {
                for (int i = 0; i < mesh_cols_; ++i) {
                    tensor_shards.push_back(t);
                }
            }
        } else {
            for (const auto& t : row_tensors) {
                auto col_chunks = experimental::xtensor::chunk(t, mesh_cols_, *col_dim);
                tensor_shards.insert(tensor_shards.end(), col_chunks.begin(), col_chunks.end());
            }
        }

        TT_FATAL(
            static_cast<int>(tensor_shards.size()) == mesh_rows_ * mesh_cols_,
            "ShardTensorTo2dMesh: Sharding failed. Number of shards should match the product of the mesh "
            "dimensions. Size: {}, rows: {}, cols: {}",
            tensor_shards.size(),
            mesh_rows_,
            mesh_cols_);

        return tensor_shards;
    }

    tt::tt_metal::DistributedTensorConfig config() const override {
        return tt::tt_metal::DistributedTensorConfig{
            tt::tt_metal::ShardTensor2D{tt::tt_metal::ShardMesh{mesh_rows_, mesh_cols_}}};
    }

private:
    size_t mesh_rows_ = 0;
    size_t mesh_cols_ = 0;
    Shard2dConfig config_;
};

class ConcatMeshToTensor : public MeshToTensor {
public:
    ConcatMeshToTensor(int dim) : concat_dim_(dim) {}

    Tensor compose(const std::vector<Tensor>& tensors) const override {
        return experimental::xtensor::concat(tensors, concat_dim_);
    }

private:
    int concat_dim_ = -1;
};

class Concat2dMeshToTensor : public MeshToTensor {
public:
    Concat2dMeshToTensor(size_t mesh_rows, size_t mesh_cols, const Concat2dConfig& config) :
        mesh_rows_(mesh_rows), mesh_cols_(mesh_cols), config_(config) {}

    Tensor compose(const std::vector<Tensor>& tensors) const override {
        const auto [row_dim, col_dim] = config_;

        std::vector<Tensor> row_concatenated;
        row_concatenated.reserve(mesh_rows_);
        for (int i = 0; i < mesh_rows_; ++i) {
            auto row_start = tensors.begin() + i * mesh_cols_;
            auto row_end = row_start + mesh_cols_;
            std::vector<Tensor> row_tensors(row_start, row_end);
            row_concatenated.push_back(experimental::xtensor::concat(row_tensors, col_dim));
        }

        return experimental::xtensor::concat(row_concatenated, row_dim);
    }

private:
    size_t mesh_rows_ = 0;
    size_t mesh_cols_ = 0;
    Concat2dConfig config_;
};

}  // namespace

std::unique_ptr<TensorToMesh> replicate_tensor_to_mesh_mapper(MeshDevice& mesh_device) {
    return std::make_unique<ReplicateTensorToMesh>(mesh_device.num_devices());
}

std::unique_ptr<TensorToMesh> shard_tensor_to_mesh_mapper(MeshDevice& mesh_device, int dim) {
    return std::make_unique<ShardTensorToMesh>(mesh_device.num_devices(), dim);
}

std::unique_ptr<TensorToMesh> shard_tensor_to_2d_mesh_mapper(
    MeshDevice& mesh_device, const MeshShape& mesh_shape, const Shard2dConfig& config) {
    TT_FATAL(
        config.row_dim.has_value() || config.col_dim.has_value(),
        "Sharding a tensor to 2D mesh requires at least one dimension to shard");
    TT_FATAL(mesh_shape.dims() == 2, "Mesh shape is not 2D: {}", mesh_shape);
    TT_FATAL(mesh_device.shape().dims() == 2, "Mesh device is not configured as a 2D mesh: {}", mesh_device.shape());
    TT_FATAL(
        mesh_shape[0] <= mesh_device.shape()[0] &&  //
            mesh_shape[1] <= mesh_device.shape()[1],
        "Device mesh shape {} does not match the provided mesh shape ({}, {}).",
        mesh_device.shape(),
        mesh_shape[0],
        mesh_shape[1]);
    return std::make_unique<ShardTensorTo2dMesh>(mesh_shape[0], mesh_shape[1], config);
}

std::unique_ptr<MeshToTensor> concat_mesh_to_tensor_composer(int dim) {
    return std::make_unique<ConcatMeshToTensor>(dim);
}

std::unique_ptr<MeshToTensor> concat_2d_mesh_to_tensor_composer(MeshDevice& mesh_device, const Concat2dConfig& config) {
    TT_FATAL(
        config.row_dim != config.col_dim,
        "Dimensions in 'dims' must be different; got row_dim: {}, col_dim: {}",
        config.row_dim,
        config.col_dim);
    TT_FATAL(mesh_device.shape().dims() == 2, "Mesh device is not configured as a 2D mesh: {}", mesh_device.shape());
    return std::make_unique<Concat2dMeshToTensor>(mesh_device.shape()[0], mesh_device.shape()[1], config);
}

Tensor distribute_tensor(
    const Tensor& tensor, const TensorToMesh& mapper, std::optional<std::reference_wrapper<MeshDevice>> mesh_device) {
    TT_FATAL(
        tensor.storage_type() != tt::tt_metal::StorageType::MULTI_DEVICE &&
            tensor.storage_type() != tt::tt_metal::StorageType::MULTI_DEVICE_HOST,
        "TensorToMesh does not support multi-device or multi-device host tensors; got storage type: {}",
        tensor.storage_type());
    std::vector<Tensor> tensors = mapper.map(tensor);
    Tensor output = aggregate_as_tensor(tensors, mapper.config());
    if (mesh_device.has_value()) {
        return output.to_device(&(mesh_device->get()), output.memory_config());
    }
    return output;
}

Tensor aggregate_tensor(const Tensor& tensor, const MeshToTensor& composer) {
    return is_multi_device_tensor(tensor) ? composer.compose(get_tensors_from_multi_device_storage(tensor))
                                          : composer.compose({tensor});
}

}  // namespace ttnn::distributed
