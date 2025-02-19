// SPDX-FileCopyrightText: © 2024 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include "tt-metalium/mesh_device.hpp"
#include "ttnn/tensor/tensor.hpp"
#include "ttnn/distributed/types.hpp"
#include "ttnn/distributed/api.hpp"
#include "ttnn/distributed/distributed_tensor_config.hpp"
#include "ttnn/distributed/types.hpp"
#include "ttnn/tensor/xtensor/partition.hpp"
#include <algorithm>
#include <tt-metalium/assert.hpp>

namespace ttnn::distributed {

// Mapper interface that distributes a host tensor onto a multi-device configuration.
class TensorToMesh {
public:
    virtual ~TensorToMesh() = default;
    virtual std::vector<Tensor> map(const Tensor& tensor) const = 0;
    virtual tt::tt_metal::DistributedTensorConfig config() const = 0;
};

// Composer interface that aggregates a multi-device tensor into a host tensor.
class MeshToTensor {
public:
    virtual ~MeshToTensor() = default;
    virtual Tensor compose(const std::vector<Tensor>& tensors) const = 0;
};

struct Shard2dConfig {
    std::optional<int> row_dim;
    std::optional<int> col_dim;
};

struct Concat2dConfig {
    int row_dim = -1;
    int col_dim = -1;
};

class ReplicateTensorToMesh : public TensorToMesh {
public:
    ReplicateTensorToMesh(size_t num_devices) : num_devices_(num_devices) {}

    ReplicateTensorToMesh(MeshDevice& mesh_device) : num_devices_(mesh_device.num_devices()) {}

    std::vector<Tensor> map(const Tensor& tensor) const override {
        std::vector<Tensor> tensors;
        tensors.reserve(num_devices_);
        std::fill_n(std::back_inserter(tensors), num_devices_, tensor);
        return tensors;
    }

    tt::tt_metal::DistributedTensorConfig config() const override {
        return tt::tt_metal::DistributedTensorConfig{ReplicateTensor{num_devices_}};
    }

private:
    size_t num_devices_ = 0;
};

class ShardTensorToMesh : public TensorToMesh {
public:
    ShardTensorToMesh(size_t num_devices, int dim) : num_devices_(num_devices), shard_dim_(dim) {}

    ShardTensorToMesh(MeshDevice& mesh_device, int dim) : num_devices_(mesh_device.num_devices()), shard_dim_(dim) {}

    std::vector<Tensor> map(const Tensor& tensor) const override {
        return experimental::xtensor::chunk(tensor, num_devices_, shard_dim_);
    }

    tt::tt_metal::DistributedTensorConfig config() const override {
        return tt::tt_metal::DistributedTensorConfig{ShardTensor{shard_dim_}};
    }

private:
    size_t num_devices_ = 0;
    int shard_dim_ = -1;
};

class ShardTensor2dMesh : public TensorToMesh {
public:
    ShardTensor2dMesh(MeshDevice& mesh_device, const MeshShape& mesh_shape, const Shard2dConfig& config) :
        mesh_shape_(mesh_shape), config_(config) {
        TT_FATAL(
            config.row_dim.has_value() || config.col_dim.has_value(),
            "Sharding a tensor to 2D mesh requires at least one dimension to shard");
        TT_FATAL(
            mesh_shape.num_rows <= mesh_device.shape().num_rows &&  //
                mesh_shape.num_cols <= mesh_device.shape().num_cols,
            "Device mesh shape does not match the provided mesh shape.");
    }

    ShardTensor2dMesh(const MeshShape& mesh_shape, const Shard2dConfig& config) :
        mesh_shape_(mesh_shape), config_(config) {
        TT_FATAL(
            config.row_dim.has_value() || config.col_dim.has_value(),
            "Sharding a tensor to 2D mesh requires at least one dimension to shard");
    }

    std::vector<Tensor> map(const Tensor& tensor) const override {
        const auto [rows, cols] = mesh_shape_;
        const auto [row_dim, col_dim] = config_;

        std::vector<Tensor> row_tensors;

        // Shard along rows
        if (!row_dim.has_value()) {
            row_tensors.reserve(rows);
            for (int i = 0; i < rows; ++i) {
                row_tensors.push_back(tensor);
            }
        } else {
            row_tensors = experimental::xtensor::chunk(tensor, rows, *row_dim);
        }

        std::vector<Tensor> tensor_shards;
        tensor_shards.reserve(rows * cols);
        // Shard along columns
        if (!col_dim.has_value()) {
            for (const auto& t : row_tensors) {
                for (int i = 0; i < cols; ++i) {
                    tensor_shards.push_back(t);
                }
            }
        } else {
            for (const auto& t : row_tensors) {
                auto col_chunks = experimental::xtensor::chunk(t, cols, *col_dim);
                tensor_shards.insert(tensor_shards.end(), col_chunks.begin(), col_chunks.end());
            }
        }

        TT_FATAL(
            static_cast<int>(tensor_shards.size()) == rows * cols,
            "ShardTensor2dMesh: Sharding failed. Number of shards should match the product of the mesh "
            "dimensions. Size: {}, rows: {}, cols: {}",
            tensor_shards.size(),
            rows,
            cols);

        return tensor_shards;
    }

    tt::tt_metal::DistributedTensorConfig config() const override {
        return DistributedTensorConfig{ShardTensor2D{ShardMesh{.y = mesh_shape_.num_rows, .x = mesh_shape_.num_cols}}};
    }

private:
    MeshShape mesh_shape_;
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

class DeviceConcatMeshToTensor : public ConcatMeshToTensor {
public:
    DeviceConcatMeshToTensor(MeshDevice mesh_device, int dim) : mesh_device_(mesh_device), concat_dim_(dim) {}

    Tensor compose(const Tensor& tensor) {
        return experimental::xtensor::concat(get_device_tensors(tensor), concat_dim_);
    }

private:
    MeshDevice mesh_device_;
    int concat_dim_ = -1;
};

class ConcatMesh2dToTensor : public MeshToTensor {
public:
    ConcatMesh2dToTensor(MeshDevice& mesh_device, const Concat2dConfig& config) :
        mesh_shape_(mesh_device.shape()), config_(config) {}

    Tensor compose(const std::vector<Tensor>& tensors) const override {
        const auto [rows, cols] = mesh_shape_;
        const auto [row_dim, col_dim] = config_;

        std::vector<Tensor> row_concatenated;
        row_concatenated.reserve(rows);
        for (int i = 0; i < rows; ++i) {
            auto row_start = tensors.begin() + i * cols;
            auto row_end = row_start + cols;
            std::vector<Tensor> row_tensors(row_start, row_end);
            row_concatenated.push_back(experimental::xtensor::concat(row_tensors, col_dim));
        }

        return experimental::xtensor::concat(row_concatenated, row_dim);
    }

private:
    MeshShape mesh_shape_;
    Concat2dConfig config_;
};

// Creates a mapper that replicates a tensor across all devices.
std::unique_ptr<TensorToMesh> replicate_tensor_to_mesh_mapper(MeshDevice& mesh_device);

// Creates a mapper that shards a tensor along a single dimension.
std::unique_ptr<TensorToMesh> shard_tensor_to_mesh_mapper(MeshDevice& mesh_device, int dim);

// Creates a mapper that shards a tensor along two dimensions, which will be intepreted as rows and columns.
// If either dimension is not specified, the tensor is replicated along that dimension.
std::unique_ptr<TensorToMesh> shard_tensor_to_2d_mesh_mapper(
    MeshDevice& mesh_device, const MeshShape& mesh_shape, const Shard2dConfig& config);

// Creates a composer that concatenates a tensor across a single dimension.
std::unique_ptr<MeshToTensor> concat_mesh_to_tensor_composer(int dim);

// Creates a composer that concatenates a tensor across two dimensions.

std::unique_ptr<MeshToTensor> concat_2d_mesh_to_tensor_composer(MeshDevice& mesh_device, const Concat2dConfig& config);

// Distributes a host tensor onto multi-device configuration according to the `mapper`.
Tensor distribute_tensor(
    const Tensor& tensor,
    const TensorToMesh& mapper,
    std::optional<std::reference_wrapper<MeshDevice>> mesh_device = std::nullopt);

// Aggregates a multi-device tensor into a host tensor according to the `composer`.
Tensor aggregate_tensor(const Tensor& tensor, const MeshToTensor& composer);

Shard2dConfig get_shard2d_config(const std::unordered_map<std::string, std::string>& metadata);

Concat2dConfig get_concat2d_config(const std::unordered_map<std::string, std::string>& metadata);

}  // namespace ttnn::distributed
