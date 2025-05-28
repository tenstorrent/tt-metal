// SPDX-FileCopyrightText: © 2024 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include "tt_stl/overloaded.hpp"
#include "ttnn/distributed/api.hpp"
#include "ttnn/distributed/distributed_tensor.hpp"
#include <tt-metalium/assert.hpp>
#include "ttnn/distributed/distributed_tensor_config.hpp"
#include "ttnn/distributed/types.hpp"
#include "ttnn/tensor/xtensor/partition.hpp"

namespace ttnn::distributed {
namespace {

class NdTensorToMesh : public TensorToMesh {
public:
    NdTensorToMesh(
        const ttnn::MeshShape& shape,
        const MeshMapperConfig& config,
        const tt::tt_metal::DistributedTensorConfig& distributed_tensor_config) :
        shape_(shape), config_(config), distributed_tensor_config_(distributed_tensor_config) {}

    std::vector<Tensor> map(const Tensor& tensor) const override {
        std::vector<Tensor> current_tensors = {tensor};

        for (size_t mesh_dim_idx = 0; mesh_dim_idx < shape_.dims(); ++mesh_dim_idx) {
            std::vector<Tensor> next_tensors;
            const size_t mesh_dim_size = shape_[mesh_dim_idx];
            const auto& placement = config_.placements[mesh_dim_idx];
            next_tensors.reserve(current_tensors.size() * mesh_dim_size);

            for (const auto& current_tensor : current_tensors) {
                std::visit(
                    tt::stl::overloaded{
                        [&](const MeshMapperConfig::Replicate&) {
                            for (size_t i = 0; i < mesh_dim_size; ++i) {
                                next_tensors.push_back(current_tensor);
                            }
                        },
                        [&](const MeshMapperConfig::Shard& shard) {
                            auto chunks = experimental::xtensor::chunk(current_tensor, mesh_dim_size, shard.dim);
                            TT_FATAL(
                                shape_.dims() == 1 || chunks.size() == mesh_dim_size,
                                "ND sharding requires the number of chunks {} to match the mesh dimension size {}",
                                chunks.size(),
                                mesh_dim_size);
                            next_tensors.insert(
                                next_tensors.end(),
                                std::make_move_iterator(chunks.begin()),
                                std::make_move_iterator(chunks.end()));
                        },
                    },
                    placement);
            }
            current_tensors = std::move(next_tensors);
        }

        TT_FATAL(
            current_tensors.size() <= shape_.mesh_size(),
            "NdTensorToMesh: Mapping failed. Expected at most {} tensors for mesh shape {}, but got {}.",
            shape_.mesh_size(),
            shape_,
            current_tensors.size());

        return current_tensors;
    }

    tt::tt_metal::DistributedTensorConfig config() const override { return distributed_tensor_config_; }

private:
    ttnn::MeshShape shape_;
    MeshMapperConfig config_;
    tt::tt_metal::DistributedTensorConfig distributed_tensor_config_;
};

class NdMeshToTensor : public MeshToTensor {
public:
    NdMeshToTensor(const ttnn::MeshShape& shape, const MeshComposerConfig& config) : shape_(shape), config_(config) {}

    Tensor compose(const std::vector<Tensor>& tensors) const override {
        TT_FATAL(
            shape_.dims() == 1 || tensors.size() == shape_.mesh_size(),
            "ND composition requires the number of tensors {} to match the mesh shape {}",
            tensors.size(),
            shape_);

        std::vector<Tensor> current_tensors = tensors;
        size_t outer_stride = shape_.dims() == 1 ? tensors.size() : shape_.mesh_size();

        for (int mesh_dim_idx = shape_.dims() - 1; mesh_dim_idx >= 0; --mesh_dim_idx) {
            const size_t mesh_dim_size = shape_.dims() == 1 ? tensors.size() : shape_[mesh_dim_idx];
            const int concat_dim = config_.dims[mesh_dim_idx];
            outer_stride /= mesh_dim_size;

            std::vector<Tensor> next_tensors;
            next_tensors.reserve(outer_stride);

            for (size_t outer_idx = 0; outer_idx < outer_stride; ++outer_idx) {
                std::vector<Tensor> group_to_concat;
                group_to_concat.reserve(mesh_dim_size);
                size_t group_start_idx = outer_idx * mesh_dim_size;
                for (size_t inner_idx = 0; inner_idx < mesh_dim_size; ++inner_idx) {
                    group_to_concat.push_back(current_tensors[outer_idx * mesh_dim_size + inner_idx]);
                }
                next_tensors.push_back(experimental::xtensor::concat(group_to_concat, concat_dim));
            }
            current_tensors = std::move(next_tensors);
        }

        TT_FATAL(
            current_tensors.size() == 1,
            "NdMeshToTensor: Composition failed. Expected 1 final tensor, but got {}.",
            current_tensors.size());
        return current_tensors[0];
    }

private:
    ttnn::MeshShape shape_;
    MeshComposerConfig config_;
};

}  // namespace

std::unique_ptr<TensorToMesh> replicate_tensor_to_mesh_mapper(MeshDevice& mesh_device) {
    return std::make_unique<NdTensorToMesh>(
        MeshShape(mesh_device.num_devices()),
        MeshMapperConfig{
            .placements =
                {
                    MeshMapperConfig::Replicate{},
                }},
        tt::tt_metal::DistributedTensorConfig{tt::tt_metal::ReplicateTensor{mesh_device.num_devices()}});
}

std::unique_ptr<TensorToMesh> shard_tensor_to_mesh_mapper(MeshDevice& mesh_device, int dim) {
    return std::make_unique<NdTensorToMesh>(
        MeshShape(mesh_device.num_devices()),
        MeshMapperConfig{
            .placements =
                {
                    MeshMapperConfig::Shard{dim},
                }},
        tt::tt_metal::DistributedTensorConfig{tt::tt_metal::ShardTensor{dim}});
}

std::unique_ptr<MeshToTensor> concat_mesh_to_tensor_composer(MeshDevice& mesh_device, int dim) {
    return std::make_unique<NdMeshToTensor>(
        MeshShape(mesh_device.num_devices()),
        MeshComposerConfig{
            .dims = {dim},
        });
}

std::unique_ptr<TensorToMesh> create_mesh_mapper(
    MeshDevice& mesh_device, const MeshMapperConfig& config, const std::optional<ttnn::MeshShape>& shape) {
    const auto distributed_shape = shape.value_or(mesh_device.shape());
    TT_FATAL(
        distributed_shape.mesh_size() <= mesh_device.shape().mesh_size(),
        "The size of the supplied mesh shape {} does not match the device shape size {}",
        distributed_shape,
        mesh_device.shape());
    TT_FATAL(
        distributed_shape.dims() == config.placements.size(),
        "The number of dimensions in the mesh shape {} does not match the "
        "number of placements in the config {}",
        distributed_shape,
        config);

    // TODO: #22258 - `DistributedTensorConfig` will be replaced by distributed host buffer, which can be used directly
    // in Tensor storage.
    tt::tt_metal::DistributedTensorConfig distributed_tensor_config;
    if (distributed_shape.dims() == 2) {
        distributed_tensor_config = tt::tt_metal::DistributedTensorConfig{
            tt::tt_metal::ShardTensor2D{tt::tt_metal::ShardMesh{.y = distributed_shape[0], .x = distributed_shape[1]}}};
    } else {
        distributed_tensor_config = tt::tt_metal::DistributedTensorConfig{tt::tt_metal::AllGatherTensor{}};
    }

    return std::make_unique<NdTensorToMesh>(distributed_shape, config, distributed_tensor_config);
}

std::unique_ptr<MeshToTensor> create_mesh_composer(
    MeshDevice& mesh_device, const MeshComposerConfig& config, const std::optional<ttnn::MeshShape>& shape) {
    const auto distributed_shape = shape.value_or(mesh_device.shape());
    TT_FATAL(
        distributed_shape.mesh_size() <= mesh_device.shape().mesh_size(),
        "The size of the supplied mesh shape {} does not match the device shape size {}",
        distributed_shape,
        mesh_device.shape());
    TT_FATAL(
        distributed_shape.dims() == config.dims.size(),
        "The number of dimensions in the mesh shape {} does not match the "
        "number of dimensions in the config {}",
        distributed_shape,
        config);

    return std::make_unique<NdMeshToTensor>(distributed_shape, config);
}

Tensor distribute_tensor(
    const Tensor& tensor, const TensorToMesh& mapper, std::optional<std::reference_wrapper<MeshDevice>> mesh_device) {
    TT_FATAL(
        tensor.storage_type() == tt::tt_metal::StorageType::HOST,
        "TensorToMesh only supports host tensors; got storage type: {}",
        tensor.storage_type());
    std::vector<Tensor> tensors = mapper.map(tensor);
    Tensor output = aggregate_as_tensor(tensors, mapper.config());
    if (mesh_device.has_value()) {
        return output.to_device(&(mesh_device->get()), output.memory_config());
    }
    return output;
}

Tensor aggregate_tensor(const Tensor& tensor, const MeshToTensor& composer) {
    return composer.compose(get_device_tensors(tensor.cpu()));
}

}  // namespace ttnn::distributed
