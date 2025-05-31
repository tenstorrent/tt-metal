// SPDX-FileCopyrightText: © 2024 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include "tensor/host_buffer/functions.hpp"
#include "tt-metalium/shape.hpp"
#include "tt-metalium/mesh_coord.hpp"
#include "tt-metalium/small_vector.hpp"
#include "tt-metalium/tilize_utils.hpp"
#include "tt_stl/overloaded.hpp"
#include "ttnn/distributed/api.hpp"
#include "ttnn/distributed/distributed_tensor.hpp"
#include <tt-metalium/assert.hpp>
#include <type_traits>
#include <xtensor/containers/xadapt.hpp>
#include <xtensor/containers/xarray.hpp>
#include <xtensor/core/xstrides.hpp>
#include "ttnn/distributed/distributed_tensor_config.hpp"
#include "ttnn/distributed/types.hpp"
#include "ttnn/tensor/xtensor/conversion_utils.hpp"
#include "ttnn/tensor/xtensor/partition.hpp"

namespace ttnn::distributed {
namespace {

using ::tt::tt_metal::DistributedHostBuffer;
using ::tt::tt_metal::distributed::MeshContainer;

// Increments `indices` in-place given `limits`, to support row-major order iteration.
bool increment_indices(const tt::stl::SmallVector<int>& limits, tt::stl::SmallVector<int>& indices) {
    for (int i = static_cast<int>(indices.size()) - 1; i >= 0; --i) {
        if (++indices[i] < limits[i]) {
            return true;
        }
        indices[i] = 0;
    }
    return false;
}

// Computes tensor spec for shards supplied in `xtensor_shards_views`.
// Note the shapes of all shards must be the same; resulting in a uniform tensor spec.
TensorSpec compute_tensor_spec_for_shards(
    const auto& xtensor_shards_views, const tt::tt_metal::TensorLayout& global_layout) {
    std::optional<ttnn::Shape> shard_shape;
    for (const auto& [_, xtensor_view] : xtensor_shards_views) {
        if (!xtensor_view.has_value()) {
            continue;
        }
        auto xtensor_shard_shape = experimental::xtensor::get_shape_from_xarray(xtensor_view->get());
        if (shard_shape.has_value()) {
            TT_FATAL(
                shard_shape.value() == xtensor_shard_shape,
                "Shard shape mismatch: expected {} but got {}",
                shard_shape.value(),
                xtensor_shard_shape);
        } else {
            shard_shape = xtensor_shard_shape;
        }
    }
    TT_FATAL(shard_shape.has_value(), "No shards were produced");
    return TensorSpec(*shard_shape, global_layout);
}

class NdTensorToMesh : public TensorToMesh {
public:
    NdTensorToMesh(
        const ttnn::MeshShape& shape,
        const MeshMapperConfig& config,
        const tt::tt_metal::DistributedTensorConfig& distributed_tensor_config) :
        shape_(shape), config_(config), distributed_tensor_config_(distributed_tensor_config) {}

    Tensor operator()(const Tensor& tensor) const override {
        switch (tensor.tensor_spec().data_type()) {
            case tt::tt_metal::DataType::BFLOAT8_B:
            case tt::tt_metal::DataType::BFLOAT4_B:
            case tt::tt_metal::DataType::FLOAT32: return shard_tensor_typed<float>(tensor);
            case tt::tt_metal::DataType::BFLOAT16: return shard_tensor_typed<bfloat16>(tensor);
            case tt::tt_metal::DataType::UINT32: return shard_tensor_typed<uint32_t>(tensor);
            case tt::tt_metal::DataType::UINT8: return shard_tensor_typed<uint8_t>(tensor);
            case tt::tt_metal::DataType::UINT16: return shard_tensor_typed<uint16_t>(tensor);
            case tt::tt_metal::DataType::INT32: return shard_tensor_typed<int32_t>(tensor);
            case tt::tt_metal::DataType::INVALID: TT_THROW("Invalid data type: {}", tensor.tensor_spec().data_type());
        }
        TT_THROW("Unreachable");
    }

    tt::tt_metal::DistributedTensorConfig config() const override { return distributed_tensor_config_; }

private:
    template <typename T>
    Tensor shard_tensor_typed(const tt::tt_metal::Tensor& tensor) const {
        std::vector<size_t> shape_vec(tensor.logical_shape().cbegin(), tensor.logical_shape().cend());
        std::vector<T> logical_data;
        auto input_xtensor = [&]() {
            const bool data_viewable = tensor.tensor_spec().layout() == tt::tt_metal::Layout::ROW_MAJOR &&
                                       tensor.tensor_spec().physical_shape() == tensor.tensor_spec().logical_2d_shape();
            if (data_viewable) {
                tt::tt_metal::HostBuffer buffer = tt::tt_metal::host_buffer::get_host_buffer(tensor);
                auto span = buffer.view_as<T>();
                return xt::adapt(span.data(), span.size(), xt::no_ownership(), shape_vec);
            } else {
                logical_data = tensor.to_vector<T>();
                auto span = tt::stl::make_span(logical_data);
                return xt::adapt(span.data(), span.size(), xt::no_ownership(), shape_vec);
            }
        }();

        // Perform sharding, followed by replication.
        tt::stl::SmallVector<size_t> shard_dims;
        tt::stl::SmallVector<int> num_chunks_per_dim;
        tt::stl::SmallVector<int> tensor_dims;
        tt::stl::SmallVector<size_t> replicate_dims;
        size_t sharded_mesh_size = 1;
        for (size_t mesh_dim_idx = 0; mesh_dim_idx < shape_.dims(); ++mesh_dim_idx) {
            const auto& placement = config_.placements[mesh_dim_idx];
            if (const auto* shard_placement = std::get_if<MeshMapperConfig::Shard>(&placement)) {
                shard_dims.push_back(mesh_dim_idx);
                num_chunks_per_dim.push_back(shape_[mesh_dim_idx]);
                tensor_dims.push_back(shard_placement->dim);
                sharded_mesh_size *= shape_[mesh_dim_idx];
            } else {
                replicate_dims.push_back(mesh_dim_idx);
            }
        }

        auto chunks = experimental::xtensor::chunk_ndim(input_xtensor, num_chunks_per_dim, tensor_dims);
        TT_FATAL(chunks.size() >= 1, "No chunks were produced");
        TT_FATAL(
            shape_.dims() == 1 || chunks.size() == sharded_mesh_size,
            "ND sharding requires the number of chunks {} to match the mesh dimension size {}",
            chunks.size(),
            sharded_mesh_size);

        using StridedViewRef = std::reference_wrapper<experimental::xtensor::StridedView<decltype(input_xtensor)>>;
        MeshContainer<std::optional<StridedViewRef>> sharded_xtensor_views(shape_, std::nullopt);

        // Distribute chunks to appropriate mesh coordinates.
        size_t chunk_idx = 0;
        tt::stl::SmallVector<int> shard_indices(shard_dims.size(), 0);
        do {
            tt::stl::SmallVector<uint32_t> mesh_coords(shape_.dims(), 0);
            for (size_t i = 0; i < shard_dims.size(); ++i) {
                mesh_coords[shard_dims[i]] = shard_indices[i];
            }
            MeshCoordinate coord(mesh_coords);
            if (chunk_idx < chunks.size()) {
                sharded_xtensor_views.at(coord) = chunks[chunk_idx];
            }
            chunk_idx++;
        } while (increment_indices(num_chunks_per_dim, shard_indices));

        tt::stl::SmallVector<int> replicate_sizes;
        for (size_t replicate_mesh_dim : replicate_dims) {
            replicate_sizes.push_back(shape_[replicate_mesh_dim]);
        }

        // Fill in gaps along replicated dimensions:
        // Treat shards placed at the beginning of each replication axes as "replication sources";
        // for each one, copy its value to all other shards along the axes.
        if (!replicate_dims.empty()) {
            for (const auto& [coord, xtensor_view] : sharded_xtensor_views) {
                const bool replication_source =
                    std::all_of(replicate_dims.begin(), replicate_dims.end(), [&](size_t replicate_mesh_dim) {
                        return coord[replicate_mesh_dim] == 0;
                    });
                if (xtensor_view.has_value() && replication_source) {
                    tt::stl::SmallVector<int> replicate_indices(replicate_dims.size(), 0);
                    do {
                        tt::stl::SmallVector<uint32_t> mesh_coords(coord.coords().begin(), coord.coords().end());
                        for (size_t i = 0; i < replicate_dims.size(); ++i) {
                            mesh_coords[replicate_dims[i]] = replicate_indices[i];
                        }
                        sharded_xtensor_views.at(MeshCoordinate(mesh_coords)) = *xtensor_view;
                    } while (increment_indices(replicate_sizes, replicate_indices));
                }
            }
        }

        const TensorSpec shard_spec =
            compute_tensor_spec_for_shards(sharded_xtensor_views, tensor.tensor_spec().tensor_layout());

        // TODO: #22169 - For multi-host, supply mesh device global/local shape, along with local offset.
        auto distributed_buffer =
            tt::tt_metal::DistributedHostBuffer::create(shape_, shape_, MeshCoordinate::zero_coordinate(shape_.dims()));
        for (const auto& [coord, xtensor_view] : sharded_xtensor_views) {
            if (xtensor_view.has_value()) {
                distributed_buffer.emplace_shard(coord, [&xtensor_view, &shard_spec, &coord]() {
                    xt::xarray<T> data(xtensor_view->get());
                    auto shard_tensor = experimental::xtensor::from_xtensor<T>(data, shard_spec);
                    return tt::tt_metal::host_buffer::get_host_buffer(shard_tensor);
                });
            }
        }

        // TODO: #22169 - Directly create a multi-host distributed tensor from the distributed host buffer.
        std::vector<Tensor> tensors;
        for (const auto& coord : MeshCoordinateRange(shape_)) {
            auto shard = distributed_buffer.get_shard(coord);
            if (shard.has_value() && !shard->view_bytes().empty()) {
                tensors.push_back(Tensor(std::move(*shard), shard_spec));
            }
        }

        return aggregate_as_tensor(tensors, config());
    }

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
    Tensor output = mapper(tensor);
    if (mesh_device.has_value()) {
        return output.to_device(&(mesh_device->get()), output.memory_config());
    }
    return output;
}

Tensor aggregate_tensor(const Tensor& tensor, const MeshToTensor& composer) {
    return composer.compose(get_device_tensors(tensor.cpu()));
}

}  // namespace ttnn::distributed
