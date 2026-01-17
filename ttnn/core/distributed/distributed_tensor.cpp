// SPDX-FileCopyrightText: © 2024 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include <tt_stl/assert.hpp>
#include <tt-metalium/distributed_host_buffer.hpp>
#include <tt-metalium/shape.hpp>
#include <tt-metalium/mesh_coord.hpp>
#include <tt-metalium/tilize_utils.hpp>

#include <tt_stl/small_vector.hpp>
#include <tt_stl/overloaded.hpp>
#include "tensor/host_buffer/functions.hpp"
#include "tensor/storage.hpp"
#include "tensor/tensor_impl.hpp"
#include <algorithm>
#include "ttnn/core.hpp"
#include "ttnn/distributed/api.hpp"
#include "ttnn/distributed/distributed_tensor.hpp"
#include <type_traits>
#include <xtensor/containers/xadapt.hpp>
#include <xtensor/containers/xarray.hpp>
#include <xtensor/core/xstrides.hpp>
#include <xtensor/core/xtensor_forward.hpp>
#include "ttnn/distributed/types.hpp"
#include "ttnn/tensor/xtensor/conversion_utils.hpp"
#include "ttnn/tensor/xtensor/partition.hpp"
#include "ttnn/distributed/tensor_topology.hpp"
#include "ttnn/distributed/host_ccl.hpp"
#include "distribution_mode.hpp"

namespace ttnn::distributed {
namespace {

// Returns a function that remaps a mesh coordinates from the mesh mapper / composer distribution shape to the device
// shape. `global_range` must outlive the use of the returned function.
auto get_remap_fn(DistributionMode distribution_mode, const MeshCoordinateRange* global_range) {
    return [distribution_mode, row_major_dst = global_range->begin()](const MeshCoordinate& src_coord) mutable {
        switch (distribution_mode) {
            case DistributionMode::ROW_MAJOR: return *(row_major_dst++);
            case DistributionMode::SUBMESH: return src_coord;
        }
        TT_THROW("Unreachable");
    };
}

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
    std::optional<Shape> shard_shape;
    for (const auto& [_, xtensor_view] : xtensor_shards_views) {
        if (!xtensor_view.has_value()) {
            continue;
        }
        auto xtensor_shard_shape = tt::tt_metal::experimental::xtensor::get_shape_from_xarray(xtensor_view->get());
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

// Creates a host buffer from a span, with the following optimizations:
// - If the span is mutable and the physical data matches the logical data, the span is used directly.
// - Otherwise, a copy of the data is created.
template <typename T>
tt::tt_metal::HostBuffer create_host_buffer_from_span(
    tt::stl::Span<T> span, const tt::tt_metal::MemoryPin& buffer_pin, const TensorSpec& tensor_spec, T pad_value) {
    if constexpr (!std::is_const_v<T>) {
        if (tensor_spec.layout() == tt::tt_metal::Layout::ROW_MAJOR &&
            tensor_spec.physical_shape() == tensor_spec.logical_2d_shape() &&
            tensor_spec.data_type() == tt::tt_metal::convert_to_data_type<T>()) {
            return tt::tt_metal::HostBuffer(span, buffer_pin);
        }
    }

    return tt::tt_metal::host_buffer::get_host_buffer(Tensor::from_span(
        tt::stl::make_const_span(span),
        tensor_spec,
        /*device=*/nullptr,
        /*cq_id=*/std::nullopt,
        pad_value));
}

}  // namespace

class TensorToMesh::Impl {
public:
    Impl(
        const MeshDevice& mesh_device,
        DistributionMode distribution_mode,
        const MeshShape& distribution_shape,
        const MeshMapperConfig& config) :
        mesh_device_view_(mesh_device.get_view()),
        global_range_(mesh_device_view_.shape()),
        distribution_mode_(distribution_mode),
        distribution_shape_(distribution_shape),
        config_(config) {}

    Tensor operator()(const Tensor& tensor) const {
        auto extract_logical_data = [this]<typename T>(const tt::tt_metal::Tensor& tensor) -> Tensor {
            const bool data_viewable = tensor.tensor_spec().layout() == tt::tt_metal::Layout::ROW_MAJOR &&
                                       tensor.tensor_spec().physical_shape() == tensor.tensor_spec().logical_2d_shape();
            tt::tt_metal::HostBuffer host_buffer = data_viewable ? tt::tt_metal::host_buffer::get_host_buffer(tensor)
                                                                 : tt::tt_metal::HostBuffer(tensor.to_vector<T>());
            return (*this)(
                host_buffer.view_as<T>(),
                tensor.tensor_spec().logical_shape(),
                host_buffer.pin(),
                tensor.tensor_spec().tensor_layout());
        };

        switch (tensor.tensor_spec().data_type()) {
            case tt::tt_metal::DataType::BFLOAT8_B:
            case tt::tt_metal::DataType::BFLOAT4_B:
            case tt::tt_metal::DataType::FLOAT32: return extract_logical_data.template operator()<float>(tensor);
            case tt::tt_metal::DataType::BFLOAT16: return extract_logical_data.template operator()<bfloat16>(tensor);
            case tt::tt_metal::DataType::UINT32: return extract_logical_data.template operator()<uint32_t>(tensor);
            case tt::tt_metal::DataType::UINT8: return extract_logical_data.template operator()<uint8_t>(tensor);
            case tt::tt_metal::DataType::UINT16: return extract_logical_data.template operator()<uint16_t>(tensor);
            case tt::tt_metal::DataType::INT32: return extract_logical_data.template operator()<int32_t>(tensor);
            case tt::tt_metal::DataType::INVALID: TT_THROW("Invalid data type: {}", tensor.tensor_spec().data_type());
        }
        TT_THROW("Unreachable");
    }

    template <typename T>
    Tensor operator()(
        tt::stl::Span<T> span,
        const Shape& shape,
        const tt::tt_metal::MemoryPin& buffer_pin,
        const tt::tt_metal::TensorLayout& layout,
        T pad_value = 0) const {
        size_t volume = shape.volume();
        TT_FATAL(
            span.size() == volume, "Current buffer size is {} different from shape volume {}", span.size(), volume);

        // Perform sharding, followed by replication.
        tt::stl::SmallVector<size_t> shard_dims;
        tt::stl::SmallVector<int> num_chunks_per_dim;
        tt::stl::SmallVector<int> tensor_dims;
        tt::stl::SmallVector<size_t> replicate_dims;
        size_t sharded_mesh_size = 1;
        for (size_t mesh_dim_idx = 0; mesh_dim_idx < distribution_shape_.dims(); ++mesh_dim_idx) {
            const auto& placement = config_.placements[mesh_dim_idx];
            if (const auto* shard_placement = std::get_if<MeshMapperConfig::Shard>(&placement)) {
                shard_dims.push_back(mesh_dim_idx);
                num_chunks_per_dim.push_back(distribution_shape_[mesh_dim_idx]);
                tensor_dims.push_back(shard_placement->dim);
                sharded_mesh_size *= distribution_shape_[mesh_dim_idx];
            } else {
                replicate_dims.push_back(mesh_dim_idx);
            }
        }

        // Optimize a fully replicated path, which can use the same buffer for all shards.
        if (shard_dims.empty()) {
            const TensorSpec tensor_spec(shape, layout);
            auto replicated_buffer = create_host_buffer_from_span<T>(span, buffer_pin, tensor_spec, pad_value);

            auto distributed_buffer = tt::tt_metal::DistributedHostBuffer::create(mesh_device_view_);
            auto remap_fn = get_remap_fn(distribution_mode_, &global_range_);
            std::vector<MeshCoordinate> buffer_coords;
            for (const auto& coord : MeshCoordinateRange(distribution_shape_)) {
                const auto mapped_coord = remap_fn(coord);
                buffer_coords.push_back(mapped_coord);
                distributed_buffer.emplace_shard(mapped_coord, [&b = replicated_buffer]() { return b; });
            }

            const auto tensor_topology =
                tt::tt_metal::TensorTopology(distribution_shape_, config_.placements, buffer_coords);

            return Tensor(tt::tt_metal::HostStorage(std::move(distributed_buffer)), tensor_spec, tensor_topology);
        }

        // Otherwise, use xtensor to chunk the data into shards.
        auto input_xtensor =
            tt::tt_metal::experimental::xtensor::adapt(span, std::vector<size_t>(shape.cbegin(), shape.cend()));

        auto chunks = tt::tt_metal::experimental::xtensor::chunk_ndim(input_xtensor, num_chunks_per_dim, tensor_dims);
        TT_FATAL(chunks.size() >= 1, "No chunks were produced");
        TT_FATAL(
            distribution_shape_.dims() == 1 || chunks.size() == sharded_mesh_size,
            "ND sharding requires the number of chunks {} to match the mesh dimension size {}",
            chunks.size(),
            sharded_mesh_size);

        using StridedViewRef =
            std::reference_wrapper<tt::tt_metal::experimental::xtensor::StridedView<decltype(input_xtensor)>>;
        tt::tt_metal::distributed::MeshContainer<std::optional<StridedViewRef>> sharded_xtensor_views(
            distribution_shape_, std::nullopt);

        // Distribute chunks to appropriate mesh coordinates.
        size_t chunk_idx = 0;
        tt::stl::SmallVector<int> shard_indices(shard_dims.size(), 0);
        do {
            tt::stl::SmallVector<uint32_t> mesh_coords(distribution_shape_.dims(), 0);
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
            replicate_sizes.push_back(distribution_shape_[replicate_mesh_dim]);
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

        return create_tensor<T>(sharded_xtensor_views, layout, pad_value);
    }

private:
    template <typename T>
    Tensor create_tensor(
        const auto& sharded_xtensor_views, const tt::tt_metal::TensorLayout& layout, T pad_value) const {
        const TensorSpec shard_spec = compute_tensor_spec_for_shards(sharded_xtensor_views, layout);

        auto distributed_buffer = tt::tt_metal::DistributedHostBuffer::create(mesh_device_view_);
        auto remap_fn = get_remap_fn(distribution_mode_, &global_range_);

        // Deduplicate processing of replicated buffers, by keeping a cache of already converted buffers.
        using XTensorViewKey = decltype(&sharded_xtensor_views.values().front()->get());
        std::unordered_map<XTensorViewKey, tt::tt_metal::HostBuffer> converted_buffers;

        std::vector<MeshCoordinate> buffer_coords;
        size_t num_views_with_value = 0;
        for (const auto& [coord, xtensor_view] : sharded_xtensor_views) {
            if (xtensor_view.has_value()) {
                const auto mapped_coord = remap_fn(coord);
                buffer_coords.push_back(mapped_coord);
                distributed_buffer.emplace_shard(
                    mapped_coord, [&converted_buffers, &xtensor_view, &shard_spec, &coord, pad_value]() {
                        // The callable makes a copy from the strided xtensor view to a vector; on multi-host systems,
                        // executed only for shards that are local to this host.

                        auto it = converted_buffers.find(&xtensor_view->get());
                        if (it != converted_buffers.end()) {
                            return it->second;
                        }
                        std::vector<std::remove_const_t<T>> data_vec(
                            xtensor_view->get().begin(), xtensor_view->get().end());
                        Tensor shard_tensor = Tensor::from_vector(
                            std::move(data_vec),
                            shard_spec,
                            /*device=*/nullptr,
                            std::nullopt,
                            pad_value);
                        auto buffer = tt::tt_metal::host_buffer::get_host_buffer(shard_tensor);
                        converted_buffers.emplace(&xtensor_view->get(), buffer);
                        return buffer;
                    });
                num_views_with_value++;
            }
        }

        // If the distribution shape is 1D and we have less shards than devices, set the distribution shape to the
        // number of chunks.
        const auto actual_distribution_shape =
            (distribution_shape_.dims() == 1) ? MeshShape(num_views_with_value) : distribution_shape_;

        const auto tensor_topology =
            tt::tt_metal::TensorTopology(actual_distribution_shape, config_.placements, buffer_coords);

        return Tensor(tt::tt_metal::HostStorage(std::move(distributed_buffer)), shard_spec, tensor_topology);
    }

    // MeshDevice parameters.
    MeshDeviceView mesh_device_view_;
    MeshCoordinateRange global_range_;
    DistributionMode distribution_mode_ = DistributionMode::ROW_MAJOR;

    // Distribution parameters.
    MeshShape distribution_shape_;
    MeshMapperConfig config_;
};

class MeshToTensor::Impl {
public:
    Impl(
        const MeshDevice& mesh_device,
        DistributionMode distribution_mode,
        const MeshShape& distribution_shape,
        const MeshComposerConfig& config) :
        global_range_(mesh_device.shape()),
        distribution_mode_(distribution_mode),
        distribution_shape_(distribution_shape),
        config_(config) {}

    template <typename T>
    std::pair<std::vector<T>, Shape> compose(const Tensor& tensor) const {
        const auto cpu_tensor = tensor.cpu();
        auto all_gather_tensor = host_ccl::all_gather(cpu_tensor);
        const auto& src_buffer = all_gather_tensor.host_storage().buffer();

        auto remap_fn = get_remap_fn(distribution_mode_, &global_range_);
        auto dst_buffer = tt::tt_metal::DistributedHostBuffer::create(distribution_shape_);

        for (const auto& dst_coord : MeshCoordinateRange(dst_buffer.shape())) {
            auto shard_opt = src_buffer.get_shard(remap_fn(dst_coord));
            if (shard_opt.has_value()) {
                dst_buffer.emplace_shard(dst_coord, [&shard_opt]() { return *shard_opt; });
            }
        }

        // Convert individual shards to logical data of the correct type `T`, if needed.
        if (!tt::tt_metal::logical_matches_physical(tensor.tensor_spec())) {
            dst_buffer = dst_buffer.transform(
                [&tensor](const tt::tt_metal::HostBuffer& shard) {
                    return tt::tt_metal::HostBuffer(Tensor(shard, tensor.tensor_spec()).to_vector<T>());
                },
                tt::tt_metal::DistributedHostBuffer::ProcessShardExecutionPolicy::PARALLEL);
        }

        // Convert shards into a linear buffer of xtensor views.
        std::vector<tt::tt_metal::experimental::xtensor::AdaptedView<const T>> xtensor_views;
        xtensor_views.reserve(distribution_shape_.mesh_size());
        std::vector<size_t> shard_shape(tensor.logical_shape().cbegin(), tensor.logical_shape().cend());
        dst_buffer.apply([&xtensor_views, &shard_shape](const tt::tt_metal::HostBuffer& shard) {
            xtensor_views.push_back(tt::tt_metal::experimental::xtensor::adapt(shard.view_as<const T>(), shard_shape));
        });

        tt::stl::SmallVector<int> num_chunks;
        // Scalar (0-dim tensor)
        bool is_single_views = xtensor_views.size() == 1;
        if (!is_single_views) {
            if (config_.dims.size() == 1) {
                num_chunks.push_back(xtensor_views.size());
            } else {
                TT_FATAL(
                    xtensor_views.size() == distribution_shape_.mesh_size(),
                    "ND composition requires the number of tensors {} to match the mesh shape {}",
                    xtensor_views.size(),
                    distribution_shape_);
                for (size_t i = 0; i < distribution_shape_.dims(); ++i) {
                    num_chunks.push_back(distribution_shape_[i]);
                }
            }
        }

        auto xtensor_adapter =
            tt::tt_metal::experimental::xtensor::concat_ndim(xtensor_views, num_chunks, config_.dims);
        auto&& shape = tt::tt_metal::experimental::xtensor::get_shape_from_xarray(xtensor_adapter.expr());
        return {std::move(xtensor_adapter).data(), std::move(shape)};
    }

    Tensor compose(const Tensor& tensor) const {
        auto dispatch_to_concrete = [this]<typename T>(const Tensor& tensor) {
            auto [data, shape] = compose<T>(tensor);
            TensorSpec spec(shape, tensor.tensor_spec().tensor_layout());
            return Tensor::from_vector(std::move(data), spec);
        };

        switch (tensor.dtype()) {
            case tt::tt_metal::DataType::BFLOAT8_B:
            case tt::tt_metal::DataType::BFLOAT4_B:
            case tt::tt_metal::DataType::FLOAT32: return dispatch_to_concrete.template operator()<float>(tensor);
            case tt::tt_metal::DataType::BFLOAT16: return dispatch_to_concrete.template operator()<bfloat16>(tensor);
            case tt::tt_metal::DataType::UINT32: return dispatch_to_concrete.template operator()<uint32_t>(tensor);
            case tt::tt_metal::DataType::UINT8: return dispatch_to_concrete.template operator()<uint8_t>(tensor);
            case tt::tt_metal::DataType::UINT16: return dispatch_to_concrete.template operator()<uint16_t>(tensor);
            case tt::tt_metal::DataType::INT32: return dispatch_to_concrete.template operator()<int32_t>(tensor);
            case tt::tt_metal::DataType::INVALID: TT_THROW("Invalid data type: {}", tensor.dtype());
        }
        TT_THROW("Unreachable");
    }

private:
    // MeshDevice parameters.
    MeshCoordinateRange global_range_;

    // Distribution parameters.
    DistributionMode distribution_mode_;
    MeshShape distribution_shape_;
    MeshComposerConfig config_;
};

TensorToMesh::TensorToMesh(std::unique_ptr<Impl> impl) : impl_(std::move(impl)) {}
TensorToMesh::~TensorToMesh() = default;
TensorToMesh::TensorToMesh(TensorToMesh&& other) noexcept = default;
TensorToMesh& TensorToMesh::operator=(TensorToMesh&& other) noexcept = default;
Tensor TensorToMesh::operator()(const Tensor& tensor) const { return (*impl_)(tensor); }

template <typename T>
Tensor TensorToMesh::operator()(
    tt::stl::Span<T> buffer,
    const ttnn::Shape& shape,
    const tt::tt_metal::MemoryPin& buffer_pin,
    const tt::tt_metal::TensorLayout& layout,
    T pad_value) const {
    return (*impl_).template operator()<T>(buffer, shape, buffer_pin, layout, pad_value);
}

TensorToMesh TensorToMesh::create(const MeshDevice& mesh_device, const MeshMapperConfig& config) {
    const auto distributed_shape = config.mesh_shape_override.value_or(mesh_device.shape());
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

    return TensorToMesh(std::make_unique<TensorToMesh::Impl>(
        mesh_device,
        compute_distribution_mode(config.mesh_shape_override, mesh_device.shape()),
        distributed_shape,
        config));
}

MeshToTensor::MeshToTensor(std::unique_ptr<Impl> impl) : impl_(std::move(impl)) {}
MeshToTensor::~MeshToTensor() = default;
MeshToTensor::MeshToTensor(MeshToTensor&& other) noexcept = default;
MeshToTensor& MeshToTensor::operator=(MeshToTensor&& other) noexcept = default;
Tensor MeshToTensor::compose(const Tensor& tensor) const { return impl_->compose(tensor); }

template <typename T>
std::pair<std::vector<T>, Shape> MeshToTensor::compose(const Tensor& tensor) const {
    return impl_->compose<T>(tensor);
}

MeshToTensor MeshToTensor::create(const MeshDevice& mesh_device, const MeshComposerConfig& config) {
    const auto distributed_shape = config.mesh_shape_override.value_or(mesh_device.shape());
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

    return MeshToTensor(std::make_unique<Impl>(
        mesh_device,
        compute_distribution_mode(config.mesh_shape_override, mesh_device.shape()),
        distributed_shape,
        config));
}

std::unique_ptr<TensorToMesh> replicate_tensor_to_mesh_mapper(MeshDevice& mesh_device) {
    return std::make_unique<TensorToMesh>(TensorToMesh::create(
        mesh_device,
        MeshMapperConfig{
            .placements = {MeshMapperConfig::Replicate{}},
            .mesh_shape_override = MeshShape(mesh_device.num_devices())}));
}

std::unique_ptr<TensorToMesh> shard_tensor_to_mesh_mapper(MeshDevice& mesh_device, int dim) {
    return std::make_unique<TensorToMesh>(TensorToMesh::create(
        mesh_device,
        MeshMapperConfig{
            .placements = {MeshMapperConfig::Shard{dim}},
            .mesh_shape_override = MeshShape(mesh_device.num_devices())}));
}

std::unique_ptr<TensorToMesh> create_mesh_mapper(MeshDevice& mesh_device, const MeshMapperConfig& config) {
    return std::make_unique<TensorToMesh>(TensorToMesh::create(mesh_device, config));
}

std::unique_ptr<MeshToTensor> concat_mesh_to_tensor_composer(MeshDevice& mesh_device, int dim) {
    return std::make_unique<MeshToTensor>(MeshToTensor::create(
        mesh_device, MeshComposerConfig{.dims = {dim}, .mesh_shape_override = MeshShape(mesh_device.num_devices())}));
}

std::unique_ptr<MeshToTensor> create_mesh_composer(MeshDevice& mesh_device, const MeshComposerConfig& config) {
    return std::make_unique<MeshToTensor>(MeshToTensor::create(mesh_device, config));
}

Tensor distribute_tensor(
    const Tensor& tensor,
    const TensorToMesh& mapper,
    std::optional<std::reference_wrapper<MeshDevice>> mesh_device,
    std::optional<ttnn::QueueId> cq_id) {
    TT_FATAL(
        tensor.storage_type() == tt::tt_metal::StorageType::HOST,
        "TensorToMesh only supports host tensors; got storage type: {}",
        tensor.storage_type());
    Tensor output = mapper(tensor);
    if (mesh_device.has_value()) {
        return output.to_device(&(mesh_device->get()), output.memory_config(), cq_id);
    }
    return output;
}

template <typename T>
Tensor create_distributed_tensor(
    tt::stl::Span<T> buffer,
    const ttnn::Shape& global_shape,
    const tt::tt_metal::MemoryPin& buffer_pin,
    const tt::tt_metal::TensorLayout& shard_layout,
    const TensorToMesh& mapper,
    std::optional<std::reference_wrapper<MeshDevice>> mesh_device,
    std::optional<ttnn::QueueId> cq_id,
    T pad_value) {
    Tensor output = mapper(buffer, global_shape, buffer_pin, shard_layout, pad_value);
    if (mesh_device.has_value()) {
        return output.to_device(&(mesh_device->get()), output.memory_config(), cq_id);
    }
    return output;
}

template <typename T>
Tensor create_distributed_tensor(
    tt::stl::Span<const T> buffer,
    const ttnn::Shape& global_shape,
    const tt::tt_metal::TensorLayout& shard_layout,
    const TensorToMesh& mapper,
    std::optional<std::reference_wrapper<MeshDevice>> mesh_device,
    std::optional<ttnn::QueueId> cq_id,
    T pad_value) {
    Tensor output =
        mapper.template operator()<const T>(buffer, global_shape, tt::tt_metal::MemoryPin(), shard_layout, pad_value);
    if (mesh_device.has_value()) {
        return output.to_device(&(mesh_device->get()), output.memory_config(), cq_id);
    }
    return output;
}

#define INSTANTIATE_CREATE_DISTRIBUTED_TENSOR(TYPE)                    \
    template Tensor create_distributed_tensor<TYPE>(                   \
        tt::stl::Span<TYPE> buffer,                                    \
        const ttnn::Shape& global_shape,                               \
        const tt::tt_metal::MemoryPin& buffer_pin,                     \
        const tt::tt_metal::TensorLayout& shard_layout,                \
        const TensorToMesh& mapper,                                    \
        std::optional<std::reference_wrapper<MeshDevice>> mesh_device, \
        std::optional<ttnn::QueueId> cq_id,                            \
        TYPE pad_value);                                               \
    template Tensor create_distributed_tensor<TYPE>(                   \
        tt::stl::Span<const TYPE> buffer,                              \
        const ttnn::Shape& global_shape,                               \
        const tt::tt_metal::TensorLayout& shard_layout,                \
        const TensorToMesh& mapper,                                    \
        std::optional<std::reference_wrapper<MeshDevice>> mesh_device, \
        std::optional<ttnn::QueueId> cq_id,                            \
        TYPE pad_value);

INSTANTIATE_CREATE_DISTRIBUTED_TENSOR(bfloat16)
INSTANTIATE_CREATE_DISTRIBUTED_TENSOR(float)
INSTANTIATE_CREATE_DISTRIBUTED_TENSOR(int32_t)
INSTANTIATE_CREATE_DISTRIBUTED_TENSOR(uint8_t)
INSTANTIATE_CREATE_DISTRIBUTED_TENSOR(uint16_t)
INSTANTIATE_CREATE_DISTRIBUTED_TENSOR(uint32_t)

#undef INSTANTIATE_CREATE_DISTRIBUTED_TENSOR

Tensor aggregate_tensor(const Tensor& tensor, const MeshToTensor& composer) { return composer.compose(tensor); }

template std::pair<std::vector<uint32_t>, Shape> MeshToTensor::compose<uint32_t>(const Tensor& tensor) const;
template std::pair<std::vector<float>, Shape> MeshToTensor::compose<float>(const Tensor& tensor) const;
template std::pair<std::vector<bfloat16>, Shape> MeshToTensor::compose<bfloat16>(const Tensor& tensor) const;
template std::pair<std::vector<int32_t>, Shape> MeshToTensor::compose<int32_t>(const Tensor& tensor) const;
template std::pair<std::vector<uint8_t>, Shape> MeshToTensor::compose<uint8_t>(const Tensor& tensor) const;
template std::pair<std::vector<uint16_t>, Shape> MeshToTensor::compose<uint16_t>(const Tensor& tensor) const;

}  // namespace ttnn::distributed
