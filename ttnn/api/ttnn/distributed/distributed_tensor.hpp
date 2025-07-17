// SPDX-FileCopyrightText: © 2024 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <memory>

#include <tt_stl/small_vector.hpp>
#include <tt-metalium/memory_pin.hpp>

#include "tt_stl/span.hpp"
#include "ttnn/tensor/tensor.hpp"
#include "ttnn/distributed/types.hpp"

namespace ttnn::distributed {

struct MeshMapperConfig {
    // Specifies the tensor should be replicated across devices.
    struct Replicate {};

    // Specifies the tensor should be sharded along the specified dimension.
    struct Shard {
        int dim = 0;
    };

    // Specifies placements for each dimension of the shape.
    // The size of `placements` must match the dimensions of the shape.
    //
    // For example, sharding a 2x8 tensor over 2x2 mesh with {Replicate(), Shard{1}} will yield the following result:
    //
    //    Input Tensor [2, 8]:
    // +----+----+----+----+----+----+---+-----+
    // |  0 |  1 |  2 |  3 |  4 |  5 |  6 |  7 |
    // |----+----+----+----+----+----+---+-----+
    // |  8 |  9 | 10 | 11 | 12 | 13 | 14 | 15 |
    // +----+----+----+----+----+----+---+-----+
    //
    //    Shape [2, 2]:
    // +-------+-------+
    // | (0,0) | (0,1) |
    // +-------+-------+
    // | (1,0) | (1,1) |
    // +-------+-------+
    //
    // Distributed Tensor on Mesh (placements = {Replicate{}, Shard{1}}):
    //
    // +-----------------------------+-----------------------------+
    // |     (0,0)                   |     (0,1)                   |
    // |    +----+----+----+----+    |    +----+----+----+----+    |
    // |    |  0 |  1 |  2 |  3 |    |    |  4 |  5 |  6 |  7 |    |
    // |    +----+----+----+----+    |    +----+----+----+----+    |
    // |    |  8 |  9 | 10 | 11 |    |    | 12 | 13 | 14 | 15 |    |
    // |    +----+----+----+----+    |    +----+----+----+----+    |
    // +-----------------------------+-----------------------------+
    // |     (1,0)                   |     (1,1)                   |
    // |    +----+----+----+----+    |    +----+----+----+----+    |
    // |    |  0 |  1 |  2 |  3 |    |    |  4 |  5 |  6 |  7 |    |
    // |    +----+----+----+----+    |    +----+----+----+----+    |
    // |    |  8 |  9 | 10 | 11 |    |    | 12 | 13 | 14 | 15 |    |
    // |    +----+----+----+----+    |    +----+----+----+----+    |
    // +-----------------------------+-----------------------------+
    //

    using Placement = std::variant<Replicate, Shard>;
    tt::stl::SmallVector<Placement> placements;

    // If provided, the sharding will be performed according to this shape, but re-mapped to the mesh device shape in
    // either row-major order, or preserving the original coordinates (if the shape fits within the mesh device
    // entirely).
    std::optional<ttnn::MeshShape> mesh_shape_override = std::nullopt;
};

std::ostream& operator<<(std::ostream& os, const MeshMapperConfig::Placement& placement);
std::ostream& operator<<(std::ostream& os, const MeshMapperConfig& config);

// Mapper interface used for distributing a tensor onto a mesh.
class TensorToMesh {
public:
    ~TensorToMesh();
    TensorToMesh(TensorToMesh&& other) noexcept;
    TensorToMesh& operator=(TensorToMesh&& other) noexcept;
    TensorToMesh(const TensorToMesh&) = delete;
    TensorToMesh& operator=(const TensorToMesh&) = delete;

    static TensorToMesh create(const MeshDevice& mesh_device, const MeshMapperConfig& config);

    // Maps a tensor onto a mesh.
    // The input tensor is expected to be host-side tensor consisting of 1 device shard (i.e., mapped to 1x1 mesh).
    // The output tensor will be a host-side tensor mapped to a mesh of the same shape as the mesh device.
    Tensor operator()(const Tensor& tensor) const;

    // Overload that takes in a span of logical data; used in situations where the tensor object might not be
    // materialized.
    template <typename T>
    Tensor operator()(
        tt::stl::Span<T> buffer,
        const ttnn::Shape& shape,
        const tt::tt_metal::MemoryPin& buffer_pin,
        const tt::tt_metal::TensorLayout& layout,
        T pad_value = 0) const;

    tt::tt_metal::DistributedTensorConfig config() const;

private:
    class Impl;

    explicit TensorToMesh(std::unique_ptr<Impl> impl);

    std::unique_ptr<Impl> impl_;
};

// Creates an ND mesh mapper that distributes a tensor according to the `config`.
std::unique_ptr<TensorToMesh> create_mesh_mapper(MeshDevice& mesh_device, const MeshMapperConfig& config);

// Creates a mapper that replicates a tensor across all devices.
// Shorthand for specifying a MeshMapperConfig that replicates the tensor over the entire mesh.
std::unique_ptr<TensorToMesh> replicate_tensor_to_mesh_mapper(MeshDevice& mesh_device);

// Creates a mapper that shards a tensor along a single dimension.
// Shorthand for specifying a MeshMapperConfig with 1D mesh shape, and sharding the tensor along a single dimension of
// the tensor.
std::unique_ptr<TensorToMesh> shard_tensor_to_mesh_mapper(MeshDevice& mesh_device, int dim);

struct MeshComposerConfig {
    // Specifies dimension of the tensor to concatenate.
    tt::stl::SmallVector<int> dims;

    // If provided, the concatenation will be performed according to this shape, but re-mapped to the mesh device shape
    // in either row-major order, or preserving the original coordinates (if the shape fits within the mesh device
    // entirely).
    std::optional<ttnn::MeshShape> mesh_shape_override = std::nullopt;
};

std::ostream& operator<<(std::ostream& os, const MeshComposerConfig& config);

// Composer interface used for aggregating a tensor distributed over a mesh.
class MeshToTensor {
public:
    ~MeshToTensor();
    MeshToTensor(MeshToTensor&& other) noexcept;
    MeshToTensor& operator=(MeshToTensor&& other) noexcept;
    MeshToTensor(const MeshToTensor&) = delete;
    MeshToTensor& operator=(const MeshToTensor&) = delete;

    static MeshToTensor create(const MeshDevice& mesh_device, const MeshComposerConfig& config);

    // Composes a tensor distributed over a mesh.
    // The input tensor is expected to be distributed over a mesh of the same shape as the mesh device.
    // The output tensor will be a host-side tensor consisting of 1 device shard (i.e., mapped to 1x1 mesh).
    Tensor compose(const Tensor& tensor) const;

    // Overload that returns a pair of logical data and its shape, composed from a tensor distributed over a mesh.
    template <typename T>
    std::pair<std::vector<T>, Shape> compose(const Tensor& tensor) const;

private:
    class Impl;

    explicit MeshToTensor(std::unique_ptr<Impl> impl);

    std::unique_ptr<Impl> impl_;
};

// Creates an ND mesh composer that aggregates a tensor according to the `config`.
std::unique_ptr<MeshToTensor> create_mesh_composer(MeshDevice& mesh_device, const MeshComposerConfig& config);

// Creates a composer that concatenates a tensor across a single dimension.
// Shorthand for specifying a MeshComposerConfig with 1D mesh shape, and concatenating the tensor across a single
// dimension.
std::unique_ptr<MeshToTensor> concat_mesh_to_tensor_composer(MeshDevice& mesh_device, int dim);

// Distributes a host tensor onto multi-device configuration according to the `mapper`.
Tensor distribute_tensor(
    const Tensor& tensor,
    const TensorToMesh& mapper,
    std::optional<std::reference_wrapper<MeshDevice>> mesh_device = std::nullopt,
    ttnn::QueueId cq_id = ttnn::DefaultQueueId);

// Creates a distributed tensor from a span of logical data specified in `buffer`.
// `global_shape` must match the size of `buffer`; shapes of shards will be derived automatically based on the `mapper`,
// and the `shard_layout` will be applied subsequently. `buffer` may be re-used to create tensors directly, taking
// `buffer_pin` as the RAII to retain reference count to the object.
template <typename T>
Tensor create_distributed_tensor(
    tt::stl::Span<T> buffer,
    const ttnn::Shape& global_shape,
    const tt::tt_metal::MemoryPin& buffer_pin,
    const tt::tt_metal::TensorLayout& shard_layout,
    const TensorToMesh& mapper,
    std::optional<std::reference_wrapper<MeshDevice>> mesh_device = std::nullopt,
    ttnn::QueueId cq_id = ttnn::DefaultQueueId,
    T pad_value = 0);

// Overload for unowned spans of data.
template <typename T>
Tensor create_distributed_tensor(
    tt::stl::Span<const T> buffer,
    const ttnn::Shape& global_shape,
    const tt::tt_metal::TensorLayout& shard_layout,
    const TensorToMesh& mapper,
    std::optional<std::reference_wrapper<MeshDevice>> mesh_device = std::nullopt,
    ttnn::QueueId cq_id = ttnn::DefaultQueueId,
    T pad_value = 0);

// Aggregates a multi-device tensor into a host tensor according to the `composer`.
Tensor aggregate_tensor(const Tensor& tensor, const MeshToTensor& composer);

}  // namespace ttnn::distributed
