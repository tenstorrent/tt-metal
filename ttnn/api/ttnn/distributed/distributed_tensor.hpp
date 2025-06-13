// SPDX-FileCopyrightText: © 2024 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <tt_stl/small_vector.hpp>
#include "ttnn/tensor/tensor.hpp"
#include "ttnn/distributed/types.hpp"

namespace ttnn::distributed {

// Mapper interface that distributes a host tensor onto a multi-device configuration.
// TODO: #22258 - Make this class concrete.
class TensorToMesh {
public:
    virtual ~TensorToMesh() = default;
    virtual Tensor operator()(const Tensor& tensor) const = 0;
    virtual tt::tt_metal::DistributedTensorConfig config() const = 0;
};

// Composer interface that aggregates a multi-device tensor into a host tensor.
class MeshToTensor {
public:
    virtual ~MeshToTensor() = default;
    virtual Tensor compose(const std::vector<Tensor>& tensors) const = 0;
};

// Creates a mapper that replicates a tensor across all devices.
// Shorthand for specifying a MeshMapperConfig that replicates the tensor over the entire mesh.
std::unique_ptr<TensorToMesh> replicate_tensor_to_mesh_mapper(MeshDevice& mesh_device);

// Creates a mapper that shards a tensor along a single dimension.
// Shorthand for specifying a MeshMapperConfig with 1D mesh shape, and sharding the tensor along a single dimension of
// the tensor.
std::unique_ptr<TensorToMesh> shard_tensor_to_mesh_mapper(MeshDevice& mesh_device, int dim);

// Creates a composer that concatenates a tensor across a single dimension.
// Shorthand for specifying a MeshComposerConfig with 1D mesh shape, and concatenating the tensor across a single
// dimension.
std::unique_ptr<MeshToTensor> concat_mesh_to_tensor_composer(MeshDevice& mesh_device, int dim);

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
    // +-------------------------------------+---------------------------------------+
    // |  (0,0)                              |  (0,1)                                |
    // | +---+---+---+---+---+---+----+----+ | +---+---+---+---+----+----+----+----+ |
    // | | 0 | 1 | 2 | 3 | 8 | 9 | 10 | 11 | | | 4 | 5 | 6 | 7 | 12 | 13 | 14 | 15 | |
    // | +---+---+---+---+---+---+----+----+ | +---+---+---+---+----+----+----+----+ |
    // +-------------------------------------+---------------------------------------+
    // |  (1,0)                              |  (1,1)                                |
    // | +---+---+---+---+---+---+----+----+ | +---+---+---+---+----+----+----+----+ |
    // | | 0 | 1 | 2 | 3 | 8 | 9 | 10 | 11 | | | 4 | 5 | 6 | 7 | 12 | 13 | 14 | 15 | |
    // | +---+---+---+---+---+---+----+----+ | +---+---+---+---+----+----+----+----+ |
    // +-------------------------------------+---------------------------------------+
    //
    tt::stl::SmallVector<std::variant<Replicate, Shard>> placements;
};

// Creates an ND mesh mapper that distributes a tensor according to the `config`.
// If `shape` is not provided, the shape of `mesh_device` is used.
// Otherwise, the size of the shape must be smaller than the mesh device shape.
std::unique_ptr<TensorToMesh> create_mesh_mapper(
    MeshDevice& mesh_device,
    const MeshMapperConfig& config,
    const std::optional<ttnn::MeshShape>& shape = std::nullopt);

struct MeshComposerConfig {
    // Specifies dimension of the tensor to concatenate.
    std::vector<int> dims;
};

// Creates an ND mesh composer that aggregates a tensor according to the `config`.
// If `shape` is not provided, the shape of `mesh_device` is used.
// Otherwise, the size of the shape must match the size of the mesh device shape.
std::unique_ptr<MeshToTensor> create_mesh_composer(
    MeshDevice& mesh_device,
    const MeshComposerConfig& config,
    const std::optional<ttnn::MeshShape>& shape = std::nullopt);

// Distributes a host tensor onto multi-device configuration according to the `mapper`.
Tensor distribute_tensor(
    const Tensor& tensor,
    const TensorToMesh& mapper,
    std::optional<std::reference_wrapper<MeshDevice>> mesh_device = std::nullopt);

// Aggregates a multi-device tensor into a host tensor according to the `composer`.
Tensor aggregate_tensor(const Tensor& tensor, const MeshToTensor& composer);

}  // namespace ttnn::distributed
