
// SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#include "rand.hpp"

#include "ttnn/operations/rand/device/rand_device_operation.hpp"
#include "ttnn/operations/core/core.hpp"
#include "ttnn/operations/copy/typecast/typecast.hpp"
#include "ttnn/tensor/types.hpp"
#include <ttnn/distributed/tensor_topology.hpp>
#include <tt-metalium/mesh_coord.hpp>

namespace ttnn {

namespace {

ttnn::Shape compute_shard_shape(
    const ttnn::Shape& logical_shape,
    const tt::tt_metal::distributed::MeshMapperConfig& config,
    const tt::tt_metal::distributed::MeshShape& mesh_shape) {
    ttnn::Shape::Container shard_dims(logical_shape.view().begin(), logical_shape.view().end());
    for (size_t i = 0; i < config.placements.size() && i < mesh_shape.dims(); ++i) {
        if (auto* shard = std::get_if<tt::tt_metal::distributed::MeshMapperConfig::Shard>(&config.placements[i])) {
            auto dim = static_cast<size_t>(shard->dim);
            TT_FATAL(
                dim < shard_dims.size(),
                "ttnn::rand: MeshMapperConfig shard dim {} exceeds tensor rank {}",
                dim,
                shard_dims.size());
            TT_FATAL(
                shard_dims[dim] % mesh_shape[i] == 0,
                "ttnn::rand: shape[{}]={} is not divisible by mesh dimension size {}",
                dim,
                shard_dims[dim],
                mesh_shape[i]);
            shard_dims[dim] /= mesh_shape[i];
        }
    }
    return ttnn::Shape(std::move(shard_dims));
}

bool has_shard_placement(const tt::tt_metal::distributed::MeshMapperConfig& config) {
    for (const auto& p : config.placements) {
        if (std::holds_alternative<tt::tt_metal::distributed::MeshMapperConfig::Shard>(p)) {
            return true;
        }
    }
    return false;
}

}  // namespace

Tensor rand(
    const ttnn::Shape& shape,
    MeshDevice& device,
    const DataType dtype,
    const Layout layout,
    const MemoryConfig& memory_config,
    float from,
    float to,
    uint32_t seed,
    const std::optional<tt::tt_metal::distributed::MeshMapperConfig>& mesh_mapper) {
    TT_FATAL(dtype != DataType::UINT8, "[ttnn::rand] DataType::UINT8 is not supported.");

    ttnn::Shape device_shape = shape;
    bool unique_per_device = false;
    if (mesh_mapper.has_value()) {
        const auto& config = mesh_mapper.value();
        auto mesh_shape = config.mesh_shape_override.value_or(device.shape());
        TT_FATAL(
            config.placements.size() == mesh_shape.dims(),
            "ttnn::rand: placements size ({}) must match mesh dimensions ({})",
            config.placements.size(),
            mesh_shape.dims());
        device_shape = compute_shard_shape(shape, config, mesh_shape);
        unique_per_device = has_shard_placement(config);
    }

    auto tensor = ttnn::prim::uniform(
        device_shape, DataType::FLOAT32, Layout::TILE, memory_config, device, from, to, seed, unique_per_device);
    if (dtype != DataType::FLOAT32) {
        tensor = ttnn::typecast(tensor, dtype);
    }
    if (layout != Layout::TILE) {
        tensor = ttnn::to_layout(tensor, layout);
    }

    if (mesh_mapper.has_value()) {
        const auto& config = mesh_mapper.value();
        auto mesh_shape = config.mesh_shape_override.value_or(device.shape());

        std::vector<tt::tt_metal::distributed::MeshCoordinate> coords;
        coords.reserve(mesh_shape.mesh_size());
        for (const auto& coord : tt::tt_metal::distributed::MeshCoordinateRange(mesh_shape)) {
            coords.push_back(coord);
        }

        tt::tt_metal::TensorTopology topology(mesh_shape, config.placements, coords);
        tensor = tensor.with_tensor_topology(std::move(topology));
    }

    return tensor;
}

}  // namespace ttnn
