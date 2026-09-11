
// SPDX-FileCopyrightText: © 2025 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include "rand.hpp"

#include <cstdint>

#include "ttnn/operations/copy/typecast/typecast.hpp"
#include "ttnn/operations/core/core.hpp"
#include "ttnn/operations/rand/device/rand_device_operation.hpp"
#include "ttnn/operations/uniform/uniform_range.hpp"
#include "ttnn/core/distributed/distribution_mode.hpp"
#include "ttnn/tensor/types.hpp"
#include <ttnn/distributed/tensor_topology.hpp>

namespace ttnn {

namespace {

ttnn::Shape compute_shard_shape(
    const ttnn::Shape& logical_shape,
    const tt::tt_metal::distributed::MeshMapperConfig& config,
    const tt::tt_metal::distributed::MeshShape& mesh_shape) {
    ttnn::Shape::Container shard_dims(logical_shape.view().begin(), logical_shape.view().end());
    for (size_t i = 0; i < config.placements.size() && i < mesh_shape.dims(); ++i) {
        if (const auto* shard =
                std::get_if<tt::tt_metal::distributed::MeshMapperConfig::Shard>(&config.placements[i])) {
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

ttsl::SmallVector<bool> build_shard_mask(const tt::tt_metal::distributed::MeshMapperConfig& config) {
    ttsl::SmallVector<bool> mask;
    mask.reserve(config.placements.size());
    for (const auto& p : config.placements) {
        mask.push_back(std::holds_alternative<tt::tt_metal::distributed::MeshMapperConfig::Shard>(p));
    }
    return mask;
}

constexpr bool is_supported_output_dtype(DataType dtype) {
    switch (dtype) {
        case DataType::BFLOAT16:
        case DataType::FLOAT32:
        case DataType::UINT32:
        case DataType::BFLOAT8_B:
        case DataType::BFLOAT4_B:
        case DataType::UINT16:
        case DataType::INT32:
        case DataType::INT8: return true;
        default: return false;
    }
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
    std::uint32_t seed,
    const std::optional<tt::tt_metal::distributed::MeshMapperConfig>& mesh_mapper) {
    TT_FATAL(is_supported_output_dtype(dtype), "[ttnn::rand] Output dtype {} is not supported.", dtype);

    const bool needs_typecast = dtype != DataType::FLOAT32 && dtype != DataType::BFLOAT16;
    const DataType generation_dtype = needs_typecast ? DataType::FLOAT32 : dtype;

    ttnn::Shape device_shape = shape;
    ttsl::SmallVector<bool> mesh_dim_is_sharded;
    std::optional<tt::tt_metal::TensorTopology> tensor_topology;
    if (mesh_mapper.has_value()) {
        const auto& config = mesh_mapper.value();
        auto mesh_shape = config.mesh_shape_override.value_or(device.shape());
        TT_FATAL(
            config.placements.size() == mesh_shape.dims(),
            "ttnn::rand: placements size ({}) must match mesh dimensions ({})",
            config.placements.size(),
            mesh_shape.dims());
        TT_FATAL(
            mesh_shape.mesh_size() <= device.num_devices(),
            "ttnn::rand: distribution mesh size ({}) exceeds device mesh size ({})",
            mesh_shape.mesh_size(),
            device.num_devices());
        device_shape = compute_shard_shape(shape, config, mesh_shape);
        mesh_dim_is_sharded = build_shard_mask(config);
        tensor_topology.emplace(
            mesh_shape,
            config.placements,
            ttnn::distributed::compute_distribution_to_mesh_mapping(mesh_shape, device.shape()));
    }

    const auto output_range = ttnn::operations::uniform::make_inclusive_output_range(from, to, generation_dtype);

    auto tensor = ttnn::prim::uniform(
        device_shape,
        generation_dtype,
        Layout::TILE,
        memory_config,
        device,
        output_range.lower_bound,
        output_range.upper_bound,
        seed,
        std::move(mesh_dim_is_sharded),
        tensor_topology);
    if (needs_typecast) {
        tensor = ttnn::typecast(tensor, dtype);
    }
    if (layout != Layout::TILE) {
        tensor = ttnn::to_layout(tensor, layout);
    }

    if (tensor_topology.has_value()) {
        tensor.update_tensor_topology(*tensor_topology);
    }

    return tensor;
}

}  // namespace ttnn
