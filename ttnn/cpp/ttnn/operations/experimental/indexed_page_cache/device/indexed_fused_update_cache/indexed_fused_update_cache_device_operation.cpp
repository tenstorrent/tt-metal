// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
// SPDX-License-Identifier: Apache-2.0

#include "indexed_fused_update_cache_device_operation.hpp"

#include <array>
#include <cstddef>
#include <cstdint>
#include <limits>
#include <variant>

#include <tt-metalium/constants.hpp>

#include "ttnn/device_operation.hpp"

namespace ttnn::experimental::prim::indexed_fused_update_cache {

namespace {

constexpr uint32_t max_update_rows = 256;
constexpr uint32_t max_positions_page_bytes = max_update_rows * sizeof(int32_t);
constexpr uint64_t max_indexable_cache_rows = static_cast<uint64_t>(std::numeric_limits<int32_t>::max()) + 1;
constexpr int cache_heads_dim = 1;
constexpr int cache_head_dim = 3;

void validate_device_tensor(const Tensor& tensor, const char* name) {
    TT_FATAL(tensor.storage_type() == StorageType::DEVICE, "{} must be on device", name);
    TT_FATAL(tensor.buffer() != nullptr, "{} must have a device buffer", name);
}

void validate_mesh_topologies(const IndexedFusedUpdateCacheInputs& args) {
    using MeshMapperConfig = tt::tt_metal::distributed::MeshMapperConfig;

    const auto& cache_topology = args.cache_tensor1.tensor_topology();
    TT_FATAL(
        args.cache_tensor2.tensor_topology() == cache_topology &&
            args.input_tensor1.tensor_topology() == cache_topology &&
            args.input_tensor2.tensor_topology() == cache_topology,
        "cache_tensor1, cache_tensor2, input_tensor1, and input_tensor2 must use identical mesh topology");

    for (const auto& placement : cache_topology.placements()) {
        if (const auto* shard = std::get_if<MeshMapperConfig::Shard>(&placement)) {
            TT_FATAL(
                shard->dim == cache_heads_dim || shard->dim == cache_head_dim,
                "cache and input tensors may only be mesh-sharded over heads (dim 1) or head dimension (dim 3); "
                "cache page/row sharding requires physical-index remapping");
        }
    }

    const auto& positions_topology = args.physical_update_idxs_tensor.tensor_topology();
    TT_FATAL(
        positions_topology.distribution_shape() == cache_topology.distribution_shape() &&
            positions_topology.mesh_coords() == cache_topology.mesh_coords(),
        "physical_update_idxs_tensor must use the same mesh shape and coordinates as the cache tensors");
    for (const auto& placement : positions_topology.placements()) {
        TT_FATAL(
            std::holds_alternative<MeshMapperConfig::Replicate>(placement),
            "physical_update_idxs_tensor must be replicated across mesh axes; sharded physical positions require "
            "cache ownership and index-remapping support");
    }
}

void validate_distinct_bound_tensors(const IndexedFusedUpdateCacheInputs& args) {
    struct NamedTensor {
        const char* name;
        const Tensor* tensor;
    };

    const std::array<NamedTensor, 5> tensors = {
        {{"cache_tensor1", &args.cache_tensor1},
         {"input_tensor1", &args.input_tensor1},
         {"cache_tensor2", &args.cache_tensor2},
         {"input_tensor2", &args.input_tensor2},
         {"physical_update_idxs_tensor", &args.physical_update_idxs_tensor}}};
    for (std::size_t first = 0; first < tensors.size(); ++first) {
        for (std::size_t second = first + 1; second < tensors.size(); ++second) {
            const auto& first_tensor = *tensors[first].tensor;
            const auto& second_tensor = *tensors[second].tensor;
            const bool aliases =
                first_tensor.memory_config().buffer_type() == second_tensor.memory_config().buffer_type() &&
                first_tensor.mesh_buffer().address() == second_tensor.mesh_buffer().address();
            TT_FATAL(
                !aliases, "{} and {} must not alias the same device buffer", tensors[first].name, tensors[second].name);
        }
    }
}

void validate_default_tile(const Tensor& tensor, const char* name) {
    const auto& tile = tensor.tensor_spec().tile();
    TT_FATAL(
        tile.get_height() == tt::constants::TILE_HEIGHT && tile.get_width() == tt::constants::TILE_WIDTH &&
            !tile.get_transpose_within_face() && !tile.get_transpose_of_faces(),
        "{} must use the default non-transposed 32x32 tile",
        name);
}

void validate_cache_input_pair(
    const Tensor& cache, const Tensor& input, const char* cache_name, const char* input_name) {
    TT_FATAL(cache.device() == input.device(), "{} and {} must be on the same device", cache_name, input_name);
    TT_FATAL(cache.layout() == Layout::TILE, "{} must use TILE layout", cache_name);
    TT_FATAL(input.layout() == Layout::TILE, "{} must use TILE layout", input_name);
    validate_default_tile(cache, cache_name);
    validate_default_tile(input, input_name);
    TT_FATAL(cache.dtype() == DataType::BFLOAT16, "{} must use BFLOAT16", cache_name);
    TT_FATAL(input.dtype() == DataType::BFLOAT16, "{} must use BFLOAT16", input_name);
    TT_FATAL(
        cache.memory_config().memory_layout() == TensorMemoryLayout::INTERLEAVED,
        "{} must use interleaved memory",
        cache_name);
    TT_FATAL(
        input.memory_config().memory_layout() == TensorMemoryLayout::INTERLEAVED,
        "{} must use interleaved memory",
        input_name);
    TT_FATAL(cache.logical_shape().size() == 4, "{} must have rank 4", cache_name);
    TT_FATAL(input.logical_shape().size() == 4, "{} must have rank 4", input_name);
    TT_FATAL(cache.logical_shape()[0] > 0, "{} must contain at least one physical page", cache_name);
    TT_FATAL(cache.logical_shape()[1] > 0, "{} must contain at least one head", cache_name);
    TT_FATAL(cache.logical_shape()[2] > 0, "{} physical pages must contain at least one row", cache_name);
    TT_FATAL(cache.logical_shape()[3] > 0, "{} head dimension must be nonzero", cache_name);
    TT_FATAL(input.logical_shape()[0] == 1, "{} dim 0 must be 1", input_name);
    TT_FATAL(
        cache.logical_shape()[1] == input.logical_shape()[1],
        "{} and {} must have the same number of heads",
        cache_name,
        input_name);
    TT_FATAL(
        cache.logical_shape()[3] == input.logical_shape()[3],
        "{} and {} must have the same head dimension",
        cache_name,
        input_name);
    TT_FATAL(cache.logical_shape()[2] % 32 == 0, "{} rows per physical page must be tile aligned", cache_name);
    TT_FATAL(cache.logical_shape()[3] % 32 == 0, "{} head dimension must be tile aligned", cache_name);
    TT_FATAL(
        cache.logical_shape()[2] == cache.padded_shape()[2] && cache.logical_shape()[3] == cache.padded_shape()[3],
        "{} cache page rows and head dimension must not require tile padding",
        cache_name);
    TT_FATAL(input.logical_shape()[2] > 0, "{} must contain at least one source row", input_name);
    TT_FATAL(input.logical_shape()[2] <= max_update_rows, "{} supports at most 256 source rows", input_name);
}

}  // namespace

IndexedFusedUpdateCacheDeviceOperation::program_factory_t
IndexedFusedUpdateCacheDeviceOperation::select_program_factory(const operation_attributes_t&, const tensor_args_t&) {
    return IndexedFusedUpdateCacheProgramFactory{};
}

void IndexedFusedUpdateCacheDeviceOperation::validate_on_program_cache_miss(
    const operation_attributes_t&, const tensor_args_t& args) {
    validate_device_tensor(args.cache_tensor1, "cache_tensor1");
    validate_device_tensor(args.input_tensor1, "input_tensor1");
    validate_device_tensor(args.cache_tensor2, "cache_tensor2");
    validate_device_tensor(args.input_tensor2, "input_tensor2");
    validate_device_tensor(args.physical_update_idxs_tensor, "physical_update_idxs_tensor");

    TT_FATAL(
        args.cache_tensor1.device() == args.input_tensor1.device() &&
            args.cache_tensor1.device() == args.cache_tensor2.device() &&
            args.cache_tensor1.device() == args.input_tensor2.device() &&
            args.cache_tensor1.device() == args.physical_update_idxs_tensor.device(),
        "all indexed_fused_update_cache tensors must be on the same device");
    const auto arch = args.cache_tensor1.device()->arch();
    TT_FATAL(
        arch == tt::ARCH::WORMHOLE_B0 || arch == tt::ARCH::BLACKHOLE,
        "indexed_fused_update_cache supports only Wormhole and Blackhole, got {}",
        arch);
    validate_mesh_topologies(args);
    validate_distinct_bound_tensors(args);

    validate_cache_input_pair(args.cache_tensor1, args.input_tensor1, "cache_tensor1", "input_tensor1");
    validate_cache_input_pair(args.cache_tensor2, args.input_tensor2, "cache_tensor2", "input_tensor2");
    TT_FATAL(
        args.cache_tensor1.logical_shape() == args.cache_tensor2.logical_shape(),
        "cache_tensor1 and cache_tensor2 must have identical shapes");
    TT_FATAL(
        args.input_tensor1.logical_shape() == args.input_tensor2.logical_shape(),
        "input_tensor1 and input_tensor2 must have identical shapes");
    TT_FATAL(
        args.cache_tensor1.memory_config() == args.cache_tensor2.memory_config(),
        "cache_tensor1 and cache_tensor2 must have identical memory configurations");
    TT_FATAL(
        args.input_tensor1.memory_config() == args.input_tensor2.memory_config(),
        "input_tensor1 and input_tensor2 must have identical memory configurations");

    const auto& positions = args.physical_update_idxs_tensor;
    TT_FATAL(positions.dtype() == DataType::INT32, "physical_update_idxs_tensor must use INT32");
    TT_FATAL(positions.layout() == Layout::ROW_MAJOR, "physical_update_idxs_tensor must use ROW_MAJOR layout");
    TT_FATAL(
        positions.memory_config().memory_layout() == TensorMemoryLayout::INTERLEAVED,
        "physical_update_idxs_tensor must use interleaved memory");
    TT_FATAL(
        positions.logical_shape().size() == 2 && positions.logical_shape()[0] == 1,
        "physical_update_idxs_tensor must have shape [1, num_rows]");
    TT_FATAL(
        positions.logical_shape()[1] >= args.input_tensor1.logical_shape()[2],
        "physical_update_idxs_tensor has fewer entries than the packed input has rows");
    TT_FATAL(
        positions.logical_shape()[1] <= max_update_rows, "physical_update_idxs_tensor supports at most 256 entries");
    TT_FATAL(
        positions.buffer()->num_dev_pages() == 1, "physical_update_idxs_tensor must fit in one row-major device page");
    TT_FATAL(
        positions.buffer()->aligned_page_size() <= max_positions_page_bytes,
        "physical_update_idxs_tensor aligned page size must not exceed 1024 bytes");

    const uint64_t total_cache_rows =
        static_cast<uint64_t>(args.cache_tensor1.logical_shape()[0]) * args.cache_tensor1.logical_shape()[2];
    TT_FATAL(
        total_cache_rows <= max_indexable_cache_rows,
        "flattened cache row count exceeds the range addressable by INT32 physical indices");
}

IndexedFusedUpdateCacheDeviceOperation::spec_return_value_t
IndexedFusedUpdateCacheDeviceOperation::compute_output_specs(const operation_attributes_t&, const tensor_args_t& args) {
    return {args.cache_tensor1.tensor_spec(), args.cache_tensor2.tensor_spec()};
}

IndexedFusedUpdateCacheDeviceOperation::topology_return_value_t
IndexedFusedUpdateCacheDeviceOperation::compute_output_topologies(
    const operation_attributes_t&, const tensor_args_t& args) {
    return {args.cache_tensor1.tensor_topology(), args.cache_tensor2.tensor_topology()};
}

IndexedFusedUpdateCacheDeviceOperation::tensor_return_value_t
IndexedFusedUpdateCacheDeviceOperation::create_output_tensors(
    const operation_attributes_t&, const tensor_args_t& args) {
    return std::make_tuple(args.cache_tensor1, args.cache_tensor2);
}

}  // namespace ttnn::experimental::prim::indexed_fused_update_cache

namespace ttnn::prim {

ttnn::experimental::prim::indexed_fused_update_cache::IndexedFusedUpdateCacheResult indexed_fused_update_cache(
    const Tensor& cache_tensor1,
    const Tensor& input_tensor1,
    const Tensor& cache_tensor2,
    const Tensor& input_tensor2,
    const Tensor& physical_update_idxs_tensor) {
    using Operation = ttnn::experimental::prim::indexed_fused_update_cache::IndexedFusedUpdateCacheDeviceOperation;
    return ttnn::device_operation::launch<Operation>(
        Operation::operation_attributes_t{},
        Operation::tensor_args_t{
            .cache_tensor1 = cache_tensor1,
            .input_tensor1 = input_tensor1,
            .cache_tensor2 = cache_tensor2,
            .input_tensor2 = input_tensor2,
            .physical_update_idxs_tensor = physical_update_idxs_tensor});
}

}  // namespace ttnn::prim
