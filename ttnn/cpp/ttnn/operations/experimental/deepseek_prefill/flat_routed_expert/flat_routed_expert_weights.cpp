// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include "flat_routed_expert_weights.hpp"

#include <cstring>

#include <tt-metalium/distributed_host_buffer.hpp>
#include <tt-metalium/experimental/distributed_tensor/distributed_tensor_apis.hpp>
#include <tt-metalium/host_buffer.hpp>
#include <tt-metalium/mesh_coord.hpp>
#include <tt_stl/assert.hpp>

namespace ttnn::operations::experimental::deepseek_prefill::flat_routed_expert {

namespace {
// Tile bytes of `spec`, and its tile count: padded 2D shape, higher dims folded into rows.
std::pair<size_t, size_t> tile_geometry(const tt::tt_metal::TensorSpec& spec) {
    const auto& tile = spec.tile();
    const size_t tile_bytes = tile.get_tile_size(tt::tt_metal::datatype_to_dataformat_converter(spec.data_type()));
    const auto& padded = spec.padded_shape();
    const size_t width = padded[-1];
    const size_t height = padded.volume() / width;
    TT_FATAL(
        height % tile.get_height() == 0 && width % tile.get_width() == 0,
        "gather_host_tiles: padded shape {} is not whole tiles",
        padded);
    return {tile_bytes, (height / tile.get_height()) * (width / tile.get_width())};
}
}  // namespace

ttnn::Tensor gather_host_tiles(
    const std::vector<ttnn::Tensor>& sources,
    const std::vector<int64_t>& tile_map,
    const ttnn::Shape& shape,
    const tt::tt_metal::MemoryConfig& memory_config) {
    using namespace tt::tt_metal;
    TT_FATAL(!sources.empty(), "gather_host_tiles: no source tensors");
    const auto& ref = sources.front();
    const DataType dtype = ref.dtype();

    std::vector<size_t> source_tiles(sources.size());
    size_t tile_bytes = 0;
    for (size_t i = 0; i < sources.size(); ++i) {
        const auto& s = sources[i];
        TT_FATAL(s.storage_type() == StorageType::HOST, "gather_host_tiles: source {} is not on host", i);
        TT_FATAL(s.layout() == Layout::TILE, "gather_host_tiles: source {} is not TILE", i);
        TT_FATAL(s.dtype() == dtype, "gather_host_tiles: source {} is {}, not {}", i, s.dtype(), dtype);
        TT_FATAL(
            s.memory_config().memory_layout() == TensorMemoryLayout::INTERLEAVED,
            "gather_host_tiles: source {} must be interleaved (its tiles in row-major order)",
            i);
        TT_FATAL(
            s.host_storage().buffer().shard_coords() == ref.host_storage().buffer().shard_coords(),
            "gather_host_tiles: source {} is distributed over different shards",
            i);
        const auto [bytes, count] = tile_geometry(s.tensor_spec());
        TT_FATAL(i == 0 || bytes == tile_bytes, "gather_host_tiles: source {} tile size differs", i);
        tile_bytes = bytes;
        source_tiles[i] = count;
    }

    const TensorSpec dst_spec(shape, TensorLayout(dtype, PageConfig(Layout::TILE), memory_config));
    const auto [dst_tile_bytes, dst_tiles] = tile_geometry(dst_spec);
    TT_FATAL(dst_tile_bytes == tile_bytes, "gather_host_tiles: destination tile size differs from the sources'");
    TT_FATAL(
        tile_map.size() == dst_tiles,
        "gather_host_tiles: tile_map has {} entries for {} destination tiles",
        tile_map.size(),
        dst_tiles);
    TT_FATAL(
        dst_tiles * tile_bytes == dst_spec.compute_packed_buffer_size_bytes(),
        "gather_host_tiles: destination packed size {} is not {} tiles of {} B",
        dst_spec.compute_packed_buffer_size_bytes(),
        dst_tiles,
        tile_bytes);
    for (const int64_t entry : tile_map) {
        if (entry < 0) {
            continue;
        }
        const auto src = static_cast<size_t>(entry >> 32);
        const auto tile = static_cast<size_t>(entry & 0xFFFFFFFF);
        TT_FATAL(src < sources.size(), "gather_host_tiles: source index {} out of range", src);
        TT_FATAL(tile < source_tiles[src], "gather_host_tiles: tile {} out of range for source {}", tile, src);
    }
    TT_FATAL(tile_bytes % sizeof(uint32_t) == 0, "gather_host_tiles: tile size {} not word aligned", tile_bytes);

    const auto& ref_buffer = ref.host_storage().buffer();
    auto dst_buffer = DistributedHostBuffer::create(ref_buffer.shape());
    const std::vector<distributed::MeshCoordinate> coords(
        ref_buffer.shard_coords().begin(), ref_buffer.shard_coords().end());
    dst_buffer.emplace_shards(
        coords,
        [&](const distributed::MeshCoordinate& coord) {
            std::vector<HostBuffer> shards;
            shards.reserve(sources.size());
            for (const auto& s : sources) {
                auto shard = s.host_storage().buffer().get_shard(coord);
                TT_FATAL(shard.has_value(), "gather_host_tiles: a source has no shard at {}", coord);
                shards.push_back(std::move(*shard));
            }
            std::vector<uint32_t> words(dst_tiles * tile_bytes / sizeof(uint32_t), 0);
            auto* dst = reinterpret_cast<std::byte*>(words.data());
            for (size_t t = 0; t < dst_tiles; ++t) {
                const int64_t entry = tile_map[t];
                if (entry < 0) {
                    continue;  // zero tile: zero exponents and mantissas decode to 0 in every block-float format
                }
                const auto src = static_cast<size_t>(entry >> 32);
                const auto tile = static_cast<size_t>(entry & 0xFFFFFFFF);
                const auto bytes = std::as_const(shards[src]).view_bytes();
                std::memcpy(dst + t * tile_bytes, bytes.data() + tile * tile_bytes, tile_bytes);
            }
            return HostBuffer(std::move(words));
        },
        DistributedHostBuffer::ProcessShardExecutionPolicy::PARALLEL);

    // Sharded on the leading dim over the mesh, as ttnn.from_torch(..., ShardTensorToMesh(dim=0)) of the stacked
    // per-device tensors would be; the sources' own topology names their [rows, cols, K, N] dims, not these.
    auto topology = TensorTopology::create_sharded_tensor_topology(ref_buffer.shape(), /*shard_dim=*/0);
    return ttnn::Tensor(host_tensor_from_buffer_with_topology(std::move(dst_buffer), dst_spec, std::move(topology)));
}

}  // namespace ttnn::operations::experimental::deepseek_prefill::flat_routed_expert
