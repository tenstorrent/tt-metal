// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include "ttnn/tensor/topology_check.hpp"

#include <tt-metalium/distributed_host_buffer.hpp>
#include <tt-metalium/experimental/distributed_tensor/distributed_tensor_apis.hpp>
#include <tt-metalium/host_buffer.hpp>
#include <tt-metalium/mesh_coord.hpp>
#include <tt_stl/assert.hpp>

#include "ttnn/distributed/tensor_topology.hpp"
#include "ttnn/distributed/types.hpp"
#include "ttnn/tensor/storage.hpp"
#include "ttnn/tensor/tensor_utils.hpp"

#include <cstddef>
#include <cstdint>
#include <cstring>
#include <optional>
#include <set>
#include <sstream>
#include <string>
#include <algorithm>
#include <variant>
#include <vector>

namespace ttnn {
namespace {

std::string describe(const tt::tt_metal::TensorTopology& topology) {
    std::ostringstream os;
    os << topology;
    return os.str();
}

// Whether a shard's bytes are all data. A tile-layout tensor whose logical shape is not tile-aligned, or a row-major
// one with padding, carries bytes (tile padding, alignment) that hold whatever the producing op left there; two
// replicas can differ there while their values agree, so those bytes take no part in the comparison.
bool bytes_are_all_data(const tt::tt_metal::TensorSpec& spec) {
    return spec.logical_2d_shape() == spec.physical_shape();
}

// Bytes per element for the formats whose elements are stored one by one; block floats pack rows of elements with a
// shared exponent and have no fixed element size.
std::optional<size_t> fixed_element_size(tt::tt_metal::DataType dtype) {
    switch (dtype) {
        case tt::tt_metal::DataType::FLOAT32:
        case tt::tt_metal::DataType::UINT32:
        case tt::tt_metal::DataType::INT32: return 4;
        case tt::tt_metal::DataType::BFLOAT16:
        case tt::tt_metal::DataType::UINT16: return 2;
        case tt::tt_metal::DataType::UINT8:
        case tt::tt_metal::DataType::INT8:
        case tt::tt_metal::DataType::FP8_E4M3: return 1;
        default: return std::nullopt;
    }
}

// Compares the logical values of two shards in place, walking the logical region through the physical layout
// (row-major tiles, row-major faces within a tile, rows within a face) and comparing each contiguous run of logical
// elements, so padding takes no part and nothing is decoded or allocated. A rank-0 or rank-1 shape is one row, as
// TensorSpec::logical_2d_shape() reads it. nullopt when the layout is not one this walk describes: block floats,
// transposed tile faces, padded leading dimensions, or a physical shape that is not the padded last two dimensions
// times the leading ones; the caller then compares decoded values instead.
std::optional<bool> logical_values_equal_in_place(
    const tt::tt_metal::HostBuffer& a, const tt::tt_metal::HostBuffer& b, const tt::tt_metal::TensorSpec& spec) {
    const auto elem = fixed_element_size(spec.data_type());
    if (!elem.has_value()) {
        return std::nullopt;
    }
    const auto& logical = spec.logical_shape();
    const auto& padded = spec.padded_shape();
    // Shape indexes with int32_t (a negative index counts from the end). A dimension a shape lacks counts as 1, so a
    // rank-1 tile-layout tensor (one row, padded to a rank-2 tile) lines up with its padded shape.
    const int32_t rank = static_cast<int32_t>(logical.rank());
    const int32_t padded_rank = static_cast<int32_t>(padded.rank());
    const auto extent = [](const tt::tt_metal::Shape& shape, int32_t shape_rank, int32_t i) -> size_t {
        return i >= -shape_rank ? shape[i] : 1;
    };
    const size_t E = *elem;
    const size_t H = extent(logical, rank, -2), W = extent(logical, rank, -1);
    const size_t PH = extent(padded, padded_rank, -2), PW = extent(padded, padded_rank, -1);
    size_t batch = 1;
    for (int32_t i = -3; i >= -std::max(rank, padded_rank); --i) {
        if (extent(logical, rank, i) != extent(padded, padded_rank, i)) {
            return std::nullopt;
        }
        batch *= extent(logical, rank, i);
    }
    const auto bytes_a = a.view_bytes();
    const auto bytes_b = b.view_bytes();
    if (H > PH || W > PW || spec.physical_shape() != tt::tt_metal::Shape2D(batch * PH, PW) ||
        bytes_a.size() != bytes_b.size() || bytes_a.size() != batch * PH * PW * E) {
        return std::nullopt;
    }
    const std::byte* pa = bytes_a.data();
    const std::byte* pb = bytes_b.data();
    if (spec.layout() == tt::tt_metal::Layout::ROW_MAJOR) {
        for (size_t n = 0; n < batch; ++n) {
            for (size_t r = 0; r < H; ++r) {
                const size_t offset = ((n * PH + r) * PW) * E;
                if (std::memcmp(pa + offset, pb + offset, W * E) != 0) {
                    return false;
                }
            }
        }
        return true;
    }
    const tt::tt_metal::Tile tile = spec.tile();
    if (tile.get_transpose_within_face() || tile.get_transpose_of_faces()) {
        return std::nullopt;
    }
    const size_t TH = tile.get_tile_shape()[0], TW = tile.get_tile_shape()[1];
    const size_t FH = tile.get_face_shape()[0], FW = tile.get_face_shape()[1];
    if (PH % TH != 0 || PW % TW != 0 || TH % FH != 0 || TW % FW != 0) {
        return std::nullopt;
    }
    const size_t tiles_per_row = PW / TW, tile_elems = TH * TW, face_elems = FH * FW;
    for (size_t n = 0; n < batch; ++n) {
        for (size_t r = 0; r < H; ++r) {
            const size_t row = n * PH + r;
            const size_t tile_row = row / TH, row_in_tile = row % TH;
            const size_t face_row = row_in_tile / FH, row_in_face = row_in_tile % FH;
            for (size_t c = 0; c < W;) {
                const size_t tile_col = c / TW, col_in_tile = c % TW;
                const size_t face_col = col_in_tile / FW, col_in_face = col_in_tile % FW;
                const size_t run = std::min(FW - col_in_face, W - c);
                const size_t offset = ((tile_row * tiles_per_row + tile_col) * tile_elems + face_row * (FH * TW) +
                                       face_col * face_elems + row_in_face * FW + col_in_face) *
                                      E;
                if (std::memcmp(pa + offset, pb + offset, run * E) != 0) {
                    return false;
                }
                c += run;
            }
        }
    }
    return true;
}

// The shard's logical values as bytes, in row-major order of the logical shape: the comparison for layouts the
// in-place walk does not describe. Block floats are unpacked to float first, so the comparison is on values. FP8_E4M3
// has no host decoder, so its one-byte codes are read as UINT8 through the same layout and compared as codes.
std::vector<std::byte> logical_bytes(const tt::tt_metal::HostBuffer& shard, const tt::tt_metal::TensorSpec& spec) {
    const tt::tt_metal::HostTensor host = tt::tt_metal::HostTensor::from_buffer(shard, spec);
    const auto as_bytes = [](const auto& values) {
        std::vector<std::byte> out(values.size() * sizeof(values[0]));
        if (!out.empty()) {
            std::memcpy(out.data(), values.data(), out.size());
        }
        return out;
    };
    switch (spec.data_type()) {
        case tt::tt_metal::DataType::FLOAT32:
        case tt::tt_metal::DataType::BFLOAT8_B:
        case tt::tt_metal::DataType::BFLOAT4_B: return as_bytes(host.to_vector<float>());
        case tt::tt_metal::DataType::BFLOAT16: return as_bytes(host.to_vector<bfloat16>());
        case tt::tt_metal::DataType::UINT32: return as_bytes(host.to_vector<uint32_t>());
        case tt::tt_metal::DataType::INT32: return as_bytes(host.to_vector<int32_t>());
        case tt::tt_metal::DataType::UINT16: return as_bytes(host.to_vector<uint16_t>());
        case tt::tt_metal::DataType::UINT8: return as_bytes(host.to_vector<uint8_t>());
        case tt::tt_metal::DataType::INT8: return as_bytes(host.to_vector<int8_t>());
        case tt::tt_metal::DataType::FP8_E4M3: {
            const auto bytes = shard.view_bytes();
            std::vector<uint8_t> codes(bytes.size());
            if (!codes.empty()) {
                std::memcpy(codes.data(), bytes.data(), codes.size());
            }
            const tt::tt_metal::TensorSpec codes_spec(
                spec.logical_shape(),
                tt::tt_metal::TensorLayout(
                    tt::tt_metal::DataType::UINT8,
                    spec.page_config(),
                    spec.memory_config(),
                    spec.tensor_layout().get_alignment()));
            return as_bytes(
                tt::tt_metal::HostTensor::from_buffer(tt::tt_metal::HostBuffer(std::move(codes)), codes_spec)
                    .to_vector<uint8_t>());
        }
        default: TT_THROW("verify_shards_match_topology: unsupported data type {}", spec.data_type());
    }
}

}  // namespace

size_t num_replica_groups(const tt::tt_metal::TensorTopology& topology) {
    const auto& placements = topology.placements();
    const auto& mesh_shape = topology.distribution_shape();
    size_t groups = 1;
    for (size_t dim = 0; dim < placements.size(); ++dim) {
        if (std::holds_alternative<tt::tt_metal::distributed::MeshMapperConfig::Shard>(placements[dim])) {
            groups *= mesh_shape[dim];
        }
    }
    return groups;
}

size_t replica_group_key(
    const tt::tt_metal::TensorTopology& topology, const tt::tt_metal::distributed::MeshCoordinate& distribution_coord) {
    const auto& placements = topology.placements();
    const auto& mesh_shape = topology.distribution_shape();
    size_t key = 0;
    for (size_t dim = 0; dim < placements.size(); ++dim) {
        if (std::holds_alternative<tt::tt_metal::distributed::MeshMapperConfig::Shard>(placements[dim])) {
            key = key * mesh_shape[dim] + distribution_coord[dim];
        }
    }
    return key;
}

void verify_shards_match_topology(const Tensor& tensor, bool compare_replica_contents) {
    TT_FATAL(
        !is_device_tensor(tensor),
        "verify_shards_match_topology checks host shards; move the tensor to the host with cpu() first");
    const auto& distributed_buffer = tensor.host_storage().buffer();
    const auto& topology = tensor.tensor_topology();
    const auto& placements = topology.placements();
    const auto& mesh_shape = topology.distribution_shape();
    const auto& mesh_coords = topology.mesh_coords();
    TT_FATAL(
        mesh_coords.size() == mesh_shape.mesh_size(),
        "Topology mesh coords size {} should match distribution shape size {}",
        mesh_coords.size(),
        mesh_shape.mesh_size());
    TT_FATAL(
        placements.size() == mesh_shape.dims(),
        "Tensor topology has {} placement(s) for a {}-dimensional distribution shape ({}): one placement per "
        "distribution dimension is required. Relabel the tensor with update_tensor_topology() so the label matches "
        "how the shards were produced.",
        placements.size(),
        mesh_shape.dims(),
        describe(topology));

    // Every populated local shard has to be reachable through the label, or whatever keeps one copy per replica group
    // leaves it out. Remote coordinates (multi-host) hold no data on this host and are exempt.
    const std::set<tt::tt_metal::distributed::MeshCoordinate> labelled_coords(mesh_coords.begin(), mesh_coords.end());
    size_t num_populated_local_shards = 0;
    for (const auto& shard_coord : distributed_buffer.shard_coords()) {
        if (!distributed_buffer.is_local(shard_coord) || !distributed_buffer.get_shard(shard_coord).has_value()) {
            continue;
        }
        ++num_populated_local_shards;
        TT_FATAL(
            labelled_coords.contains(shard_coord),
            "Host storage holds a shard at mesh coordinate {} that the tensor topology does not cover ({}; host "
            "storage shape {}). The topology label does not describe the data, so this shard would be left out. "
            "Relabel the tensor with update_tensor_topology() so the label matches how the shards were produced.",
            shard_coord,
            describe(topology),
            distributed_buffer.shape());
    }

    // Replica groups (replica_group_key): coordinates that differ only along Replicate dimensions. The first member
    // met stands for its group; every later member must match it.
    struct Representative {
        tt::tt_metal::distributed::MeshCoordinate coord;
        const std::byte* data = nullptr;
        size_t size = 0;
        std::vector<std::byte> logical;  // Decoded once per group when the bytes are not all data.
    };
    const bool compare_bytes = bytes_are_all_data(tensor.tensor_spec());
    std::vector<std::optional<Representative>> groups(num_replica_groups(topology));
    size_t dist_idx = 0;
    for (const auto& dist_coord : tt::tt_metal::distributed::MeshCoordinateRange(mesh_shape)) {
        const auto& coord = mesh_coords[dist_idx++];
        const auto buffer = distributed_buffer.get_shard(coord);
        if (!buffer.has_value()) {
            // A labelled coordinate without a shard is only legitimate when the shard lives on another host.
            TT_FATAL(
                !distributed_buffer.is_local(coord),
                "Tensor topology lists mesh coordinate {} but the host storage has no shard there ({}; host storage "
                "shape {} with {} populated shard(s)). The topology label describes data that does not exist, for "
                "example a LOCAL-mode partial file re-dumped on a single host (re-dump from the multi-host job "
                "instead). Relabel the tensor with update_tensor_topology() so the label matches how the shards were "
                "produced.",
                coord,
                describe(topology),
                distributed_buffer.shape(),
                num_populated_local_shards);
            continue;
        }
        const auto bytes = buffer->view_bytes();
        auto& group = groups[replica_group_key(topology, dist_coord)];
        if (!group.has_value()) {
            group = Representative{.coord = coord, .data = bytes.data(), .size = bytes.size()};
            continue;
        }
        TT_FATAL(
            group->size == bytes.size(),
            "Tensor topology labels mesh coordinates {} and {} as replicas of each other ({}; host storage shape {}), "
            "but their shard sizes differ ({} vs {} bytes). The topology label does not describe the data. Relabel the "
            "tensor with update_tensor_topology() so the label matches how the shards were produced.",
            group->coord,
            coord,
            describe(topology),
            distributed_buffer.shape(),
            group->size,
            bytes.size());
        // Two views of one allocation hold the same bytes by construction (the fully replicated mapper path aliases
        // one buffer); only distinct buffers are compared. Equal bytes settle it either way. Bytes that differ settle
        // it when every byte is data; otherwise (tile padding is not data) the logical values decide: compared in
        // place where the walk describes the layout, through decoded copies otherwise. Padding that agrees between
        // replicas therefore costs one memcmp, and padding that differs one more pass over the two buffers.
        if (compare_replica_contents && group->data != bytes.data()) {
            bool same = std::memcmp(group->data, bytes.data(), bytes.size()) == 0;
            if (!same && !compare_bytes) {
                const auto representative = distributed_buffer.get_shard(group->coord);
                if (const auto in_place = logical_values_equal_in_place(*representative, *buffer, tensor.tensor_spec());
                    in_place.has_value()) {
                    same = *in_place;
                } else {
                    try {
                        if (group->logical.empty()) {
                            group->logical = logical_bytes(*representative, tensor.tensor_spec());
                        }
                        const std::vector<std::byte> member = logical_bytes(*buffer, tensor.tensor_spec());
                        same = member.size() == group->logical.size() &&
                               std::memcmp(group->logical.data(), member.data(), member.size()) == 0;
                    } catch (const std::exception& e) {
                        TT_THROW(
                            "Could not decode the shards at mesh coordinates {} and {} to compare them as replicas "
                            "({}; host storage shape {}): {}",
                            group->coord,
                            coord,
                            describe(topology),
                            distributed_buffer.shape(),
                            e.what());
                    }
                }
            }
            TT_FATAL(
                same,
                "Tensor topology labels mesh coordinates {} and {} as replicas of each other ({}; host storage shape "
                "{}), but their shard contents differ. The topology label does not describe the data, and keeping one "
                "copy for both would lose the shard at {}. Relabel the tensor with update_tensor_topology() so the "
                "label matches how the shards were produced.",
                group->coord,
                coord,
                describe(topology),
                distributed_buffer.shape(),
                coord);
        }
    }
}

}  // namespace ttnn
