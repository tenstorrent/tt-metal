// SPDX-FileCopyrightText: © 2025 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include "squeeze.hpp"
#include <algorithm>
#include <tt_stl/small_vector.hpp>
#include <ttnn/distributed/distributed_configs.hpp>
#include <ttnn/distributed/tensor_topology.hpp>
#include "ttnn/operations/core/core.hpp"
#include "ttnn/tensor/tensor.hpp"

namespace ttnn {

ttnn::Tensor squeeze(const ttnn::Tensor& input_tensor, const ttsl::SmallVector<int>& dim) {
    const auto& original_logical_shape = input_tensor.logical_shape();
    const auto& padded_shape = input_tensor.padded_shape();
    auto input_tensor_rank = original_logical_shape.rank();

    ttsl::SmallVector<uint32_t> new_logical_shape(original_logical_shape.cbegin(), original_logical_shape.cend());
    ttsl::SmallVector<uint32_t> new_padded_shape(padded_shape.cbegin(), padded_shape.cend());

    // Explicitly copy dim to avoid modifying the input
    auto dims = dim;

    // handle negative dimensions
    for (int& dim : dims) {
        if (dim < 0) {
            dim += input_tensor_rank;
        }
    }
    // Sort the dimensions in descending order to avoid issues with modifying new_shape in loop
    std::sort(dims.rbegin(), dims.rend());

    // Special ugly case for 0-ranked input
    if (input_tensor_rank == 0) [[unlikely]] {
        if (dims.empty() || (dims.size() == 1 && (dims[0] == 0 || dims[0] == -1))) {
            return input_tensor;
        }
        TT_THROW("Dimension out of range (expected to be of [-1, 0], but got {})", dims[0]);
    }

    // Dims actually erased below (a requested dim is skipped if its size isn't 1) -- needed after the
    // reshape to shift any Shard placement's dim down by however many erased dims precede it.
    ttsl::SmallVector<int32_t> erased_dims;
    for (size_t i = 0; i < dims.size(); ++i) {
        const auto dim = dims[i];
        // Check duplicate dimensions
        if (i > 0) {
            TT_FATAL(dim != dims[i - 1], "dim {} appears multiple times in the list of dims", dim);
        }
        TT_FATAL(
            (dim >= 0) && (dim < input_tensor_rank),
            "Dimension out of range (expected to be in range of [{},{}], but got {})",
            -static_cast<std::ptrdiff_t>(input_tensor_rank),
            input_tensor_rank - 1,
            dim);

        // If original dimension was not of size 1, include all dimensions
        if (original_logical_shape[dim] != 1) {
            continue;
        }

        new_logical_shape.erase(new_logical_shape.begin() + dim);
        new_padded_shape.erase(new_padded_shape.begin() + dim);
        erased_dims.push_back(dim);
    }

    // Note: don't have to check padded too
    if (new_logical_shape == original_logical_shape) {
        return input_tensor;
    }

    auto output = ttnn::reshape(
        input_tensor, ttnn::Shape(std::move(new_logical_shape)), ttnn::Shape(std::move(new_padded_shape)));

    // reshape()'s underlying view() carries the input's mesh TensorTopology forward unchanged (it's a
    // pure metadata reinterpretation -- no cross-device data movement), but squeezing removes dims from
    // the logical shape, so a Shard recorded on a dim above an erased one is now off by however many
    // erased dims precede it. Shift those dims down to keep the topology valid for the squeezed rank.
    std::sort(erased_dims.begin(), erased_dims.end());
    const auto& input_topology = input_tensor.tensor_topology();
    auto placements = input_topology.placements();
    bool topology_changed = false;
    for (auto& placement : placements) {
        if (auto* shard = std::get_if<tt::tt_metal::distributed::MeshMapperConfig::Shard>(&placement)) {
            // Shard::dim is unnormalized by construction (may be negative); normalize against the
            // PRE-squeeze rank, since that's what it was recorded against.
            const auto normalized_dim = static_cast<int32_t>(original_logical_shape.get_normalized_index(shard->dim));
            if (std::find(erased_dims.begin(), erased_dims.end(), normalized_dim) != erased_dims.end()) {
                // A squeezed dim is size 1 per device (e.g. an evenly-divided expert/dispatch-group
                // axis, one slice per device); once it's erased there's no tensor dim left to anchor
                // the placement to, so Replicate is the only placement that still accurately
                // describes this mesh axis.
                placement = tt::tt_metal::distributed::MeshMapperConfig::Replicate{};
                topology_changed = true;
                continue;
            }
            const int32_t shift = static_cast<int32_t>(
                std::count_if(erased_dims.begin(), erased_dims.end(), [normalized_dim](int32_t erased) {
                    return erased < normalized_dim;
                }));
            const int32_t new_dim = normalized_dim - shift;
            if (new_dim != shard->dim) {
                shard->dim = new_dim;
                topology_changed = true;
            }
        }
    }
    if (topology_changed) {
        output.update_tensor_topology(tt::tt_metal::TensorTopology(
            input_topology.distribution_shape(), placements, input_topology.mesh_coords()));
    }

    return output;
}

ttnn::Tensor squeeze(const ttnn::Tensor& input_tensor, int dim) {
    ttsl::SmallVector<int> dims{dim};
    return squeeze(input_tensor, dims);
}

ttnn::Tensor squeeze(const ttnn::Tensor& input_tensor) {
    auto input_tensor_rank = input_tensor.logical_shape().rank();
    ttsl::SmallVector<int> dims(input_tensor_rank);
    std::iota(dims.begin(), dims.end(), 0);
    return squeeze(input_tensor, dims);
}

}  // namespace ttnn
