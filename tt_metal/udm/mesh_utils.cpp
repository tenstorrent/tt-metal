// SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#include "tt_metal/udm/mesh_utils.hpp"
#include "tt_metal/udm/mesh_tensor_builder.hpp"
#include <tt_stl/assert.hpp>
#include <cmath>

namespace tt::tt_metal::experimental::udm {

void compute_strides(
    const std::array<uint32_t, MAX_RANK>& shape, uint32_t rank, std::array<uint32_t, MAX_RANK>& strides) {
    uint32_t stride = 1;
    for (int i = rank - 1; i >= 0; --i) {
        strides[i] = stride;
        stride *= shape[i];
    }
}

void adjust_array_to_rank(
    std::array<uint32_t, MAX_RANK>& arr, uint32_t& current_rank, uint32_t target_rank, uint32_t fill_value) {
    if (current_rank >= target_rank) {
        return;
    }

    uint32_t diff = target_rank - current_rank;
    // Shift existing elements to the right
    for (int i = static_cast<int>(current_rank) - 1; i >= 0; --i) {
        arr[diff + i] = arr[i];
    }
    // Fill beginning with fill_value
    for (uint32_t i = 0; i < diff; ++i) {
        arr[i] = fill_value;
    }
    current_rank = target_rank;
}

void adjust_shape_ranks(
    std::array<uint32_t, MAX_RANK>& arr1,
    uint32_t& rank1,
    std::array<uint32_t, MAX_RANK>& arr2,
    uint32_t& rank2,
    uint32_t fill_value) {
    if (rank1 == rank2) {
        return;
    }

    if (rank1 < rank2) {
        // Prepend fill_value to arr1
        uint32_t diff = rank2 - rank1;
        for (int i = rank1 - 1; i >= 0; --i) {
            arr1[i + diff] = arr1[i];
        }
        for (uint32_t i = 0; i < diff; ++i) {
            arr1[i] = fill_value;
        }
        rank1 = rank2;
    } else {
        // Prepend fill_value to arr2
        uint32_t diff = rank1 - rank2;
        for (int i = rank2 - 1; i >= 0; --i) {
            arr2[i + diff] = arr2[i];
        }
        for (uint32_t i = 0; i < diff; ++i) {
            arr2[i] = fill_value;
        }
        rank2 = rank1;
    }
}

namespace {

/**
 * @brief Factor num_cores into N factors, trying to keep them balanced
 *
 * For example: factor_cores(12, 2) might return {3, 4} or {4, 3}
 */
std::vector<uint32_t> factor_cores_into_dims(uint32_t num_cores, size_t num_dims) {
    TT_FATAL(num_cores > 0 && num_dims > 0, "num_cores and num_dims must be positive");

    std::vector<uint32_t> factors(num_dims, 1);

    if (num_dims == 1) {
        factors[0] = num_cores;
        return factors;
    }

    // Try to make factors as balanced as possible
    // Start with target = num_cores^(1/num_dims)
    uint32_t remaining = num_cores;
    double target = std::pow(static_cast<double>(num_cores), 1.0 / static_cast<double>(num_dims));

    for (size_t d = 0; d < num_dims - 1; ++d) {
        // Find the largest factor <= target that divides remaining
        uint32_t factor = static_cast<uint32_t>(target);
        while (factor > 1 && remaining % factor != 0) {
            factor--;
        }
        if (factor == 1 && remaining > 1) {
            factor = remaining;  // Use all remaining if no good factor found
        }

        factors[d] = factor;
        remaining /= factor;

        // Update target for remaining dimensions
        if (d < num_dims - 2) {
            target = std::pow(static_cast<double>(remaining), 1.0 / static_cast<double>(num_dims - d - 1));
        }
    }

    // Last dimension gets whatever's left
    factors[num_dims - 1] = remaining;

    return factors;
}

/**
 * @brief Calculate the number of pages for a given core in a partitioned dimension
 *
 * Distributes dim_size_in_pages across num_cores:
 * - Each core gets at least (dim_size_in_pages / num_cores) pages
 * - The first (dim_size_in_pages % num_cores) cores get one extra page
 *
 * @return pair<num_pages, base_pages> where:
 *   - num_pages: actual pages for this core
 *   - base_pages: base page count per core (for offset calculation)
 */
std::pair<uint32_t, uint32_t> get_core_pages(uint32_t dim_size_in_pages, uint32_t num_cores, uint32_t core_coord) {
    // Standard distribution: floor division + remainder distribution
    uint32_t base_pages = dim_size_in_pages / num_cores;  // floor division
    uint32_t remainder = dim_size_in_pages % num_cores;   // extra pages to distribute

    uint32_t num_pages;
    if (core_coord < remainder) {
        // First 'remainder' cores get one extra page
        num_pages = base_pages + 1;
    } else {
        // Remaining cores get base_pages (may be 0 if more cores than pages)
        num_pages = base_pages;
    }

    // For offset calculation, use the size of cores that get extra pages
    uint32_t offset_base = base_pages + 1;

    return {num_pages, offset_base};
}

/**
 * @brief Find the partition index for a given dimension
 *
 * Returns the index in the partition_dims array, or -1 if not partitioned
 */
int find_partition_index(int dim, const std::vector<int>& partition_dims) {
    for (size_t i = 0; i < partition_dims.size(); ++i) {
        if (partition_dims[i] == dim) {
            return static_cast<int>(i);
        }
    }
    return -1;
}

/**
 * @brief Convert linear core index to multi-dimensional coordinates (row-major)
 */
void compute_core_coords(
    uint32_t core_idx, const std::vector<uint32_t>& cores_per_dim, std::vector<uint32_t>& core_coords) {
    uint32_t temp = core_idx;
    for (int d = cores_per_dim.size() - 1; d >= 0; --d) {
        core_coords[d] = temp % cores_per_dim[d];
        temp /= cores_per_dim[d];
    }
}

/**
 * @brief Check if a core would be assigned any work
 *
 * Returns false if the core would get 0 pages in any partitioned dimension
 */
bool core_has_work(
    const std::vector<uint32_t>& core_coords,
    const std::vector<uint32_t>& dim_sizes,
    const std::vector<uint32_t>& cores_per_dim) {
    for (size_t d = 0; d < core_coords.size(); ++d) {
        auto [num_pages, base_pages] = get_core_pages(dim_sizes[d], cores_per_dim[d], core_coords[d]);
        if (num_pages == 0) {
            return false;
        }
    }
    return true;
}

}  // anonymous namespace

GcoresInfo map_tensor_to_gcores(
    const MeshTensorBuilder& tensor_builder, const std::vector<Gcore>& gcores, int partition_dim) {
    // Single-dimension partitioning - just call the ND version
    return map_tensor_to_gcores_nd(tensor_builder, gcores, {partition_dim});
}

GcoresInfo map_tensor_to_gcores_nd(
    const MeshTensorBuilder& tensor_builder, const std::vector<Gcore>& gcores, const std::vector<int>& partition_dims) {
    TT_FATAL(!gcores.empty(), "gcores cannot be empty");
    TT_FATAL(!partition_dims.empty(), "partition_dims cannot be empty");

    // Get the mesh tensor shape in pages (for proper work distribution based on memory pages)
    const auto& mesh_tensor_shape_in_pages = tensor_builder.get_mesh_tensor_shape_in_pages();
    uint32_t rank = mesh_tensor_shape_in_pages.rank();

    // Compute row-major strides for the mesh tensor shape
    std::vector<uint32_t> mesh_tensor_strides(rank);
    uint32_t stride = 1;
    for (int i = rank - 1; i >= 0; --i) {
        mesh_tensor_strides[i] = stride;
        stride *= mesh_tensor_shape_in_pages[i];
    }

    // Validate and normalize partition dimensions
    std::vector<int> normalized_dims;
    std::vector<uint32_t> dim_sizes;
    for (size_t i = 0; i < partition_dims.size(); ++i) {
        int dim = partition_dims[i];
        if (dim < 0) {
            dim = rank + dim;
        }
        TT_FATAL(
            dim >= 0 && static_cast<size_t>(dim) < rank,
            "partition_dim {} out of bounds for tensor rank {}",
            dim,
            rank);
        normalized_dims.push_back(dim);
        dim_sizes.push_back(mesh_tensor_shape_in_pages[dim]);
    }

    // Automatically factor the cores into an N-dimensional grid
    uint32_t num_cores = gcores.size();
    std::vector<uint32_t> cores_per_dim = factor_cores_into_dims(num_cores, partition_dims.size());

    GcoresInfo info;
    info.num_cores = 0;  // Will be updated as we assign cores
    info.partition_dims = normalized_dims;

    // Generate multi-dimensional grid assignment
    std::vector<uint32_t> core_coords(partition_dims.size(), 0);
    uint32_t assigned_cores = 0;

    for (uint32_t core_idx = 0; core_idx < num_cores; ++core_idx) {
        // Calculate multi-dimensional core coordinate from linear index (row-major)
        compute_core_coords(core_idx, cores_per_dim, core_coords);

        // Skip cores with no work (more cores than tensor chunks)
        if (!core_has_work(core_coords, dim_sizes, cores_per_dim)) {
            continue;
        }

        // Assign gcore (in order from the provided vector)
        info.gcores.push_back(gcores[core_idx]);

        // Add new entry for this assigned core
        std::vector<uint32_t> offsets, pages, strides;
        offsets.reserve(rank);
        pages.reserve(rank);
        strides.reserve(rank);

        for (uint32_t d = 0; d < rank; ++d) {
            // Check if this dimension is partitioned
            int partition_idx = find_partition_index(static_cast<int>(d), normalized_dims);

            uint32_t num_pages, offset, stride;

            if (partition_idx >= 0) {
                // Partitioned dimension: compute assignment for this core
                auto [core_pages, base_pages] =
                    get_core_pages(dim_sizes[partition_idx], cores_per_dim[partition_idx], core_coords[partition_idx]);
                num_pages = core_pages;
                offset = core_coords[partition_idx] * base_pages;
            } else {
                // Non-partitioned dimension: process entire dimension
                num_pages = mesh_tensor_shape_in_pages[d];
                offset = 0;
            }

            // Use row-major stride for all dimensions
            stride = mesh_tensor_strides[d];

            offsets.push_back(offset);
            pages.push_back(num_pages);
            strides.push_back(stride);
        }

        // Add this core's dimension info to the result
        info.dim_offsets.push_back(std::move(offsets));
        info.dim_pages.push_back(std::move(pages));
        info.dim_strides.push_back(std::move(strides));

        assigned_cores++;
    }

    info.num_cores = assigned_cores;
    return info;
}

}  // namespace tt::tt_metal::experimental::udm
