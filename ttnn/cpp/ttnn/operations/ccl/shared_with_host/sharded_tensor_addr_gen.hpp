// SPDX-FileCopyrightText: © 2023 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0
#pragma once

#include <cmath>
#include <cstdint>
#include <array>
#include <bit>
#include "tt_metal/impl/buffers/buffer_constants.hpp"

/*
 *    ------   ATTENTION  ATTENTION  ATTENTION  ATTENTION  ATTENTION   ------
 * This file is intended to be useable across both host and device code. Therefore.
 *
 * DO NOT include any headers that are not host/device agnostic.
 * DO NOT use any types that do not have fixed sizes across host and device.
 * e.g. int32_t -> good (always 32 bits), int -> bad (size depends on platform)
 *
 * The reason for dual inclusion across host/device is because this code is used
 * on device, but is further tested on host through gtests. This enables us to
 * sweep functionality quickly and easily without involving end-to-end device kernel
 * invocations and program creation.
 */

using noc_grid_index_t = std::uint8_t;

namespace tt {
namespace tt_metal {

namespace address_generators {

template <typename ArchSpecificWorkerToNocLookup>
struct WorkerToNocCoordLookup {
    noc_grid_index_t get_noc_x_from_worker_x(noc_grid_index_t worker_x) const {
        return ArchSpecificWorkerToNocLookup::get_noc_x_from_worker_x(worker_x);
    }

    noc_grid_index_t get_noc_y_from_worker_y(noc_grid_index_t worker_y) const {
        return ArchSpecificWorkerToNocLookup::get_noc_y_from_worker_y(worker_y);
    }
};

/* A worker coord to noc coord lookup
 * It is marked "Harvested" in the type name because a non-harvested Wormhole part has a
 * fixed coordinate mapping, whereas the harvested part has potentially unique mapping per device
 * Since we, in general, don't know if we will be running on harvested parts when writing
 * our kernels, we call this HarvestedWormholeWorkerToNocLookup to indicate it can be
 * used on both harvested and non-harvested parts.
 */
struct HarvestedWormholeWorkerToNocLookup : WorkerToNocCoordLookup<HarvestedWormholeWorkerToNocLookup> {
    HarvestedWormholeWorkerToNocLookup(uint32_t nrows,
                                       const uint32_t* const row_map,
                                       uint32_t ncols,
                                       const uint32_t* const col_map)
        : nrows(nrows), row_map(row_map), ncols(ncols), col_map(col_map) {}

    noc_grid_index_t get_noc_x_from_worker_x(noc_grid_index_t worker_x) const {
        // ASSERT worker_x < worker_to_routing_x_wormhole.size()
        return col_map[worker_x];
    }

    noc_grid_index_t get_noc_y_from_worker_y(noc_grid_index_t worker_y) const {
        // ASSERT worker_y < worker_to_routing_y_wormhole.size()
        return row_map[worker_y];
    }

    uint32_t nrows;
    const uint32_t* const row_map;
    uint32_t ncols;
    const uint32_t* const col_map;
};

struct device_core_location_t {
    noc_grid_index_t noc_y;
    noc_grid_index_t noc_x;
};
struct test_shard_location_t {
    device_core_location_t core_location;
    std::uint32_t page_offset;
};
struct test_shard_location_with_contig_t {
    device_core_location_t core_location;
    std::uint32_t page_offset;
    std::uint32_t contig_pages_in_row;
};

/* Similar to interleaved address generators found in dataflow API, this acts
 * as a somewhat standardized interface to generate addresses for sharded tensors
 * to go from global tile IDs to concrete noc addresses.
 *
 * This is the base class, to be inherited by each shard strategy */
template <typename shard_type_device_shard_spec_t>
struct device_shard_spec_t {
    constexpr device_shard_spec_t(uint8_t shard_grid_height,
                                  uint8_t shard_grid_width,
                                  uint8_t shard_grid_start_y_logical,
                                  uint8_t shard_grid_start_x_logical,
                                  bool transposed_grid)
        : shard_grid_height(shard_grid_height),
          shard_grid_width(shard_grid_width),
          shard_grid_start_y_logical(shard_grid_start_y_logical),
          shard_grid_start_x_logical(shard_grid_start_x_logical),
          transposed_grid(transposed_grid) {}

    uint8_t shard_grid_height;
    uint8_t shard_grid_width;
    uint8_t shard_grid_start_y_logical;
    uint8_t shard_grid_start_x_logical;
    bool transposed_grid;

    constexpr uint32_t get_pages_per_shard() const {
        return shard_type_device_shard_spec_t::get_pages_per_shard_x() *
               shard_type_device_shard_spec_t::get_pages_per_shard_y();
    }
    constexpr uint16_t get_shard_grid_num_cores() const {
        return shard_grid_height * shard_grid_width;
    }

    constexpr uint16_t get_shard_grid_width() const {
        return shard_grid_width;
    }
    constexpr uint16_t get_shard_grid_height() const {
        return shard_grid_height;
    }
};

/*
 * Width sharded tensor spec - acts as helper for WidthShardedAddressGenerator
 */
struct DeviceWidthShardSpec : public device_shard_spec_t<DeviceWidthShardSpec> {
    constexpr DeviceWidthShardSpec(uint16_t pages_per_shard_y,
                                   uint16_t pages_per_shard_x,
                                   uint8_t shard_grid_height,
                                   uint8_t shard_grid_width,
                                   uint8_t shard_grid_start_y_logical,
                                   uint8_t shard_grid_start_x_logical,
                                   bool transposed_grid)
        : device_shard_spec_t<DeviceWidthShardSpec>(shard_grid_height,
                                                    shard_grid_width,
                                                    shard_grid_start_y_logical,
                                                    shard_grid_start_x_logical,
                                                    transposed_grid),
          pages_per_shard_y(pages_per_shard_y),
          pages_per_shard_x(pages_per_shard_x) {}

    uint16_t pages_per_shard_y;
    uint16_t pages_per_shard_x;

    constexpr uint32_t get_pages_per_shard_x() const {
        return pages_per_shard_x;
    }
    constexpr uint32_t get_pages_per_shard_y() const {
        return pages_per_shard_y;
    }
    constexpr uint32_t get_pages_per_tensor_x() const {
        return pages_per_shard_x * get_shard_grid_num_cores();
    }
    constexpr uint32_t get_pages_per_tensor_y() const {
        return pages_per_shard_y;
    }

    constexpr uint32_t get_shard_grid_inner_dim() const {
        return (!transposed_grid * shard_grid_width) + (transposed_grid * shard_grid_height);
    }
    constexpr uint32_t get_shard_grid_outer_dim() const {
        return (!transposed_grid * shard_grid_height) + (transposed_grid * shard_grid_width);
    }
};

template <typename T>
constexpr std::pair<T, T> flat_index_to_2d(std::uint32_t index, T inner_dim_size) {
    std::uint32_t outer_dim_index = index / inner_dim_size;
    std::uint32_t inner_dim_index = index - (outer_dim_index * inner_dim_size);

    return std::make_pair(inner_dim_index, outer_dim_index);
}

/*
 * Implements a tensor global page_id to noc address lookup, for width sharded tensors
 * Doesn't assume anything about padding and only operates on whole page boundaries.
 */
template <typename worker_to_noc_lookup_t, typename DEVICE_SHARD_SPEC_T>
struct WidthShardedAddressGenerator {
    worker_to_noc_lookup_t worker_to_noc_lookup;
    DEVICE_SHARD_SPEC_T tensor_shard_spec;
    uint32_t page_size;
    uint32_t bank_base_address;

public:
    constexpr WidthShardedAddressGenerator(worker_to_noc_lookup_t lookup,
                                           DEVICE_SHARD_SPEC_T const& tensor_shard_spec,
                                           uint32_t page_size,
                                           uint32_t base_address)
        : worker_to_noc_lookup(lookup),
          tensor_shard_spec(tensor_shard_spec),
          page_size(page_size),
          bank_base_address(base_address) {}

    /*
     * This function is an alternative API that allows the caller to implement a more efficient traversal/iteration of
     * their tensor by returning the noc address of the specified `global_page_id` along with the number of contiguous
     * pages starting at this address until the end of the logical row, in the same bank. This can be more performant
     * for several reasons:
     *
     * 1. Fewer noc commands required
     * 2. Fewer calls to this function
     *
     * For example, consider a shard size of [3,4] pages per shard, where the `global_page_id` resolves to an address
     * pointing to the page marked with 'x' below. In this case, the function would return the address of the page
     * marked with 'x' and 3, for the 3 contiguous pages, starting at 'x' until the end of the row. ┌─┬─┬─┬─┐ │ │ │ │ │
     *  ├─┼─┼─┼─┤
     *  │ │x│ │ │
     *  ├─┼─┼─┼─┤
     *  │ │ │ │ │
     *  └─┴─┴─┴─┘
     *     ◄───►
     *     3 contiguous pages until end of row
     */
    test_shard_location_with_contig_t get_page_location_with_contiguous_pages_in_row_in_bank(
        std::uint32_t global_page_id) const {
        // With width sharding, the tensor is fractured along width, but can be mapped onto a 2D grid, in such a case
        // the logical tensor is still fractured only along 1 dimension but the placement/allocation snakes through the
        // grid.
        std::uint32_t global_pages_per_row_logical = tensor_shard_spec.get_pages_per_tensor_x();

        std::uint32_t page_global_outer_dim = global_page_id / global_pages_per_row_logical;
        std::uint32_t page_global_inner_dim = global_page_id - (page_global_outer_dim * global_pages_per_row_logical);

        // might be able to save on some divides here if we can multiply out the pages per shard_x and shards per row
        // ... think about it
        // likewise maybe we can also take it a step further and do the same sort of thing above too to get away with a
        // single divide for the full function
        std::uint32_t global_shard_index = page_global_inner_dim / tensor_shard_spec.get_pages_per_shard_x();

        auto [shard_grid_inner_dim_index, shard_grid_outer_dim_index] =
            flat_index_to_2d<std::uint32_t>(global_shard_index, tensor_shard_spec.get_shard_grid_inner_dim());

        std::uint32_t worker_y_offset = (!tensor_shard_spec.transposed_grid * shard_grid_outer_dim_index) +
                                        (tensor_shard_spec.transposed_grid * shard_grid_inner_dim_index);
        std::uint32_t worker_x_offset = (!tensor_shard_spec.transposed_grid * shard_grid_inner_dim_index) +
                                        (tensor_shard_spec.transposed_grid * shard_grid_outer_dim_index);

        std::uint32_t page_in_shard_x =
            page_global_inner_dim - (global_shard_index * tensor_shard_spec.get_pages_per_shard_x());
        std::uint32_t page_in_shard_y = page_global_outer_dim;
        std::uint32_t page_offset_in_shard =
            (page_global_outer_dim * tensor_shard_spec.get_pages_per_shard_x()) + page_in_shard_x;

        std::uint32_t worker_x_logical = tensor_shard_spec.shard_grid_start_x_logical + worker_x_offset;
        std::uint32_t worker_y_logical = tensor_shard_spec.shard_grid_start_y_logical + worker_y_offset;

        noc_grid_index_t noc_x = worker_to_noc_lookup.get_noc_x_from_worker_x(worker_x_logical);
        noc_grid_index_t noc_y = worker_to_noc_lookup.get_noc_y_from_worker_y(worker_y_logical);

        return test_shard_location_with_contig_t{device_core_location_t{noc_y, noc_x},
                                                 page_offset_in_shard,
                                                 tensor_shard_spec.get_pages_per_shard_x() - page_in_shard_x};
    }

    /*
     * Return the noc address for the specified global page id, where global_page_id is a flat index of the page if
     * iterating through the tensor in a row-major order.
     */
    test_shard_location_t get_page_location(std::uint32_t global_page_id) const {
        auto const& result = get_page_location_with_contiguous_pages_in_row_in_bank(global_page_id);
        return test_shard_location_t{result.core_location, result.page_offset};
    }

    // Upon support of macros that indicate if compiling for host or device, we can enable this by stubbing out
    // `get_noc_addr` uint64_t get_noc_addr(std::uint32_t global_page_id, std::uint32_t offset = 0) const {
    //     auto const&[noc_yx, page_offset] = this->get_page_location(global_page_id);
    //     return get_noc_addr(static_cast<uint32_t>(noc_yx.noc_x), noc_yx.noc_y, this->bank_base_address + (page_offset
    //     * this->page_size) + offset);
    // }
};

/*
 * Height sharded tensor spec - acts as helper for WidthShardedAddressGenerator
 */
struct DeviceHeightShardSpec : public device_shard_spec_t<DeviceHeightShardSpec> {
    constexpr DeviceHeightShardSpec(uint16_t pages_per_shard_y,
                                    uint16_t pages_per_shard_x,
                                    uint8_t shard_grid_height,
                                    uint8_t shard_grid_width,
                                    uint8_t shard_grid_start_y_logical,
                                    uint8_t shard_grid_start_x_logical,
                                    bool transposed_grid)
        : device_shard_spec_t<DeviceHeightShardSpec>(shard_grid_height,
                                                     shard_grid_width,
                                                     shard_grid_start_y_logical,
                                                     shard_grid_start_x_logical,
                                                     transposed_grid),
          pages_per_shard_y(pages_per_shard_y),
          pages_per_shard_x(pages_per_shard_x) {}

    uint16_t pages_per_shard_y;
    uint16_t pages_per_shard_x;

    constexpr uint32_t get_pages_per_shard_x() const {
        return pages_per_shard_x;
    }
    constexpr uint32_t get_pages_per_shard_y() const {
        return pages_per_shard_y;
    }
    constexpr uint32_t get_pages_per_tensor_x() const {
        return pages_per_shard_x;
    }
    constexpr uint32_t get_pages_per_tensor_y() const {
        return pages_per_shard_y * get_shard_grid_num_cores();
    }
    constexpr uint32_t get_shard_grid_inner_dim() const {
        return (!transposed_grid * shard_grid_height) + (transposed_grid * shard_grid_width);
    }
    constexpr uint32_t get_shard_grid_outer_dim() const {
        return (!transposed_grid * shard_grid_width) + (transposed_grid * shard_grid_height);
    }
};

/*
 * Implements a tensor global page_id to noc address lookup, for height sharded tensors
 * Doesn't assume anything about padding and only operates on whole page boundaries.
 */
template <typename worker_to_noc_lookup_t, typename DEVICE_SHARD_SPEC_T>
struct HeightShardedAddressGenerator {
    worker_to_noc_lookup_t worker_to_noc_lookup;
    DEVICE_SHARD_SPEC_T tensor_shard_spec;
    uint32_t page_size;
    uint32_t bank_base_address;

public:
    constexpr HeightShardedAddressGenerator(worker_to_noc_lookup_t lookup,
                                            DEVICE_SHARD_SPEC_T const& tensor_shard_spec,
                                            uint32_t page_size,
                                            uint32_t base_address)
        : worker_to_noc_lookup(lookup),
          tensor_shard_spec(tensor_shard_spec),
          page_size(page_size),
          bank_base_address(base_address) {}

    /*
     * This function is an alternative API that allows the caller to implement a more efficient traversal/iteration of
     * their tensor by returning the noc address of the specified `global_page_id` along with the number of contiguous
     * pages starting at this address until the end of the logical row, in the same bank. This can be more performant
     * for several reasons:
     *
     * 1. Fewer noc commands required
     * 2. Fewer calls to this function
     *
     * For example, consider a shard size of [3,4] pages per shard, where the `global_page_id` resolves to an address
     * pointing to the page marked with 'x' below. In this case, the function would return the address of the page
     * marked with 'x' and 3, for the 3 contiguous pages, starting at 'x' until the end of the row. ┌─┬─┬─┬─┐ │ │ │ │ │
     *  ├─┼─┼─┼─┤
     *  │ │x│ │ │
     *  ├─┼─┼─┼─┤
     *  │ │ │ │ │
     *  └─┴─┴─┴─┘
     *     ◄───►
     *     3 contiguous pages until end of row
     */
    test_shard_location_with_contig_t get_page_location_with_contiguous_pages_in_row_in_bank(
        std::uint32_t global_page_id) const {
        // With height sharding, the tensor is fractured along height, but can be mapped onto a 2D grid, in such a case
        // the logical tensor is still fractured only along 1 dimension but the placement/allocation snakes through the
        // grid.
        std::uint32_t pages_per_shard =
            tensor_shard_spec.get_pages_per_shard_y() * tensor_shard_spec.get_pages_per_shard_x();

        std::uint32_t global_shard_index = global_page_id / pages_per_shard;

        std::uint32_t page_offset_in_shard = global_page_id - (global_shard_index * pages_per_shard);

        // might be able to save on some divides here if we can multiply out the pages per shard_x and shards per row
        // ... think about it
        // likewise maybe we can also take it a step further and do the same sort of thing above too to get away with a
        // single divide for the full function
        auto [shard_grid_inner_dim_index, shard_grid_outer_dim_index] =
            flat_index_to_2d<std::uint32_t>(global_shard_index, tensor_shard_spec.get_shard_grid_inner_dim());

        std::uint32_t worker_y_offset = (!tensor_shard_spec.transposed_grid * shard_grid_inner_dim_index) +
                                        (tensor_shard_spec.transposed_grid * shard_grid_outer_dim_index);
        std::uint32_t worker_x_offset = (!tensor_shard_spec.transposed_grid * shard_grid_outer_dim_index) +
                                        (tensor_shard_spec.transposed_grid * shard_grid_inner_dim_index);

        std::uint32_t worker_x_logical = tensor_shard_spec.shard_grid_start_x_logical + worker_x_offset;
        std::uint32_t worker_y_logical = tensor_shard_spec.shard_grid_start_y_logical + worker_y_offset;

        noc_grid_index_t noc_x = worker_to_noc_lookup.get_noc_x_from_worker_x(worker_x_logical);
        noc_grid_index_t noc_y = worker_to_noc_lookup.get_noc_y_from_worker_y(worker_y_logical);

        return test_shard_location_with_contig_t{device_core_location_t{noc_y, noc_x},
                                                 page_offset_in_shard,
                                                 1};  // tensor_shard_spec.get_pages_per_shard_x() - page_in_shard_x};
    }

    /*
     * Return the noc address for the specified global page id, where global_page_id is a flat index of the page if
     * iterating through the tensor in a row-major order.
     */
    test_shard_location_t get_page_location(std::uint32_t global_page_id) const {
        auto const& result = get_page_location_with_contiguous_pages_in_row_in_bank(global_page_id);
        return test_shard_location_t{result.core_location, result.page_offset};
    }

    // Upon support of macros that indicate if compiling for host or device, we can enable this by stubbing out
    // `get_noc_addr` uint64_t get_noc_addr(std::uint32_t global_page_id, std::uint32_t offset = 0) const {
    //     auto const&[noc_yx, page_offset] = this->get_page_location(global_page_id);
    //     return get_noc_addr(static_cast<uint32_t>(noc_yx.noc_x), noc_yx.noc_y, this->bank_base_address + (page_offset
    //     * this->page_size) + offset);
    // }
};

/*
 * Block sharded tensor spec - acts as helper for WidthShardedAddressGenerator
 */
struct DeviceBlockShardSpec : public device_shard_spec_t<DeviceBlockShardSpec> {
    constexpr DeviceBlockShardSpec(uint16_t pages_per_shard_y,
                                   uint16_t pages_per_shard_x,
                                   uint8_t shard_grid_height,
                                   uint8_t shard_grid_width,
                                   uint8_t shard_grid_start_y_logical,
                                   uint8_t shard_grid_start_x_logical,
                                   bool transposed_grid)
        : device_shard_spec_t<DeviceBlockShardSpec>(shard_grid_height,
                                                    shard_grid_width,
                                                    shard_grid_start_y_logical,
                                                    shard_grid_start_x_logical,
                                                    transposed_grid),
          pages_per_shard_y(pages_per_shard_y),
          pages_per_shard_x(pages_per_shard_x) {}

    uint16_t pages_per_shard_y;
    uint16_t pages_per_shard_x;

    constexpr uint32_t get_pages_per_shard_x() const {
        return pages_per_shard_x;
    }
    constexpr uint32_t get_pages_per_shard_y() const {
        return pages_per_shard_y;
    }
    constexpr uint32_t get_pages_per_tensor_x() const {
        return pages_per_shard_x * get_shard_grid_width();
    }
    constexpr uint32_t get_pages_per_tensor_y() const {
        return pages_per_shard_y * get_shard_grid_height();
    }
    constexpr uint32_t get_shard_grid_inner_dim() const {
        return (!transposed_grid * shard_grid_width) + (transposed_grid * shard_grid_height);
    }
    constexpr uint32_t get_shard_grid_outer_dim() const {
        return (!transposed_grid * shard_grid_height) + (transposed_grid * shard_grid_width);
    }
};

/*
 * Implements a tensor global page_id to noc address lookup, for block sharded tensors
 * Doesn't assume anything about padding and only operates on whole page boundaries.
 */
template <typename worker_to_noc_lookup_t, typename DEVICE_SHARD_SPEC_T>
struct BlockShardedAddressGenerator {
    worker_to_noc_lookup_t worker_to_noc_lookup;
    DEVICE_SHARD_SPEC_T tensor_shard_spec;
    uint32_t page_size;
    uint32_t bank_base_address;

public:
    constexpr BlockShardedAddressGenerator(worker_to_noc_lookup_t lookup,
                                           DEVICE_SHARD_SPEC_T const& tensor_shard_spec,
                                           uint32_t page_size,
                                           uint32_t base_address)
        : worker_to_noc_lookup(lookup),
          tensor_shard_spec(tensor_shard_spec),
          page_size(page_size),
          bank_base_address(base_address) {}

    /*
     * This function is an alternative API that allows the caller to implement a more efficient traversal/iteration of
     * their tensor by returning the noc address of the specified `global_page_id` along with the number of contiguous
     * pages starting at this address until the end of the logical row, in the same bank. This can be more performant
     * for several reasons:
     *
     * 1. Fewer noc commands required
     * 2. Fewer calls to this function
     *
     * For example, consider a shard size of [3,4] pages per shard, where the `global_page_id` resolves to an address
     * pointing to the page marked with 'x' below. In this case, the function would return the address of the page
     * marked with 'x' and 3, for the 3 contiguous pages, starting at 'x' until the end of the row. ┌─┬─┬─┬─┐ │ │ │ │ │
     *  ├─┼─┼─┼─┤
     *  │ │x│ │ │
     *  ├─┼─┼─┼─┤
     *  │ │ │ │ │
     *  └─┴─┴─┴─┘
     *     ◄───►
     *     3 contiguous pages until end of row
     */
    test_shard_location_with_contig_t get_page_location_with_contiguous_pages_in_row_in_bank(
        std::uint32_t global_page_id) const {
        // With block sharding, the tensor is fractured along height and width.

        // For now we don't support transposed grid for block sharding

        std::uint32_t global_pages_per_row_logical = tensor_shard_spec.get_pages_per_tensor_x();

        std::uint32_t page_global_outer_dim = global_page_id / global_pages_per_row_logical;
        std::uint32_t page_global_inner_dim = global_page_id - (page_global_outer_dim * global_pages_per_row_logical);

        std::uint32_t shard_grid_inner_dim_index = page_global_inner_dim / tensor_shard_spec.get_pages_per_shard_x();
        std::uint32_t shard_grid_outer_dim_index = page_global_outer_dim / tensor_shard_spec.get_pages_per_shard_y();

        std::uint32_t page_offset_in_shard_x =
            page_global_inner_dim - (shard_grid_inner_dim_index * tensor_shard_spec.get_pages_per_shard_x());
        std::uint32_t page_offset_in_shard_y =
            page_global_outer_dim - (shard_grid_outer_dim_index * tensor_shard_spec.get_pages_per_shard_y());

        std::uint32_t page_offset_in_shard =
            (page_offset_in_shard_y * tensor_shard_spec.get_pages_per_shard_x()) + page_offset_in_shard_x;

        // might be able to save on some divides here if we can multiply out the pages per shard_x and shards per row
        // ... think about it
        // likewise maybe we can also take it a step further and do the same sort of thing above too to get away with a
        // single divide for the full function

        std::uint32_t worker_y_offset = (!tensor_shard_spec.transposed_grid * shard_grid_outer_dim_index) +
                                        (tensor_shard_spec.transposed_grid * shard_grid_inner_dim_index);
        std::uint32_t worker_x_offset = (!tensor_shard_spec.transposed_grid * shard_grid_inner_dim_index) +
                                        (tensor_shard_spec.transposed_grid * shard_grid_outer_dim_index);

        std::uint32_t worker_x_logical = tensor_shard_spec.shard_grid_start_x_logical + worker_x_offset;
        std::uint32_t worker_y_logical = tensor_shard_spec.shard_grid_start_y_logical + worker_y_offset;

        noc_grid_index_t noc_x = worker_to_noc_lookup.get_noc_x_from_worker_x(worker_x_logical);
        noc_grid_index_t noc_y = worker_to_noc_lookup.get_noc_y_from_worker_y(worker_y_logical);

        return test_shard_location_with_contig_t{device_core_location_t{noc_y, noc_x},
                                                 page_offset_in_shard,
                                                 tensor_shard_spec.get_pages_per_shard_x() - page_offset_in_shard_x};
    }

    /*
     * Return the noc address for the specified global page id, where global_page_id is a flat index of the page if
     * iterating through the tensor in a row-major order.
     */
    test_shard_location_t get_page_location(std::uint32_t global_page_id) const {
        auto const& result = get_page_location_with_contiguous_pages_in_row_in_bank(global_page_id);
        return test_shard_location_t{result.core_location, result.page_offset};
    }

    // Upon support of macros that indicate if compiling for host or device, we can enable this by stubbing out
    // `get_noc_addr` uint64_t get_noc_addr(std::uint32_t global_page_id, std::uint32_t offset = 0) const {
    //     auto const&[noc_yx, page_offset] = this->get_page_location(global_page_id);
    //     return get_noc_addr(static_cast<uint32_t>(noc_yx.noc_x), noc_yx.noc_y, this->bank_base_address + (page_offset
    //     * this->page_size) + offset);
    // }
};

/*
 * Implement the global `get_noc_addr` interface overload that takes a page ID and returns noc address, but for the
 * ShardSpecAddressGenerator
 */
template <typename worker_to_noc_lookup_t, typename ShardedAddrgenT>
inline std::uint64_t get_noc_addr(const uint32_t id, const ShardedAddrgenT& s, uint32_t offset = 0) {
    return s.get_noc_addr(id, offset);
}

/*
 * Below are several type resolving helpers to streamline address generator creation and to start
 * unifiyng instantiation of sharded address generators with interleaved address generators.
 */
template <TensorMemoryLayout layout>
struct is_sharded_layout {
    static constexpr bool value = false;
};
template <>
struct is_sharded_layout<TensorMemoryLayout::BLOCK_SHARDED> {
    static constexpr bool value = true;
};
template <>
struct is_sharded_layout<TensorMemoryLayout::WIDTH_SHARDED> {
    static constexpr bool value = true;
};
template <>
struct is_sharded_layout<TensorMemoryLayout::HEIGHT_SHARDED> {
    static constexpr bool value = true;
};

template <TensorMemoryLayout layout, typename worker_to_noc_lookup_t, typename DEVICE_SHARD_SPEC_T>
using sharded_addrgen_builder_t =
    std::conditional_t<layout == TensorMemoryLayout::WIDTH_SHARDED,
                       WidthShardedAddressGenerator<worker_to_noc_lookup_t, DEVICE_SHARD_SPEC_T>,
                       std::conditional_t<layout == TensorMemoryLayout::HEIGHT_SHARDED,
                                          HeightShardedAddressGenerator<worker_to_noc_lookup_t, DEVICE_SHARD_SPEC_T>,
                                          BlockShardedAddressGenerator<worker_to_noc_lookup_t, DEVICE_SHARD_SPEC_T>>>;

template <TensorMemoryLayout layout, typename worker_to_noc_lookup_t, typename DEVICE_SHARD_SPEC_T>
constexpr auto build_sharded_addr_gen(worker_to_noc_lookup_t const& worker_to_noc_lookup,
                                      DEVICE_SHARD_SPEC_T const& device_shard_spec,
                                      uint32_t page_size,
                                      uint32_t base_address)
    -> sharded_addrgen_builder_t<layout, worker_to_noc_lookup_t, DEVICE_SHARD_SPEC_T> {
    if constexpr (layout == TensorMemoryLayout::WIDTH_SHARDED) {
        return WidthShardedAddressGenerator<worker_to_noc_lookup_t, DEVICE_SHARD_SPEC_T>(
            worker_to_noc_lookup, device_shard_spec, page_size, base_address);

    } else if constexpr (layout == TensorMemoryLayout::HEIGHT_SHARDED) {
        return HeightShardedAddressGenerator<worker_to_noc_lookup_t, DEVICE_SHARD_SPEC_T>(
            worker_to_noc_lookup, device_shard_spec, page_size, base_address);
    } else {
        static_assert(layout == TensorMemoryLayout::BLOCK_SHARDED);
        return BlockShardedAddressGenerator<worker_to_noc_lookup_t, DEVICE_SHARD_SPEC_T>(
            worker_to_noc_lookup, device_shard_spec, page_size, base_address);
    }
}

template <TensorMemoryLayout layout>
struct DeviceShardSpecTypeGetter {
    using type = nullptr_t;
};
template <>
struct DeviceShardSpecTypeGetter<TensorMemoryLayout::WIDTH_SHARDED> {
    using type = DeviceWidthShardSpec;
};
template <>
struct DeviceShardSpecTypeGetter<TensorMemoryLayout::HEIGHT_SHARDED> {
    using type = DeviceHeightShardSpec;
};
template <>
struct DeviceShardSpecTypeGetter<TensorMemoryLayout::BLOCK_SHARDED> {
    using type = DeviceBlockShardSpec;
};

using DefaultWidthShardedAddressGenerator =
    WidthShardedAddressGenerator<HarvestedWormholeWorkerToNocLookup, DeviceWidthShardSpec>;
using DefaultHeightShardedAddressGenerator =
    HeightShardedAddressGenerator<HarvestedWormholeWorkerToNocLookup, DeviceHeightShardSpec>;
using DefaultBlockShardedAddressGenerator =
    BlockShardedAddressGenerator<HarvestedWormholeWorkerToNocLookup, DeviceBlockShardSpec>;

}  // namespace address_generators
}  // namespace tt_metal
}  // namespace tt
