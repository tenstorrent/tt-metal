// SPDX-FileCopyrightText: © 2023 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include <cstdint>
#include <array>

using noc_grid_index_t = std::uint8_t;



template <typename ArchSpecificWorkerToNocLookup>
struct WorkerToNocCoordLookup {
    noc_grid_index_t get_noc_x_from_worker_x(noc_grid_index_t worker_x) const {
        return ArchSpecificWorkerToNocLookup::get_noc_x_from_worker_x(worker_x);
    }

    noc_grid_index_t get_noc_y_from_worker_y(noc_grid_index_t worker_y) const {
        return ArchSpecificWorkerToNocLookup::get_noc_y_from_worker_y(worker_y);
    }
};

struct UnharvestedWormholeWorkerToNocLookup : WorkerToNocCoordLookup<UnharvestedWormholeWorkerToNocLookup>{
    static constexpr std::array<noc_grid_index_t, 8> worker_to_routing_x = {
        1,2,3,4,6,7,8,9
    };
    static constexpr std::array<noc_grid_index_t, 10> worker_to_routing_y = {
        1,2,3,4,5,7,8,9,10,11
    };

    noc_grid_index_t get_noc_x_from_worker_x(noc_grid_index_t worker_x) const {
        // ASSERT worker_x < worker_to_routing_x_wormhole.size()
        return worker_to_routing_x[worker_x];
    }

    noc_grid_index_t get_noc_y_from_worker_y(noc_grid_index_t worker_y) const {
        // ASSERT worker_y < worker_to_routing_y_wormhole.size()
        return worker_to_routing_y[worker_y];
    }
};

struct HarvestedWormholeWorkerToNocLookup : WorkerToNocCoordLookup<UnharvestedWormholeWorkerToNocLookup>{
    HarvestedWormholeWorkerToNocLookup(uint32_t nrows, const uint32_t *const row_map, uint32_t ncols, const uint32_t *const col_map) :
        nrows(nrows), row_map(row_map), ncols(ncols), col_map(col_map) {}

    noc_grid_index_t get_noc_x_from_worker_x(noc_grid_index_t worker_x) const {
        // ASSERT worker_x < worker_to_routing_x_wormhole.size()
        return col_map[worker_x];
    }

    noc_grid_index_t get_noc_y_from_worker_y(noc_grid_index_t worker_y) const {
        // ASSERT worker_y < worker_to_routing_y_wormhole.size()
        return row_map[worker_y];
    }

    uint32_t nrows;
    const uint32_t *const row_map;
    uint32_t ncols;
    const uint32_t *const col_map;
};


struct device_core_location_t {
    noc_grid_index_t noc_y;
    noc_grid_index_t noc_x;
};
struct test_shard_location_t {
    device_core_location_t core_location;
    std::uint32_t page_offset;
};
struct device_shard_spec_t {
    uint8_t shard_grid_height;
    uint8_t shard_grid_width;
    uint8_t shard_grid_start_y_logical;
    uint8_t shard_grid_start_x_logical;
    uint8_t pages_per_shard;
    bool transposed_grid;
};

template <typename worker_to_noc_lookup_t>
struct WidthShardedAddressGenerator {
    worker_to_noc_lookup_t worker_to_noc_lookup;
    device_shard_spec_t tensor_shard_spec;
    uint32_t page_size;
    uint32_t bank_base_address;

   public:
    WidthShardedAddressGenerator(worker_to_noc_lookup_t lookup, device_shard_spec_t const& tensor_shard_spec, uint32_t page_size, uint32_t base_address) : worker_to_noc_lookup(lookup), tensor_shard_spec(tensor_shard_spec), page_size(page_size), bank_base_address(base_address) {}

    test_shard_location_t get_page_location(std::size_t global_page_id) const {
        // branchless
        std::size_t inner_shard_grid_dim_size =
            ((tensor_shard_spec.transposed_grid * tensor_shard_spec.shard_grid_height) +
             (!tensor_shard_spec.transposed_grid * tensor_shard_spec.shard_grid_width));
        // branchless
        std::size_t outer_shard_grid_dim_size =
            ((!tensor_shard_spec.transposed_grid * tensor_shard_spec.shard_grid_height) +
             (tensor_shard_spec.transposed_grid * tensor_shard_spec.shard_grid_width));

        std::size_t pages_per_shard_grid_inner_dim =tensor_shard_spec.pages_per_shard * inner_shard_grid_dim_size;

        // Sorry for the division... but for now lets get something out the door. We can easily optimize this for power of 2
        // grid sizes (which are probably common)
        std::size_t shard_index_outer_dim = global_page_id / pages_per_shard_grid_inner_dim;
        std::size_t shard_index_inner_dim = (global_page_id - (shard_index_outer_dim * pages_per_shard_grid_inner_dim)) / tensor_shard_spec.pages_per_shard;
        std::size_t page_offset_in_shard = global_page_id - ((shard_index_outer_dim * pages_per_shard_grid_inner_dim) + (shard_index_inner_dim * tensor_shard_spec.pages_per_shard));

        std::size_t worker_x_offset = (!tensor_shard_spec.transposed_grid * shard_index_inner_dim) + (tensor_shard_spec.transposed_grid * shard_index_outer_dim);
        std::size_t worker_y_offset = (!tensor_shard_spec.transposed_grid * shard_index_outer_dim) + (tensor_shard_spec.transposed_grid * shard_index_inner_dim);

        std::size_t worker_x_logical = tensor_shard_spec.shard_grid_start_x_logical + worker_x_offset;
        std::size_t worker_y_logical = tensor_shard_spec.shard_grid_start_y_logical + worker_y_offset;

        noc_grid_index_t noc_x = worker_to_noc_lookup.get_noc_x_from_worker_x(worker_x_logical);
        noc_grid_index_t noc_y = worker_to_noc_lookup.get_noc_y_from_worker_y(worker_y_logical);
        return test_shard_location_t{device_core_location_t{noc_y, noc_x}, page_offset_in_shard};
    }

    // uint64_t get_noc_addr(std::size_t global_page_id, std::size_t offset = 0) const {
    //     auto const&[noc_yx, page_offset] = this->get_page_location(global_page_id);
    //     return get_noc_addr(static_cast<uint32_t>(noc_yx.noc_x), noc_yx.noc_y, this->bank_base_address + (page_offset * this->page_size) + offset);
    // }
};

template <typename worker_to_noc_lookup_t>
inline std::uint64_t get_noc_addr(const uint32_t id, const WidthShardedAddressGenerator<worker_to_noc_lookup_t>& s, uint32_t offset = 0) {
    /*
        Alternative API for getting the noc address when we are reading using a swizzled
        layout. This version assumes bank unit size can be arbitrary size. Use
        get_noc_addr(const uint32_t id, InterleavedPow2AddrGen s) for optimized algorithm in which stick size
        is a power of 2.

        id: Unique id for the bank_unit you want to read, assuming row major order. We use this to compute the
        bank for this unit of data.

        InterleavedAddrGen: Check struct for attribute definitions.
    */
    return s.get_noc_addr(id, offset);
}
