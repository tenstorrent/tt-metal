// SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#include <cstdint>
#include "accessor/tensor_accessor.h"
#include "dataflow_api.h"

// Function to iterate over all page IDs in a shard
template <typename DSpec, typename Func>
void iterate_pages_in_shard(const TensorAccessor<DSpec>& accessor, uint32_t shard_id, Func&& process_page) {
    const auto& dspec = accessor.dspec();

    // Assume static rank
    constexpr uint32_t rank = DSpec::rank_ct;

    // Convert shard_id to shard coordinates using shard_grid
    std::array<uint32_t, rank> shard_coord;
    uint32_t remaining_shard_id = shard_id;
    for (int i = rank - 1; i >= 0; --i) {
        shard_coord[i] = remaining_shard_id % dspec.shard_grid()[i];
        remaining_shard_id /= dspec.shard_grid()[i];
    }

    // Function to convert coordinates to page_id
    auto coords_to_page_id = [&](const std::array<uint32_t, rank>& coords) -> uint32_t {
        uint32_t page_id = 0;
        for (uint32_t i = 0; i < rank; ++i) {
            page_id = page_id * dspec.tensor_shape()[i] + coords[i];
        }
        return page_id;
    };

    // Recursively iterate through all combinations of page coordinates within the shard
    std::array<uint32_t, rank> page_coord_within_shard{};
    std::array<uint32_t, rank> global_page_coord;

    auto iterate_dimension = [&](auto&& self, uint32_t dim) -> void {
        if (dim == rank) {
            // Convert shard-relative coordinates to global coordinates
            for (uint32_t i = 0; i < rank; ++i) {
                global_page_coord[i] = shard_coord[i] * dspec.shard_shape()[i] + page_coord_within_shard[i];

                // Check bounds - some shards at edges might have fewer pages
                if (global_page_coord[i] >= dspec.tensor_shape()[i]) {
                    return;  // Skip this page as it's outside tensor bounds
                }
            }

            // Convert to page_id and process
            uint32_t page_id = coords_to_page_id(global_page_coord);
            process_page(page_id);
            return;
        }

        // Iterate through all positions in current dimension
        for (uint32_t i = 0; i < dspec.shard_shape()[dim]; ++i) {
            page_coord_within_shard[dim] = i;
            self(self, dim + 1);
        }
    };

    // Start the recursive iteration
    iterate_dimension(iterate_dimension, 0);
}

// Kernel that iterates over local shards of sharded src tensor, reads each page, and writes it to the destination
// tensor.
void kernel_main() {
    auto args_src = TensorAccessorArgs<0, 0>();
    auto args_dst =
        TensorAccessorArgs<args_src.next_compile_time_args_offset(), args_src.next_common_runtime_args_offset()>();
    constexpr uint32_t base_idx_cta = args_dst.next_compile_time_args_offset();
    constexpr uint32_t base_idx_crta = args_dst.next_common_runtime_args_offset();

    constexpr uint32_t page_size = get_compile_time_arg_val(base_idx_cta);

    const uint32_t bank_base_address_src = get_common_arg_val<uint32_t>(base_idx_crta);
    const uint32_t bank_base_address_dst = get_common_arg_val<uint32_t>(base_idx_crta + 1);
    const uint32_t num_shards = get_common_arg_val<uint32_t>(base_idx_crta + 2);
    const uint32_t shard_id_stride = get_common_arg_val<uint32_t>(base_idx_crta + 3);

    const uint32_t first_shard_id = get_arg_val<uint32_t>(0);

    auto accessor_src = TensorAccessor(args_src, bank_base_address_src, page_size);
    auto accessor_dst = TensorAccessor(args_dst, bank_base_address_dst, page_size);

    for (uint32_t shard_id = first_shard_id; shard_id < num_shards; shard_id += shard_id_stride) {
        iterate_pages_in_shard(accessor_src, shard_id, [&](uint32_t page_id) {
            // TODO: local noc_addr calculation can be optimized
            ASSERT(accessor_src.is_local_page(page_id));
            noc_async_write_page(page_id, accessor_dst, accessor_src.get_noc_addr(page_id));
            noc_async_writes_flushed();
        });
    }
    noc_async_write_barrier();
}
