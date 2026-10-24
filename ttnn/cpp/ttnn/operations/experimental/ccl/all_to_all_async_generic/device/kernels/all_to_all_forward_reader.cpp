// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include "api/dataflow/dataflow_api.h"
#include "api/dataflow/noc.h"
#include "api/dataflow/noc_semaphore.h"
#include "api/dataflow/circular_buffer.h"
#include "api/core_local_mem.h"
#include "api/tensor/noc_traits.h"
#include <cstdint>

using address_t = uint32_t;

constexpr uint32_t cb0_id = get_compile_time_arg_val(0);
constexpr uint32_t page_size = get_compile_time_arg_val(1);
constexpr uint32_t max_pages_per_packet = get_compile_time_arg_val(2);
constexpr uint32_t concat_dim_size = get_compile_time_arg_val(3);
constexpr uint32_t inner_dims_size = get_compile_time_arg_val(4);
constexpr uint32_t concat_num_tiles = get_compile_time_arg_val(5);
constexpr uint32_t full_block_offset = get_compile_time_arg_val(6);
constexpr auto staging_tensor_args = TensorAccessorArgs<7>();

void kernel_main() {
    const address_t staging_address = get_common_arg_val<address_t>(0);
    const uint32_t forward_semaphore_addr = get_common_arg_val<uint32_t>(1);

    size_t arg_idx = 0;
    const uint32_t expected_completions = get_arg_val<uint32_t>(arg_idx++);
    const uint32_t num_ranges = get_arg_val<uint32_t>(arg_idx++);

    auto staging_addrgen = TensorAccessor(staging_tensor_args, staging_address);
    Noc noc_obj;
    CircularBuffer cb0(cb0_id);

    auto* forward_semaphore_ptr = reinterpret_cast<volatile tt_l1_ptr uint32_t*>(forward_semaphore_addr);
    noc_semaphore_wait(forward_semaphore_ptr, expected_completions);
    noc_semaphore_set(forward_semaphore_ptr, 0);

    for (uint32_t range = 0; range < num_ranges; ++range) {
        uint32_t block_idx = get_arg_val<uint32_t>(arg_idx++);
        const uint32_t block_end_id = get_arg_val<uint32_t>(arg_idx++);
        const uint32_t block_stride = get_arg_val<uint32_t>(arg_idx++);
        while (block_idx < block_end_id) {
            cb0.reserve_back(1);
            CoreLocalMem<uint8_t> dst(cb0.get_write_ptr());
            for (uint32_t page = 0; page < max_pages_per_packet && block_idx < block_end_id; ++page) {
                const uint32_t o = block_idx / (concat_num_tiles * inner_dims_size);
                const uint32_t c = (block_idx / inner_dims_size) % concat_num_tiles;
                const uint32_t i = block_idx % inner_dims_size;
                const uint32_t tile_id =
                    o * inner_dims_size * concat_dim_size + (c + full_block_offset) * inner_dims_size + i;
                noc_obj.async_read(
                    staging_addrgen, dst, page_size, {.page_id = tile_id}, {.offset_bytes = page * page_size});
                block_idx += block_stride;
            }
            noc_obj.async_read_barrier();
            cb0.push_back(1);
        }
    }
}
