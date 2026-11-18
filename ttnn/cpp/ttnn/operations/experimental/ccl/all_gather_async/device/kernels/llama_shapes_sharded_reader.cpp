// SPDX-FileCopyrightText: © 2025 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include "api/dataflow/dataflow_api.h"
#include "api/dataflow/noc.h"
#include "api/dataflow/dataflow_buffer.h"
#include "api/dataflow/noc_semaphore.h"
#include "api/dataflow/endpoints.h"
#include "api/core_local_mem.h"
#include "api/buffer_types.hpp"
#include "api/tensor/local_tensor_accessor.h"
#include "internal/local_mem/address_access.h"
#include "experimental/kernel_args.h"
#include <cstdint>
#include <utility>

using address_t = uint32_t;

///////////////////////////////////////////////////
// COMPILE TIME ARGS
///////////////////////////////////////////////////

constexpr uint32_t my_chip_id = get_arg(args::my_chip_id);
constexpr uint32_t tensor0_page_size = get_arg(args::tensor0_page_size);

/*
 * CCL Send will present various operating modes. Although there is only a single send kernel, it may (compile time)
 * dispatch implementations depending on those invocation parameters.
 */
void kernel_main() {
    ///////////////////////////////////////////////////
    // ARGS
    ///////////////////////////////////////////////////

    // Load the input tensor spec
    const LocalTensorAccessor<uint8_t> input(tensor::input);
    address_t tensor_address0 = static_cast<address_t>(local_mem::detail::AddressAccess::address(input.local_mem()));
    uint32_t num_tiles_per_core = get_arg(args::num_tiles_per_core);
    uint32_t num_tiles_to_read = get_arg(args::num_tiles_to_read);
    uint32_t first_core_tile_start_offset = get_arg(args::first_core_tile_start_offset);
    uint32_t num_cores = get_arg(args::num_cores);
    // The input cores' NoC x coordinates are varargs [0, num_cores), their y coordinates [num_cores, 2 * num_cores).

    // interleaved addrgen

    Noc noc_obj;
    DataflowBuffer dfb0(dfb::data);

    uint32_t tiles_read = 0;
    uint32_t shard_tile_id = first_core_tile_start_offset;
    uint32_t core_id = 0;
    while (tiles_read < num_tiles_to_read) {
        uint32_t num_tiles_to_read_this_core =
            std::min(num_tiles_per_core - shard_tile_id, num_tiles_to_read - tiles_read);
        dfb0.reserve_back(num_tiles_to_read_this_core);
        const uint32_t l1_write_addr = dfb0.get_write_ptr();
        const uint32_t shard_addr = tensor_address0 + shard_tile_id * tensor0_page_size;

        noc_obj.async_read(
            UnicastEndpoint{},
            CoreLocalMem<uint8_t>(l1_write_addr),
            num_tiles_to_read_this_core * tensor0_page_size,
            {.noc_x = get_vararg(core_id), .noc_y = get_vararg(num_cores + core_id), .addr = shard_addr},
            {});
        noc_obj.async_read_barrier();

        dfb0.push_back(num_tiles_to_read_this_core);
        tiles_read += num_tiles_to_read_this_core;
        shard_tile_id = 0;
        core_id++;
    }
}
