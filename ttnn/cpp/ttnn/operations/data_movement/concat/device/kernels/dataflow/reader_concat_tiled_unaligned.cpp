// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include <stdint.h>
#include "api/dataflow/dataflow_api.h"
#include "api/dataflow/noc.h"
#include "api/dataflow/dataflow_buffer.h"
#include "api/core_local_mem.h"
#include "api/tensor/noc_traits.h"

// Streams tile-rows ("bands") of every input into cb_in, band-major then input-major:
// for each band, input 0's tiles, then input 1's, etc. The compute kernel untilizes them
// in the same order.
void kernel_main() {
    constexpr uint32_t cb_in = get_compile_time_arg_val(0);
    constexpr uint32_t num_tensors = get_compile_time_arg_val(1);
    constexpr auto tensor_accessor_args = make_tensor_accessor_args_tuple<num_tensors, 2>();

    const uint32_t num_bands = get_arg_val<uint32_t>(0);
    const uint32_t band_start = get_arg_val<uint32_t>(1);
    constexpr uint32_t addr_base_idx = 2;

    auto tensor_accessors_tuple = make_tensor_accessor_tuple(tensor_accessor_args, addr_base_idx);
    auto accessors = make_abstract_tensor_accessor_wrappers(tensor_accessors_tuple);

    uint32_t wt[num_tensors];
    tt_l1_ptr uint32_t* arg_ptr = (tt_l1_ptr uint32_t*)get_arg_addr(addr_base_idx + num_tensors);
    for (uint32_t i = 0; i < num_tensors; ++i) {
        wt[i] = arg_ptr[i];
    }

    const uint32_t tile_bytes = get_tile_size(cb_in);

    DataflowBuffer dfb_in(cb_in);
    Noc noc;

    for (uint32_t b = 0; b < num_bands; ++b) {
        const uint32_t band = band_start + b;
        for (uint32_t i = 0; i < num_tensors; ++i) {
            const uint32_t band_first_tile = band * wt[i];
            dfb_in.reserve_back(wt[i]);
            uint32_t l1_write_addr = dfb_in.get_write_ptr();
            for (uint32_t t = 0; t < wt[i]; ++t) {
                noc.async_read(
                    accessors[i],
                    CoreLocalMem<uint8_t>(l1_write_addr),
                    tile_bytes,
                    {.page_id = band_first_tile + t},
                    {});
                l1_write_addr += tile_bytes;
            }
            noc.async_read_barrier();
            dfb_in.push_back(wt[i]);
        }
    }
}
