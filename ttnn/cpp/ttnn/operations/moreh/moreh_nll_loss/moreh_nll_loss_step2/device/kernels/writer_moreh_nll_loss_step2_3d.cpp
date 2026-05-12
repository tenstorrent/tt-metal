// SPDX-FileCopyrightText: © 2023 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include "api/dataflow/dataflow_api.h"
#include "ttnn/kernel/dataflow/moreh_common.hpp"
#include "experimental/noc.h"
#include "experimental/circular_buffer.h"
#include "experimental/tensor.h"

void kernel_main() {
    using namespace tt::constants;
    uint32_t i = 0;
    auto output_addr = get_arg_val<uint32_t>(i++);
    auto num_tiles_per_core = get_arg_val<uint32_t>(i++);
    auto start_id = get_arg_val<uint32_t>(i++);
    auto W = get_arg_val<uint32_t>(i++);
    auto element_size = get_arg_val<uint32_t>(i++);

    constexpr uint32_t cb_output = tt::CBIndex::c_16;

    constexpr auto output_args = TensorAccessorArgs<0>();

    const auto output_addrg = TensorAccessor(output_args, output_addr);

    uint32_t Wf = (W + FACE_WIDTH - 1) / FACE_WIDTH;
    uint32_t Wt = (W + TILE_WIDTH - 1) / TILE_WIDTH;

    constexpr uint32_t onetile = 1;

    experimental::Noc noc;
    experimental::CircularBuffer cb_out(cb_output);

    uint32_t end_id = start_id + num_tiles_per_core;
    for (uint32_t i = start_id; i < end_id; ++i) {
        cb_out.wait_front(onetile);
        uint32_t n = i / Wf;
        uint32_t w = (i % Wf) * FACE_WIDTH;
        uint32_t nt = n / TILE_HEIGHT;
        uint32_t wt = w / TILE_WIDTH;

        uint32_t noc_id = nt * Wt + wt;
        uint32_t noc_offset;
        get_noc_offset(n, w, element_size, noc_offset);

        noc.async_write(
            cb_out,
            output_addrg,
            NOC_MINIMUM_READ_SIZE,
            {.offset_bytes = 0},
            {.page_id = noc_id, .offset_bytes = noc_offset});
        noc.async_write_barrier();

        cb_out.pop_front(onetile);
    }
}
