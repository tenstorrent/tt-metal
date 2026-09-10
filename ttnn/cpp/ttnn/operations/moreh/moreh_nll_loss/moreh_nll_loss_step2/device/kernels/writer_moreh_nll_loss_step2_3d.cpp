// SPDX-FileCopyrightText: © 2023 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include "api/dataflow/dataflow_api.h"
#include "ttnn/kernel/dataflow/moreh_common.hpp"
#include "api/dataflow/noc.h"
#include "api/dataflow/dataflow_buffer.h"
#include "api/tensor/noc_traits.h"
#include "experimental/kernel_args.h"

void kernel_main() {
    using namespace tt::constants;
    auto num_tiles_per_core = get_arg(args::num_tiles_per_core);
    auto start_id = get_arg(args::start_id);
    auto W = get_arg(args::W);
    auto element_size = get_arg(args::element_size);

    const auto output_addrg = TensorAccessor(tensor::output);

    uint32_t Wf = (W + FACE_WIDTH - 1) / FACE_WIDTH;
    uint32_t Wt = (W + TILE_WIDTH - 1) / TILE_WIDTH;

    constexpr uint32_t onetile = 1;

    Noc noc;
    DataflowBuffer dfb_out(dfb::output);

    uint32_t end_id = start_id + num_tiles_per_core;
    for (uint32_t i = start_id; i < end_id; ++i) {
        dfb_out.wait_front(onetile);
        uint32_t n = i / Wf;
        uint32_t w = (i % Wf) * FACE_WIDTH;
        uint32_t nt = n / TILE_HEIGHT;
        uint32_t wt = w / TILE_WIDTH;

        uint32_t noc_id = nt * Wt + wt;
        uint32_t noc_offset;
        get_noc_offset(n, w, element_size, noc_offset);

        noc.async_write(
            dfb_out,
            output_addrg,
            NOC_MINIMUM_READ_SIZE,
            {.offset_bytes = 0},
            {.page_id = noc_id, .offset_bytes = noc_offset});
        noc.async_write_barrier();

        dfb_out.pop_front(onetile);
    }
}
