// SPDX-FileCopyrightText: © 2023 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include "api/dataflow/dataflow_api.h"
#include "api/dataflow/noc.h"
#include "api/dataflow/dataflow_buffer.h"
#include "api/tensor/noc_traits.h"
#include "experimental/kernel_args.h"

void kernel_main() {
    uint32_t num_tiles = get_arg(args::num_tiles);
    uint32_t start_id = get_arg(args::start_id);

    // single-tile ublocks
    constexpr uint32_t onetile = 1;

    Noc noc;

    // input_grad (out0) and other_grad (out1) are optional outputs. When absent, the host omits
    // the tensor binding entirely, so tensor::s0 / tensor::s1 do not exist; the HAS_INPUT_GRAD /
    // HAS_OTHER_GRAD compile-time defines (emitted by the host only when the output is present)
    // gate the accessor construction and the write block. The out0 / out1 DFBs stay bound 1P+1C.
#ifdef HAS_INPUT_GRAD
    const auto s0 = TensorAccessor(tensor::s0);
    DataflowBuffer dfb_out0(dfb::out0);
    const auto out0_tile_bytes = dfb_out0.get_entry_size();
#endif

#ifdef HAS_OTHER_GRAD
    const auto s1 = TensorAccessor(tensor::s1);
    DataflowBuffer dfb_out1(dfb::out1);
    const auto out1_tile_bytes = dfb_out1.get_entry_size();
#endif

    uint32_t end_id = start_id + num_tiles;
    for (uint32_t i = start_id; i < end_id; i++) {
#ifdef HAS_INPUT_GRAD
        dfb_out0.wait_front(onetile);
        noc.async_write(dfb_out0, s0, out0_tile_bytes, {.offset_bytes = 0}, {.page_id = i});
        noc.async_write_barrier();
        dfb_out0.pop_front(onetile);
#endif

#ifdef HAS_OTHER_GRAD
        dfb_out1.wait_front(onetile);
        noc.async_write(dfb_out1, s1, out1_tile_bytes, {.offset_bytes = 0}, {.page_id = i});
        noc.async_write_barrier();
        dfb_out1.pop_front(onetile);
#endif
    }
}
