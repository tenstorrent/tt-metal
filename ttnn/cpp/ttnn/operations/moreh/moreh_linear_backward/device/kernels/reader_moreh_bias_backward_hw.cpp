// SPDX-FileCopyrightText: © 2023 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include "ttnn/kernel/dataflow/moreh_common.hpp"
#include "api/dataflow/noc.h"
#include "api/dataflow/dataflow_buffer.h"
#include "api/tensor/noc_traits.h"
#include "experimental/kernel_args.h"

void kernel_main() {
    const uint32_t num_tiles = get_arg(args::num_tiles);
    const uint32_t start_id = get_arg(args::start_id);
    const uint32_t mask_h = get_arg(args::mask_h);
    const uint32_t mask_w = get_arg(args::mask_w);
    const bool do_mask_h = (get_arg(args::do_mask_h) == 1);
    const bool do_mask_w = (get_arg(args::do_mask_w) == 1);

    union {
        float f;
        uint32_t u;
    } scaler;
    scaler.f = 1.0f;
    DataflowBuffer dfb_scaler(dfb::scaler);
    fill_cb_with_value(dfb_scaler, scaler.u);

    // The mask buffer is only allocated when a mask applies, so the host binds it — and defines
    // DO_MASK_H_W — on exactly that condition. Without the binding there is no dfb::mask_h_w token
    // to name, hence the preprocessor gate around the otherwise-unchanged runtime check.
#ifdef DO_MASK_H_W
    if (do_mask_h || do_mask_w) {
        DataflowBuffer dfb_mask_h_w(dfb::mask_h_w);
        generate_mask_h_w(dfb_mask_h_w, mask_h, mask_w);
    }
#endif

    const auto s0 = TensorAccessor(tensor::src);

    Noc noc;
    DataflowBuffer dfb_in0(dfb::in0);
    const auto in0_tile_bytes = dfb_in0.get_tile_size();

    constexpr uint32_t onetile = 1;
    for (uint32_t i = start_id; i < start_id + num_tiles; i++) {
        dfb_in0.reserve_back(onetile);
        noc.async_read(s0, dfb_in0, in0_tile_bytes, {.page_id = i}, {.offset_bytes = 0});
        noc.async_read_barrier();
        dfb_in0.push_back(onetile);
    }
}
