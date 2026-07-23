// SPDX-FileCopyrightText: © 2023 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

// Metal 2.0 fork of writer_unary_stick_layout_interleaved_start_id.cpp. Identical dataflow logic; the
// output CB index becomes a dfb:: binding, the destination TensorAccessor a tensor:: binding (the
// dst_addr runtime arg is gone), and the remaining runtime args become named. The dead legacy CTA at
// index 1 (the passed-but-unread output page size) is dropped. The legacy copy is retained for the
// not-yet-ported consumers (data_movement/concat, data_movement/slice).

#include <stdint.h>
#include "api/dataflow/dataflow_api.h"
#include "api/dataflow/noc.h"
#include "api/dataflow/dataflow_buffer.h"
#include "api/tensor/noc_traits.h"
#include "experimental/kernel_args.h"

void kernel_main() {
    auto stick_size = get_arg(args::stick_size);
    auto num_sticks = get_arg(args::num_sticks);
    auto start_id = get_arg(args::start_id);

    const auto s0 = TensorAccessor(tensor::output);

    Noc noc;
    DataflowBuffer dfb_out0(dfb::out0);

#ifdef BACKWARDS
    uint32_t end_id = start_id - num_sticks;
    for (uint32_t i = start_id; i != end_id; --i) {
#else
    uint32_t end_id = start_id + num_sticks;
    for (uint32_t i = start_id; i < end_id; ++i) {
#endif
        dfb_out0.wait_front(1);
        noc.async_write(dfb_out0, s0, stick_size, {.offset_bytes = 0}, {.page_id = i, .offset_bytes = 0});
        noc.async_write_barrier();
        dfb_out0.pop_front(1);
    }
}
