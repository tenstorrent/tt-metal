// SPDX-FileCopyrightText: © 2023 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

// NOTE: This is the Metal 2.0 fork of writer_unary_stick_layout_interleaved_start_id.cpp, which lives
// beside it. Ops ported to Metal 2.0 bind this file; the original serves the consumers still on the
// legacy API. Until the last of them migrates and the original is retired, changes here likely belong
// there too.
//
// The binding names below (dfb::out0, tensor::dst) and the named argument set (stick_size, num_sticks,
// start_id) are this fork's interface: every later consumer inherits them, so they are taken from the
// kernel's own vocabulary rather than any one op's locals, and are not renamed once a consumer exists.

#include <stdint.h>
#include "api/dataflow/dataflow_api.h"
#include "api/dataflow/noc.h"
#include "api/dataflow/dataflow_buffer.h"
#include "api/tensor/noc_traits.h"
#include "experimental/kernel_args.h"

void kernel_main() {
    uint32_t stick_size = get_arg(args::stick_size);
    uint32_t num_sticks = get_arg(args::num_sticks);
    uint32_t start_id = get_arg(args::start_id);

    const auto s0 = TensorAccessor(tensor::dst);

    Noc noc;
    // dfb_out0 holds the sticks to drain to the destination tensor; the host binds this kernel as its
    // consumer. The write size comes from stick_size rather than the buffer's entry size, because a
    // producer may stage each stick in an allocator-aligned entry that is wider than the stick.
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
