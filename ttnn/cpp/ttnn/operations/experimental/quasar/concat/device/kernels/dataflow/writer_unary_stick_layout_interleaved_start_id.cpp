// SPDX-FileCopyrightText: © 2023 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

// Quasar concat's copy of the shared Metal 2.0 writer
// ttnn/cpp/ttnn/kernel/dataflow/writer_unary_stick_layout_interleaved_start_id_metal2.cpp, kept here so
// the experimental/quasar op does not change when the shared kernel does. The binding names
// (dfb::out0, tensor::dst) and the named argument set (stick_size, num_sticks, start_id) are that
// kernel's interface.

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
