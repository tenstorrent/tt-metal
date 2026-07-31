// SPDX-FileCopyrightText: © 2023 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

// Metal 2.0 fork of writer_unary_sharded.cpp.
//
// Same logic, expressed against the Metal 2.0 named-binding APIs: the output CB index CTA became the
// `dfb::out` DFB binding and the unit-count RTA became a named argument.
//
// This fork exists because the legacy original is bound by several op directories that cannot all
// convert at once; it lives alongside the original rather than replacing it.
//
// Binding vocabulary a Metal 2.0 KernelSpec must supply for this source:
//   dfb::out        — the output DFB, bound CONSUMER
//   args::num_units — runtime arg

#include <stdint.h>
#include "api/dataflow/dataflow_api.h"
#include "api/dataflow/dataflow_buffer.h"
#include "experimental/kernel_args.h"

void kernel_main() {
    const uint32_t num_units = get_arg(args::num_units);

    DataflowBuffer dfb_out(dfb::out);

    dfb_out.wait_front(num_units);
    // Output is sharded in place, so the data is already where it needs to be; the
    // wait above is only a readiness handshake. Pop to leave the DFB balanced.
    dfb_out.pop_front(num_units);
}
