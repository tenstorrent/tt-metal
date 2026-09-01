// SPDX-FileCopyrightText: © 2023 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

// KEEP IN SYNC WITH: writer_unary_sharded.cpp (this directory)
//
// This is the Metal 2.0 fork of that kernel. Same logic, expressed against the Metal 2.0
// named-binding APIs: the output CB index CTA became the `dfb::out` DFB binding and the unit-count
// RTA became a named argument. A behavioural change to either one must be mirrored in the other.
//
// The fork exists because the legacy original is bound by 8 op directories that cannot all convert at
// once; it lives alongside the original rather than replacing it. Once the last legacy consumer is
// ported, delete the original and rename this file over it.
//
// TODO(#52228): retire this duplication. The issue records why it exists, the full consumer
// list, and the sunset plan: https://github.com/tenstorrent/tt-metal/issues/52228
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
