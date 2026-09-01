// SPDX-FileCopyrightText: © 2023 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

// KEEP IN SYNC WITH: writer_unary_sharded_metal2.cpp (this directory)
//
// That file is the Metal 2.0 fork of this kernel: identical dataflow logic, with the resource
// plumbing expressed as named bindings (dfb::/args::) instead of positional compile-time and runtime
// args. Ops whose program factory has been ported to the Metal 2.0 host API bind the fork; ops still
// on the legacy host API bind this file. A behavioural change to either one must be mirrored in the
// other.
//
// The duplication is temporary. Once the last legacy consumer is ported, delete this file and rename
// the fork over it.
//
// TODO(#52228): retire this duplication. The issue records why it exists, the full consumer
// list, and the sunset plan: https://github.com/tenstorrent/tt-metal/issues/52228

#include <stdint.h>
#include "api/dataflow/dataflow_api.h"
#include "api/dataflow/dataflow_buffer.h"

void kernel_main() {
    const uint32_t num_units = get_arg_val<uint32_t>(0);

    constexpr uint32_t dfb_id_out = get_compile_time_arg_val(0);

    DataflowBuffer dfb_out(dfb_id_out);

    dfb_out.wait_front(num_units);
    // Output is sharded in place, so the data is already where it needs to be; the
    // wait above is only a readiness handshake. Pop to leave the CB balanced.
    dfb_out.pop_front(num_units);
}
