// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
// SPDX-License-Identifier: Apache-2.0

// nomic_moe_dispatch reader_noc0 (RISCV_0 / NOC0): gathers rows [0, rows_noc0) of every block of this core's output
// shards into cb_rows_head. The body is shared with reader_noc1 (nomic_moe_dispatch_gather.hpp); the row range, CB
// and ids span come from the named CT args.

#include "nomic_moe_dispatch_gather.hpp"

void kernel_main() { nomic_moe_dispatch::gather_reader(); }
