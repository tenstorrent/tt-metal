// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
// SPDX-License-Identifier: Apache-2.0

// nomic_moe_dispatch reader_noc1 (RISCV_1 / NOC1): gathers rows [rows_noc0, 32) of every block of this core's output
// shards into cb_rows_tail. NOC1 carries no writes in this op (the output is produced in its home L1), so the
// RISC a writer would use reads the other half of every block. Body: nomic_moe_dispatch_gather.hpp.

#include "nomic_moe_dispatch_gather.hpp"

void kernel_main() { nomic_moe_dispatch::gather_reader(); }
