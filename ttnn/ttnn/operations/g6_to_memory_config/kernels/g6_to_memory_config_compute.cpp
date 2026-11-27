// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

// g6_to_memory_config has no compute stage and the program descriptor launches no compute kernel:
// pages move whole over the NoC, and a pass through DEST (unpack -> pack) would re-derive the shared
// exponents of bfloat8_b / bfloat4_b pages, breaking the bit-exact contract (op_design.md, regime R8).

void kernel_main() {}
