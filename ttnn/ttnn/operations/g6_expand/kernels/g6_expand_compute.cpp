// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

// g6_expand has no compute stage and the program descriptor does not dispatch this kernel: every
// output byte is a NoC byte copy of x's row 0 (reader + writer). Routing the values through
// unpack/DEST/pack could change bits (denormal flush, NaN payloads), which the bit-exact contract
// forbids, and an idle TRISC kernel would only add launch cost and kernel-config footprint.

void kernel_main() {}
