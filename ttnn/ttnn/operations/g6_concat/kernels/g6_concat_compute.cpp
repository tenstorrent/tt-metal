// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

// Not attached to the g6_concat program. The op only moves bytes: routing a value through
// unpack / DEST / pack would risk changing NaN payloads or subnormals and would add an L1
// crossing (op_design.md, Overview). The reader and writer are the whole program.

void kernel_main() {}
