// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

// Not attached to any program: rows_tile is a pure byte move done by the reader and writer over the
// NoC. A pass through the unpacker/DEST/packer may flush denormals or rewrite NaN payloads, which the
// bit-exact bfloat16 contract forbids (op_design.md, X4). A regime that needs compute (bfloat8_b
// requantization, R4) builds its kernel here.

void kernel_main() {}
