// SPDX-FileCopyrightText: © 2026 Tenstorrent Inc.
// SPDX-License-Identifier: Apache-2.0

// g6_concat compute kernel — intentionally EMPTY and NOT attached to the program.
//
// op_design.md classifies g6_concat as pure data movement: values move as bytes from the input
// buffers into cb_out_tiles (reader) and out to the output buffer (writer); routing them through
// unpack -> DEST -> pack would be a forbidden format path (f32 low bits / bf8b groups) and an
// extra L1 pass. The program descriptor therefore registers only the reader and the writer.
// This file exists only so the op directory has the conventional three-kernel layout.

void kernel_main() {}
