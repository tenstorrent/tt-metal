// SPDX-FileCopyrightText: © 2026 Tenstorrent Inc.
// SPDX-License-Identifier: Apache-2.0

// g6_expand compute — intentionally NOT part of the program.
//
// op_design.md classifies g6_expand as data_movement: the broadcast tile B(n, w) is built by
// self-aimed L1->L1 NoC reads in the reader (byte moves, bit-exact by construction). Routing the
// values through unpack/DEST/pack (rejected regime R5) would risk the bit identity of NaN
// payloads and denormals, so g6_expand_program_descriptor.py registers no compute kernel.
// This file exists only to document that decision; it is never compiled.

void kernel_main() {}
