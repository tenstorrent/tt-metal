// SPDX-FileCopyrightText: © 2026 Tenstorrent Inc.
// SPDX-License-Identifier: Apache-2.0

// g6_slice compute kernel -- intentionally NOT attached to the program.
//
// op_design.md classifies g6_slice as pure data movement ("no compute kernel, no arithmetic").
// A compute-kernel copy (unpack -> math -> pack) is design regime R13, rejected: it passes values
// through unpacker/packer formats (int32/uint32 above 2**24 change, block-format exponents
// regroup). The reader moves raw bytes (R1) and assembles shifted tiles in L1 (R2); the writer
// stores them. This file exists only so the op directory has the conventional three kernel
// roles; g6_slice_program_descriptor.py never references it.

void kernel_main() {}
