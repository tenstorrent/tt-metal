// SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0
//
// all_gather — compute kernel (INTENTIONALLY UNUSED).
//
// all_gather is a CCL collective: it is PURE DATA MOVEMENT (an identity gather,
// no arithmetic — PCC ~1.0). There is no unpack/math/pack stage, so the program
// descriptor wires only the reader (NCRISC) and writer (BRISC) dataflow kernels
// and does NOT instantiate any compute (TRISC) kernel. This file exists to
// document that absence; it is not referenced by all_gather_program_descriptor.py.
//
// If a future refinement fuses an elementwise/reduction step into the gather
// (e.g. an all_reduce-style collective), that compute would live here and be
// added to the ProgramDescriptor's kernels list with a ComputeConfigDescriptor.

void kernel_main() {
    // No-op: all_gather performs no on-device arithmetic.
}
