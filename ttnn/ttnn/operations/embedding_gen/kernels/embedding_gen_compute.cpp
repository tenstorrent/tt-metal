// SPDX-FileCopyrightText: © 2026 Tenstorrent Inc.
// SPDX-License-Identifier: Apache-2.0

// embedding_gen has no compute kernel (op_design.md: classification data_movement, thread plan (R, C, W) =
// (1, 0, 1)). The lookup only moves bytes: the reader places every selected table piece directly in output layout
// and the writer stores it, so no NEO is engaged and this file is not bound into the program
// (embedding_gen_program_artifacts.py builds a reader and a writer only). It exists so the op's kernel set has the
// usual three names; a future refinement that needs a compute stage starts here.

void kernel_main() {}
