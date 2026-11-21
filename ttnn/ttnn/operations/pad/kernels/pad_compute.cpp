// SPDX-FileCopyrightText: © 2026 Tenstorrent Inc.
// SPDX-License-Identifier: Apache-2.0

// pad has no compute kernel, by design (op_design.md: classification data_movement, thread plan
// (R, C, W) = (1, 0, 1)). The op moves row bytes and writes a constant; nothing passes through the NEOs, so
// no element is ever converted (rule R1). This file is not bound into the ProgramSpec
// (pad_program_artifacts.py binds pad_reader.cpp and pad_writer.cpp only); it exists to record that choice.

void kernel_main() {}
