// SPDX-FileCopyrightText: © 2026 Tenstorrent Inc.
// SPDX-License-Identifier: Apache-2.0

// upsample has no compute stage, so this file is intentionally not bound to the program
// (upsample_program_artifacts.py builds a reader and a writer only; thread plan (R, C, W) = (2, 0, 4)).
//
// Why: every output element is an input element, unchanged (spec DEFINITION, rule R1). Routing row-major
// sticks through unpack -> DEST -> pack would add a stage and risk a format conversion R1 forbids (a float32
// value must not pass through a narrower format). The writer writes each resident input stick
// scale_h * scale_w times straight from the reader's ring (op_design.md, Block Operation Realization).
//
// A compute kernel returns with the deferred `tile_layout` regime (op_design.md, Regimes), where W
// replication of TILE-layout data may go through the untilize / tilize helpers.
