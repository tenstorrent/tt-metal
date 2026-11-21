// SPDX-FileCopyrightText: © 2026 Tenstorrent Inc.
// SPDX-License-Identifier: Apache-2.0

// paged_fill_cache has NO compute stage, and this kernel is NOT part of its program.
//
// The op moves whole tiles unchanged (rule R1: no element passes through another format), so every tile
// goes DRAM -> dfb_input_tiles -> DRAM between the reader and the writer data-movement kernels and never
// enters a NEO. The design's thread plan is (R, C, W) = (1, 0, 1) (op_design.md, Work Distribution), as
// Quasar's own tile `slice` (reader + writer only). Routing tiles through a NEO would add a ring, an
// unpack/pack round trip and a format path that a bit-exact copy does not need.
//
// The file exists so the op's kernel set is complete as listed; paged_fill_cache_program_artifacts.py
// does not load it.

void kernel_main() {}
