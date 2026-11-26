// SPDX-FileCopyrightText: © 2025 Tenstorrent Inc.
// SPDX-License-Identifier: Apache-2.0

// g6_to_memory_config has NO compute phase (op_design.md: classification data_movement — pages
// move as bytes and are never unpacked; unpacking/repacking would re-quantize block formats).
// This file exists only for the standard op layout; the program descriptor never attaches it, so
// no TRISC is launched.

void kernel_main() {}
