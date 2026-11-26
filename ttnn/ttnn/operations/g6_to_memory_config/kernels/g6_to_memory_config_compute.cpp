// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

// g6_to_memory_config has no compute stage and its program descriptor attaches no compute kernel:
// routing a page through the unpacker/packer would change bfloat8_b/bfloat4_b bits, and there is no
// arithmetic. Kept only as the empty TRISC role of the op's kernel set.

void kernel_main() {}
