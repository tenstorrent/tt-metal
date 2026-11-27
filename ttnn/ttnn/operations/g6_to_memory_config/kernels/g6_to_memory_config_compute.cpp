// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

// g6_to_memory_config has no compute stage and the program descriptor attaches no compute kernel:
// pages are moved whole by the data-movement RISCs, and routing them through unpack/pack would change
// bfloat8_b / bfloat4_b bits. This file exists only to complete the op's reader/compute/writer set.

void kernel_main() {}
