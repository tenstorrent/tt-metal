// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
// SPDX-License-Identifier: Apache-2.0

// g6_expand has no compute stage and the program descriptor dispatches no compute kernel: the
// contract is bit-exact, and a pass through unpack/DEST/pack could alter NaN payloads, signed zeros
// or denormals. Every output byte is a NoC copy made by the reader and writer kernels.

void kernel_main() {}
