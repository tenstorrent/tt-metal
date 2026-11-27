// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

// g6_to_memory_config has no compute phase and the program descriptor never launches this kernel:
// pages move as bytes between the NoC and L1, and a page passed through unpack/pack would no longer
// be bit-exact for the block formats. Launching an empty TRISC kernel would only add launch cost.

void kernel_main() {}
