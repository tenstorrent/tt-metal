// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

// Not attached to the program. g6_concat only moves bytes: an unpack -> DEST -> pack round trip
// would be the value conversion the op must never apply (fp32 low bits under a 16-bit DEST,
// bfloat8_b requantization), so the program descriptor dispatches only the reader and writer.

void kernel_main() {}
