// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

// Not dispatched. g6_slice is a pure data-movement op (op_design.md, Classification): every value
// is copied bit for bit in x's storage format, and a pass through unpack -> DEST -> pack could
// neither shift rows/columns nor keep int32/uint32 above 2^24 and NaN payloads exact
// (op_design.md, Regimes: compute_copy rejected). The program descriptor builds no compute kernel.

void kernel_main() {}
