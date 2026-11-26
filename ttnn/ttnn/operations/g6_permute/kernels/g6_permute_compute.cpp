// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

// rows_tile has no compute phase and the program descriptor dispatches no compute kernel: values move only through
// the NoC, so bfloat16 bits (NaN payloads, -0, denormals) survive. A pass through unpack/DEST/pack may flush
// denormals and rewrite NaN. This translation unit exists for the regimes that need one (bfloat8_b requantization).

void kernel_main() {}
