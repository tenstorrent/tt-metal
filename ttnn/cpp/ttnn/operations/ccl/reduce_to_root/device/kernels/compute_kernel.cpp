// SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC
// SPDX-License-Identifier: Apache-2.0

// this kernel receives l, m, s tensors from the reader and perform the following computations
// - inputs: l1, s1, m1 and l2, s2, m2; output: l, s, m
//----> m = max(m1, m2)
//- P1 = exp((m1 - m) * scale) (called exp_max_diff)
//- P2 = exp((m2 - m) * scale)
//----> s = s1 * P1 + s2 * P2
//----> l = l1 * P1 + l2 * P2
// writes the tensors l, s, m to the writer buffers

// for last round of device 1 add extra compute:
// out = v / s
