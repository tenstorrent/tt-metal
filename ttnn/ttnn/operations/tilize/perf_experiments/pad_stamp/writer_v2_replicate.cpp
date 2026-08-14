// SPDX-FileCopyrightText: © 2026 Tenstorrent Inc.
// SPDX-License-Identifier: Apache-2.0
// EXPERIMENT ONLY (perf bake-off "pad_stamp") — arm: v2_replicate
// Stamp once, replicate into the CB via a local L1->L1 NoC transfer (option 1).
// A distinct source FILE per arm, not just a define: the generic-op program hash
// and the JIT kernel hash both key on the kernel source path, so two arms can
// never collide in either cache inside one pytest process.
#define PS_FILL 0
#define PS_PADTILE 1
#include "ttnn/ttnn/operations/tilize/perf_experiments/pad_stamp/pad_stamp_writer.inc"
