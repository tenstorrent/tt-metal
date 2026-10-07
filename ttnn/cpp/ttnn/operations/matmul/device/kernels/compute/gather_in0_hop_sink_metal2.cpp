// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

// The compute side of a Metal 2.0 gather_in0 hop core. A hop core only passes in0 shards on, out of its
// in2 (reader_bmm_tile_layout_in0_ring_hop_metal2.cpp), and in2 has to sit at the ring workers' address
// on it. A local buffer needs a consumer on every core it lives on, and on the workers that consumer is
// compute, so the hop binds this kernel as in2's consumer. It reads nothing.

void kernel_main() {}
