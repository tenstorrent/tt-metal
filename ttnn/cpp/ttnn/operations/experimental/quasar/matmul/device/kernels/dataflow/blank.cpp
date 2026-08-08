// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

// No-op data-movement kernel for the matmul's idle ("noop") cores. Co-located with the quasar matmul op so
// the factory doesn't depend on the deprecated tt_metal/kernels/{dataflow,compute}/blank.cpp (moved to
// tests/ by #44980). See NOOP_DM_KERNEL_PATH in matmul_multicore_reuse_mcast_1d_program_factory.cpp.
void kernel_main() {}
