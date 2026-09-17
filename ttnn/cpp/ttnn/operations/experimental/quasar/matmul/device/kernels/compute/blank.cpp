// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

// No-op compute kernel for the matmul's idle ("noop") cores. Co-located with the quasar matmul op so the
// factory doesn't depend on the deprecated tt_metal/kernels/compute/blank.cpp (moved to tests/ by #44980).
// See NOOP_COMPUTE_KERNEL_PATH in matmul_multicore_reuse_mcast_1d_program_factory.cpp.
#include "api/compute/blank.h"

void kernel_main() {}
