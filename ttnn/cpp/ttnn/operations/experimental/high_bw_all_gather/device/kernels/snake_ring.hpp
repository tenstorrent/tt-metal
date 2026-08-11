// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include "cpp/ttnn/operations/ccl/shared_with_host/snake_ring.hpp"

// Compatibility alias for the high_bw_all_gather device kernels. New users
// should include the CCL-owned header and use ttnn::ccl::snake_ring directly.
namespace ttnn::operations::experimental::high_bw_all_gather {
namespace snake_ring = ttnn::ccl::snake_ring;
}
