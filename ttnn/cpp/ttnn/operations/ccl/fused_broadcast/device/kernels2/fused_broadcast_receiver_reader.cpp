// SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#include "dataflow_api.h"
#include <cstdint>
#include <utility>
#include "ttnn/operations/ccl/shared_with_host/sharded_tensor_addr_gen.hpp"
#include "ttnn/operations/ccl/kernel_common/sharding_addrgen.hpp"

using address_t = uint32_t;

///////////////////////////////////////////////////
// COMPILE TIME ARGS
///////////////////////////////////////////////////

// nothing it just receives data
void kernel_main() {
    // Debug: Confirm this kernel is correctly empty
    DPRINT << "RECEIVER READER: This kernel should be empty - fabric writes directly to CB" << ENDL();

    // This kernel is intentionally empty - the fabric connection
    // writes directly to the circular buffer, no reader needed
}
