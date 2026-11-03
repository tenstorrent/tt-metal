// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include <cstdint>

#include "api/compute/common.h"
#include "experimental/kernel_args.h"

// Consumer of the gathered A on the bounding-box cores that hold no weight. The hubs multicast A
// over the whole box at full_in0's offset, so those cores carry full_in0 too; its consumer has to
// be a compute kernel everywhere because the weight cores' consumer is the matmul.
void kernel_main() {
    constexpr uint32_t full_num_tiles = get_arg(args::full_num_tiles);
    cb_wait_front(dfb::full_in0, full_num_tiles);
    cb_pop_front(dfb::full_in0, full_num_tiles);
}
