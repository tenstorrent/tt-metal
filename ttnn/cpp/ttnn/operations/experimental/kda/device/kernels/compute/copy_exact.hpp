// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <cstdint>
#include <type_traits>

#include "api/compute/common.h"
#include "api/compute/pack.h"
#include "api/compute/reconfig_data_format.h"
#include "api/compute/tile_move_copy.h"
#include "api/dataflow/dataflow_buffer.h"

namespace kda {

// Copy Count tiles of `in` to `out` and every buffer in `more`, one tile per DST acquire. An UnpackToDest input in the
// outputs' format makes the copy bit-exact; `in` is consumed.
template <uint32_t Count, typename... More>
FORCE_INLINE void copy_exact(DataflowBuffer& in, DataflowBuffer& out, More&... more) {
    static_assert((std::is_same_v<More, DataflowBuffer> && ...), "copy_exact outputs are DataflowBuffers");
    const uint32_t in_id = in.get_id();
    in.wait_front(Count);
    out.reserve_back(Count);
    (more.reserve_back(Count), ...);
    pack_reconfig_data_format(out.get_id());
    reconfig_data_format_srca(in_id);
    copy_init(in_id);
    for (uint32_t tile = 0; tile < Count; ++tile) {
        tile_regs_acquire();
        copy_tile(in_id, tile, 0);
        tile_regs_commit();
        tile_regs_wait();
        pack_tile(0, out.get_id(), tile);
        (pack_tile(0, more.get_id(), tile), ...);
        tile_regs_release();
    }
    out.push_back(Count);
    (more.push_back(Count), ...);
    in.pop_front(Count);
}

}  // namespace kda
