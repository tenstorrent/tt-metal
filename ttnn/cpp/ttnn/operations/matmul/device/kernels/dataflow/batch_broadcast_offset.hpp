// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <stdint.h>

// Tile offset of output batch `batch` in an input whose batch dims broadcast: `dims` are the output batch dims and
// `strides` the input's tile stride per dim, 0 where it broadcasts (see utilities::add_batch_broadcast_args).
template <typename Dims, typename Strides>
FORCE_INLINE uint32_t batch_broadcast_offset(uint32_t batch, const Dims& dims, const Strides& strides) {
    uint32_t offset = 0;
    for (uint32_t index = dims.size(); index-- > 0;) {
        offset += (batch % dims[index]) * strides[index];
        batch /= dims[index];
    }
    return offset;
}
