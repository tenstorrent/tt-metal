// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <cstdint>

#include "api/compute/common.h"
#include "api/compute/eltwise_binary_sfpu.h"
#include "api/compute/reconfig_data_format.h"
#include "api/compute/tile_move_copy.h"

namespace kda {

// Converts a diagonal tile between A and its complement E = A - I in FP32 DST: dst[out] = dst[x] + I, or - I with
// Subtract. The identity tile enters DST in slot `scratch`; srcA is switched to the identity DFB and back to `srca_id`.
// The SFPU op reads only DST, so the caller's operand formats are unaffected; the caller re-issues its own unpack and
// math init afterwards.
template <bool Subtract, typename DFBIdentity>
FORCE_INLINE void apply_identity(DFBIdentity& identity, uint32_t srca_id, uint32_t x, uint32_t scratch, uint32_t out) {
    const uint32_t identity_id = identity.get_id();
    reconfig_data_format_srca(srca_id, identity_id);
    copy_init(identity_id);
    copy_tile(identity_id, 0, scratch);
    reconfig_data_format_srca(identity_id, srca_id);
    if constexpr (Subtract) {
        sub_binary_tile_init();
        sub_binary_tile(x, scratch, out);
    } else {
        add_binary_tile_init();
        add_binary_tile(x, scratch, out);
    }
}

}  // namespace kda
