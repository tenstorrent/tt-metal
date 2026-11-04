// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <cstdint>

#include "api/compute/common.h"
#include "api/compute/eltwise_binary_sfpu.h"
#include "api/compute/reconfig_data_format.h"
#include "api/compute/tile_move_copy.h"

namespace kda {

namespace detail {

// dst[out] = dst[x] + I, or - I with Subtract, in FP32 DST. The identity tile enters DST in slot `scratch`; srcA is
// switched to the identity DFB and back to `srca_id`. The SFPU op reads only DST, so the caller's operand formats are
// unaffected; the caller re-issues its own unpack and math init afterwards.
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

}  // namespace detail

// Diagonal tile of A in DST slot `a` -> its complement E = A - I in slot `e`, using slot `scratch` for I.
template <typename DFBIdentity>
FORCE_INLINE void a_to_complement(DFBIdentity& identity, uint32_t srca_id, uint32_t a, uint32_t scratch, uint32_t e) {
    detail::apply_identity<true>(identity, srca_id, a, scratch, e);
}

// Diagonal tile of E = A - I in DST slot `e` -> A = E + I in slot `a`, using slot `scratch` for I.
template <typename DFBIdentity>
FORCE_INLINE void complement_to_a(DFBIdentity& identity, uint32_t srca_id, uint32_t e, uint32_t scratch, uint32_t a) {
    detail::apply_identity<false>(identity, srca_id, e, scratch, a);
}

// Copy tile `tile` of a complement-form operand (E = A - I) into DST slot 0 and return the slot that holds A: slot 0
// off the diagonal, where A = E, and `a_slot` on it, where A = E + I is formed in FP32 with I in slot 1. Expects
// copy_init(in) and leaves it in place.
template <typename DFBIn, typename DFBIdentity>
FORCE_INLINE uint32_t copy_tile_as_a(DFBIn& in, DFBIdentity& identity, uint32_t tile, bool diagonal, uint32_t a_slot) {
    const uint32_t in_id = in.get_id();
    copy_tile(in_id, tile, 0);
    if (!diagonal) {
        return 0;
    }
    complement_to_a(identity, in_id, 0, 1, a_slot);
    copy_init(in_id);
    return a_slot;
}

}  // namespace kda
