// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
// SPDX-License-Identifier: Apache-2.0

#pragma once

/**
 * @file eltwise_optional.hpp
 * @brief Conditional / optional chain element wrappers.
 *
 * Two patterns are supported:
 *
 *  1. **Compile-time conditional** — `OptionalChainElement<bool COND, Inner>`. When
 *     COND is true the wrapper inherits Inner's constructors AND its full type / tag,
 *     so the chain pipeline sees `Inner`. When COND is false the wrapper inherits
 *     Inner's tag (so chain traits classify it the same way) but every hook is a
 *     no-op. The wrapper's variadic constructor swallows any args the caller would
 *     pass to Inner — `OptionalChainElement<COND, FillScalar>{0.5f}` compiles for
 *     COND in {true, false}.
 *
 *  2. **Runtime conditional** — kernels that need to pick a chain shape based on a
 *     runtime arg use the standard "template the inner function on a `bool`" pattern
 *     and dispatch from `kernel_main`:
 *
 *         template <bool DO_MASK>
 *         inline void run_op(uint32_t n) {
 *             eltwise_chain(n,
 *                 CopyTile<...>{},
 *                 OptionalChainElement<DO_MASK, MaskInject<...>>{},
 *                 SfpuOp<...>{},
 *                 PackTile<...>{});
 *         }
 *         void kernel_main() {
 *             const bool do_mask = get_arg_val<uint32_t>(0) != 0;
 *             if (do_mask) run_op<true>(n); else run_op<false>(n);
 *         }
 *
 * @section optional_caveats Caveats
 *
 * Mid-loop per-iteration runtime conditions (`if (col_idx == Wt-1) mask_tile(...)`)
 * are NOT supported by these helpers — they require a per-iter runtime branch
 * inside the chain pipeline, which collides with the chain's compile-time dispatch.
 * Use a separate chain invocation for the conditional iteration if needed.
 */

#include "ttnn/cpp/ttnn/kernel_lib/eltwise_chain.hpp"

namespace compute_kernel_lib {

/// Conditional chain element — when COND is false every hook is a no-op.
template <bool COND, class Inner>
struct OptionalChainElement;

// COND == true: forward to Inner.
template <class Inner>
struct OptionalChainElement<true, Inner> : Inner {
    using Inner::Inner;
};

// COND == false: inherit Inner's tag (so chain traits classify it the same way) but
// emit nothing in any hook. Tag-only type — no Inner construction, no Inner state.
template <class Inner>
struct OptionalChainElement<false, Inner>
    : std::conditional_t<
          is_pack_tile_op_v<Inner>,
          PackTileTag,
          std::conditional_t<
              is_copy_tile_op_v<Inner>,
              CopyTileTag,
              std::conditional_t<
                  is_binary_fpu_op_v<Inner>,
                  BinaryFpuTag,
                  std::conditional_t<
                      is_dest_reuse_binary_op_v<Inner>,
                      DestReuseBinaryTag,
                      std::conditional_t<
                          is_unary_bcast_op_v<Inner>,
                          UnaryBcastTag,
                          std::conditional_t<
                              is_fill_tile_op_v<Inner>,
                              FillTileTag,
                              std::conditional_t<is_rand_tile_op_v<Inner>, RandTileTag, DestOnlyTag>>>>>>> {
    static constexpr bool is_upfront = false;
    static constexpr bool clashes_with_fpu = false;
    static constexpr uint32_t cb_a_id() { return 0; }
    static constexpr uint32_t cb_b_id() { return 0; }
    static constexpr uint32_t pack_cb_id() { return 0; }
    static constexpr InputLifecycle a_policy() { return CallerManaged; }
    static constexpr InputLifecycle b_policy() { return CallerManaged; }

    // Variadic ctor swallows any args the caller would pass to the inner element
    // so kernel-side construction `OptionalChainElement<false, Inner>{...}` doesn't
    // need to gate the ctor args on COND. Provides a default ctor too.
    constexpr OptionalChainElement() noexcept = default;
    template <class... Ignored>
    constexpr explicit OptionalChainElement(Ignored&&...) noexcept {}

    static constexpr uint32_t lane_width = 1;
    static ALWI void init() {}
    ALWI void wait_per_tile(uint32_t) const {}
    ALWI void wait_upfront(uint32_t) const {}
    ALWI void exec(uint32_t, uint32_t) const {}
    ALWI void pop_per_tile(uint32_t) const {}
    ALWI void pop_upfront_end(uint32_t) const {}
    ALWI void reserve_per_tile(uint32_t) const {}
    ALWI void reserve_upfront(uint32_t) const {}
    ALWI void push_per_tile(uint32_t) const {}
    ALWI void push_at_end(uint32_t) const {}

    // 2D no-op stubs (called by `eltwise_chain(EltwiseShape, ...)` overload).
    ALWI void wait_upfront_2d(uint32_t, uint32_t) const {}
    ALWI void exec_2d(uint32_t, uint32_t, uint32_t, uint32_t) const {}
    ALWI void pop_upfront_end_2d(uint32_t, uint32_t) const {}
    ALWI void reserve_upfront_2d(uint32_t, uint32_t) const {}
    ALWI void push_at_end_2d(uint32_t, uint32_t) const {}
};

}  // namespace compute_kernel_lib
