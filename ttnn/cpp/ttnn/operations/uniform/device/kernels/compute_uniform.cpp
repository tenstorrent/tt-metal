// SPDX-FileCopyrightText: © 2024 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include "api/compute/compute_kernel_api.h"
#include "api/compute/compute_kernel_hw_startup.h"
#include "api/compute/eltwise_unary/eltwise_unary.h"
#include "api/dataflow/circular_buffer.h"
#include "ttnn/cpp/ttnn/kernel_lib/eltwise/api/chain.hpp"
#include "ttnn/cpp/ttnn/kernel_lib/eltwise/generators/rand.hpp"

namespace {

using namespace compute_kernel_lib;

struct RandArgs {
    uint32_t key;
    uint32_t key1;
    uint32_t lower_bits;
    uint32_t scale_bits;
    uint32_t start_id;
    uint32_t num_tiles;
    uint32_t epoch_lo;
    uint32_t epoch_hi;
};

// A template so the generator that is not selected is never instantiated.
template <uint32_t Generator, uint32_t OutputCb, bool HasState, bool SaltEnabled>
ALWI void generate(const RandArgs& a) {
    constexpr auto out = output(OutputCb, ReservePolicy::PerTile, PushPolicy::PerTile, DataFormatReconfig::Disabled);
    // The epoch is a 64-bit counter and Threefry-2x32 takes a 2x32 counter, so the whole epoch folds in
    // with one PRF evaluation. Both words matter: folding only the low one would repeat the stream every
    // 2^32 advances.
    uint32_t key0 = a.key;
    uint32_t key1 = a.key1;
    if constexpr (HasState) {
        const ThreefryWords k = rand_epoch_key(a.key, a.epoch_lo, a.epoch_hi);
        key0 = k.x0;
        key1 = k.x1;
    }
    if constexpr (Generator == 0) {
        // uniform shares this kernel and keeps its pre-existing values, so the position salt is opt-in.
        uint32_t salt_key = 0u;
        if constexpr (SaltEnabled) {
            salt_key = key0;
        }
        eltwise_chain(
            IterationShape::tiles(a.num_tiles),
            RandTile<Dst::D0>{a.lower_bits, a.scale_bits, key0, a.start_id, salt_key},
            PackTile<out>{});
    } else {
        eltwise_chain(
            IterationShape::tiles(a.num_tiles),
            ThreefryTile<Dst::D0>{a.lower_bits, a.scale_bits, key0, key1, a.start_id},
            PackTile<out>{});
    }
}

}  // namespace

void kernel_main() {
    constexpr uint32_t output_dfb_id = get_compile_time_arg_val(0);
    constexpr uint32_t generator = get_compile_time_arg_val(1);  // 0 = hardware LFSR, 1 = Threefry-2x32
    constexpr bool has_state = get_compile_time_arg_val(2) != 0;
    constexpr uint32_t state_cb_id = get_compile_time_arg_val(3);
    constexpr bool salt_enabled = get_compile_time_arg_val(4) != 0;

    RandArgs a{};
    a.key = get_arg_val<uint32_t>(0);
    union {
        float f;
        uint32_t u;
    } f2u_lower_bound, f2u_upper_bound, f2u_scale;
    f2u_lower_bound.u = get_arg_val<uint32_t>(1);
    f2u_upper_bound.u = get_arg_val<uint32_t>(2);
    // The host supplies inclusive endpoints that are representable in the
    // destination dtype. Choose the largest scale whose rounded endpoint does
    // not exceed the upper bound, avoiding a clamp in the SFPU hot loop.
    f2u_scale.f = f2u_upper_bound.f - f2u_lower_bound.f;
    if (f2u_lower_bound.f + f2u_scale.f > f2u_upper_bound.f && f2u_scale.u != 0) {
        --f2u_scale.u;
    }
    a.lower_bits = f2u_lower_bound.u;
    a.scale_bits = f2u_scale.u;
    a.start_id = get_arg_val<uint32_t>(3);
    a.num_tiles = get_arg_val<uint32_t>(4);
    a.key1 = get_arg_val<uint32_t>(5);

    // Startup stays the first Compute API call in the kernel: it does MMIO configuration that needs the
    // execution units idle, so the state read below must not precede it.
    compute_kernel_hw_startup(output_dfb_id, output_dfb_id);

    // 64-bit epoch of this core's state page, advanced by the writer after the last tile.
    if constexpr (has_state) {
        CircularBuffer cb_state(state_cb_id);
        cb_state.wait_front(1);
        a.epoch_lo = cb_state.read_tile_value(/*tile_index=*/0, /*element_offset=*/0);
        a.epoch_hi = cb_state.read_tile_value(/*tile_index=*/0, /*element_offset=*/1);
        cb_state.pop_front(1);
    }

    generate<generator, output_dfb_id, has_state, salt_enabled>(a);
}
