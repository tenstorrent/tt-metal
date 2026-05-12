// SPDX-FileCopyrightText: © 2024 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include <cstdint>
#include "api/compute/common.h"
#include "api/compute/eltwise_unary/dropout.h"
#include "ttnn/cpp/ttnn/kernel_lib/eltwise_chain.hpp"

namespace {

template <compute_kernel_lib::Dst Slot = compute_kernel_lib::Dst::D0>
struct Dropout : compute_kernel_lib::UnaryOp<Dropout<Slot>, Slot> {
    uint32_t int_probability;
    uint32_t int_scale_factor;
    constexpr Dropout(uint32_t p, uint32_t s) noexcept : int_probability(p), int_scale_factor(s) {}
    constexpr Dropout() noexcept : int_probability(0), int_scale_factor(0) {}

    static ALWI void init() { /* dropout_kernel_init handles seed; no separate per-op init */ }
    static ALWI void call(uint32_t /*idst*/) {}
    ALWI void exec(uint32_t /*i*/) const {
        dropout_tile(compute_kernel_lib::to_u32(Slot), int_probability, int_scale_factor);
    }
};

}  // namespace

void kernel_main() {
    using namespace compute_kernel_lib;

    constexpr uint32_t per_core_block_cnt = get_compile_time_arg_val(0);
    constexpr uint32_t per_core_block_dim = get_compile_time_arg_val(1);
    constexpr uint32_t int_probability = get_compile_time_arg_val(2);
    constexpr uint32_t int_scale_factor = get_compile_time_arg_val(3);

    uint32_t seed = get_arg_val<uint32_t>(0);
    constexpr auto cb_in = tt::CBIndex::c_0;
    constexpr auto cb_out = tt::CBIndex::c_2;
    constexpr uint32_t total_tiles = per_core_block_cnt * per_core_block_dim;

    // D5/D8: caller-side BIG init at the top of MAIN().
    compute_kernel_hw_startup(cb_in, cb_in, cb_out);
    // Dropout requires a one-time seed init beyond the chain element's per-tile init.
    dropout_kernel_init(seed);

    eltwise_chain(
        total_tiles,
        CopyTile<cb_in, Dst::D0, CopyTilePolicy::WaitAndPop>{},
        Dropout<Dst::D0>{int_probability, int_scale_factor},
        PackTile<cb_out, Dst::D0, PackTilePolicy::PerTileReserveAndPush>{});
}
