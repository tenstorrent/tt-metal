// SPDX-FileCopyrightText: © 2025 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include "api/compute/common.h"
#include "api/compute/eltwise_unary/typecast.h"
#include "ttnn/cpp/ttnn/kernel_lib/eltwise_chain.hpp"

namespace {

// Wrap host-defined TYPECAST_LLK / TYPECAST_LLK_INIT macros in a chain element.
struct TypecastSfpu : compute_kernel_lib::DestOnlyTag {
    static constexpr compute_kernel_lib::Dst dst_idx = compute_kernel_lib::Dst::D0;
    static ALWI void init() { TYPECAST_LLK_INIT(); }
    ALWI void exec(uint32_t /*i*/) const { TYPECAST_LLK(0); }
};

}  // namespace

void kernel_main() {
    using namespace compute_kernel_lib;

    constexpr uint32_t per_core_block_cnt = get_compile_time_arg_val(0);
    constexpr uint32_t per_core_block_dim = get_compile_time_arg_val(1);
    constexpr uint32_t input_cb = get_compile_time_arg_val(2);
    constexpr uint32_t output_cb = get_compile_time_arg_val(3);
    constexpr uint32_t num_tiles = per_core_block_cnt * per_core_block_dim;

    // D5/D8: caller-side BIG init at the top of MAIN().
    eltwise_chain_with_init(
        num_tiles,
        CopyTile<input_cb, Dst::D0, CopyTilePolicy::WaitAndPop>{},
        TypecastSfpu{},
        PackTile<output_cb, Dst::D0, PackTilePolicy::PerTileReserveAndPush>{});
}
