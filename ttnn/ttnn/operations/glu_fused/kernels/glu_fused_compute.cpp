// SPDX-FileCopyrightText: © 2026 Tenstorrent Inc.
// SPDX-License-Identifier: Apache-2.0

// glu_fused — Compute kernel.
//
// One fused chain per output tile:
//   D0 = A tile          (from cb_input_a)
//   D1 = B tile          (from cb_input_b)
//   D1 = sigmoid(D1)     (Approx::Exact — accurate sigmoid, NOT fast-approx)
//   D0 = D0 * D1         (in-DST multiply)
//   pack(D0) -> cb_output_tiles
//
// Chain stride = 2 (D0, D1). In fp32 half-sync with fp32_dest_acc_en, DEST
// holds 4 slots → auto-batch fills DEST with up to 2 chain iterations per
// tile_regs_acquire.
//
// CT args: [cb_input_a, cb_input_b, cb_output_tiles]
// RT args: [num_output_tiles_this_core]

#include <cstdint>

#include "ttnn/cpp/ttnn/kernel_lib/sfpu_helpers.hpp"

void kernel_main() {
    const uint32_t num_output_tiles = get_arg_val<uint32_t>(0);

    constexpr uint32_t cb_input_a = get_compile_time_arg_val(0);
    constexpr uint32_t cb_input_b = get_compile_time_arg_val(1);
    constexpr uint32_t cb_output_tiles = get_compile_time_arg_val(2);

    using namespace compute_kernel_lib;

    // init_sfpu does the full hw startup (replaces compute_kernel_hw_startup
    // for SFPU-only kernels). The unpacker is anchored to cb_input_a — its
    // float32 format also covers cb_input_b — and the packer to cb_output_tiles.
    init_sfpu(cb_input_a, cb_output_tiles);

    auto chain = sfpu_chain(
        Load<cb_input_a, Dst::D0>{},
        Load<cb_input_b, Dst::D1>{},
        Sigmoid<Approx::Exact, Dst::D1>{},
        SfpuMul<Dst::D0, Dst::D1, Dst::D0>{});

    sfpu_pipeline<
        SfpuBatching::Auto,
        SfpuInputPolicy::WaitAndPopPerTile,
        SfpuOutputPolicy::PerTile,
        SfpuDataFormatReconfig::INPUT_AND_OUTPUT>(chain, cb_output_tiles, num_output_tiles);
}
