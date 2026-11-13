// SPDX-FileCopyrightText: © 2024 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include <cstdint>
#include "api/compute/compute_kernel_hw_startup.h"
#include "ttnn/cpp/ttnn/kernel_lib/eltwise/api/chain.hpp"
#include "ttnn/cpp/ttnn/kernel_lib/eltwise/unary/scalar.hpp"  // Dropout owns PRNG initialization.
#include "experimental/kernel_args.h"

namespace ckl = compute_kernel_lib;

namespace {
// MurmurHash3's 32-bit finalizer.
inline uint32_t fmix32(uint32_t h) {
    h ^= h >> 16;
    h *= 0x85EBCA6Bu;
    h ^= h >> 13;
    h *= 0xC2B2AE35u;
    h ^= h >> 16;
    return h;
}
}  // namespace

void kernel_main() {
    constexpr uint32_t per_core_block_cnt = get_arg(args::per_core_block_cnt);
    constexpr uint32_t per_core_block_dim = get_arg(args::per_core_block_dim);
    constexpr uint32_t int_probability = get_arg(args::int_probability);
    constexpr uint32_t int_scale_factor = get_arg(args::int_scale_factor);

    // Every core's SFPU PRNG starts from its own seed, so cores sharing a seed would emit identical masks: the
    // call's seed (common), offset by the device's id when each device draws its own mask (common, set per device;
    // 0 otherwise), hashed with the core's first tile (per core). Hashed rather than strided: Quasar's
    // init_prng_seed seeds lane i with seed + i * 0x9E3779B9, so a linear per-core stride would make one core's
    // lanes shifted copies of another's. All-ones is the XNOR-LFSR lock-up state and dropout_init passes the seed to
    // init_prng_seed unfiltered.
    uint32_t seed = fmix32(
        (get_arg(args::seed) + get_arg(args::device_seed_offset)) ^ fmix32(get_arg(args::tile_offset) + 0x632BE5ABu));
    if (seed == 0xFFFFFFFFu) {
        seed = 0xFFFFFFFEu;
    }

    DataflowBuffer dfb_in0_obj(dfb::in0);
    DataflowBuffer dfb_out_obj(dfb::out);

    compute_kernel_hw_startup(dfb::in0, dfb::out);

    constexpr uint32_t total_tiles = per_core_block_cnt * per_core_block_dim;
    ckl::eltwise_chain(
        ckl::IterationShape::tiles(total_tiles).block_size(per_core_block_dim),
        ckl::CopyTile<
            ckl::input(dfb::in0, ckl::WaitPolicy::PerTile, ckl::PopPolicy::PerTile, ckl::DataFormatReconfig::Disabled),
            ckl::Dst::D0>{dfb_in0_obj},
        ckl::Dropout<ckl::Dst::D0>{int_probability, int_scale_factor, seed},
        ckl::PackTile<ckl::output(
            dfb::out,
            ckl::ReservePolicy::PerBlockSize,
            ckl::PushPolicy::PerBlockSize,
            ckl::DataFormatReconfig::Disabled)>{dfb_out_obj});
}
