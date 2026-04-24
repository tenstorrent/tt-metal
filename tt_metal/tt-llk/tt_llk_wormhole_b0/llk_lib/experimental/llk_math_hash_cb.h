// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0
//
// Wormhole B0 mirror of the SFPU-backed CB hash. Left as a placeholder in this
// initial PR -- the Blackhole variant is shipping first and Wormhole parity is
// out of scope until the Blackhole implementation has been hardware-validated
// and the exact SFPU op sequence is frozen. A port should be near-mechanical
// once that is true; the same SFPMUL24 / SFPXOR / SFPSHFT2 / SFPLOAD-INT32
// primitives exist on WH B0, but semantic checks (UnshuffleFP32, Dst SM vs 2sc,
// SFPLOAD mode numbers) must be re-verified against WH ISA docs.
//
// The compute-API shim gates the SFPU variant on ARCH_BLACKHOLE, so including
// this empty header on WH is safe; the empty inlines simply never get called.

#pragma once

#include <cstdint>

namespace ckernel::sfpu
{

inline void _llk_math_hash_cb_init_()
{
    // Intentionally empty -- SFPU hash not supported on WH B0 in this PR.
}

inline void _llk_math_hash_cb_tile_(std::uint32_t /*dst_tile_idx*/)
{
    // Intentionally empty -- SFPU hash not supported on WH B0 in this PR.
}

inline void _llk_math_hash_cb_reduce_and_store_(std::uint32_t /*dst_tile_idx*/)
{
    // Intentionally empty -- SFPU hash not supported on WH B0 in this PR.
}

} // namespace ckernel::sfpu
