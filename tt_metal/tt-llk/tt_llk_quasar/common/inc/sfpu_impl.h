// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <cstdint>

namespace ckernel::sfpu
{

/// Which implementation runs an SFPU op. Every value must give bit-identical results for the same math policy.
enum class SfpuImpl : std::uint8_t
{
    Sfpi,      // SFPI-compiled loads, ops and stores
    LoadMacro, // SFPLOADMACRO sequences replayed from the replay buffer
};

/// Falls back to Sfpi under DISABLE_SFPLOADMACRO (ttsim). LoadMacro without a LoadMacro version fails to compile.
template <SfpuImpl REQUESTED, bool HAS_LOADMACRO>
constexpr SfpuImpl resolve_sfpu_impl()
{
    static_assert(REQUESTED != SfpuImpl::LoadMacro || HAS_LOADMACRO, "This SFPU op has no SFPLOADMACRO version for the selected math policy");
#ifdef DISABLE_SFPLOADMACRO
    return SfpuImpl::Sfpi;
#else
    return REQUESTED;
#endif
}

} // namespace ckernel::sfpu
