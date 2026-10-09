// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <cstdint>

namespace ckernel::sfpu
{

/// How an SFPU op is issued. Every value must give bit-identical results for the same math policy.
enum class SfpuIssue : std::uint8_t
{
    Sfpi,      // SFPI-compiled loads, ops and stores
    LoadMacro, // SFPLOADMACRO sequences replayed from the replay buffer
};

/// Falls back to Sfpi under DISABLE_SFPLOADMACRO (ttsim). LoadMacro without a LoadMacro version fails to compile.
template <SfpuIssue REQUESTED, bool HAS_LOADMACRO>
constexpr SfpuIssue resolve_sfpu_issue()
{
    static_assert(REQUESTED != SfpuIssue::LoadMacro || HAS_LOADMACRO, "This SFPU op has no SFPLOADMACRO version for the selected math policy");
#ifdef DISABLE_SFPLOADMACRO
    return SfpuIssue::Sfpi;
#else
    return REQUESTED;
#endif
}

} // namespace ckernel::sfpu
