// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <cstdint>

namespace ckernel::sfpu
{

/**
 * @brief How an SFPU op's instructions are issued, independent of the register file it runs on.
 *
 * Selects the issue mechanism, never the math: for the same math policy every value must give
 * bit-identical results. A sequence that computes something different is a different math
 * policy, not a new value here.
 */
enum class SfpuIssue : std::uint8_t
{
    Sfpi,      // SFPI-compiled loads, ops and stores
    LoadMacro, // SFPLOADMACRO sequences replayed from the replay buffer
};

/**
 * @brief Resolve a requested issue mechanism against the op and the build.
 *
 * Builds without SFPLOADMACRO (DISABLE_SFPLOADMACRO, e.g. ttsim) fall back to Sfpi; that is the
 * only fallback. Requesting LoadMacro for an op or math policy that has no LoadMacro version
 * fails to compile in every build, so ttsim and hardware builds accept the same requests.
 *
 * @tparam REQUESTED: Issue mechanism asked for, values = <Sfpi/LoadMacro>
 * @tparam HAS_LOADMACRO: Whether the op has a LoadMacro version for the selected math policy.
 */
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
