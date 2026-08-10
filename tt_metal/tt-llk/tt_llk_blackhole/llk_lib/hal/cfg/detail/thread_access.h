// SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <cstdint>

#include "../access_types.h"
#include "../registers.h"
#include "ckernel_ops.h"

namespace hal::cfg
{

/**
 * @brief Write a prepacked thread-CFG word selected by a runtime section.
 *
 * This overload is dependency-light because ckernel address-modifier types are
 * defined before the complete CFG HAL. It is still part of the same public
 * cfg::write() interface; normal field writes should use the compile-time
 * section overload from cfg/access.h.
 */
template <Access A, const Field& F>
inline __attribute__((always_inline)) void write(const std::uint32_t section, const std::uint32_t value)
{
    static_assert(A == Access::TensixCfgUnit, "runtime-section thread CFG writes require Access::TensixCfgUnit");
    static_assert(F.file == RegisterFile::Thread, "SETC16 targets thread CFG only");
    static_assert(F.shamt(Sec::S0) == 0, "prepacked thread CFG anchor must begin at bit zero");

    TTI_SETC16(F.addr32(static_cast<Sec>(section)), value);
}

} // namespace hal::cfg
