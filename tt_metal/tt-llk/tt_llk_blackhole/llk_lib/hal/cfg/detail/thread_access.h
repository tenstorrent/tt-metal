// SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <cstdint>

#include "../access_types.h"
#include "../registers.h"
#include "ckernel_ops.h"
#include "llk_assert.h"

namespace hal::cfg
{

/**
 * @brief Write a prepacked thread-CFG word selected by a constant-propagated section.
 *
 * This overload is dependency-light because ckernel address-modifier types are
 * defined before the complete CFG HAL. It exists for that early include path;
 * normal callers must use the compile-time section overload from cfg/access.h.
 * SETC16 encodes its address as an immediate, so an arbitrary runtime section
 * cannot be emitted even though this compatibility signature accepts an integer.
 *
 * @tparam A: Access path, values = <TensixCfgUnit>.
 * @tparam F: Thread-CFG field anchoring the prepacked word.
 * @param section: Section index that the caller must make compile-time constant through inlining; assert-enabled builds require it to be smaller than F.count.
 * @param value: Prepacked thread-CFG word.
 */
template <Access A, const Field& F>
inline __attribute__((always_inline)) void write(const std::uint32_t section, const std::uint32_t value)
{
    static_assert(A == Access::TensixCfgUnit, "constant-propagated thread CFG writes require Access::TensixCfgUnit");
    static_assert(F.file == RegisterFile::Thread, "SETC16 targets thread CFG only");
    static_assert(F.shamt(Sec::S0) == 0, "prepacked thread CFG anchor must begin at bit zero");
    LLK_ASSERT(section < F.count, "section index out of range for this register");

    TTI_SETC16(F.addr32(static_cast<Sec>(section)), value);
}

} // namespace hal::cfg
