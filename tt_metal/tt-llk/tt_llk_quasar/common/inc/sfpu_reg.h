// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <cstdint>

namespace ckernel::sfpu
{

// Register space for one SFPU operand. Availability is architecture-specific.
enum class SfpuReg : std::uint8_t
{
    Dest,
    SrcS,
};

} // namespace ckernel::sfpu
