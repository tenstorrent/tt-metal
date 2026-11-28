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

// Index of a tile in Dest, written DstIndex{n}. A scoped enum, so a bare number is not accepted where a Dest index is
// meant, and it compiles to the plain integer it wraps.
enum class DstIndex : std::uint32_t
{
};

} // namespace ckernel::sfpu
