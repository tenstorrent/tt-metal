// SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#define WELFORD_SFPU_DST_ADDR_MOD ckernel::ADDR_MOD_7
// Blackhole auto-stalls after SFPSHFT2, but an explicit NOP avoids the
// extra presentation cycle even when the following shuffle is independent.
#define WELFORD_SFPU_INDEPENDENT_SHFT2_NOP() TTI_SFPNOP
#define WELFORD_SFPU_ONLINE_HAZARD_NOP()
#define WELFORD_SFPU_INSTR_PER_ROW 6
#include "ckernel_sfpu_welfords_common.h"
#undef WELFORD_SFPU_INSTR_PER_ROW
#undef WELFORD_SFPU_ONLINE_HAZARD_NOP
#undef WELFORD_SFPU_INDEPENDENT_SHFT2_NOP
#undef WELFORD_SFPU_DST_ADDR_MOD
