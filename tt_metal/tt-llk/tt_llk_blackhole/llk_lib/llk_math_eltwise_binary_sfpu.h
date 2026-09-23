// SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <cstdint>
#include <type_traits>

#include "ckernel_globals.h"
#include "ckernel_include.h"
#include "ckernel_ops.h"
#include "ckernel_sfpu.h"
#include "cmath_common.h"
#include "llk_math_common.h"
#include "llk_math_eltwise_sfpu_common.h"

using namespace ckernel;
// The shared init for binary SFPU ops is _llk_math_eltwise_sfpu_init_() in llk_math_eltwise_sfpu_common.h;
// ops that need ADDR_MOD_6 (dest auto-increment) program it in their own init.
