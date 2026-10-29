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
#include "llk_sfpu_types.h"

using namespace ckernel;

template <SfpuType sfpu_op>
inline void _llk_math_eltwise_unary_sfpu_init_()
{
    sfpu::_init_sfpu_config_reg();
    math::reset_counters(p_setrwc::SET_ABD_F);
}

// Kernel-invariant SFPU init. Runs the parts of the per-op init that are identical for every SFPU op and
// therefore only need to run once per kernel: the SFPU config register (SFPCONFIG(0, 0xF, 1)). ADDR_MOD_7 is
// not set here: _llk_math_hw_configure_ sets it to zero advance and every op leaves it there. Hoisted out of
// the per-op path so that the self-contained per-op init (ckernel::sfpu::_init_<op>_) does not re-run it on
// every op init. Metal wires this into every
// "full init" entry point (compute_kernel_hw_startup, init_sfpu, unary_op_init_common),
// and the tt-llk standalone SFPU test harness wires it into its init prelude.
inline void _llk_math_eltwise_unary_sfpu_init_once_()
{
    sfpu::_init_sfpu_config_reg();
}
