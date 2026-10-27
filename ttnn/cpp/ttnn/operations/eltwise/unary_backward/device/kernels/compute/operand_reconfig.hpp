// SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include "ttnn/cpp/ttnn/kernel_lib/eltwise/api/chain.hpp"

// Unpacker format switching for the unary-backward compute kernels' two operand buffers.
//
// The unpacker only has to switch format between grad_output (c_0) and input (c_1) when they
// carry different formats, which the program factory signals with MIXED_OPERAND_DATA_FORMATS. For
// the same-dtype case -- nearly every call -- the single configuration compute_kernel_hw_startup()
// installs already covers both buffers, and a reconfiguration per tile transition is pure
// overhead, so it stays disabled. Both operand dtypes are in the program hash, so this can be a
// compile-time decision.
#ifdef MIXED_OPERAND_DATA_FORMATS
constexpr auto operand_reconfig = compute_kernel_lib::DataFormatReconfig::Enabled;
#else
constexpr auto operand_reconfig = compute_kernel_lib::DataFormatReconfig::Disabled;
#endif
