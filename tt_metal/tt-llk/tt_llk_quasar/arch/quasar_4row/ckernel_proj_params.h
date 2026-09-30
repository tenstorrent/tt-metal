// SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0
#pragma once

// Quasar variant with a 4-row FPU. Start from the base Quasar parameters and override what differs.
#include_next <ckernel_proj_params.h>

#undef MATH_ROWS
#define MATH_ROWS 0x00000004 // = 4 in decimal
