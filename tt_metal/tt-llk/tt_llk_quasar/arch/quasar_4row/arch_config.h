// SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0
#pragma once

#define MATH_ROWS 4

#include "../quasar/arch_config.h"

static_assert(ckernel::arch::fpu_rows == 4, "quasar_4row requires a four-row FPU");
