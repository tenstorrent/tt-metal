// SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0
#pragma once

#include <ckernel_proj_params_base.h>

// Overrides from the selected Quasar variant (tt_llk_quasar/arch/<variant>), if any.
#if __has_include(<ckernel_proj_params_variant.h>)
#include <ckernel_proj_params_variant.h>
#endif
