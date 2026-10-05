// SPDX-FileCopyrightText: © 2025 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0
#pragma once

#include <cfg_defines_base.h>

// Overrides from the selected Quasar variant (tt_llk_quasar/arch/<variant>), if any.
#if __has_include(<cfg_defines_variant.h>)
#include <cfg_defines_variant.h>
#endif
