// SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <cstdint>

#include "api/compile_time_args.h"

#define CT_ARG(name) get_named_compile_time_arg_val(name)
#define RT_ARG(name) get_arg_val<uint32_t>(get_named_compile_time_arg_val("rt_" name))
