// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
// SPDX-License-Identifier: Apache-2.0

#pragma once

/**
 * @file eltwise_helpers.hpp
 * @brief Aggregator — pulls in every eltwise helper header so callers can include one file.
 *
 * Prefer including only the family header(s) the kernel actually uses to keep build times
 * lean (each family header pulls in its own LLK includes). This aggregator is for kernels
 * that mix many ops and don't care about the include surface.
 */

#include "ttnn/cpp/ttnn/kernel_lib/eltwise_chain.hpp"
#include "ttnn/cpp/ttnn/kernel_lib/eltwise_activations.hpp"
#include "ttnn/cpp/ttnn/kernel_lib/eltwise_math.hpp"
#include "ttnn/cpp/ttnn/kernel_lib/eltwise_misc.hpp"
#include "ttnn/cpp/ttnn/kernel_lib/eltwise_predicates.hpp"
#include "ttnn/cpp/ttnn/kernel_lib/eltwise_rounding.hpp"
#include "ttnn/cpp/ttnn/kernel_lib/eltwise_scalar.hpp"
#include "ttnn/cpp/ttnn/kernel_lib/eltwise_special.hpp"
#include "ttnn/cpp/ttnn/kernel_lib/eltwise_trig.hpp"
#include "ttnn/cpp/ttnn/kernel_lib/eltwise_fill.hpp"
#include "ttnn/cpp/ttnn/kernel_lib/eltwise_rand.hpp"
#include "ttnn/cpp/ttnn/kernel_lib/eltwise_binary_sfpu.hpp"
#include "ttnn/cpp/ttnn/kernel_lib/eltwise_optional.hpp"
#include "ttnn/cpp/ttnn/kernel_lib/eltwise_convenience.hpp"
