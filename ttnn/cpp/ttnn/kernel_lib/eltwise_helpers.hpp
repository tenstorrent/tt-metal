// SPDX-FileCopyrightText: © 2026 Tenstorrent Inc.
// SPDX-License-Identifier: Apache-2.0

#pragma once

/**
 * @file eltwise_helpers.hpp
 * @brief Aggregator for the eltwise helper family.
 *
 * Pulls in the core (chain combinator + pipeline) plus every Tier 1 op
 * category. Tier 2 category files (eltwise_trig, eltwise_rounding, …) get
 * added here as they ship.
 *
 * Usage:
 *   #include "ttnn/cpp/ttnn/kernel_lib/eltwise_helpers.hpp"
 *   using namespace compute_kernel_lib::eltwise;
 *
 *   binary_op_init_common(cb_a, cb_b, cb_out);   // caller still does this
 *
 *   eltwise_pipeline(eltwise_chain(CopyTile<cb_a, Dst::D0>{}, Exp<>{}),
 *                    cb_out, num_tiles);
 *
 * Lives at `compute_kernel_lib::eltwise` so it does not collide with the
 * legacy `compute_kernel_lib::Sfpu*` / `compute_kernel_lib::add` names that
 * still exist in `sfpu_helpers.hpp` and `binary_op_helpers.hpp`. Both can
 * be included in the same kernel during migration.
 */

#include "ttnn/cpp/ttnn/kernel_lib/eltwise_chain.hpp"
#include "ttnn/cpp/ttnn/kernel_lib/eltwise_binary.hpp"
#include "ttnn/cpp/ttnn/kernel_lib/eltwise_mask.hpp"
#include "ttnn/cpp/ttnn/kernel_lib/eltwise_math.hpp"
#include "ttnn/cpp/ttnn/kernel_lib/eltwise_activations.hpp"
#include "ttnn/cpp/ttnn/kernel_lib/eltwise_predicates.hpp"
#include "ttnn/cpp/ttnn/kernel_lib/eltwise_trig.hpp"
#include "ttnn/cpp/ttnn/kernel_lib/eltwise_rounding.hpp"
#include "ttnn/cpp/ttnn/kernel_lib/eltwise_special.hpp"
#include "ttnn/cpp/ttnn/kernel_lib/eltwise_scalar.hpp"
#include "ttnn/cpp/ttnn/kernel_lib/eltwise_misc.hpp"
#include "ttnn/cpp/ttnn/kernel_lib/eltwise_ternary.hpp"
