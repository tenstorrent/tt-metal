// SPDX-FileCopyrightText: (c) 2026 Tenstorrent AI ULC
// SPDX-License-Identifier: Apache-2.0

#pragma once

/**
 * @file eltwise_helpers.hpp
 * @brief Aggregator for the V2 eltwise helper family.
 *
 * Includes the chain core and op-category headers so callers can pull in the
 * full surface from one place. Phase 1 ships the core + math ops only; binary,
 * ternary, activations, predicates, etc. land as follow-up work.
 */

#include "ttnn/cpp/ttnn/kernel_lib/eltwise_chain.hpp"
#include "ttnn/cpp/ttnn/kernel_lib/eltwise_math.hpp"
