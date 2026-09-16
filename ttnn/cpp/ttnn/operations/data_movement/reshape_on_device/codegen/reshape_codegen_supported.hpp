// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include "ttnn/tensor/tensor.hpp"

namespace ttnn::operations::data_movement::reshape_codegen {

// Correctness gate for a ROW_MAJOR reshape from `input` to a tensor whose last dimension is
// `out_last_dim_elements`, both interleaved (DRAM or L1) and sharing `input`'s buffer type. This
// is the transport's only shape it can serve: it moves ROW_MAJOR pages byte-for-byte through a
// changed last-dimension width, so it never touches TILE layout, sharding, or a placement change.
// Consulted by the routed public entry (as the correctness half of `supported && !demoted`) and by
// `prim::reshape_codegen`'s validate; the two must agree exactly.
bool supported_by_codegen(const Tensor& input, uint32_t out_last_dim_elements, const MemoryConfig& output_mem_config);

// Perf-demotion gate: correct but not worth the codegen path. Routing-only -- consulted by
// `ttnn::reshape` alone, never by validate and never by `reshape_force_codegen`.
bool is_demoted(const Tensor& input, uint32_t out_last_dim_elements, const MemoryConfig& output_mem_config);

}  // namespace ttnn::operations::data_movement::reshape_codegen
