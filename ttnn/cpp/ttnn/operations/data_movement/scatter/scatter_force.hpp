// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <optional>
#include <string>

#include "ttnn/types.hpp"

namespace ttnn::operations::data_movement::detail {

// Verification-only entry points that pin which scatter implementation runs.
//
// `ttnn.scatter` chooses on its own and takes no argument to override that: which implementation
// serves a call is an internal decision. These exist because the choice still has to be *checkable*
// -- the guarantees are bit-exactness against the existing implementation over the supported scope
// and a device-time win, and neither is observable through an entry point that silently picks one.
// They are bound only under `ttnn._ttnn.operations.data_movement` and are deliberately not
// registered into the `ttnn.*` namespace, so they are reachable from tests and benchmarking without
// being part of the public API. This header is likewise kept out of the installed `api` file set, and
// they sit in `detail` rather than alongside the real op entries so that a caller reaching for one
// has to say so.
//
// Prefer `ttnn::scatter` everywhere else: it already declines the cases the second entry rejects.

// The existing native implementation, unconditionally.
Tensor scatter_force_native(
    const Tensor& input_tensor,
    const int32_t& dim,
    const Tensor& index_tensor,
    const Tensor& source_tensor,
    const std::optional<MemoryConfig>& output_memory_config = std::nullopt,
    const std::optional<std::string>& opt_reduction_string = std::nullopt,
    const std::optional<CoreRangeSet>& sub_core_grid = std::nullopt);

// The generated implementation, unconditionally. Throws for a case outside its support scope instead
// of falling back, so that a comparison against the native path cannot silently end up measuring
// native twice and reporting it as agreement.
Tensor scatter_force_codegen(
    const Tensor& input_tensor,
    const int32_t& dim,
    const Tensor& index_tensor,
    const Tensor& source_tensor,
    const std::optional<MemoryConfig>& output_memory_config = std::nullopt,
    const std::optional<std::string>& opt_reduction_string = std::nullopt,
    const std::optional<CoreRangeSet>& sub_core_grid = std::nullopt);

}  // namespace ttnn::operations::data_movement::detail
