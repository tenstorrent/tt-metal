// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <optional>

#include "ttnn/types.hpp"

namespace ttnn::operations::data_movement::detail {

// Verification-only entry points that pin which permute implementation runs.
//
// `ttnn.permute` chooses on its own and takes no argument to override that: which implementation
// serves a call is an internal decision. These exist because the choice still has to be
// *checkable* -- the guarantees are bit-exactness against the existing implementation over the
// supported scope and a device-time win, and neither is observable through an entry point that
// silently picks one. They are bound only under `ttnn._ttnn.operations.data_movement` and are
// deliberately not registered into the `ttnn.*` namespace, so they are reachable from tests and
// benchmarking without being part of the public API. This header is likewise kept out of the
// installed `api` file set, and they sit in `detail` rather than alongside the real op entry so
// that a caller reaching for one has to say so.
//
// Prefer `ttnn::permute` everywhere else: it already declines the cases the second entry rejects.

// The existing native implementation, unconditionally. Front-end answers that belong to it -- the
// identity/no-op shortcut, the rank<4 promotion, the transpose decompositions -- still apply, so
// this is what a routed-to-native call does.
ttnn::Tensor permute_force_native(
    const ttnn::Tensor& input_tensor,
    const ttsl::SmallVector<int64_t>& dims,
    const std::optional<MemoryConfig>& memory_config = std::nullopt,
    float pad_value = 0.0f);

// The generated implementation, unconditionally. Throws for a case outside its support scope
// instead of falling back, so that a comparison against the native path cannot silently end up
// measuring native twice and reporting it as agreement. Dispatches past the no-op shortcut too:
// forced means a program runs.
ttnn::Tensor permute_force_codegen(
    const ttnn::Tensor& input_tensor,
    const ttsl::SmallVector<int64_t>& dims,
    const std::optional<MemoryConfig>& memory_config = std::nullopt,
    float pad_value = 0.0f);

}  // namespace ttnn::operations::data_movement::detail
