// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <optional>

#include "ttnn/types.hpp"

namespace ttnn::operations::data_movement::detail {

// Verification-only entry points that pin which repeat_interleave implementation runs.
//
// `ttnn.repeat_interleave` chooses on its own and takes no argument to override that: which
// implementation serves a call is an internal decision. These exist because the choice still has
// to be *checkable* -- the guarantees are bit-exactness against the existing implementation over
// the supported scope and a device-time win, and neither is observable through an entry point that
// silently picks one. They are bound only under `ttnn._ttnn.operations.data_movement` and are
// deliberately not registered into the `ttnn.*` namespace, so they are reachable from tests and
// benchmarking without being part of the public API. This header is likewise kept out of the
// installed `api` file set, and they sit in `detail` rather than alongside the real op entry so
// that a caller reaching for one has to say so.
//
// Prefer `ttnn::repeat_interleave` everywhere else: it already declines the cases the second entry
// rejects.

// The existing composite/native implementation, unconditionally.
ttnn::Tensor repeat_interleave_force_native(
    const ttnn::Tensor& input_a,
    uint32_t repeats,
    int32_t dim,
    const std::optional<MemoryConfig>& output_mem_config = std::nullopt);

// The generated implementation, unconditionally. Throws for a case outside its support scope
// instead of falling back, so that a comparison against the native path cannot silently end up
// measuring native twice and reporting it as agreement.
ttnn::Tensor repeat_interleave_force_codegen(
    const ttnn::Tensor& input_a,
    uint32_t repeats,
    int32_t dim,
    const std::optional<MemoryConfig>& output_mem_config = std::nullopt);

}  // namespace ttnn::operations::data_movement::detail
