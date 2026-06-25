// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

// Content hash of a Metal 2.0 ProgramSpec, for use as a program-cache key.
//
// The ProgramSpec fully defines a compiled Program, so hashing it keys the cache
// at program identity: two specs collide iff they describe the same Program. This
// is the correct-by-construction replacement for hand-maintained per-op
// compute_program_hash, which has to be kept in sync with the program by hand.
//
// ProgramSpec is reflection-hashable via ttsl::hash out of the box, so almost all
// of it is hashed generically. The single semantic generic reflection can't express
// is shape *relaxation*: a TensorParameter may declare dynamic_tensor_shape /
// match_padded_shape_only (see advanced_options.hpp), meaning the Program is
// invariant to (part of) the tensor shape, so shapes that differ only where the
// Program doesn't care share a cache entry instead of fragmenting it. Every other
// field is folded in via reflection, so adding a field to
// KernelSpec/DataflowBufferSpec/etc. is picked up automatically; a static_assert in
// the .cpp guards the one place that isn't (the top-level ProgramSpec field set), so
// a new ProgramSpec member can't silently slip out of the key.

#include <cstddef>

#include <tt-metalium/experimental/metal2_host_api/program_spec.hpp>

namespace ttnn::device_operation {

// Content hash of `spec`, suitable as a program-cache key. Combine with the op's
// type hash at the call site so distinct ops with coincidentally-identical specs
// don't collide.
std::size_t program_spec_cache_key(const tt::tt_metal::experimental::ProgramSpec& spec);

}  // namespace ttnn::device_operation
