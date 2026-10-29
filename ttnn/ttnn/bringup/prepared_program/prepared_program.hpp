// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

// A generic_op program prepared once and dispatched with only its tensor addresses patched.
//
// generic_op takes a MeshProgramDescriptor every call and, on a program-cache hit, re-applies every runtime arg of
// every core of every chip from it (the descriptor slow path) - and its callers rebuild the descriptor to rebind
// their CBs to arenas taken again each call. A PreparedProgram keeps the descriptor for good: prepare() records
// which runtime-arg slots hold an io tensor's address (exact matches only) and which CBs an io tensor backs; each
// run patches just those in the cached program (the program factory's override_runtime_arguments - the
// descriptor-era cache-hit hook), so a call's host cost is the patch count, not the program size.
//
// Contract: every per-call tensor whose address a kernel takes (a runtime arg holding it, or a CB on it) is passed
// in io_tensors - at prepare and at every run, at the same positions; addresses of tensors outside io_tensors and
// every other runtime arg value are frozen at prepare. A constant equal to an io tensor's address is taken for it
// (prepare refuses two distinct io tensors at one address).

#include <cstdint>
#include <memory>
#include <vector>

#include <tt-metalium/experimental/mesh_program_descriptor.hpp>
#include <tt_stl/reflection.hpp>

#include "ttnn/tensor/tensor.hpp"

namespace ttnn::operations::bringup::prepared_program {

struct RtPatch {  // runtime arg `arg` of kernel `kernel` on core (x, y) := address of io tensor `io`
    uint32_t kernel = 0, x = 0, y = 0, arg = 0, io = 0;
};
struct CommonRtPatch {  // common runtime arg `arg` of kernel `kernel` := address of io tensor `io`
    uint32_t kernel = 0, arg = 0, io = 0;
};
struct ValPatch {  // runtime arg `arg` of kernel `kernel` on core (x, y) := run value `slot` (value_slots option)
    uint32_t kernel = 0, x = 0, y = 0, arg = 0, slot = 0;
};
struct CbPatch {  // CB `cb` (descriptor order) onto io tensor `io`'s buffer, its descriptor's address offset kept
    uint32_t cb = 0, io = 0;
};

struct Prepared {
    tt::tt_metal::experimental::MeshProgramDescriptor desc;
    // per mesh program (desc.mesh_programs order)
    std::vector<std::vector<RtPatch>> rt;
    std::vector<std::vector<CommonRtPatch>> crt;
    std::vector<std::vector<CbPatch>> cbs;
    std::vector<std::vector<ValPatch>> val;  // (value_slots: empty unless prepared with value slots)
    // per mesh program, per io tensor: its rt / crt patches (indices); a hit rewrites an io tensor's slots only when
    // the first of them does not already hold its address (they are always written together)
    std::vector<std::vector<std::vector<uint32_t>>> rt_of_io, crt_of_io;
    size_t n_io = 0;
    size_t n_values = 0;  // (value_slots option: the values each run passes)
    ttsl::hash::hash_t hash = 0;

    // the mesh program covering coord (desc.mesh_programs index), or -1
    int program_of(const ttnn::MeshCoordinate& coord) const;
    size_t patch_count() const;
};

// A per-run value (option value_slots): runtime arg `arg` of kernel `kernel` on every core of every mesh program
// where that kernel has it := run value `slot`. Such a slot is never taken for an io tensor's address and is left out
// of the cache key (a scalar that changes every call, e.g. a write offset, without a new prepare).
struct ValueSlot {
    uint32_t kernel = 0, arg = 0, slot = 0;
};

// Records the patches from the descriptor and the io tensors it was built against (see the contract above).
// value_slots (default none: the behaviour before the option) adds the per-run values.
std::shared_ptr<const Prepared> prepare(
    tt::tt_metal::experimental::MeshProgramDescriptor desc,
    const std::vector<Tensor>& io_tensors,
    const std::vector<ValueSlot>& value_slots = {});

// Dispatches the prepared program on these io tensors (the output is the last, as generic_op); `values` holds one
// value per value slot (prepared.n_values; none by default).
Tensor run(
    const std::shared_ptr<const Prepared>& prepared,
    const std::vector<Tensor>& io_tensors,
    const std::vector<uint32_t>& values = {});

}  // namespace ttnn::operations::bringup::prepared_program
