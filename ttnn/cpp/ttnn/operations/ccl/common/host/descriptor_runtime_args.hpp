// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <cstddef>
#include <initializer_list>
#include <vector>

#include <tt-metalium/buffer.hpp>
#include <tt-metalium/program_descriptors.hpp>
#include <tt_stl/assert.hpp>

namespace ttnn::ccl {

// Fabric helpers append onto vector<uint32_t>, so leading tensor-address slots stay 0u until bound here.
inline void emplace_runtime_args_with_leading_buffers(
    tt::tt_metal::KernelDescriptor& kernel,
    const tt::tt_metal::CoreCoord& core,
    std::initializer_list<tt::tt_metal::Buffer*> leading_buffers,
    const std::vector<uint32_t>& raw_args) {
    TT_FATAL(
        raw_args.size() >= leading_buffers.size(),
        "runtime-arg vector ({}) is shorter than the {} leading Buffer* bindings",
        raw_args.size(),
        leading_buffers.size());
    size_t slot = 0;
    tt::tt_metal::KernelDescriptor::RTArgList args;
    args.reserve(raw_args.size());
    for (tt::tt_metal::Buffer* buffer : leading_buffers) {
        TT_FATAL(raw_args[slot] == 0u, "runtime-arg slot {} is reserved for a Buffer* binding and must stay 0", slot);
        TT_FATAL(buffer != nullptr, "runtime-arg slot {} Buffer* is null", slot);
        args.push_back(buffer);
        ++slot;
    }
    for (size_t arg_idx = slot; arg_idx < raw_args.size(); ++arg_idx) {
        args.push_back(raw_args[arg_idx]);
    }
    kernel.emplace_runtime_args(core, args);
}

}  // namespace ttnn::ccl
