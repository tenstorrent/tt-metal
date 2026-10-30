// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <cstdint>
#include <limits>
#include <string>
#include <variant>

#include <tt-metalium/experimental/metal2_host_api/program_spec.hpp>

// Experimental: what an op lets a command list patch between replays.
namespace ttnn::experimental {

// A host scalar the op passes to one kernel as a named common runtime argument.
struct CommonRuntimeArgTarget {
    tt::tt_metal::experimental::KernelSpecName kernel;
    std::string arg_name;
    // Largest value the op's program and tensors support.
    uint32_t capacity = std::numeric_limits<uint32_t>::max();
};

// A tensor the op binds to a program tensor parameter. A replacement must have the recorded tensor's spec.
struct TensorTarget {
    tt::tt_metal::experimental::TensorParamName parameter;
};

// One patchable argument of an op. Several entries may share a name when one argument reaches several kernels.
struct CommandListParameter {
    // The op's own name for the argument. The caller of CommandListCapture::record() maps it to a list parameter.
    std::string name;
    std::variant<CommonRuntimeArgTarget, TensorTarget> target;
};

}  // namespace ttnn::experimental
