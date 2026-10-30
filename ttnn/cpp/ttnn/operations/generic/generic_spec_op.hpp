// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <optional>
#include <string>
#include <utility>
#include <vector>

#include <tt-metalium/experimental/metal2_host_api/program_run_args.hpp>
#include <tt-metalium/experimental/metal2_host_api/program_spec.hpp>
#include <tt_stl/reflection.hpp>  // ttsl::hash::hash_t

#include "ttnn/types.hpp"

namespace ttnn {

// Run-time half of a Metal 2.0 generic_op call. Unlike experimental::ProgramRunArgs it binds
// tensors by value, so it can be built from Python; generic_op maps each tensor back to its
// position in io_tensors.
struct ProgramSpecRunArgs {
    std::vector<tt::tt_metal::experimental::KernelRunArgs> kernel_run_args;
    // TensorParameter name -> tensor. Every tensor must also appear in io_tensors.
    std::vector<std::pair<std::string, Tensor>> tensor_args;
    // Overrides the structural hash of the ProgramSpec as the program-cache key.
    std::optional<ttsl::hash::hash_t> custom_program_hash;
};

// Metal 2.0 generic_op: runs a caller-built ProgramSpec on every device of the io tensors' mesh.
// The last element of io_tensors is the preallocated output and is returned.
Tensor generic_op(
    const std::vector<Tensor>& io_tensors,
    const tt::tt_metal::experimental::ProgramSpec& program_spec,
    const ProgramSpecRunArgs& run_args);

}  // namespace ttnn
