// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <cstddef>
#include <functional>
#include <string_view>

#include "ttnn-nanobind/nanobind_fwd.hpp"

namespace tt::tt_metal {
struct ProgramRunArgs;
}  // namespace tt::tt_metal

namespace ttnn {
class Tensor;
}  // namespace ttnn

namespace ttnn::experimental {
struct MeshProgramRunArgs;
}  // namespace ttnn::experimental

namespace ttnn::program_spec {
namespace nb = nanobind;

// Python classes for the Metal 2.0 ProgramSpec and ProgramRunArgs types (tt::tt_metal) and for generic_op's
// MeshProgramSpec and MeshProgramRunArgs, re-exported under ttnn.experimental.
void py_module_types(nb::module_& mod);

// Calls visit with the name and the ttnn.Tensor of each tensor argument of a ProgramRunArgs made in Python, in
// tensor_args order. The C++ run args keep only the tensor's MeshTensor, which the per-device views of one tensor
// share. Calls nothing for run args made in C++. Needs the GIL; visit must run no Python code and must not keep the
// name or the tensor after it returns.
void for_each_python_tensor_arg(
    const tt::tt_metal::ProgramRunArgs& run_args,
    const std::function<void(std::string_view name, const ttnn::Tensor& tensor)>& visit);

// The same for each program of a MeshProgramRunArgs made in Python, with the program's index in its programs.
void for_each_python_tensor_arg(
    const ttnn::experimental::MeshProgramRunArgs& run_args,
    const std::function<void(std::size_t program, std::string_view name, const ttnn::Tensor& tensor)>& visit);

}  // namespace ttnn::program_spec
