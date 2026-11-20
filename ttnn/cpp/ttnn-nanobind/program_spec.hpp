// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <string_view>
#include <utility>
#include <vector>

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

// A tensor argument of run args made in Python: the tensor parameter's name and the ttnn.Tensor bound to it. The C++
// run args keep only the tensor's MeshTensor, which the per-device views of one tensor share.
using PythonTensorArg = std::pair<std::string_view, const ttnn::Tensor*>;

// The tensor arguments of a ProgramRunArgs made in Python, in tensor_args order; none for run args made in C++. The
// names and tensors belong to the Python object: use them only while holding the GIL, and run no Python code meanwhile.
std::vector<PythonTensorArg> python_tensor_args(const tt::tt_metal::ProgramRunArgs& run_args);

// The tensor arguments of each program of a MeshProgramRunArgs made in Python, in the order of its programs; none for
// run args made in C++. The same rules apply.
std::vector<std::vector<PythonTensorArg>> python_tensor_args(const ttnn::experimental::MeshProgramRunArgs& run_args);

}  // namespace ttnn::program_spec
