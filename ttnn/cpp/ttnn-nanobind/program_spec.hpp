// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include "ttnn-nanobind/nanobind_fwd.hpp"
#include "ttnn/operations/generic/generic_op.hpp"

namespace nanobind {
class object;
}  // namespace nanobind

namespace ttnn::program_spec {
namespace nb = nanobind;

// Python classes for the Metal 2.0 ProgramSpec and ProgramRunArgs types (tt::tt_metal) and for generic_op's
// MeshProgramSpec and MeshProgramRunArgs, re-exported under ttnn.experimental.
void py_module_types(nb::module_& mod);

// The tensor_args dict (tensor parameter name to ttnn.Tensor) of a ProgramRunArgs made in Python, or None for run
// args made in C++. The C++ object keeps only the tensors' MeshTensors, which per-device views of one tensor share,
// while the dict keeps the tensors themselves. Read it only, with the GIL held.
nb::object python_tensor_args(const ttnn::experimental::ProgramRunArgs& run_args);

// For a MeshProgramRunArgs made in Python, the list of those dicts, one per program in the order of its programs; None
// for run args made in C++. Read it only, with the GIL held.
nb::object python_tensor_args(const ttnn::experimental::MeshProgramRunArgs& run_args);

}  // namespace ttnn::program_spec
