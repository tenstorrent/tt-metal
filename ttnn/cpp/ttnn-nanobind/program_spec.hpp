// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include "ttnn-nanobind/nanobind_fwd.hpp"

namespace nanobind {
class handle;
class str;
}  // namespace nanobind

namespace ttnn::program_spec {
namespace nb = nanobind;

// Python classes for the Metal 2.0 ProgramSpec and ProgramRunArgs types (tt::tt_metal::experimental),
// re-exported under ttnn.experimental.
void py_module_types(nb::module_& mod);

// __setattr__ for the classes that keep their tensors alive in a Python attribute (and so need nb::dynamic_attr()).
// On an instance of bound_type itself it sets the class's own attributes and that one, and rejects any other name with
// AttributeError, as the classes without dynamic_attr do; a Python subclass may set any attribute.
void set_known_attribute(nb::handle self, const nb::str& name, nb::handle value, nb::handle bound_type);

// The attribute in which a Python ProgramRunArgs holds its tensor_args dict. The C++ object refers to the tensors'
// MeshTensors, so the dict keeps them alive.
inline constexpr const char* TENSOR_ARGS_ATTRIBUTE = "_tensor_args";

}  // namespace ttnn::program_spec
