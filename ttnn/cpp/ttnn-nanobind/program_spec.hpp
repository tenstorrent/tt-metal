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
// On an instance of bound_type itself it sets the class's own attributes and rejects any other name with
// AttributeError, as the classes without dynamic_attr do; a Python subclass may set attributes of its own. The
// attribute that holds the tensors (_tensor_args) and the instance __dict__ that holds it are read-only for both: only
// set_tensor_owners writes them.
void set_known_attribute(nb::handle self, const nb::str& name, nb::handle value, nb::handle bound_type);

// __delattr__ for the same classes: deleting _tensor_args or __dict__ raises AttributeError, as assigning them does.
void delete_known_attribute(nb::handle self, const nb::str& name);

// Stores owners, the Python objects that keep the tensors of a run args object alive, in its _tensor_args attribute.
// PyObject_GenericSetAttr skips type(self).__setattr__, so a Python subclass's __setattr__ cannot drop them. Call it
// before the C++ object refers to the new tensors: if it fails, the object is unchanged.
void set_tensor_owners(nb::handle self, nb::handle owners);

}  // namespace ttnn::program_spec
