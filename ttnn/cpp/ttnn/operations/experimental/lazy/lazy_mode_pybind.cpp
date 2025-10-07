// SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#include "lazy_mode_pybind.hpp"

#include <pybind11/pybind11.h>

#include "ttnn/experimental/lazy/lazy_mode.hpp"

namespace py = pybind11;

namespace ttnn::experimental::lazy {

void bind_lazy_mode(py::module& module) {
    // auto m_lazy = module.def_submodule("lazy", "Lazy evaluation mode control");

    module.def(
        "is_lazy_enabled",
        &is_lazy_enabled,
        R"doc(
            Check if lazy evaluation is currently enabled.

            Returns:
                bool: True if lazy mode is enabled, False otherwise.

            Example:
                >>> import ttnn
                >>> if ttnn.experimental.is_lazy_enabled():
                ...     print("Lazy mode is active")
        )doc");

    module.def(
        "lazy_enable",
        &enable,
        R"doc(
            Enable lazy evaluation mode.

            When lazy mode is enabled, operations are not executed immediately.
            Instead, they build a computation graph that can be optimized and
            executed later.

            Example:
                >>> import ttnn
                >>> ttnn.experimental.lazy.enable()
        )doc");

    module.def(
        "lazy_disable",
        &disable,
        R"doc(
            Disable lazy evaluation mode.

            When lazy mode is disabled, operations are executed eagerly.

            Example:
                >>> import ttnn
                >>> ttnn.experimental.lazy.disable()
        )doc");
}

}  // namespace ttnn::experimental::lazy
