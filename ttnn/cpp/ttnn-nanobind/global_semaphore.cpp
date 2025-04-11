// SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#include "global_semaphore.hpp"

#include <cstdint>

#include <nanobind/nanobind.h>

#include "ttnn/global_semaphore.hpp"
#include <tt-metalium/global_semaphore.hpp>

namespace ttnn::global_semaphore {

void py_module_types(nb::module_& mod) {
    nb::class_<GlobalSemaphore>(mod, "global_sempahore");
}

void py_module(nb::module_& mod) {
    // Single Device Creation API
    mod.def(
        "create_global_semaphore",
        nb::overload_cast<IDevice*, const CoreRangeSet&, uint32_t, BufferType>(
            &ttnn::global_semaphore::create_global_semaphore),
        nb::arg("device"),
        nb::arg("cores"),
        nb::arg("initial_value"),
        nb::arg("buffer_type") = tt::tt_metal::BufferType::L1,
        R"doc(
            Create a GlobalSemaphore Object on a single device.

            Args:
                device (Device): The device on which to create the global semaphore.
                cores (CoreRangeSet): The cores on which the global semaphore will be used for synchronization.
                initial_value (int): The initial value of the global semaphore.
                buffer_type (BufferType): The type of buffer to use for the global semaphore.
            )doc");

    // MeshDevice Creation API
    mod.def(
        "create_global_semaphore",
        nb::overload_cast<MeshDevice*, const CoreRangeSet&, uint32_t, BufferType>(
            &ttnn::global_semaphore::create_global_semaphore),
        nb::arg("mesh_device"),
        nb::arg("cores"),
        nb::arg("initial_value"),
        nb::arg("buffer_type") = tt::tt_metal::BufferType::L1,
        R"doc(
            Create a GlobalSemaphore Object on a single device.

            Args:
                mesh_device (MeshDevice): The mesh device on which to create the global semaphore.
                cores (CoreRangeSet): The cores on which the global semaphore will be used for synchronization.
                initial_value (int): The initial value of the global semaphore.
                buffer_type (BufferType): The type of buffer to use for the global semaphore.
            )doc");

    mod.def(
        "get_global_semaphore_address",
        nb::overload_cast<const GlobalSemaphore&>(&get_global_semaphore_address),
        nb::arg("global_semaphore"),
        R"doc(
            Get the address of the global semaphore.

            Args:
                global_semaphore (GlobalSemaphore): The global semaphore object.
            )doc");

    mod.def(
        "reset_global_semaphore_value",
        nb::overload_cast<const GlobalSemaphore&, uint32_t>(&reset_global_semaphore_value),
        nb::arg("global_semaphore"),
        nb::arg("reset_value"),
        R"doc(
            Reset the value of the global semaphore.

            Args:
                global_semaphore (GlobalSemaphore): The global semaphore object.
                reset_value (int): The value to reset the global semaphore to.
            )doc");
}

}  // namespace ttnn::global_semaphore
