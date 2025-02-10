// SPDX-FileCopyrightText: © 2024 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include "global_semaphore.hpp"

#include <tt-metalium/global_semaphore.hpp>
#include "cpp/ttnn/global_semaphore.hpp"
#include "pybind11/pybind11.h"
#include "pybind11/stl.h"

namespace ttnn::global_semaphore {

void py_module_types(py::module& module) {
    py::class_<GlobalSemaphore, std::shared_ptr<GlobalSemaphore>>(module, "global_sempahore");
    py::class_<MultiDeviceGlobalSemaphore>(module, "multi_device_global_semaphore");
}

void py_module(py::module& module) {
    // Single Device APIs
    module.def(
        "create_global_semaphore",
        py::overload_cast<IDevice*, const CoreRangeSet&, uint32_t, BufferType>(
            &ttnn::global_semaphore::create_global_semaphore),
        py::arg("device"),
        py::arg("cores"),
        py::arg("initial_value"),
        py::arg("buffer_type") = tt::tt_metal::BufferType::L1,
        R"doc(
            Create a GlobalSemaphore Object on a single device.

            Args:
                device (Device): The device on which to create the global semaphore.
                cores (CoreRangeSet): The cores on which the global semaphore will be used for synchronization.
                initial_value (int): The initial value of the global semaphore.
                buffer_type (BufferType): The type of buffer to use for the global semaphore.
            )doc");

    module.def(
        "get_global_semaphore_address",
        py::overload_cast<const GlobalSemaphore&>(&get_global_semaphore_address),
        py::arg("global_semaphore"),
        R"doc(
            Get the address of the global semaphore.

            Args:
                global_semaphore (GlobalSemaphore): The global semaphore object.
            )doc");

    module.def(
        "reset_global_semaphore_value",
        py::overload_cast<const GlobalSemaphore&, uint32_t>(&reset_global_semaphore_value),
        py::arg("global_semaphore"),
        py::arg("reset_value"),
        R"doc(
            Reset the value of the global semaphore.

            Args:
                global_semaphore (GlobalSemaphore): The global semaphore object.
                reset_value (int): The value to reset the global semaphore to.
            )doc");

    // Multi Device APIs
    module.def(
        "create_global_semaphore",
        py::overload_cast<MeshDevice*, const CoreRangeSet&, uint32_t, BufferType>(
            &ttnn::global_semaphore::create_global_semaphore),
        py::arg("mesh_device"),
        py::arg("cores"),
        py::arg("initial_value"),
        py::arg("buffer_type") = tt::tt_metal::BufferType::L1,
        R"doc(
            Create a GlobalSemaphore Object on a single device.

            Args:
                mesh_device (MeshDevice): The mesh device on which to create the global semaphore.
                cores (CoreRangeSet): The cores on which the global semaphore will be used for synchronization.
                initial_value (int): The initial value of the global semaphore.
                buffer_type (BufferType): The type of buffer to use for the global semaphore.
            )doc");

    module.def(
        "create_global_semaphore_with_same_address",
        &ttnn::global_semaphore::create_global_semaphore_with_same_address,
        py::arg("mesh_device"),
        py::arg("cores"),
        py::arg("initial_value"),
        py::arg("buffer_type") = tt::tt_metal::BufferType::L1,
        py::arg("attempts") = 1000,
        py::arg("search_max") = false,
        R"doc(
            Create a GlobalSemaphore Object on multiple devices with the same address by iteratively creating global semaphore until alignment is found.
            Fails if the address is not the same on all devices after the specified number of attempts.
            Note: Temperary API until mesh allocator is implemented.

            Args:
                mesh_device (MeshDevice): The mesh device on which to create the global semaphore.
                cores (CoreRangeSet): The cores on which the global semaphore will be used for synchronization.
                initial_value (int): The initial value of the global semaphore.
                buffer_type (BufferType): The type of buffer to use for the global semaphore.
                attempts (int): The number of attempts to create the global semaphore with the same address.
                search_max (bool): Whether to search for the maximum address. (default: False, which searches for the minimum address)
            )doc");

    module.def(
        "get_global_semaphore_address",
        py::overload_cast<const MultiDeviceGlobalSemaphore&>(&get_global_semaphore_address),
        py::arg("global_semaphore"),
        R"doc(
            Get the address of the global semaphore.

            Args:
                global_semaphore (GlobalSemaphore): The global semaphore object.
            )doc");

    module.def(
        "reset_global_semaphore_value",
        py::overload_cast<const MultiDeviceGlobalSemaphore&, uint32_t>(&reset_global_semaphore_value),
        py::arg("global_semaphore"),
        py::arg("reset_value"),
        R"doc(
            Reset the value of the global semaphore.

            Args:
                global_semaphore (GlobalSemaphore): The global semaphore object.
                reset_value (int): The value to reset the global semaphore to.
            )doc");
}

}  // namespace ttnn::global_semaphore
