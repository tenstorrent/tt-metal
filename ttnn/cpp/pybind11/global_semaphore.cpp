// SPDX-FileCopyrightText: © 2024 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include "global_semaphore.hpp"

#include "tt_metal/impl/buffers/global_semaphore.hpp"
#include "ttnn/cpp/ttnn/global_semaphore.hpp"
#include "pybind11/pybind11.h"

namespace ttnn::global_semaphore {

void py_module_types(py::module& module) {
    py::class_<GlobalSemaphore, std::shared_ptr<GlobalSemaphore>>(module, "global_sempahore");
    py::class_<MultiDeviceGlobalSemaphore>(module, "multi_device_global_semaphore");
}

void py_module(py::module& module) {
    // Single Device APIs
    module.def(
        "create_global_semaphore",
        [](Device* device,
           const CoreRangeSet& cores,
           uint32_t initial_value,
           BufferType buffer_type,
           const std::vector<SubDeviceId>& sub_device_ids) {
            return ttnn::global_semaphore::create_global_semaphore(
                device, cores, initial_value, buffer_type, sub_device_ids);
        },
        py::arg("device"),
        py::arg("cores"),
        py::arg("initial_value"),
        py::arg("buffer_type") = tt::tt_metal::BufferType::L1,
        py::arg("sub_device_ids") = std::vector<SubDeviceId>(),
        R"doc(
            Create a GlobalSemaphore Object on a single device.

            Args:
                device (Device): The device on which to create the global semaphore.
                cores (CoreRangeSet): The cores on which the global semaphore will be used for synchronization.
                initial_value (int): The initial value of the global semaphore.
                buffer_type (BufferType): The type of buffer to use for the global semaphore.
                sub_device_ids (List[ttnn.SubDeviceIds]): Sub-device IDs to wait on before writing the global semaphore value to device.
                Defaults to waiting on all sub-devices.
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
        [](const GlobalSemaphore& global_semaphore,
           uint32_t reset_value,
           const std::vector<SubDeviceId>& sub_device_ids) {
            ttnn::global_semaphore::reset_global_semaphore_value(global_semaphore, reset_value, sub_device_ids);
        },
        py::arg("global_semaphore"),
        py::arg("reset_value"),
        py::arg("sub_device_ids") = std::vector<SubDeviceId>(),
        R"doc(
            Reset the value of the global semaphore.

            Args:
                global_semaphore (GlobalSemaphore): The global semaphore object.
                reset_value (int): The value to reset the global semaphore to.
                sub_device_ids (List[ttnn.SubDeviceIds]): Sub-device IDs to wait on before writing the global semaphore value to device.
                Defaults to waiting on all sub-devices.
            )doc");

    // Multi Device APIs
    module.def(
        "create_global_semaphore",
        [](MeshDevice* mesh_device,
           const CoreRangeSet& cores,
           uint32_t initial_value,
           BufferType buffer_type,
           const std::vector<SubDeviceId>& sub_device_ids) {
            return ttnn::global_semaphore::create_global_semaphore(
                mesh_device, cores, initial_value, buffer_type, sub_device_ids);
        },
        py::arg("mesh_device"),
        py::arg("cores"),
        py::arg("initial_value"),
        py::arg("buffer_type") = tt::tt_metal::BufferType::L1,
        py::arg("sub_device_ids") = std::vector<SubDeviceId>(),
        R"doc(
            Create a GlobalSemaphore Object on a single device.

            Args:
                mesh_device (MeshDevice): The mesh device on which to create the global semaphore.
                cores (CoreRangeSet): The cores on which the global semaphore will be used for synchronization.
                initial_value (int): The initial value of the global semaphore.
                buffer_type (BufferType): The type of buffer to use for the global semaphore.
                sub_device_ids (List[ttnn.SubDeviceIds]): Sub-device IDs to wait on before writing the global semaphore value to device.
                Defaults to waiting on all sub-devices.
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
        [](const MultiDeviceGlobalSemaphore& global_semaphore,
           uint32_t reset_value,
           const std::vector<SubDeviceId>& sub_device_ids) {
            ttnn::global_semaphore::reset_global_semaphore_value(global_semaphore, reset_value, sub_device_ids);
        },
        py::arg("global_semaphore"),
        py::arg("reset_value"),
        py::arg("sub_device_ids") = std::vector<SubDeviceId>(),
        R"doc(
            Reset the value of the global semaphore.

            Args:
                global_semaphore (GlobalSemaphore): The global semaphore object.
                reset_value (int): The value to reset the global semaphore to.
                sub_device_ids (List[ttnn.SubDeviceIds]): Sub-device IDs to wait on before writing the global semaphore value to device.
            )doc");
}

}  // namespace ttnn::global_semaphore
