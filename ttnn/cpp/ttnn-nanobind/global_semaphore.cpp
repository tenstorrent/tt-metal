// SPDX-FileCopyrightText: © 2025 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include "global_semaphore.hpp"

#include <tt_stl/reflection.hpp>
#include <cstdint>

#include <nanobind/nanobind.h>

#include "ttnn/global_semaphore.hpp"
#include <tt-metalium/global_semaphore.hpp>
#include <tt-metalium/experimental/forge_backdoor/global_semaphore.hpp>
#include <nanobind/stl/optional.h>

namespace ttnn::global_semaphore {

void py_module_types(nb::module_& mod) { nb::class_<GlobalSemaphore>(mod, "global_semaphore"); }

void py_module(nb::module_& mod) {
    mod.def(
        "experimental_create_global_semaphore_at_address",
        [](tt::tt_metal::distributed::MeshDevice& mesh,
           const tt::tt_metal::CoreRangeSet& cores,
           std::optional<uint32_t> initial_value,
           tt::tt_metal::BufferType buffer_type,
           uint64_t address) {
            // The Metal entry point validates every destination before initializing storage.
            using namespace tt::tt_metal;
            TT_FATAL(buffer_type == BufferType::L1, "Address-based semaphores require ordinary SRAM");
            return tt::tt_metal::experimental::CreateGlobalSemaphore(mesh, cores, initial_value, buffer_type, address);
        },
        nb::arg("mesh_device"),
        nb::arg("cores"),
        nb::arg("initial_value"),
        nb::arg("buffer_type"),
        nb::arg("address"),
        nb::keep_alive<0, 1>(),
        "Initialize a non-owning semaphore at one address on all supplied mesh devices and cores. "
        "Metal validates alignment and the padded write range against allocatable SRAM; the caller retains backing.");
    mod.def(
        "create_global_semaphore",
        &ttnn::global_semaphore::create_global_semaphore,
        nb::arg("mesh_device"),
        nb::arg("cores"),
        nb::arg("initial_value"),
        nb::arg("buffer_type") = nb::cast(tt::tt_metal::BufferType::L1),
        R"doc(
            Create a GlobalSemaphore Object on a device.

            Args:
                mesh_device (MeshDevice): The mesh device on which to create the global semaphore.
                cores (CoreRangeSet): The cores on which the global semaphore will be used for synchronization.
                initial_value (int): The initial value of the global semaphore.
                buffer_type (BufferType): The type of buffer to use for the global semaphore.
            )doc");

    mod.def(
        "get_global_semaphore_address",
        &get_global_semaphore_address,
        nb::arg("global_semaphore"),
        R"doc(
            Get the address of the global semaphore.

            Args:
                global_semaphore (GlobalSemaphore): The global semaphore object.
            )doc");

    mod.def(
        "reset_global_semaphore_value",
        &reset_global_semaphore_value,
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
