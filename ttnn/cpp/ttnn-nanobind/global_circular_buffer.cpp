// SPDX-FileCopyrightText: © 2025 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include "global_circular_buffer.hpp"

#include <cstdint>
#include <utility>
#include <vector>

#include <nanobind/nanobind.h>
#include <nanobind/stl/pair.h>
#include <nanobind/stl/vector.h>

#include "ttnn/global_circular_buffer.hpp"
namespace ttnn::global_circular_buffer {

void py_module_types(nb::module_& mod) {
    nb::class_<GlobalCircularBuffer>(mod, "global_circular_buffer")
        .def("size", &GlobalCircularBuffer::size)
        .def("sender_core_type", [](const GlobalCircularBuffer& gcb) {
            return gcb.sender_core_type() == tt::tt_metal::experimental::SenderCoreType::Worker ? "worker" : "dram";
        });
}

void py_module(nb::module_& mod) {
    // Single Device APIs
    mod.def(
        "create_global_circular_buffer",
        nb::overload_cast<IDevice*, const std::vector<std::pair<CoreCoord, CoreRangeSet>>&, uint32_t, BufferType>(
            &ttnn::global_circular_buffer::create_global_circular_buffer),
        nb::keep_alive<0, 1>(),  // test
        nb::arg("device"),
        nb::arg("sender_receiver_core_mapping"),
        nb::arg("size"),
        nb::arg("buffer_type") = tt::tt_metal::BufferType::L1,
        R"doc(
            Create a GlobalCircularBuffer Object on a single device.

            Args:
                device (Device): The device on which to create the global circular buffer.
                sender_receiver_core_mapping (List[Tuple[CoreCoord, CoreRangeSet]]): The mapping of remote sender to remote receiver cores for the circular buffer.
                size (int): Size of the global circular buffer per core in bytes.
                buffer_type (BufferType): The type of buffer to use for the global circular buffer.\
            )doc");

    // Multi Device APIs
    mod.def(
        "create_global_circular_buffer",
        nb::overload_cast<MeshDevice*, const std::vector<std::pair<CoreCoord, CoreRangeSet>>&, uint32_t, BufferType>(
            &ttnn::global_circular_buffer::create_global_circular_buffer),
        nb::keep_alive<0, 1>(),  // test
        nb::arg("mesh_device"),
        nb::arg("sender_receiver_core_mapping"),
        nb::arg("size"),
        nb::arg("buffer_type") = tt::tt_metal::BufferType::L1,
        R"doc(
            Create a GlobalCircularBuffer Object on a single device.

            Args:
                mesh_device (MeshDevice): The mesh device on which to create the global circular buffer.
                sender_receiver_core_mapping (List[Tuple[CoreCoord, CoreRangeSet]]): The mapping of remote sender to remote receiver cores for the circular buffer.
                size (int): Size of the global circular buffer per core in bytes.
                buffer_type (BufferType): The type of buffer to use for the global circular buffer.
            )doc");

    // DRAM-sender variants (senders are programmable DRAM cores identified by bank id).
    mod.def(
        "create_global_circular_buffer_with_dram_senders",
        nb::overload_cast<IDevice*, const std::vector<std::pair<uint32_t, CoreRangeSet>>&, uint32_t, BufferType>(
            &ttnn::global_circular_buffer::create_global_circular_buffer_with_dram_senders),
        nb::keep_alive<0, 1>(),
        nb::arg("device"),
        nb::arg("bank_to_receivers"),
        nb::arg("size"),
        nb::arg("buffer_type") = tt::tt_metal::BufferType::L1,
        R"doc(
            Create a GlobalCircularBuffer where senders are programmable DRAM cores (Blackhole DRISCs).
            Each bank id is mapped to an unused DRAM subchannel; receiver sets across senders must
            be disjoint and must not collide with the DRAM sender physical NOC coords.

            Args:
                device: The device to create the buffer on.
                bank_to_receivers: List of (bank_id, receivers) pairs.
                size: Per-receiver fifo size in bytes.
                buffer_type: Buffer type (L1 or L1_SMALL).
            )doc");

    mod.def(
        "create_global_circular_buffer_with_dram_senders",
        nb::overload_cast<MeshDevice*, const std::vector<std::pair<uint32_t, CoreRangeSet>>&, uint32_t, BufferType>(
            &ttnn::global_circular_buffer::create_global_circular_buffer_with_dram_senders),
        nb::keep_alive<0, 1>(),
        nb::arg("mesh_device"),
        nb::arg("bank_to_receivers"),
        nb::arg("size"),
        nb::arg("buffer_type") = tt::tt_metal::BufferType::L1,
        R"doc(
            MeshDevice variant of create_global_circular_buffer_with_dram_senders.
            )doc");
}

}  // namespace ttnn::global_circular_buffer
