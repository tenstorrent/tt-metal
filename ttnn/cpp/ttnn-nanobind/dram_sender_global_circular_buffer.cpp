// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include "dram_sender_global_circular_buffer.hpp"

#include <cstdint>
#include <utility>
#include <vector>

#include <nanobind/nanobind.h>
#include <nanobind/stl/pair.h>
#include <nanobind/stl/vector.h>

#include "ttnn/dram_sender_global_circular_buffer.hpp"

namespace ttnn::dram_sender_global_circular_buffer {

void py_module_types(nb::module_& mod) {
    nb::class_<DramSenderGlobalCircularBuffer>(mod, "dram_sender_global_circular_buffer")
        .def("size", &DramSenderGlobalCircularBuffer::size)
        .def("sender_cores", &DramSenderGlobalCircularBuffer::sender_cores, nb::rv_policy::reference_internal)
        .def("receiver_cores", &DramSenderGlobalCircularBuffer::receiver_cores, nb::rv_policy::reference_internal);
}

void py_module(nb::module_& mod) {
    mod.def(
        "create_dram_sender_global_circular_buffer",
        nb::overload_cast<IDevice*, const std::vector<std::pair<uint32_t, CoreRangeSet>>&, uint32_t, BufferType>(
            &ttnn::dram_sender_global_circular_buffer::create_dram_sender_global_circular_buffer),
        nb::keep_alive<0, 1>(),
        nb::arg("device"),
        nb::arg("bank_to_receivers"),
        nb::arg("size"),
        nb::arg("buffer_type") = tt::tt_metal::BufferType::L1,
        R"doc(
            Create a DramSenderGlobalCircularBuffer with senders on programmable DRAM cores.

            Picks the unused DRAM subchannel per requested bank and wires each one as a sender
            to its receiver CoreRangeSet. Receiver sets across senders must be disjoint.

            Args:
                device: The device to create the buffer on.
                bank_to_receivers: List of (bank_id, receivers) pairs.
                size: Per-receiver fifo size in bytes.
                buffer_type: Buffer type (L1 or L1_SMALL).
            )doc");

    mod.def(
        "create_dram_sender_global_circular_buffer",
        nb::overload_cast<MeshDevice*, const std::vector<std::pair<uint32_t, CoreRangeSet>>&, uint32_t, BufferType>(
            &ttnn::dram_sender_global_circular_buffer::create_dram_sender_global_circular_buffer),
        nb::keep_alive<0, 1>(),
        nb::arg("mesh_device"),
        nb::arg("bank_to_receivers"),
        nb::arg("size"),
        nb::arg("buffer_type") = tt::tt_metal::BufferType::L1,
        R"doc(
            MeshDevice variant of create_dram_sender_global_circular_buffer.
            )doc");
}

}  // namespace ttnn::dram_sender_global_circular_buffer
