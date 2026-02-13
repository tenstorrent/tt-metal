// SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#include "hd_socket.hpp"

#include <cstdint>
#include <optional>

#include <nanobind/nanobind.h>
#include <nanobind/stl/optional.h>
#include <nanobind/stl/shared_ptr.h>
#include <nanobind/stl/vector.h>

#include <tt-metalium/experimental/sockets/h2d_socket.hpp>
#include <tt-metalium/experimental/sockets/d2h_socket.hpp>

#include "ttnn/tensor/tensor.hpp"
#include "ttnn/tensor/host_buffer/functions.hpp"

namespace ttnn::hd_socket {

void py_module_types(nb::module_& mod) {
    nb::enum_<tt::tt_metal::distributed::H2DMode>(mod, "H2DMode")
        .value(
            "HOST_PUSH",
            tt::tt_metal::distributed::H2DMode::HOST_PUSH,
            "Host pushes data to device via UMD TLB writes.")
        .value(
            "DEVICE_PULL",
            tt::tt_metal::distributed::H2DMode::DEVICE_PULL,
            "Device pulls data from pinned host memory via PCIe NOC reads.");

    nb::class_<tt::tt_metal::distributed::H2DSocket>(mod, "H2DSocket")
        .def(
            nb::init<
                const std::shared_ptr<tt::tt_metal::distributed::MeshDevice>&,
                const tt::tt_metal::distributed::MeshCoreCoord&,
                tt::tt_metal::BufferType,
                uint32_t,
                tt::tt_metal::distributed::H2DMode>(),
            nb::arg("mesh_device"),
            nb::arg("recv_core"),
            nb::arg("buffer_type"),
            nb::arg("fifo_size"),
            nb::arg("h2d_mode"),
            R"doc(
                Construct an H2DSocket for streaming data from host to a device core.

                Args:
                    mesh_device (MeshDevice): The mesh device containing the target core.
                    recv_core (MeshCoreCoord): The target core coordinate to receive data.
                    buffer_type (BufferType): Memory type for the device-side FIFO buffer (L1 or DRAM).
                    fifo_size (int): Size of the circular FIFO buffer in bytes. Must be PCIe-aligned.
                    h2d_mode (H2DMode): Transfer mode: HOST_PUSH or DEVICE_PULL.
            )doc")
        .def(
            "get_page_size",
            &tt::tt_metal::distributed::H2DSocket::get_page_size,
            R"doc(
                Returns the currently configured page size.
            )doc")
        .def(
            "get_config_buffer_address",
            &tt::tt_metal::distributed::H2DSocket::get_config_buffer_address,
            R"doc(
                Returns the L1 address of the socket configuration buffer on the device.
                This address is passed to the device kernel to access socket metadata.
            )doc")
        .def(
            "set_page_size",
            &tt::tt_metal::distributed::H2DSocket::set_page_size,
            nb::arg("page_size"),
            R"doc(
                Sets the page size for subsequent write operations.

                Args:
                    page_size (int): Page size in bytes. Must be PCIe-aligned.
            )doc")
        .def(
            "write",
            &tt::tt_metal::distributed::H2DSocket::write,
            nb::arg("data"),
            nb::arg("num_pages"),
            R"doc(
                Writes data pages to the socket FIFO.

                Blocks if the FIFO does not have enough space, waiting for the device to
                acknowledge previously written data.

                Args:
                    data (int): Pointer to the source data buffer (as an integer address).
                    num_pages (int): Number of pages to write.
            )doc")
        .def(
            "write_tensor",
            [](tt::tt_metal::distributed::H2DSocket& self, tt::tt_metal::Tensor& tensor) {
                TT_FATAL(
                    tensor.storage_type() == tt::tt_metal::StorageType::HOST,
                    "write_tensor: tensor must be on host (HostStorage)");

                auto host_buffer = tt::tt_metal::host_buffer::get_host_buffer(tensor);
                auto data_span = host_buffer.view_bytes();
                uint32_t page_size = self.get_page_size();
                TT_FATAL(page_size > 0, "write_tensor: page_size must be set before calling write_tensor");
                TT_FATAL(
                    data_span.size() % page_size == 0,
                    "write_tensor: tensor data size ({}) is not a multiple of page_size ({})",
                    data_span.size(),
                    page_size);
                uint32_t num_writes = data_span.size() / page_size;
                for (uint32_t i = 0; i < num_writes; i++) {
                    self.write(data_span.data() + (i * page_size), 1);
                }
            },
            nb::arg("tensor"),
            R"doc(
                Writes a host tensor's data to the socket FIFO.

                The tensor must be on host (HostStorage). The page size must be set
                via set_page_size() before calling this method. The tensor's data size
                must be an exact multiple of the page size.

                Args:
                    tensor (Tensor): A host-resident tensor whose data will be written to the socket.
            )doc")
        .def(
            "barrier",
            &tt::tt_metal::distributed::H2DSocket::barrier,
            nb::arg("timeout_ms") = nb::none(),
            R"doc(
                Blocks until the device has acknowledged all written data.

                Args:
                    timeout_ms (int, optional): Timeout in milliseconds. Throws if not met within timeout.
            )doc")
        .def(
            "get_active_cores",
            &tt::tt_metal::distributed::H2DSocket::get_active_cores,
            R"doc(
                Returns the list of active MeshCoreCoords used by this socket.

                Returns:
                    List[MeshCoreCoord]: The active core coordinates.
            )doc")
        .def(
            "get_mesh_device",
            &tt::tt_metal::distributed::H2DSocket::get_mesh_device,
            R"doc(
                Returns the MeshDevice associated with this socket.

                Returns:
                    MeshDevice: The mesh device this socket is bound to.
            )doc")
        .def(
            "get_h2d_mode",
            &tt::tt_metal::distributed::H2DSocket::get_h2d_mode,
            R"doc(
                Returns the H2D transfer mode of this socket.

                Returns:
                    H2DMode: The transfer mode (HOST_PUSH or DEVICE_PULL).
            )doc");

    nb::class_<tt::tt_metal::distributed::D2HSocket>(mod, "D2HSocket")
        .def(
            nb::init<
                const std::shared_ptr<tt::tt_metal::distributed::MeshDevice>&,
                const tt::tt_metal::distributed::MeshCoreCoord&,
                uint32_t>(),
            nb::arg("mesh_device"),
            nb::arg("sender_core"),
            nb::arg("fifo_size"),
            R"doc(
                Construct a D2HSocket for streaming data from a device core to host.

                Args:
                    mesh_device (MeshDevice): The mesh device containing the sender core.
                    sender_core (MeshCoreCoord): The source core coordinate that sends data.
                    fifo_size (int): Size of the circular FIFO buffer in bytes. Must be PCIe-aligned.
            )doc")
        .def(
            "get_page_size",
            &tt::tt_metal::distributed::D2HSocket::get_page_size,
            R"doc(
                Returns the currently configured page size.
            )doc")
        .def(
            "get_config_buffer_address",
            &tt::tt_metal::distributed::D2HSocket::get_config_buffer_address,
            R"doc(
                Returns the L1 address of the socket configuration buffer on the device.
                This address should be passed to the device kernel so it can create a
                sender socket interface.
            )doc")
        .def(
            "set_page_size",
            &tt::tt_metal::distributed::D2HSocket::set_page_size,
            nb::arg("page_size"),
            R"doc(
                Sets the page size for subsequent read operations.

                Args:
                    page_size (int): Page size in bytes. Must be PCIe-aligned.
            )doc")
        .def(
            "read",
            &tt::tt_metal::distributed::D2HSocket::read,
            nb::arg("data"),
            nb::arg("num_pages"),
            nb::arg("notify_sender") = true,
            R"doc(
                Reads data pages from the socket FIFO.

                Blocks until the requested number of pages are available.

                Args:
                    data (int): Pointer to the destination buffer (as an integer address).
                    num_pages (int): Number of pages to read.
                    notify_sender (bool): If True, updates bytes_acked on the device to signal
                                          that buffer space is available. Default: True.
            )doc")
        .def(
            "read_tensor",
            [](tt::tt_metal::distributed::D2HSocket& self, tt::tt_metal::Tensor& tensor, bool notify_sender) {
                TT_FATAL(
                    tensor.storage_type() == tt::tt_metal::StorageType::HOST,
                    "read_tensor: tensor must be on host (HostStorage)");

                auto host_buffer = tt::tt_metal::host_buffer::get_host_buffer(tensor);
                auto data_span = host_buffer.view_bytes();
                uint32_t page_size = self.get_page_size();
                TT_FATAL(page_size > 0, "read_tensor: page_size must be set before calling read_tensor");
                TT_FATAL(
                    data_span.size() % page_size == 0,
                    "read_tensor: tensor data size ({}) is not a multiple of page_size ({})",
                    data_span.size(),
                    page_size);
                uint32_t num_pages = data_span.size() / page_size;
                std::cout << "Reading " << num_pages << " pages" << std::endl;
                self.read(data_span.data(), num_pages, notify_sender);
            },
            nb::arg("tensor"),
            nb::arg("notify_sender") = true,
            R"doc(
                Reads data from the socket FIFO into a host tensor.

                The tensor must be on host (HostStorage) and pre-allocated with the
                correct size. The page size must be set via set_page_size() before
                calling this method. The tensor's data size must be an exact multiple
                of the page size.

                Args:
                    tensor (Tensor): A pre-allocated host-resident tensor to read data into.
                    notify_sender (bool): If True, updates bytes_acked on the device to signal
                                          that buffer space is available. Default: True.
            )doc")
        .def(
            "barrier",
            &tt::tt_metal::distributed::D2HSocket::barrier,
            nb::arg("timeout_ms") = nb::none(),
            R"doc(
                Blocks until all sent data has been acknowledged.

                Args:
                    timeout_ms (int, optional): Timeout in milliseconds. Throws if not met within timeout.
            )doc")
        .def(
            "get_active_cores",
            &tt::tt_metal::distributed::D2HSocket::get_active_cores,
            R"doc(
                Returns the list of active MeshCoreCoords used by this socket.

                Returns:
                    List[MeshCoreCoord]: The active core coordinates.
            )doc")
        .def(
            "get_mesh_device",
            &tt::tt_metal::distributed::D2HSocket::get_mesh_device,
            R"doc(
                Returns the MeshDevice associated with this socket.

                Returns:
                    MeshDevice: The mesh device this socket is bound to.
            )doc");
}

void py_module(nb::module_& /* mod */) {
    // No free functions to bind currently.
    // H2DSocket and D2HSocket are fully bound via py_module_types.
}

}  // namespace ttnn::hd_socket
