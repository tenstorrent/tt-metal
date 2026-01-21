// SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#include "mpi_socket.hpp"

#include <memory>

#include <nanobind/nanobind.h>
#include <nanobind/stl/unique_ptr.h>
#include <nanobind/stl/shared_ptr.h>

#include <ttnn/distributed/isocket.hpp>
#include <ttnn/distributed/mpi_socket.hpp>

namespace ttnn::mpi_socket {

void py_module_types(nb::module_& mod) {
    nb::class_<ttnn::distributed::MPISocket, ttnn::distributed::ISocket>(mod, "MPISocket")
        .def(
            nb::init<const tt::tt_metal::distributed::MeshSocket&>(),
            nb::arg("mesh_socket"),
            R"doc(
                Initialize an MPISocket with a MeshSocket.

                MPI-based implementation of distributed tensor communication that provides
                point-to-point tensor communication between MPI ranks using the Message
                Passing Interface. This implementation wraps a MeshSocket to handle the
                underlying tensor serialization and MPI message passing.

                Args:
                    mesh_socket (MeshSocket): The underlying mesh socket for communication
            )doc")
        .def_static(
            "create",
            &ttnn::distributed::MPISocket::create,
            nb::arg("mesh_device"),
            nb::arg("rank"),
            nb::arg("socket_config"),
            R"doc(
                Create an MPISocket for communication with a specific rank.

                Args:
                    mesh_device (MeshDevice): The mesh device to use for communication
                    rank (Rank): The rank to communicate with
                    socket_config (SocketConfig): The socket configuration

                Returns:
                    MPISocket: The created MPI socket
            )doc");
}

void py_module(nb::module_& mod) {
    mod.def(
        "create",
        &ttnn::distributed::MPISocket::create,
        nb::arg("mesh_device"),
        nb::arg("rank"),
        nb::arg("socket_config"),
        R"doc(
            Create an MPISocket for communication with a specific rank.

            Factory function to create an MPI socket for point-to-point tensor
            communication between MPI ranks.

            Args:
                mesh_device (MeshDevice): The mesh device to use for communication
                rank (Rank): The rank to communicate with
                socket_config (SocketConfig): The socket configuration

            Returns:
                MPISocket: The created MPI socket
        )doc");
}

}  // namespace ttnn::mpi_socket
