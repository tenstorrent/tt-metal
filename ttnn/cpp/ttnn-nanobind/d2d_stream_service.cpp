// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include "d2d_stream_service.hpp"

#include <cstdint>
#include <memory>
#include <utility>

#include <nanobind/nanobind.h>
#include <nanobind/stl/optional.h>
#include <nanobind/stl/pair.h>
#include <nanobind/stl/shared_ptr.h>
#include <nanobind/stl/unique_ptr.h>

#include <tt-metalium/buffer_types.hpp>
#include <tt-metalium/core_coord.hpp>
#include <tt-metalium/distributed_context.hpp>
#include <tt-metalium/experimental/sockets/mesh_socket.hpp>
#include <tt-metalium/mesh_coord.hpp>
#include <tt-metalium/mesh_device.hpp>

#include "ttnn/distributed/distributed_tensor.hpp"
#include "ttnn/tensor/d2d_stream_service.hpp"
#include "ttnn/tensor/tensor.hpp"

namespace ttnn::d2d_stream_service {

namespace {

using tt::tt_metal::distributed::SocketMemoryConfig;
using tt::tt_metal::distributed::multihost::Rank;
using ttnn::D2DEndpointConfig;
using ttnn::D2DStreamConfig;
using ttnn::D2DStreamService;
using ttnn::D2DStreamServiceReceiver;
using ttnn::D2DStreamServiceSender;
using MeshDevicePtr = std::shared_ptr<tt::tt_metal::distributed::MeshDevice>;

// Assemble a D2DStreamConfig from individual kwargs. Like the H2DStreamService ctor, the
// config struct is never exposed to Python directly — it holds a move-only `mapper`
// (std::unique_ptr<TensorToMesh>) and a SocketMemoryConfig, so the factory entry points
// take the constituent fields and build it inline. `mapper` is REQUIRED (the service
// TT_FATALs on a null mapper); nanobind surrenders ownership into the unique_ptr and
// invalidates the Python-side wrapper.
D2DStreamConfig make_config(
    const tt::tt_metal::TensorSpec& global_spec,
    std::unique_ptr<ttnn::distributed::TensorToMesh> mapper,
    tt::tt_metal::BufferType socket_buffer_type,
    uint32_t fifo_size_bytes,
    const CoreRange& sender_worker_cores,
    const CoreRange& receiver_worker_cores,
    uint32_t metadata_size_bytes,
    bool share_fabric_links) {
    return D2DStreamConfig{
        .global_spec = global_spec,
        .mapper = std::move(mapper),
        .socket_mem_config = SocketMemoryConfig{socket_buffer_type, fifo_size_bytes},
        .sender_worker_cores = sender_worker_cores,
        .receiver_worker_cores = receiver_worker_cores,
        .metadata_size_bytes = metadata_size_bytes,
        .share_fabric_links = share_fabric_links,
    };
}

}  // namespace

void py_module_types(nb::module_& mod) {
    // ----- Sender handle (created only via D2DStreamService factories) -----
    nb::class_<D2DStreamServiceSender>(mod, "D2DStreamServiceSender")
        .def(
            "get_backing_tensor",
            &D2DStreamServiceSender::get_backing_tensor,
            nb::rv_policy::reference_internal,
            R"doc(
                The outbound device tensor a producing op writes into; the sender service
                forwards its contents over fabric once the op signals data_ready and the
                fabric-link lease is granted. Same instance across calls.
            )doc")
        .def(
            "get_per_shard_spec",
            &D2DStreamServiceSender::get_per_shard_spec,
            nb::rv_policy::reference_internal,
            "The per-coord tt::tt_metal::TensorSpec of the backing tensor (mapper output).")
        .def(
            "get_worker_cores",
            &D2DStreamServiceSender::get_worker_cores,
            "Worker CoreRange the producing op runs on; sizes the data_ready ack count.")
        .def(
            "get_service_core",
            &D2DStreamServiceSender::get_service_core,
            nb::arg("coord"),
            "Logical CoreCoord of the sender service core on this coord's device.")
        .def(
            "get_data_ready_counter_addr",
            &D2DStreamServiceSender::get_data_ready_counter_addr,
            nb::arg("coord"),
            R"doc(
                L1 address (on this coord's service core) of the data_ready counter the
                producing op atomic-incs once per worker after writing the backing tensor;
                the sender forwards when it reaches num_workers and the lease is granted.
            )doc")
        .def(
            "get_consumed_sem_addr",
            &D2DStreamServiceSender::get_consumed_sem_addr,
            R"doc(
                L1 address (on every worker core) of the consumed semaphore the sender
                multicasts after a forward drains. A separate gate op waits on it before
                the next iteration overwrites the backing tensor.
            )doc")
        .def(
            "get_metadata_addr",
            &D2DStreamServiceSender::get_metadata_addr,
            nb::arg("coord"),
            "L1 address of the metadata buffer on this coord's service core (metadata mode).")
        .def(
            "wait_for_fabric_links",
            &D2DStreamServiceSender::wait_for_fabric_links,
            R"doc(
                Block until the sender service has released its fabric link(s) (its granted
                forward is complete). Pair with release_fabric_links to arbitrate the link
                against model-graph fabric ops on the same device.
            )doc")
        .def(
            "release_fabric_links",
            &D2DStreamServiceSender::release_fabric_links,
            R"doc(
                Grant the sender service one transfer over its fabric link(s). The service
                forwards when it also sees data_ready, then releases the link back.
            )doc");

    // ----- Receiver handle (created only via D2DStreamService factories) -----
    nb::class_<D2DStreamServiceReceiver>(mod, "D2DStreamServiceReceiver")
        .def(
            "get_backing_tensor",
            &D2DStreamServiceReceiver::get_backing_tensor,
            nb::rv_policy::reference_internal,
            R"doc(
                The inbound device tensor the receiver service writes incoming fabric data
                into; a consuming op reads it once data_ready is signaled. Same instance
                across calls.
            )doc")
        .def(
            "get_per_shard_spec",
            &D2DStreamServiceReceiver::get_per_shard_spec,
            nb::rv_policy::reference_internal,
            "The per-coord tt::tt_metal::TensorSpec of the backing tensor (mapper output).")
        .def(
            "get_worker_cores",
            &D2DStreamServiceReceiver::get_worker_cores,
            "Worker CoreRange the consuming op runs on; sizes the consumed ack count.")
        .def(
            "get_service_core",
            &D2DStreamServiceReceiver::get_service_core,
            nb::arg("coord"),
            "Logical CoreCoord of the receiver service core on this coord's device.")
        .def(
            "get_data_ready_sem_addr",
            &D2DStreamServiceReceiver::get_data_ready_sem_addr,
            R"doc(
                L1 address (on every worker core) of the data_ready semaphore the receiver
                multicasts after a transfer lands. The consuming op polls it, then reads
                the backing tensor.
            )doc")
        .def(
            "get_consumed_counter_addr",
            &D2DStreamServiceReceiver::get_consumed_counter_addr,
            nb::arg("coord"),
            R"doc(
                L1 address (on this coord's service core) of the consumed counter the
                consuming op atomic-incs once per worker after reading the backing tensor;
                the receiver may then accept the next transfer.
            )doc")
        .def(
            "get_metadata_addr",
            &D2DStreamServiceReceiver::get_metadata_addr,
            "L1 address of the metadata buffer on every worker core (metadata mode).")
        .def(
            "wait_for_fabric_links",
            &D2DStreamServiceReceiver::wait_for_fabric_links,
            "Block until the receiver service has released its fabric link(s).")
        .def(
            "release_fabric_links",
            &D2DStreamServiceReceiver::release_fabric_links,
            "Grant the receiver service one transfer over its fabric link(s).");

    // ----- Factory (static methods only; not instantiable) -----
    nb::class_<D2DStreamService>(mod, "D2DStreamService")
        .def_static(
            "create_pair",
            // Single-process: both meshes live in this process, so the socket is built
            // without cross-process rendezvous. Returns (sender, receiver).
            [](const MeshDevicePtr& sender_mesh,
               const MeshDevicePtr& receiver_mesh,
               const tt::tt_metal::TensorSpec& global_spec,
               std::unique_ptr<ttnn::distributed::TensorToMesh> mapper,
               uint32_t fifo_size_bytes,
               const CoreRange& sender_worker_cores,
               const CoreRange& receiver_worker_cores,
               tt::tt_metal::BufferType socket_buffer_type,
               uint32_t metadata_size_bytes,
               bool share_fabric_links) {
                auto cfg = make_config(
                    global_spec,
                    std::move(mapper),
                    socket_buffer_type,
                    fifo_size_bytes,
                    sender_worker_cores,
                    receiver_worker_cores,
                    metadata_size_bytes,
                    share_fabric_links);
                return D2DStreamService::create_pair(sender_mesh, receiver_mesh, std::move(cfg));
            },
            nb::arg("sender_mesh"),
            nb::arg("receiver_mesh"),
            nb::arg("global_spec"),
            nb::arg("mapper"),
            nb::arg("fifo_size_bytes"),
            nb::arg("sender_worker_cores"),
            nb::arg("receiver_worker_cores"),
            nb::arg("socket_buffer_type") = tt::tt_metal::BufferType::L1,
            nb::arg("metadata_size_bytes") = 0u,
            nb::arg("share_fabric_links") = true,
            R"doc(
                Build a sender+receiver pair in a SINGLE process (both meshes owned here).

                Returns:
                    Tuple[D2DStreamServiceSender, D2DStreamServiceReceiver]
            )doc")
        .def_static(
            "create_sender",
            // Multi-process: must run on the sender rank. The peer (receiver) is created
            // in the receiver-rank process via create_receiver; the per-endpoint
            // MeshSocket ctor does the cross-process rendezvous over the current world.
            [](const MeshDevicePtr& sender_mesh,
               const tt::tt_metal::TensorSpec& global_spec,
               std::unique_ptr<ttnn::distributed::TensorToMesh> mapper,
               uint32_t fifo_size_bytes,
               const CoreRange& sender_worker_cores,
               const CoreRange& receiver_worker_cores,
               int sender_rank,
               int receiver_rank,
               tt::tt_metal::BufferType socket_buffer_type,
               uint32_t metadata_size_bytes,
               bool share_fabric_links) {
                auto cfg = make_config(
                    global_spec,
                    std::move(mapper),
                    socket_buffer_type,
                    fifo_size_bytes,
                    sender_worker_cores,
                    receiver_worker_cores,
                    metadata_size_bytes,
                    share_fabric_links);
                // distributed_context left null -> the service uses get_current_world().
                D2DEndpointConfig endpoints{
                    .sender_rank = Rank{sender_rank},
                    .receiver_rank = Rank{receiver_rank},
                    .distributed_context = nullptr};
                return D2DStreamService::create_sender(sender_mesh, std::move(cfg), endpoints);
            },
            nb::arg("sender_mesh"),
            nb::arg("global_spec"),
            nb::arg("mapper"),
            nb::arg("fifo_size_bytes"),
            nb::arg("sender_worker_cores"),
            nb::arg("receiver_worker_cores"),
            nb::arg("sender_rank"),
            nb::arg("receiver_rank"),
            nb::arg("socket_buffer_type") = tt::tt_metal::BufferType::L1,
            nb::arg("metadata_size_bytes") = 0u,
            nb::arg("share_fabric_links") = true,
            R"doc(
                Build the SENDER endpoint in this (sender-rank) process. Must be called on
                the rank equal to `sender_rank`; the matching `create_receiver` runs in the
                `receiver_rank` process. Uses the current distributed world for the
                cross-process socket rendezvous.

                Returns:
                    D2DStreamServiceSender
            )doc")
        .def_static(
            "create_receiver",
            // Multi-process: must run on the receiver rank.
            [](const MeshDevicePtr& receiver_mesh,
               const tt::tt_metal::TensorSpec& global_spec,
               std::unique_ptr<ttnn::distributed::TensorToMesh> mapper,
               uint32_t fifo_size_bytes,
               const CoreRange& sender_worker_cores,
               const CoreRange& receiver_worker_cores,
               int sender_rank,
               int receiver_rank,
               tt::tt_metal::BufferType socket_buffer_type,
               uint32_t metadata_size_bytes,
               bool share_fabric_links) {
                auto cfg = make_config(
                    global_spec,
                    std::move(mapper),
                    socket_buffer_type,
                    fifo_size_bytes,
                    sender_worker_cores,
                    receiver_worker_cores,
                    metadata_size_bytes,
                    share_fabric_links);
                D2DEndpointConfig endpoints{
                    .sender_rank = Rank{sender_rank},
                    .receiver_rank = Rank{receiver_rank},
                    .distributed_context = nullptr};
                return D2DStreamService::create_receiver(receiver_mesh, std::move(cfg), endpoints);
            },
            nb::arg("receiver_mesh"),
            nb::arg("global_spec"),
            nb::arg("mapper"),
            nb::arg("fifo_size_bytes"),
            nb::arg("sender_worker_cores"),
            nb::arg("receiver_worker_cores"),
            nb::arg("sender_rank"),
            nb::arg("receiver_rank"),
            nb::arg("socket_buffer_type") = tt::tt_metal::BufferType::L1,
            nb::arg("metadata_size_bytes") = 0u,
            nb::arg("share_fabric_links") = true,
            R"doc(
                Build the RECEIVER endpoint in this (receiver-rank) process. Must be called
                on the rank equal to `receiver_rank`; the matching `create_sender` runs in
                the `sender_rank` process. Uses the current distributed world for the
                cross-process socket rendezvous.

                Returns:
                    D2DStreamServiceReceiver
            )doc");
}

void py_module(nb::module_& /* mod */) {
    // No free functions; the service is exposed entirely via py_module_types.
}

}  // namespace ttnn::d2d_stream_service
