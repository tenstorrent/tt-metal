// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include "h2d_stream_service.hpp"

#include <cstddef>
#include <cstdint>
#include <memory>
#include <utility>

#include <nanobind/nanobind.h>
#include <nanobind/ndarray.h>
#include <nanobind/stl/optional.h>
#include <nanobind/stl/shared_ptr.h>
#include <nanobind/stl/unique_ptr.h>
#include <nanobind/stl/vector.h>

#include <tt-metalium/buffer_types.hpp>
#include <tt-metalium/core_coord.hpp>
#include <tt-metalium/experimental/sockets/h2d_socket.hpp>
#include <tt-metalium/mesh_coord.hpp>
#include <tt-metalium/mesh_device.hpp>

#include "ttnn/distributed/distributed_tensor.hpp"
#include "ttnn/services/h2d_socket_service.hpp"
#include "ttnn/tensor/tensor.hpp"

namespace ttnn::h2d_stream_service {

void py_module_types(nb::module_& mod) {
    nb::class_<tt::tt_metal::H2DStreamService>(mod, "H2DStreamService")
        .def(
            "__init__",
            [](tt::tt_metal::H2DStreamService* self,
               const std::shared_ptr<tt::tt_metal::distributed::MeshDevice>& mesh_device,
               const tt::tt_metal::TensorSpec& global_spec,
               uint32_t fifo_size_bytes,
               uint32_t scratch_cb_size_bytes,
               std::unique_ptr<ttnn::distributed::TensorToMesh> mapper,
               tt::tt_metal::BufferType socket_buffer_type,
               tt::tt_metal::distributed::H2DMode socket_mode,
               std::optional<CoreRange> worker_cores,
               uint32_t metadata_size_bytes) {
                tt::tt_metal::H2DStreamService::Config cfg{
                    .global_spec = global_spec,
                    .mapper = std::move(mapper),
                    .socket_buffer_type = socket_buffer_type,
                    .fifo_size_bytes = fifo_size_bytes,
                    .scratch_cb_size_bytes = scratch_cb_size_bytes,
                    .socket_mode = socket_mode,
                    .worker_cores = worker_cores,
                    .metadata_size_bytes = metadata_size_bytes,
                };
                new (self) tt::tt_metal::H2DStreamService(mesh_device, std::move(cfg));
            },
            nb::arg("mesh_device"),
            nb::arg("global_spec"),
            nb::arg("fifo_size_bytes"),
            nb::arg("scratch_cb_size_bytes"),
            nb::arg("mapper").none() = nb::none(),
            nb::arg("socket_buffer_type") = tt::tt_metal::BufferType::L1,
            nb::arg("socket_mode") = tt::tt_metal::distributed::H2DMode::DEVICE_PULL,
            nb::arg("worker_cores").none() = nb::none(),
            nb::arg("metadata_size_bytes") = 0u,
            R"doc(
                Construct a persistent H2DStreamService on the given mesh.

                The service launches one persistent receiver kernel per participating
                mesh coord and owns a backing device tensor with the per-shard spec
                derived from `global_spec` + `mapper`. Subsequent `forward_to_tensor`
                / `forward_to_tensor_bytes` calls stream data into the kernels' FIFOs
                without any per-call program build or dispatch.

                Args:
                    mesh_device (MeshDevice): The mesh device this service runs on.
                    global_spec (TensorSpec): Spec of the un-sharded source tensor.
                        Drives the mapper input shape and (after distribution) the
                        per-device tensor's layout.
                    fifo_size_bytes (int): Size of each H2D socket's FIFO buffer in
                        bytes. Must be PCIe-aligned.
                    scratch_cb_size_bytes (int): On-device scratch circular buffer
                        size per recv core in bytes. Drives the chunk-picker; must
                        be >= one tensor page.
                    mapper (TensorToMesh, optional): How the global tensor is split /
                        replicated across the mesh. Ownership transfers to the
                        service; the Python object is invalidated after this call.
                        Construct via
                        `ttnn.create_mesh_mapper(mesh_device, mesh_mapper_config)`.
                        If None (the default), the service synthesises a
                        replicate-on-every-mesh-dim mapper internally — works for
                        any mesh shape (1x1 mesh = identity, NxM mesh = full tensor
                        on every device). Sharded distributions must supply an
                        explicit mapper.
                    socket_buffer_type (BufferType, optional): Memory type for the
                        socket's device-side FIFO buffer (L1 or DRAM). Default: L1.
                    socket_mode (H2DMode, optional): Transfer mode (HOST_PUSH or
                        DEVICE_PULL). Default: DEVICE_PULL.
                    worker_cores (CoreRange, optional): When set, after every
                        transfer the persistent receiver kernel multicasts a
                        data-ready inc to a GlobalSemaphore on every worker core
                        in this CoreRange and then waits for one ack per worker
                        before proceeding. Caller's worker kernel reads the sem
                        at `get_data_ready_sem_addr()` and atomic-incs the
                        counter at `get_consumed_counter_addr(coord)`. If None
                        (the default), the kernel skips the handshake entirely.
                    metadata_size_bytes (int, optional): When non-zero, every
                        forward_to_tensor call must include exactly this many
                        bytes of trailing metadata; the service kernel multi-
                        casts them to a fixed L1 address on every worker core
                        in `worker_cores` (retrievable via `get_metadata_addr()`).
                        Requires `worker_cores` to be set; must be <= the
                        derived socket page size (TT_FATAL'd at construction).
                        Default: 0 (metadata path disabled).
            )doc")
        .def(
            "forward_to_tensor",
            // Distributed host-tensor path. Tensor must already be distributed by a
            // mapper equivalent to the one the service was constructed with (i.e.
            // its per-shard spec must equal get_per_shard_spec()). Returns once the
            // bytes are in the socket FIFOs; call barrier() to wait for the device
            // to drain them.
            //
            // `metadata` must be exactly `metadata_size_bytes` bytes long when the
            // service was constructed with metadata enabled; empty otherwise. An
            // empty bytes object always satisfies the disabled case.
            [](tt::tt_metal::H2DStreamService& self,
               const tt::tt_metal::Tensor& host_tensor,
               const nb::bytes& metadata) {
                auto meta_span = ttsl::Span<const std::byte>(
                    reinterpret_cast<const std::byte*>(metadata.c_str()), metadata.size());
                self.forward_to_tensor(host_tensor, meta_span);
            },
            nb::arg("host_tensor"),
            nb::arg("metadata") = nb::bytes("", 0),
            R"doc(
                Stream a pre-distributed host tensor's per-coord shards through the
                sockets.

                Args:
                    host_tensor (Tensor): A host tensor that has already been
                        distributed by a mapper matching the one passed at
                        construction. `host_tensor.tensor_spec()` must equal
                        `get_per_shard_spec()`.
                    metadata (bytes, optional): Inline metadata payload appended
                        to this transfer. Length must equal
                        `metadata_size_bytes` passed at construction; pass an
                        empty bytes object when metadata was not enabled.
                        Default: empty bytes.
            )doc")
        .def(
            "forward_to_tensor_bytes",
            [](tt::tt_metal::H2DStreamService& self,
               const nb::ndarray<nb::c_contig, nb::device::cpu>& data,
               const nb::bytes& metadata) {
                auto bytes = ttsl::Span<const std::byte>(
                    reinterpret_cast<const std::byte*>(data.data()), data.nbytes());
                auto meta_span = ttsl::Span<const std::byte>(
                    reinterpret_cast<const std::byte*>(metadata.c_str()), metadata.size());
                self.forward_to_tensor(bytes, meta_span);
            },
            nb::arg("data"),
            nb::arg("metadata") = nb::bytes("", 0),
            R"doc(
                Stream a contiguous CPU ndarray as the un-sharded global tensor. The
                service distributes internally via its owned mapper.

                Args:
                    data (numpy.ndarray | torch.Tensor): A contiguous CPU ndarray
                        whose total byte size equals
                        `global_spec.compute_packed_buffer_size_bytes()`. Element
                        dtype is irrelevant — the array is reinterpreted as raw
                        bytes matching `global_spec`'s layout. Only the global
                        spec's ROW_MAJOR layout is supported on this path today.
                    metadata (bytes, optional): Inline metadata payload appended
                        to this transfer. Length must equal
                        `metadata_size_bytes` passed at construction; pass an
                        empty bytes object when metadata was not enabled.
                        Default: empty bytes.
            )doc")
        .def(
            "barrier",
            &tt::tt_metal::H2DStreamService::barrier,
            R"doc(
                Block until the device has acknowledged every in-flight host write.
                Call before reading the backing tensor or destroying the service.
            )doc")
        .def(
            "get_backing_tensor",
            &tt::tt_metal::H2DStreamService::get_backing_tensor,
            nb::rv_policy::reference_internal,
            R"doc(
                The device tensor the persistent kernels write into. Same instance
                across calls — its contents are overwritten by each forward_to_tensor.

                Returns:
                    Tensor: A device-resident tensor with one shard per participating
                    mesh coord.
            )doc")
        .def(
            "get_per_shard_spec",
            &tt::tt_metal::H2DStreamService::get_per_shard_spec,
            nb::rv_policy::reference_internal,
            R"doc(
                The per-coord TensorSpec produced by the mapper at construction time.
                Same as `get_backing_tensor().tensor_spec()`.
            )doc")
        .def(
            "payload_size_bytes",
            &tt::tt_metal::H2DStreamService::payload_size_bytes,
            R"doc(
                Bytes the caller must hand to `forward_to_tensor_bytes` per call —
                equal to the packed size of one full global tensor.

                Returns:
                    int: Required payload size in bytes.
            )doc")
        .def(
            "metadata_size_bytes",
            &tt::tt_metal::H2DStreamService::metadata_size_bytes,
            R"doc(
                Bytes of metadata that must be attached to each `forward_to_tensor*`
                call. Zero means the metadata path is disabled and the no-metadata
                overload must be used.

                Returns:
                    int: Required metadata size in bytes (0 if disabled).
            )doc")
        .def(
            "get_worker_cores",
            &tt::tt_metal::H2DStreamService::get_worker_cores,
            R"doc(
                Worker CoreRange the service synchronizes with — same grid passed
                via `worker_cores` at construction. Consumers building a peer
                MeshWorkload around the service use this to size their
                `pages_per_worker` partitioning and to multicast destinations.

                Raises:
                    RuntimeError: If the service was constructed without
                        `worker_cores`.

                Returns:
                    CoreRange: The worker grid.
            )doc")
        .def(
            "get_sockets",
            &tt::tt_metal::H2DStreamService::get_sockets,
            R"doc(
                The list of H2DSocket pointers, one per participating mesh coord.
                The service owns the sockets; the returned pointers must not outlive
                the service.

                Returns:
                    List[H2DSocket]: One socket per participating coord.
            )doc")
        .def(
            "get_data_ready_sem_addr",
            &tt::tt_metal::H2DStreamService::get_data_ready_sem_addr,
            R"doc(
                L1 address of the data-ready GlobalSemaphore on every worker core
                in `worker_cores`. Same value across (device, worker core) by
                mesh-wide GlobalSemaphore construction. Workers poll their local
                copy here; the service kernel multicasts an atomic-inc after
                every transfer.

                Raises:
                    RuntimeError: If the service was constructed without
                        `worker_cores`.

                Returns:
                    int: The L1 address.
            )doc")
        .def(
            "get_consumed_counter_addr",
            &tt::tt_metal::H2DStreamService::get_consumed_counter_addr,
            nb::arg("coord"),
            R"doc(
                L1 address of the per-coord consumed-counter on this coord's
                service core. Workers send NoC atomic-incs here (one per
                consumed iteration); the service kernel polls it locally to know
                when all workers have acknowledged the iteration.

                Args:
                    coord (MeshCoordinate): The mesh coordinate whose service-core
                        L1 address to look up.

                Raises:
                    RuntimeError: If the service was constructed without
                        `worker_cores`, or `coord` does not participate.

                Returns:
                    int: The L1 address on the service core for this coord.
            )doc")
        .def(
            "get_service_core",
            &tt::tt_metal::H2DStreamService::get_service_core,
            nb::arg("coord"),
            R"doc(
                Logical CoreCoord of the service core on this coord's device.
                Combine with `get_consumed_counter_addr` to build the NoC
                destination workers atomic-inc into; the caller converts
                logical -> physical via the mesh device's
                `worker_core_from_logical_core` at workload setup time.

                Args:
                    coord (MeshCoordinate): The mesh coordinate whose service
                        core to look up.

                Raises:
                    RuntimeError: If the coord does not participate in this
                        service.

                Returns:
                    CoreCoord: The logical service-core coordinate on that device.
            )doc")
        .def(
            "get_metadata_addr",
            &tt::tt_metal::H2DStreamService::get_metadata_addr,
            R"doc(
                L1 address of the metadata destination on every worker core in
                `worker_cores`. Same address across (device, worker core) by
                mesh-wide L1-sharded Buffer construction. The service kernel
                multicasts the first `metadata_size_bytes` of every transfer's
                trailing metadata page here, BEFORE flipping the data-ready
                semaphore (so workers observing data-ready see consistent
                DRAM + L1 state).

                Raises:
                    RuntimeError: If the service was constructed with
                        `metadata_size_bytes = 0`.

                Returns:
                    int: The L1 address.
            )doc")
        .def(
            "export_descriptor",
            &tt::tt_metal::H2DStreamService::export_descriptor,
            nb::arg("service_id"),
            R"doc(
                Write the service's flatbuffer descriptor to
                `/dev/shm/tt_h2d_stream_service_<service_id>.bin` so a
                co-process can attach to this service via
                `H2DStreamService.connect(service_id)` (C++ API). The
                descriptor embeds every per-coord socket's connection
                metadata; the connector reads a single file and attaches all
                sockets in one shot. Owner-side only — `RuntimeError` if
                called on a connector-side service.

                Args:
                    service_id (str): Identifier used in the descriptor
                        filename. The same value must be passed to
                        `H2DStreamService::connect` on the connector side.

                Returns:
                    str: Full path to the written descriptor file.
            )doc")
        .def_static(
            "connect",
            [](const std::string& service_id, std::optional<uint32_t> timeout_ms) {
                return tt::tt_metal::H2DStreamService::connect(service_id, timeout_ms);
            },
            nb::arg("service_id"),
            nb::arg("timeout_ms") = nb::none(),
            R"doc(
                Attach to an exported H2DStreamService from another process.

                Reads the descriptor file at
                `/dev/shm/tt_h2d_stream_service_<service_id>.bin`,
                reconstructs the mapper from the embedded mesh shape + mapper
                config, and attaches every per-coord H2DSocket inline. The
                returned service holds NO MeshDevice handle — it talks to the
                device only through the per-socket PCIeCoreWriter paths.

                The returned instance supports `forward_to_tensor`,
                `forward_to_tensor_bytes`, `barrier`, and `get_per_shard_spec`.
                Owner-only methods (`get_backing_tensor`, the worker-sync
                getters, `export_descriptor`) raise on a connector-side
                instance.

                Args:
                    service_id (str): Identifier the owner passed to
                        `export_descriptor`.
                    timeout_ms (int, optional): Max wait time for the
                        descriptor file. Defaults to the C++ default (10s).

                Returns:
                    H2DStreamService: Connector-side service handle.
            )doc");
}

void py_module(nb::module_& /* mod */) {
    // No free functions; the service is exposed entirely via py_module_types.
}

}  // namespace ttnn::h2d_stream_service
