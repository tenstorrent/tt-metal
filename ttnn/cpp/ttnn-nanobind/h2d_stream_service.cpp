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
#include <nanobind/stl/shared_ptr.h>
#include <nanobind/stl/unique_ptr.h>
#include <nanobind/stl/vector.h>

#include <tt-metalium/buffer_types.hpp>
#include <tt-metalium/core_coord.hpp>
#include <tt-metalium/experimental/sockets/h2d_socket.hpp>
#include <tt-metalium/mesh_device.hpp>

#include "ttnn/distributed/distributed_tensor.hpp"
#include "ttnn/tensor/socket_services.hpp"
#include "ttnn/tensor/tensor.hpp"

namespace ttnn::h2d_stream_service {

void py_module_types(nb::module_& mod) {
    nb::class_<tt::tt_metal::H2DStreamService>(mod, "H2DStreamService")
        .def(
            "__init__",
            // Builds H2DStreamService::Config inline from kwargs, then placement-new
            // constructs the service. Config isn't separately exposed to Python —
            // the service ctor is the single entry point.
            //
            // `mapper` is taken as a plain `std::unique_ptr<TensorToMesh>` — nanobind
            // surrenders ownership directly into this type and invalidates the
            // Python-side wrapper as part of the transfer (no separate `release()`
            // / re-wrap step needed, which would skip nanobind's wrapper-tracking
            // cleanup and leak the Python wrapper). If None is passed the
            // unique_ptr arrives null and the C++ ctor synthesises a
            // replicate-on-every-mesh-dim default.
            [](tt::tt_metal::H2DStreamService* self,
               const std::shared_ptr<tt::tt_metal::distributed::MeshDevice>& mesh_device,
               const tt::tt_metal::TensorSpec& global_spec,
               uint32_t fifo_size_bytes,
               uint32_t scratch_cb_size_bytes,
               std::unique_ptr<ttnn::distributed::TensorToMesh> mapper,
               tt::tt_metal::BufferType socket_buffer_type,
               tt::tt_metal::distributed::H2DMode socket_mode) {
                tt::tt_metal::H2DStreamService::Config cfg{
                    .global_spec = global_spec,
                    .mapper = std::move(mapper),
                    .socket_buffer_type = socket_buffer_type,
                    .fifo_size_bytes = fifo_size_bytes,
                    .scratch_cb_size_bytes = scratch_cb_size_bytes,
                    .socket_mode = socket_mode,
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
            )doc")
        .def(
            "forward_to_tensor",
            // Distributed host-tensor path. Tensor must already be distributed by a
            // mapper equivalent to the one the service was constructed with (i.e.
            // its per-shard spec must equal get_per_shard_spec()). Returns once the
            // bytes are in the socket FIFOs; call barrier() to wait for the device
            // to drain them.
            [](tt::tt_metal::H2DStreamService& self, const tt::tt_metal::Tensor& host_tensor) {
                self.forward_to_tensor(host_tensor);
            },
            nb::arg("host_tensor"),
            R"doc(
                Stream a pre-distributed host tensor's per-coord shards through the
                sockets.

                Args:
                    host_tensor (Tensor): A host tensor that has already been
                        distributed by a mapper matching the one passed at
                        construction. `host_tensor.tensor_spec()` must equal
                        `get_per_shard_spec()`.
            )doc")
        .def(
            "forward_to_tensor_bytes",
            // Raw-bytes path. The service distributes via its internal mapper, so
            // the input is the un-sharded global tensor as raw bytes. The ndarray's
            // total nbytes must equal global_spec.compute_packed_buffer_size_bytes().
            // ROW_MAJOR-only constraint is enforced inside the C++ service.
            [](tt::tt_metal::H2DStreamService& self,
               const nb::ndarray<nb::c_contig, nb::device::cpu>& data) {
                auto bytes = ttsl::Span<const std::byte>(
                    reinterpret_cast<const std::byte*>(data.data()), data.nbytes());
                self.forward_to_tensor(bytes);
            },
            nb::arg("data"),
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
            "get_sockets",
            &tt::tt_metal::H2DStreamService::get_sockets,
            R"doc(
                The list of H2DSocket pointers, one per participating mesh coord.
                The service owns the sockets; the returned pointers must not outlive
                the service.

                Returns:
                    List[H2DSocket]: One socket per participating coord.
            )doc");
}

void py_module(nb::module_& /* mod */) {
    // No free functions; the service is exposed entirely via py_module_types.
}

}  // namespace ttnn::h2d_stream_service
