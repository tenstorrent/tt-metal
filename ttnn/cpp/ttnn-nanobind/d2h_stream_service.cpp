// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include "d2h_stream_service.hpp"

#include <cstddef>
#include <cstdint>
#include <memory>
#include <string>
#include <utility>
#include <vector>

#include <nanobind/nanobind.h>
#include <nanobind/ndarray.h>
#include <nanobind/stl/optional.h>
#include <nanobind/stl/shared_ptr.h>
#include <nanobind/stl/string.h>
#include <nanobind/stl/unique_ptr.h>
#include <nanobind/stl/vector.h>

#include <tt-metalium/core_coord.hpp>
#include <tt-metalium/mesh_coord.hpp>
#include <tt-metalium/mesh_device.hpp>

#include "ttnn/distributed/distributed_tensor.hpp"
#include "ttnn/services/d2h_socket_service.hpp"
#include "ttnn/tensor/tensor.hpp"

namespace ttnn::d2h_stream_service {

void py_module_types(nb::module_& mod) {
    nb::class_<tt::tt_metal::D2HStreamService>(mod, "D2HStreamService")
        .def(
            "__init__",
            [](tt::tt_metal::D2HStreamService* self,
               const std::shared_ptr<tt::tt_metal::distributed::MeshDevice>& mesh_device,
               // None = metadata-only mode: no DRAM payload; metadata_size_bytes must be > 0.
               std::optional<tt::tt_metal::TensorSpec> global_spec,
               uint32_t fifo_size_bytes,
               uint32_t max_socket_page_size_bytes,
               std::unique_ptr<ttnn::distributed::TensorToMesh> mapper,
               std::optional<tt::tt_metal::distributed::MeshComposerConfig> composer_config,
               std::optional<CoreRange> worker_cores,
               std::optional<CoreCoord> metadata_master_core,
               uint32_t metadata_size_bytes,
               bool parallel_host_read,
               uint32_t host_read_thread_count) {
                tt::tt_metal::D2HStreamService::Config cfg{
                    .global_spec = std::move(global_spec),
                    .mapper = std::move(mapper),
                    .composer_config = std::move(composer_config),
                    .fifo_size_bytes = fifo_size_bytes,
                    .max_socket_page_size_bytes = max_socket_page_size_bytes,
                    .worker_cores = worker_cores,
                    .metadata_master_core = metadata_master_core,
                    .metadata_size_bytes = metadata_size_bytes,
                    .parallel_host_read = parallel_host_read,
                    .host_read_thread_count = host_read_thread_count,
                };
                new (self) tt::tt_metal::D2HStreamService(mesh_device, std::move(cfg));
            },
            nb::arg("mesh_device"),
            nb::arg("global_spec").none(),
            nb::arg("fifo_size_bytes"),
            nb::arg("max_socket_page_size_bytes"),
            nb::arg("mapper").none() = nb::none(),
            nb::arg("composer_config").none() = nb::none(),
            nb::arg("worker_cores").none() = nb::none(),
            nb::arg("metadata_master_core").none() = nb::none(),
            nb::arg("metadata_size_bytes") = 0u,
            nb::arg("parallel_host_read") = true,
            nb::arg("host_read_thread_count") = 0u)
        .def(
            "read_metadata",
            [](tt::tt_metal::D2HStreamService& self) {
                std::vector<std::byte> metadata(self.metadata_size_bytes());
                self.read_metadata(ttsl::Span<std::byte>(metadata.data(), metadata.size()));
                return nb::bytes(reinterpret_cast<const char*>(metadata.data()), metadata.size());
            },
            "Metadata-only read: returns the per-transfer record as bytes; asserts cross-chip equality. Metadata-only "
            "mode only.")
        .def(
            "read_from_tensor",
            [](tt::tt_metal::D2HStreamService& self, ttnn::Tensor& host_tensor) {
                std::vector<std::byte> metadata(self.metadata_size_bytes());
                self.read_from_tensor(host_tensor, ttsl::Span<std::byte>(metadata.data(), metadata.size()));
                return nb::bytes(reinterpret_cast<const char*>(metadata.data()), metadata.size());
            },
            "Drain one transfer into host_tensor (per-shard spec); returns trailing metadata bytes. Not in "
            "metadata-only "
            "mode.",
            nb::arg("host_tensor"))
        .def(
            "read_from_tensor_bytes",
            [](tt::tt_metal::D2HStreamService& self) {
                std::vector<std::byte> out(self.payload_size_bytes());
                std::vector<std::byte> metadata(self.metadata_size_bytes());
                self.read_from_tensor(
                    ttsl::Span<std::byte>(out.data(), out.size()),
                    ttsl::Span<std::byte>(metadata.data(), metadata.size()));
                return nb::bytes(reinterpret_cast<const char*>(out.data()), out.size());
            },
            "Drain one transfer; return the composed global payload as bytes. Not in metadata-only mode.")
        .def(
            "read_from_tensor_bytes_with_metadata",
            [](tt::tt_metal::D2HStreamService& self) {
                std::vector<std::byte> out(self.payload_size_bytes());
                std::vector<std::byte> metadata(self.metadata_size_bytes());
                self.read_from_tensor(
                    ttsl::Span<std::byte>(out.data(), out.size()),
                    ttsl::Span<std::byte>(metadata.data(), metadata.size()));
                return nb::make_tuple(
                    nb::bytes(reinterpret_cast<const char*>(out.data()), out.size()),
                    nb::bytes(reinterpret_cast<const char*>(metadata.data()), metadata.size()));
            },
            "Like read_from_tensor_bytes but returns (payload_bytes, metadata_bytes).")
        .def(
            "notify_backing_ready",
            &tt::tt_metal::D2HStreamService::notify_backing_ready,
            "Host-only path: bump each device's write-ack to release one transfer. Owner-only; raises if worker_cores "
            "set.")
        .def("barrier", &tt::tt_metal::D2HStreamService::barrier, "Block until all sockets have drained.")
        .def(
            "get_backing_tensor",
            &tt::tt_metal::D2HStreamService::get_backing_tensor,
            nb::rv_policy::reference_internal,
            "Service-owned device payload tensor. Owner-only; unused in metadata-only mode.")
        .def(
            "get_per_shard_spec",
            &tt::tt_metal::D2HStreamService::get_per_shard_spec,
            nb::rv_policy::reference_internal,
            "Per-device shard spec a host read tensor must match.")
        .def(
            "payload_size_bytes",
            &tt::tt_metal::D2HStreamService::payload_size_bytes,
            "Packed payload byte size; 0 in metadata-only mode.")
        .def(
            "metadata_size_bytes",
            &tt::tt_metal::D2HStreamService::metadata_size_bytes,
            "Metadata record size in bytes (0 if disabled).")
        .def(
            "get_slot_count",
            &tt::tt_metal::D2HStreamService::get_slot_count,
            "Data-CB depth in socket-page slots (owner-only).")
        .def(
            "get_sockets",
            &tt::tt_metal::D2HStreamService::get_sockets,
            "Underlying per-device D2H sockets (diagnostic).")
        .def(
            "get_worker_cores",
            &tt::tt_metal::D2HStreamService::get_worker_cores,
            "Configured worker CoreRange; raises if worker-sync unset.")
        .def(
            "get_metadata_master_core",
            &tt::tt_metal::D2HStreamService::get_metadata_master_core,
            "Worker core that forwards the metadata record (metadata mode only).")
        .def(
            "get_transfer_done_sem_addr",
            &tt::tt_metal::D2HStreamService::get_transfer_done_sem_addr,
            "Global-sem address multicast to workers per transfer (owner-only).")
        .def(
            "get_write_ack_counter_addr",
            &tt::tt_metal::D2HStreamService::get_write_ack_counter_addr,
            "Service-core L1 write-ack counter addr for coord (owner-only).")
        .def(
            "get_worker_metadata_addr",
            &tt::tt_metal::D2HStreamService::get_worker_metadata_addr,
            "Worker-side metadata staging L1 base (owner-only).")
        .def(
            "get_service_core",
            &tt::tt_metal::D2HStreamService::get_service_core,
            "Logical service core claimed at coord (owner-only).")
        .def(
            "get_metadata_input_addr",
            &tt::tt_metal::D2HStreamService::get_metadata_input_addr,
            "Service-core L1 addr the metadata record lands at for coord (owner-only).")
        .def(
            "get_metadata_addr",
            &tt::tt_metal::D2HStreamService::get_metadata_addr,
            "Alias of get_metadata_input_addr (owner-only).")
        .def(
            "export_descriptor",
            &tt::tt_metal::D2HStreamService::export_descriptor,
            "Write this service's socket descriptor to /dev/shm and return its path for connect(). Owner-only.",
            nb::arg("service_id"))
        .def_static(
            "connect",
            [](const std::string& service_id,
               std::optional<uint32_t> timeout_ms,
               bool parallel_host_read,
               uint32_t host_read_thread_count) {
                return tt::tt_metal::D2HStreamService::connect(
                    service_id, timeout_ms, parallel_host_read, host_read_thread_count);
            },
            "Attach (connector side) by service_id; reconstructs metadata-only mode when num_socket_pages==0. "
            "Read-only, "
            "no MeshDevice.",
            nb::arg("service_id"),
            nb::arg("timeout_ms") = nb::none(),
            nb::arg("parallel_host_read") = true,
            nb::arg("host_read_thread_count") = 0u);
}

void py_module(nb::module_& /* mod */) {}

}  // namespace ttnn::d2h_stream_service
