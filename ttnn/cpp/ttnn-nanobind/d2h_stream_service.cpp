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
               const tt::tt_metal::TensorSpec& global_spec,
               uint32_t fifo_size_bytes,
               uint32_t scratch_cb_size_bytes,
               std::unique_ptr<ttnn::distributed::TensorToMesh> mapper,
               std::optional<tt::tt_metal::distributed::MeshComposerConfig> composer_config,
               std::optional<CoreRange> worker_cores,
               std::optional<CoreCoord> metadata_master_core,
               uint32_t metadata_size_bytes) {
                tt::tt_metal::D2HStreamService::Config cfg{
                    .global_spec = global_spec,
                    .mapper = std::move(mapper),
                    .composer_config = std::move(composer_config),
                    .fifo_size_bytes = fifo_size_bytes,
                    .scratch_cb_size_bytes = scratch_cb_size_bytes,
                    .worker_cores = worker_cores,
                    .metadata_master_core = metadata_master_core,
                    .metadata_size_bytes = metadata_size_bytes,
                };
                new (self) tt::tt_metal::D2HStreamService(mesh_device, std::move(cfg));
            },
            nb::arg("mesh_device"),
            nb::arg("global_spec"),
            nb::arg("fifo_size_bytes"),
            nb::arg("scratch_cb_size_bytes"),
            nb::arg("mapper").none() = nb::none(),
            nb::arg("composer_config").none() = nb::none(),
            nb::arg("worker_cores").none() = nb::none(),
            nb::arg("metadata_master_core").none() = nb::none(),
            nb::arg("metadata_size_bytes") = 0u)
        .def(
            "read_from_tensor",
            [](tt::tt_metal::D2HStreamService& self, tt::tt_metal::Tensor& host_tensor) {
                std::vector<std::byte> metadata(self.metadata_size_bytes());
                self.read_from_tensor(host_tensor, ttsl::Span<std::byte>(metadata.data(), metadata.size()));
                return nb::bytes(reinterpret_cast<const char*>(metadata.data()), metadata.size());
            },
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
            })
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
            })
        .def("notify_backing_ready", &tt::tt_metal::D2HStreamService::notify_backing_ready)
        .def("barrier", &tt::tt_metal::D2HStreamService::barrier)
        .def(
            "get_backing_tensor",
            &tt::tt_metal::D2HStreamService::get_backing_tensor,
            nb::rv_policy::reference_internal)
        .def(
            "get_per_shard_spec",
            &tt::tt_metal::D2HStreamService::get_per_shard_spec,
            nb::rv_policy::reference_internal)
        .def("payload_size_bytes", &tt::tt_metal::D2HStreamService::payload_size_bytes)
        .def("metadata_size_bytes", &tt::tt_metal::D2HStreamService::metadata_size_bytes)
        .def("get_sockets", &tt::tt_metal::D2HStreamService::get_sockets)
        .def("get_worker_cores", &tt::tt_metal::D2HStreamService::get_worker_cores)
        .def("get_metadata_master_core", &tt::tt_metal::D2HStreamService::get_metadata_master_core)
        .def("get_transfer_done_sem_addr", &tt::tt_metal::D2HStreamService::get_transfer_done_sem_addr)
        .def("get_write_ack_counter_addr", &tt::tt_metal::D2HStreamService::get_write_ack_counter_addr)
        .def("get_worker_metadata_addr", &tt::tt_metal::D2HStreamService::get_worker_metadata_addr)
        .def("get_service_core", &tt::tt_metal::D2HStreamService::get_service_core)
        .def("get_metadata_input_addr", &tt::tt_metal::D2HStreamService::get_metadata_input_addr)
        .def("get_metadata_addr", &tt::tt_metal::D2HStreamService::get_metadata_addr)
        .def("export_descriptor", &tt::tt_metal::D2HStreamService::export_descriptor, nb::arg("service_id"))
        .def_static(
            "connect",
            [](const std::string& service_id, std::optional<uint32_t> timeout_ms) {
                return tt::tt_metal::D2HStreamService::connect(service_id, timeout_ms);
            },
            nb::arg("service_id"),
            nb::arg("timeout_ms") = nb::none());
}

void py_module(nb::module_& /* mod */) {}

}  // namespace ttnn::d2h_stream_service
