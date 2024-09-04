// SPDX-FileCopyrightText: © 2024 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <pybind11/pybind11.h>
#include <pybind11/stl.h>

#include "ttnn/cpp/pybind11/decorators.hpp"
#include "untilize_with_halo_v2.hpp"

namespace ttnn::operations::data_movement::detail {
namespace py = pybind11;

void bind_untilize_with_halo_v2(py::module& module) {
    auto doc =
        R"doc(
            untilize_with_halo_v2(input_tensor: ttnn.Tensor, padding_config: ttnn.Tensor, local_config: ttnn.Tensor, remote_config: ttnn.Tensor, \*, pad_val: int, ncores_nhw: int, max_out_nsticks_per_core: int,
                                  memory_config: Optional[MemoryConfig] = None, remote_read: bool = False, transpose_mcast: bool = False, queue_id: int = 0) -> ttnn.Tensor

            Untilizes input tiled data to row major format and constructs halo'd output shards.

            Args:
                * :attr:`input_tensor`: Input Tensor.
                * :attr:`padding_config`: Padding config tensor.
                * :attr:`local_config`: Local config tensor.
                * :attr:`remote_config`: Remote config tensor.

            Keyword Args:
                * :attr:`pad_val`: Pad value.
                * :attr:`ncores_nhw`: Number of cores per NHW.
                * :attr:`max_out_nsticks_per_core`: Max output nsticks per core.
                * :attr:`memory_config`: Output memory config.
                * :attr:`remote_read`: Remote read.
                * :attr:`transpose_mcast`: Transpose mcast.
                * :attr:`queue_id`: command queue id.
        )doc";

    using OperationType = decltype(ttnn::untilize_with_halo_v2);
    ttnn::bind_registered_operation(
        module,
        ttnn::untilize_with_halo_v2,
        doc,
        ttnn::pybind_overload_t{
            [](const OperationType& self,
               const ttnn::Tensor& input_tensor,
               const ttnn::Tensor& padding_config,
               const ttnn::Tensor& local_config,
               const ttnn::Tensor& remote_config,
               const uint32_t pad_val,
               const uint32_t ncores_nhw,
               const uint32_t max_out_nsticks_per_core,
               const std::optional<MemoryConfig>& memory_config,
               const bool remote_read,
               const bool transpose_mcast,
               uint8_t queue_id) {
                return self(
                    queue_id,
                    input_tensor,
                    padding_config,
                    local_config,
                    remote_config,
                    pad_val,
                    ncores_nhw,
                    max_out_nsticks_per_core,
                    memory_config,
                    remote_read,
                    transpose_mcast);
            },
            py::arg("input_tensor"),
            py::arg("padding_config"),
            py::arg("local_config"),
            py::arg("remote_config"),
            py::kw_only(),
            py::arg("pad_val"),
            py::arg("ncores_nhw"),
            py::arg("max_out_nsticks_per_core"),
            py::arg("memory_config") = std::nullopt,
            py::arg("remote_read") = false,
            py::arg("transpose_mcast") = false,
            py::arg("queue_id") = 0,
        });
}
}  // namespace ttnn::operations::data_movement::detail
