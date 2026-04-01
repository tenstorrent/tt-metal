// SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#include "per_core_allocation.hpp"

#include <nanobind/nanobind.h>
#include <nanobind/stl/optional.h>

#include <tt-metalium/experimental/per_core_allocation/mesh_buffer.hpp>
#include <ttnn/tensor/memory_config/memory_config.hpp>
#include <ttnn/tensor/tensor.hpp>
#include <ttnn/tensor/tensor_utils.hpp>
#include <tt-metalium/mesh_buffer.hpp>

namespace nb = nanobind;
using namespace tt::tt_metal;

namespace ttnn::per_core_allocation {

void py_module(nb::module_& m) {
    // Factory that creates a MemoryConfig with per_core_allocation enabled.
    m.def(
        "MemoryConfig",
        [](TensorMemoryLayout memory_layout,
           BufferType buffer_type,
           std::optional<ShardSpec> shard_spec,
           bool per_core_allocation) {
            auto config = MemoryConfig(memory_layout, buffer_type, std::move(shard_spec));
            experimental::per_core_allocation::set_per_core_allocation(config, per_core_allocation);
            return config;
        },
        nb::arg("memory_layout") = TensorMemoryLayout::INTERLEAVED,
        nb::arg("buffer_type") = BufferType::L1,
        nb::arg("shard_spec") = nb::none(),
        nb::arg("per_core_allocation") = true,
        "Create a MemoryConfig with per-core allocation enabled.");

    m.def(
        "set_per_core_allocation",
        [](MemoryConfig& config, bool enable) {
            experimental::per_core_allocation::set_per_core_allocation(config, enable);
        },
        nb::arg("config"),
        nb::arg("enable") = true,
        "Enable or disable per-core allocation on a MemoryConfig.");

    m.def(
        "per_core_buffer_address",
        [](const Tensor& tensor, const CoreCoord& core) -> uint32_t {
            TT_FATAL(is_device_tensor(tensor), "{} doesn't support per_core_buffer_address", tensor.storage_type());
            TT_FATAL(tensor.is_allocated(), "Tensor is not allocated.");
            return experimental::per_core_allocation::get_per_core_address(tensor.mesh_buffer(), core);
        },
        nb::arg("tensor"),
        nb::arg("core"),
        "Get the per-core L1 address for a specific core from a Tensor.");
}

}  // namespace ttnn::per_core_allocation
