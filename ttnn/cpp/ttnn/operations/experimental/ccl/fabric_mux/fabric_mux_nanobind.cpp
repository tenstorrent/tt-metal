// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include "fabric_mux_nanobind.hpp"

#include <cstddef>
#include <cstdint>
#include <optional>
#include <vector>

#include <nanobind/nanobind.h>
#include <nanobind/stl/optional.h>
#include <nanobind/stl/vector.h>

#include <tt-metalium/experimental/fabric/fabric.hpp>
#include <tt-metalium/kernel_types.hpp>
#include <tt-metalium/program_descriptors.hpp>

#include "ttnn-nanobind/export_enum.hpp"
#include "ttnn/operations/ccl/ccl_common.hpp"

namespace ttnn::operations::experimental::fabric_mux {
namespace {

// The C++ helper appends arguments to an existing vector; Python callers need
// an independent value that can be inserted into a kernel descriptor.
std::vector<uint32_t> client_compile_time_args(
    uint32_t num_clients,
    tt::tt_fabric::FabricMuxChannelType channel_type,
    const tt::tt_fabric::FabricMuxConfig& config) {
    std::vector<uint32_t> arguments;
    ttnn::ccl::fabric_mux_connection_ct_args(num_clients, channel_type, config, arguments);
    return arguments;
}

// Keep semaphore allocation in TT-Metal so descriptor construction uses the
// same runtime-argument ABI and semaphore ownership rules as native CCLs.
std::vector<uint32_t> client_runtime_args(
    bool connection_valid,
    bool is_termination_master,
    tt::tt_fabric::FabricMuxChannelType channel_type,
    const tt::tt_metal::CoreCoord& mux_virtual_core,
    uint32_t client_index,
    const tt::tt_metal::CoreCoord& client_logical_core,
    const tt::tt_fabric::FabricMuxConfig& config,
    tt::tt_metal::ProgramDescriptor& program_descriptor,
    const tt::tt_metal::CoreCoord& termination_master_virtual_core,
    std::optional<uint32_t> termination_master_semaphore_id) {
    std::vector<uint32_t> arguments;
    ttnn::ccl::fabric_mux_connection_rt_args(
        connection_valid,
        is_termination_master,
        channel_type,
        mux_virtual_core,
        client_index,
        client_logical_core,
        config,
        program_descriptor,
        termination_master_virtual_core,
        arguments,
        termination_master_semaphore_id);
    return arguments;
}

// Route setup must remain in TT-Metal because it owns fabric topology state
// and appends the resulting connection resources to the descriptor.
std::vector<uint32_t> kernel_runtime_args(
    const tt::tt_fabric::FabricMuxConfig& config,
    const tt::tt_fabric::FabricNodeId& source_node_id,
    const tt::tt_fabric::FabricNodeId& destination_node_id,
    uint32_t link_index,
    tt::tt_metal::ProgramDescriptor& program_descriptor,
    const tt::tt_metal::CoreCoord& mux_logical_core) {
    return config.get_fabric_mux_run_time_args<tt::tt_metal::ProgramDescriptor>(
        source_node_id, destination_node_id, link_index, program_descriptor, mux_logical_core);
}

}  // namespace

void bind_fabric_mux(nb::module_& experimental_module) {
    auto fabric_mux_module = experimental_module.def_submodule(
        "fabric_mux", "Experimental program-descriptor configuration for a program-local fabric mux.");

    nb::enum_<tt::tt_fabric::FabricMuxChannelType>(fabric_mux_module, "ChannelType", "Fabric mux channel payload type.")
        .value("FULL_SIZE", tt::tt_fabric::FabricMuxChannelType::FULL_SIZE_CHANNEL)
        .value("HEADER_ONLY", tt::tt_fabric::FabricMuxChannelType::HEADER_ONLY_CHANNEL);

    export_enum<tt::tt_metal::KernelBuildOptLevel>(fabric_mux_module, "KernelBuildOptLevel");

    nb::class_<tt::tt_fabric::FabricMuxConfig>(
        fabric_mux_module, "Config", "Computes the kernel arguments and L1 layout for a program-local fabric mux.")
        .def(
            nb::init<uint8_t, uint8_t, uint8_t, uint8_t, size_t, size_t, tt::CoreType>(),
            nb::kw_only(),
            nb::arg("num_full_size_channels"),
            nb::arg("num_header_only_channels"),
            nb::arg("num_buffers_per_full_size_channel"),
            nb::arg("num_buffers_per_header_only_channel"),
            nb::arg("full_size_channel_buffer_size_bytes"),
            nb::arg("base_l1_address"),
            nb::arg("core_type") = nb::cast(tt::CoreType::WORKER),
            "Create a mux configuration whose private memory begins at base_l1_address.")
        .def(
            "kernel_compile_time_args",
            &tt::tt_fabric::FabricMuxConfig::get_fabric_mux_compile_time_args,
            "Return compile-time arguments for tt_fabric_mux.cpp.")
        .def(
            "kernel_runtime_args",
            &kernel_runtime_args,
            nb::kw_only(),
            nb::arg("source_node_id"),
            nb::arg("destination_node_id"),
            nb::arg("link_index"),
            nb::arg("program_descriptor"),
            nb::arg("mux_logical_core"),
            "Return mux-kernel runtime arguments and append its fabric connection resources to the descriptor.")
        .def("num_channels", &tt::tt_fabric::FabricMuxConfig::get_num_channels, nb::arg("channel_type"))
        .def("num_buffers", &tt::tt_fabric::FabricMuxConfig::get_num_buffers, nb::arg("channel_type"))
        .def("buffer_size_bytes", &tt::tt_fabric::FabricMuxConfig::get_buffer_size_bytes, nb::arg("channel_type"))
        .def("memory_map_end_address", &tt::tt_fabric::FabricMuxConfig::get_memory_map_end_address);

    fabric_mux_module.def(
        "client_compile_time_args",
        &client_compile_time_args,
        nb::kw_only(),
        nb::arg("num_clients"),
        nb::arg("channel_type"),
        nb::arg("config"),
        "Return the compile-time arguments required by a fabric-mux client kernel.");

    fabric_mux_module.def(
        "client_runtime_args",
        &client_runtime_args,
        nb::kw_only(),
        nb::arg("connection_valid"),
        nb::arg("is_termination_master"),
        nb::arg("channel_type"),
        nb::arg("mux_virtual_core"),
        nb::arg("client_index"),
        nb::arg("client_logical_core"),
        nb::arg("config"),
        nb::arg("program_descriptor"),
        nb::arg("termination_master_virtual_core"),
        nb::arg("termination_master_semaphore_id") = nb::none(),
        "Return one client's runtime arguments and append its semaphores to the descriptor.");

    fabric_mux_module.def(
        "channel_buffer_size_bytes",
        &tt::tt_fabric::get_tt_fabric_channel_buffer_size_bytes,
        "Return the maximum buffer size for one full-size mux channel.");
}

}  // namespace ttnn::operations::experimental::fabric_mux
