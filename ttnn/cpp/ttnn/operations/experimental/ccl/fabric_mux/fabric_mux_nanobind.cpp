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

std::vector<uint32_t> client_compile_time_args(
    uint32_t num_clients,
    tt::tt_fabric::FabricMuxChannelType channel_type,
    const tt::tt_fabric::FabricMuxConfig& config) {
    std::vector<uint32_t> arguments;
    ttnn::ccl::fabric_mux_connection_ct_args(num_clients, channel_type, config, arguments);
    return arguments;
}

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
            nb::init<uint8_t, uint8_t, uint8_t, uint8_t, size_t, size_t, tt::CoreType, size_t>(),
            nb::kw_only(),
            nb::arg("num_full_size_channels"),
            nb::arg("num_header_only_channels"),
            nb::arg("num_buffers_per_full_size_channel"),
            nb::arg("num_buffers_per_header_only_channel"),
            nb::arg("full_size_channel_buffer_size_bytes"),
            nb::arg("base_l1_address"),
            nb::arg("core_type") = nb::cast(tt::CoreType::WORKER),
            nb::arg("usable_l1_end_address") = 0,
            R"doc(
            Configure the mux L1 memory map.

            Args:
                num_full_size_channels: Number of full-size channels.
                num_header_only_channels: Number of header-only channels.
                num_buffers_per_full_size_channel: Buffers per full-size channel.
                num_buffers_per_header_only_channel: Buffers per header-only channel.
                full_size_channel_buffer_size_bytes: Bytes per full-size buffer.
                base_l1_address: First address of the mux memory map.
                core_type: Core type for the mux; defaults to WORKER.
                usable_l1_end_address: Exclusive L1 ceiling; 0 selects the physical L1 end.

            Raises:
                RuntimeError: No channels are configured, a buffer exceeds the supported size, the core type is
                    unsupported, or the memory map exceeds the L1 ceiling.
            )doc")
        .def(
            "kernel_compile_time_args",
            &tt::tt_fabric::FabricMuxConfig::get_fabric_mux_compile_time_args,
            R"doc(
            Return compile-time arguments for the fabric mux kernel.

            Returns:
                list[int]: Mux kernel compile-time arguments.
            )doc")
        .def(
            "kernel_runtime_args",
            &tt::tt_fabric::FabricMuxConfig::get_fabric_mux_run_time_args<tt::tt_metal::ProgramDescriptor>,
            nb::kw_only(),
            nb::arg("source_node_id"),
            nb::arg("destination_node_id"),
            nb::arg("link_index"),
            nb::arg("program_descriptor"),
            nb::arg("mux_logical_core"),
            R"doc(
            Build mux kernel runtime arguments and append fabric connection resources to the descriptor.

            Args:
                source_node_id: Fabric node hosting the mux.
                destination_node_id: Fabric node reached through the selected link.
                link_index: Fabric link to the destination node.
                program_descriptor: Descriptor that receives the connection resources.
                mux_logical_core: Logical core hosting the mux.

            Returns:
                list[int]: Mux kernel runtime arguments.
            )doc")
        .def(
            "num_channels",
            &tt::tt_fabric::FabricMuxConfig::get_num_channels,
            nb::arg("channel_type"),
            R"doc(
            Return the number of channels of the selected type.

            Args:
                channel_type: FULL_SIZE or HEADER_ONLY.

            Returns:
                int: Channel count.
            )doc")
        .def(
            "num_buffers",
            &tt::tt_fabric::FabricMuxConfig::get_num_buffers,
            nb::arg("channel_type"),
            R"doc(
            Return the number of buffers in each channel of the selected type.

            Args:
                channel_type: FULL_SIZE or HEADER_ONLY.

            Returns:
                int: Buffers per channel.
            )doc")
        .def(
            "buffer_size_bytes",
            &tt::tt_fabric::FabricMuxConfig::get_buffer_size_bytes,
            nb::arg("channel_type"),
            R"doc(
            Return the size of one buffer in a channel of the selected type.

            Args:
                channel_type: FULL_SIZE or HEADER_ONLY.

            Returns:
                int: Buffer size in bytes.
            )doc")
        .def(
            "memory_map_end_address",
            &tt::tt_fabric::FabricMuxConfig::get_memory_map_end_address,
            R"doc(
            Return the exclusive end address of the mux L1 memory map.

            Returns:
                int: First L1 address after the mux memory map.
            )doc");

    fabric_mux_module.def(
        "client_compile_time_args",
        &client_compile_time_args,
        nb::kw_only(),
        nb::arg("num_clients"),
        nb::arg("channel_type"),
        nb::arg("config"),
        R"doc(
        Build compile-time arguments for a fabric mux client kernel.

        Args:
            num_clients: Number of clients sharing the mux direction.
            channel_type: FULL_SIZE or HEADER_ONLY.
            config: Mux configuration.

        Returns:
            list[int]: Client kernel compile-time arguments.
        )doc");

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
        R"doc(
        Build one client's runtime arguments and append its new semaphores to the descriptor.

        Args:
            connection_valid: Whether this client has a mux connection.
            is_termination_master: Whether this client owns termination synchronization.
            channel_type: FULL_SIZE or HEADER_ONLY.
            mux_virtual_core: Virtual coordinates of the mux core.
            client_index: Channel index for this client.
            client_logical_core: Logical coordinates of the client core.
            config: Mux configuration.
            program_descriptor: Descriptor that receives client semaphores.
            termination_master_virtual_core: Virtual coordinates of the termination master.
            termination_master_semaphore_id: Existing termination semaphore ID to reuse, or None to allocate one.

        Returns:
            list[int]: The 17 client kernel runtime arguments.

        Raises:
            RuntimeError: The client index is outside the configured channel count, or the descriptor has no
                available semaphore IDs on the client core.
        )doc");

    fabric_mux_module.def(
        "channel_buffer_size_bytes",
        &tt::tt_fabric::get_tt_fabric_channel_buffer_size_bytes,
        R"doc(
        Return the maximum buffer size for one full-size mux channel.

        Returns:
            int: Maximum buffer size in bytes.
        )doc");
}

}  // namespace ttnn::operations::experimental::fabric_mux
