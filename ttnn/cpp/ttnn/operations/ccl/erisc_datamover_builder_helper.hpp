// SPDX-FileCopyrightText: © 2025 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <tt-metalium/erisc_datamover_builder.hpp>

#include "ttnn/operations/ccl/ccl_host_types.hpp"
#include "ttnn/distributed/types.hpp"

namespace ttnn {
namespace ccl {

class EdmLineFabricOpInterface {
public:
    enum Direction {
        // Ascending chips in the sequence
        FORWARD,

        // Descending chips in the sequence
        BACKWARD,
    };

    //   The constructor will assemble/connect the line across the specified device sequence, for all available links.
    EdmLineFabricOpInterface(
        const std::vector<tt::tt_metal::IDevice*>& device_sequence,
        const std::vector<tt::tt_metal::Program*>& program_sequence,
        std::optional<size_t> desired_num_links = std::nullopt,
        bool build_in_worker_connection_mode = false,
        Topology topology = Topology::Linear,
        bool is_galaxy = false,
        const tt::tt_fabric::FabricRouterBufferConfig& edm_buffer_config = tt::tt_fabric::FabricRouterBufferConfig{});

    // Invocable per chip if we want to collectively build the fabric by building this separately per chip
    // (and implicitly building the fabric that way)
    EdmLineFabricOpInterface(
        tt::tt_metal::IDevice* local_device,
        std::optional<tt::tt_metal::IDevice*> forward_device,
        std::optional<tt::tt_metal::IDevice*> backward_device,
        tt::tt_metal::Program* program,
        std::optional<size_t> desired_num_links,
        bool build_in_worker_connection_mode = false,
        Topology topology = Topology::Linear);

    static EdmLineFabricOpInterface build_program_builder_worker_connection_fabric(
        const std::vector<tt::tt_metal::IDevice*>& device_sequence,
        const std::vector<tt::tt_metal::Program*>& program_sequence,
        std::optional<size_t> desired_num_links = std::nullopt,
        Topology topology = Topology::Linear);
    static EdmLineFabricOpInterface build_program_builder_worker_connection_fabric(
        tt::tt_metal::IDevice* local_device,
        tt::tt_metal::IDevice* forward_device,
        tt::tt_metal::IDevice* backward_device,
        tt::tt_metal::Program* program,
        std::optional<size_t> desired_num_links = std::nullopt,
        Topology topology = Topology::Linear);

    // Will create a connection adapter for a worker which can be used to pass args to the worker kernel talking to the
    // corresponding fabric endpoint. This interface will guarantee unique connections only so requesting more unique
    // connections than available will result in an error.
    tt::tt_fabric::SenderWorkerAdapterSpec uniquely_connect_worker(tt::tt_metal::IDevice* device, Direction direction);

    // builds the ethernet kernels for all EDMs in the "fabric"
    void build_kernels() const;

    // Generates a list of target cores (for now assumed from chip 0 in the line) from farthest
    // to nearest for the sake of sending teardown/termination signals on workload completion.
    // Returns: A list of termination infos which can be passed to a terminate kernel
    // Note there is currently a small bug in that with multiple links, we don't currently know
    // who will be sending the termination signals (and which link(s) they are connected to)
    // and so a termination signal may be sent to our link first before the other eth core links
    // on the chip so multi-link isn't officially supported yet
    std::vector<tt::tt_fabric::edm_termination_info_t> generate_ordered_termination_info_farthest_to_nearest() const;

    // Generates a list of termination infos for the local chip's EDMs
    std::vector<tt::tt_fabric::edm_termination_info_t> generate_local_chip_fabric_termination_infos(
        tt::tt_metal::IDevice* device) const;

    // Accessors
    size_t get_num_links() const { return num_links; }

    size_t get_device_count() const { return device_sequence.size(); }

    size_t get_index_of_device(tt::tt_metal::IDevice* device) const {
        for (size_t i = 0; i < device_sequence.size(); i++) {
            if (device_sequence[i] == device) {
                return i;
            }
        }
        TT_THROW("Device {} not found in device sequence of line fabric", device->id());
        return -1;
    }

    size_t get_edm_buffer_size_bytes() const { return buffer_size_bytes; }

    void teardown_from_host(
        tt::tt_fabric::TerminationSignal termination_signal =
            tt::tt_fabric::TerminationSignal::IMMEDIATELY_TERMINATE) const;

    static void launch_mesh_fabric(MeshDevice* mesh_device);
    static void teardown_edm_fabric(MeshDevice* mesh_device);

    void set_firmware_context_switch_interval(size_t interval);

    // Device ID -> EDM Builders
    std::unordered_map<size_t, std::vector<tt::tt_fabric::FabricEriscDatamoverBuilder>> edm_builders_forward_direction;
    std::unordered_map<size_t, std::vector<tt::tt_fabric::FabricEriscDatamoverBuilder>> edm_builders_backward_direction;

private:
    // Device ID -> link index
    std::unordered_map<size_t, size_t> next_forward_direction_edm_available;
    std::unordered_map<size_t, size_t> next_backward_direction_edm_available;

    std::vector<tt::tt_metal::IDevice*> device_sequence;
    std::vector<tt::tt_metal::Program*> programs;

    size_t num_links;
    size_t buffer_size_bytes;
    size_t firmware_context_switch_interval =
        tt::tt_fabric::FabricEriscDatamoverBuilder::default_firmware_context_switch_interval;
};

void initialize_edm_fabric(
    distributed::MeshDevice* mesh_device,
    bool wrap_fabric_around_mesh = false,
    std::optional<size_t> context_switch_interval_override = std::nullopt,
    Topology topology = Topology::Linear);
void teardown_edm_fabric(
    distributed::MeshDevice* mesh_device, bool wrap_fabric_around_mesh = false, Topology topology = Topology::Linear);

};  // namespace ccl
};  // namespace ttnn
