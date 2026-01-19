// SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#include <hal/hal.hpp>

#include <llrt/tt_cluster.hpp>
#include <llrt/get_platform_architecture.hpp>
#include <impl/profiler/profiler_state_manager.hpp>

std::unique_ptr<tt::tt_metal::Hal> create_hal(const std::unique_ptr<tt::umd::Cluster>& cluster) {
    tt::llrt::RunTimeOptions rtoptions;
    tt::umd::ClusterDescriptor* cluster_descriptor = cluster->get_cluster_description();
    bool is_base_routing_fw_enabled = tt::Cluster::is_base_routing_fw_enabled(
        tt::Cluster::get_cluster_type_from_cluster_desc(rtoptions, cluster_descriptor));
    // Telemetry currently doesn't collect data on the granularity of each ERISC processor / RISCV.
    // is_2_erisc_mode is hardcoded to True to show 2 ERISC processors on Blackhole with the
    // Hal::get_num_risc_processors API. This parameter doesn't affect the addresses of any buffers in L1. If Metal only
    // has 1 ERISC enabled for Blackhole then telemetry will simply read empty values for the other ERISC.
    constexpr bool is_2_erisc_mode = true;
    return std::make_unique<tt::tt_metal::Hal>(
        tt::tt_metal::get_platform_architecture(rtoptions),
        is_base_routing_fw_enabled,
        is_2_erisc_mode,
        tt::tt_metal::get_profiler_dram_bank_size_for_hal_allocation(rtoptions));
}
