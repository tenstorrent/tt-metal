// SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#include <hal/hal.hpp>

#include <llrt/tt_cluster.hpp>
#include <llrt/get_platform_architecture.hpp>

std::unique_ptr<tt::tt_metal::Hal> create_hal(const std::unique_ptr<tt::umd::Cluster> &cluster) {
    tt::llrt::RunTimeOptions rtoptions;
    tt::umd::ClusterDescriptor* cluster_descriptor = cluster->get_cluster_description();
    bool is_base_routing_fw_enabled =
        tt::Cluster::is_base_routing_fw_enabled(tt::Cluster::get_cluster_type_from_cluster_desc(rtoptions, cluster_descriptor));
    // is_2_erisc_mode is true. But it can change to false on the Metal runtime to maintain backward compabaility with
    // older firmware.
    // This value doesn't affect the addresses of any buffers.
    // Telemetry doesn't care about number of ERISC processors so it's ok to not be in sync with the value on Metal. 
    bool is_2_erisc_mode = rtoptions.get_enable_2_erisc_mode();
    return std::make_unique<tt::tt_metal::Hal>(tt::tt_metal::get_platform_architecture(rtoptions), is_base_routing_fw_enabled, is_2_erisc_mode);
}
