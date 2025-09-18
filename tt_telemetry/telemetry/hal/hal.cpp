// SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#include <hal/hal.hpp>

#include <llrt/tt_cluster.hpp>
#include <llrt/get_platform_architecture.hpp>

std::unique_ptr<tt::tt_metal::Hal> create_hal(const std::unique_ptr<tt::umd::Cluster> &cluster) {
    tt::llrt::RunTimeOptions rtoptions;
    tt::umd::tt_ClusterDescriptor* cluster_descriptor = cluster->get_cluster_description();
    bool is_base_routing_fw_enabled =
        tt::Cluster::is_base_routing_fw_enabled(tt::Cluster::get_cluster_type_from_cluster_desc(rtoptions, cluster_descriptor));
    return std::make_unique<tt::tt_metal::Hal>(tt::tt_metal::get_platform_architecture(rtoptions), is_base_routing_fw_enabled);
}
