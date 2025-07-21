// SPDX-FileCopyrightText: © 2025 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include "cluster.hpp"

#include <pybind11/pybind11.h>

#include <tt-metalium/tt_metal.hpp>
#include <tt-metalium/impl/context/metal_context.hpp>
#include <tt-metalium/llrt/tt_cluster.hpp>

#include "ttnn/cluster.hpp"

using namespace tt::tt_metal;

namespace py = pybind11;

namespace {

void ttnn_cluster(py::module& module) {
    // Bind ClusterType enum
    py::enum_<tt::ClusterType>(module, "ClusterType", "Enum representing different cluster types")
        .value("INVALID", tt::ClusterType::INVALID, "Invalid cluster type")
        .value("N150", tt::ClusterType::N150, "Production N150")
        .value("N300", tt::ClusterType::N300, "Production N300")
        .value("T3K", tt::ClusterType::T3K, "Production T3K, built with 4 N300s")
        .value("GALAXY", tt::ClusterType::GALAXY, "Production Galaxy, all chips with mmio")
        .value("TG", tt::ClusterType::TG, "Will be deprecated")
        .value("P100", tt::ClusterType::P100, "Blackhole single card, ethernet disabled")
        .value("P150", tt::ClusterType::P150, "Blackhole single card, ethernet enabled")
        .value("P150_X2", tt::ClusterType::P150_X2, "2 Blackhole single card, ethernet connected")
        .value("P150_X4", tt::ClusterType::P150_X4, "4 Blackhole single card, ethernet connected")
        .value("SIMULATOR_WORMHOLE_B0", tt::ClusterType::SIMULATOR_WORMHOLE_B0, "Simulator Wormhole B0")
        .value("SIMULATOR_BLACKHOLE", tt::ClusterType::SIMULATOR_BLACKHOLE, "Simulator Blackhole")
        .value("N300_2x2", tt::ClusterType::N300_2x2, "2 N300 cards, ethernet connected to form 2x2");

    module.def(
        "get_cluster_type",
        []() -> tt::ClusterType {
            return tt::tt_metal::MetalContext::instance().get_cluster().get_cluster_type();
        },
        R"doc(
            Get the cluster type of the current cluster.
            
            Returns:
                ttnn.cluster.ClusterType: The type of the current cluster.
            
            Example:
                >>> import ttnn
                >>> cluster_type = ttnn.cluster.get_cluster_type()
                >>> print(cluster_type)
                ttnn.cluster.ClusterType.N150  # (example output)
                >>> 
                >>> # You can also compare cluster types
                >>> if cluster_type == ttnn.cluster.ClusterType.T3K:
                ...     print("Running on T3K cluster")
                >>> 
                >>> # Or use in conditional logic
                >>> is_galaxy = cluster_type in [ttnn.cluster.ClusterType.GALAXY, ttnn.cluster.ClusterType.TG]
        )doc");

    module.def(
        "is_galaxy_cluster",
        []() -> bool {
            return tt::tt_metal::MetalContext::instance().get_cluster().is_galaxy_cluster();
        },
        R"doc(
            Check if the current cluster is a Galaxy cluster.
            
            Returns:
                bool: True if the cluster is a Galaxy cluster, False otherwise.
            
            Example:
                >>> import ttnn
                >>> if ttnn.cluster.is_galaxy_cluster():
                ...     print("Running on Galaxy cluster")
                ... else:
                ...     print("Running on non-Galaxy cluster")
        )doc");

    module.def(
        "number_of_user_devices", 
        []() -> size_t {
            return tt::tt_metal::MetalContext::instance().get_cluster().number_of_user_devices();
        },
        R"doc(
            Get the number of user-accessible devices in the cluster.
            
            Returns:
                int: The number of user devices in the cluster.
            
            Note:
                For Galaxy systems, this excludes MMIO gateway chips that are only used for dispatch.
            
            Example:
                >>> import ttnn
                >>> num_devices = ttnn.cluster.number_of_user_devices()
                >>> print(f"Cluster has {num_devices} user devices")
        )doc");

    module.def(
        "serialize_cluster_descriptor",
        &ttnn::cluster::serialize_cluster_descriptor,
        R"doc(
            Serialize cluster descriptor to a file.
            
            Returns:
                str: Path to the serialized cluster descriptor file.
            
            Example:
                >>> import ttnn
                >>> descriptor_path = ttnn.cluster.serialize_cluster_descriptor()
                >>> print(f"Cluster descriptor saved to: {descriptor_path}")
        )doc");
}

}  // namespace

namespace ttnn::cluster {

void py_cluster_module(py::module& module) {
    ttnn_cluster(module);
}

}  // namespace ttnn::cluster
