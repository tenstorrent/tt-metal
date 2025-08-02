// SPDX-FileCopyrightText: © 2025 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include "cluster.hpp"

#include <pybind11/pybind11.h>

#include "ttnn/cluster.hpp"

namespace py = pybind11;

namespace {

void ttnn_cluster(py::module& module) {
    module.def(
        "get_cluster_type",
        &ttnn::cluster::get_cluster_type,
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
void py_cluster_module_types(py::module& module) {
    // Bind ClusterType enum using the public API
    py::enum_<tt::tt_metal::ClusterType>(module, "ClusterType", "Enum representing different cluster types")
        .value("INVALID", tt::tt_metal::ClusterType::INVALID, "Invalid cluster type")
        .value("N150", tt::tt_metal::ClusterType::N150, "Production N150")
        .value("N300", tt::tt_metal::ClusterType::N300, "Production N300")
        .value("T3K", tt::tt_metal::ClusterType::T3K, "Production T3K, built with 4 N300s")
        .value("GALAXY", tt::tt_metal::ClusterType::GALAXY, "Production Galaxy, all chips with mmio")
        .value("TG", tt::tt_metal::ClusterType::TG, "Will be deprecated")
        .value("P100", tt::tt_metal::ClusterType::P100, "Blackhole single card, ethernet disabled")
        .value("P150", tt::tt_metal::ClusterType::P150, "Blackhole single card, ethernet enabled")
        .value("P150_X2", tt::tt_metal::ClusterType::P150_X2, "2 Blackhole single card, ethernet connected")
        .value("P150_X4", tt::tt_metal::ClusterType::P150_X4, "4 Blackhole single card, ethernet connected")
        .value("P150_X8", tt::tt_metal::ClusterType::P150_X8, "8 Blackhole single card, ethernet connected")
        .value("SIMULATOR_WORMHOLE_B0", tt::tt_metal::ClusterType::SIMULATOR_WORMHOLE_B0, "Simulator Wormhole B0")
        .value("SIMULATOR_BLACKHOLE", tt::tt_metal::ClusterType::SIMULATOR_BLACKHOLE, "Simulator Blackhole")
        .value("N300_2x2", tt::tt_metal::ClusterType::N300_2x2, "2 N300 cards, ethernet connected to form 2x2");
}

void py_cluster_module(py::module& module) {
    ttnn_cluster(module);
}

}  // namespace ttnn::cluster
