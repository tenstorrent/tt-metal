// SPDX-FileCopyrightText: © 2025 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include "cluster.hpp"

#include <nanobind/nanobind.h>

#include <tt-metalium/tt_metal.hpp>

#include "ttnn/cluster.hpp"

using namespace tt::tt_metal;

namespace ttnn::cluster {

void py_cluster_module(nb::module_& mod) {
    mod.def(
        "serialize_cluster_descriptor",
        &ttnn::cluster::serialize_cluster_descriptor,
        R"doc(
               Serialize cluster descriptor to a file.
             )doc");
}

}  // namespace ttnn::cluster
