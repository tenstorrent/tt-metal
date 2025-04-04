// SPDX-FileCopyrightText: © 2025 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include "cluster.hpp"

#include <pybind11/operators.h>
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>

#include "small_vector_caster.hpp"  // NOLINT - for pybind11 SmallVector binding support.
#include <tt-metalium/persistent_kernel_cache.hpp>
#include <tt-metalium/memory_reporter.hpp>
#include <tt-metalium/device_impl.hpp>
#include <tt-metalium/tt_metal.hpp>
#include <tt-metalium/distributed.hpp>
#include <tt-metalium/host_api.hpp>
#include <tt-metalium/hal.hpp>
#include <tt-metalium/trace.hpp>
#include "ttnn/operations/experimental/auto_format/auto_format.hpp"
#include <tt-metalium/hal.hpp>
#include "tools/profiler/op_profiler.hpp"

using namespace tt::tt_metal;

namespace py = pybind11;

namespace ttnn {
namespace cluster {

void py_cluster_module(py::module& module) {
    module.def(
        "serialize_cluster_descriptor",
        &ttnn::cluster::serialize_cluster_descriptor,
        R"doc(
               Serialize cluster descriptor to a file.
             )doc");
}

}  // namespace cluster
}  // namespace ttnn
