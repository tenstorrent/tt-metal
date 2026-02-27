// SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#include "pipeline_module_nanobind.hpp"

#include <sstream>

#include <nanobind/nanobind.h>
#include <nanobind/stl/vector.h>

#include "tt-metalium/experimental/blitz_decode_pipeline.hpp"

namespace ttnn::pipeline_module {

void bind_blitz_decode_pipeline(nb::module_& mod) {
    using tt::tt_metal::experimental::distributed::BlitzDecodePipelineStage;

    nb::class_<BlitzDecodePipelineStage>(mod, "BlitzDecodePipelineStage")
        .def_ro("stage_index", &BlitzDecodePipelineStage::stage_index)
        .def_ro("entry_node_coord", &BlitzDecodePipelineStage::entry_node_coord)
        .def_ro("exit_node_coord", &BlitzDecodePipelineStage::exit_node_coord)
        .def("__repr__", [](const BlitzDecodePipelineStage& stage) {
            std::ostringstream repr;
            repr << "BlitzDecodePipelineStage(stage_index=" << stage.stage_index
                 << ", entry_node_coord=" << stage.entry_node_coord << ", exit_node_coord=" << stage.exit_node_coord
                 << ")";
            return repr.str();
        });

    mod.def(
        "generate_blitz_decode_pipeline",
        [](tt::tt_metal::distributed::MeshDevice& mesh_device) {
            return tt::tt_metal::experimental::distributed::generate_blitz_decode_pipeline(mesh_device);
        },
        nb::arg("mesh_device"),
        R"doc(
            Generate the Blitz decode pipeline stages for the provided mesh device.

            Args:
                mesh_device (MeshDevice): Mesh device for which to generate the pipeline stages.

            Returns:
                List[BlitzDecodePipelineStage]: Ordered pipeline stages for Blitz decode.
        )doc");
}

}  // namespace ttnn::pipeline_module
