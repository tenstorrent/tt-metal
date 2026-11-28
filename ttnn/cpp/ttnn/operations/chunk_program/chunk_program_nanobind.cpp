// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
// SPDX-License-Identifier: Apache-2.0

#include "chunk_program_nanobind.hpp"

#include <nanobind/nanobind.h>
#include <nanobind/stl/shared_ptr.h>
#include <nanobind/stl/vector.h>

#include "chunk_program.hpp"
#include "ttnn-nanobind/bind_function.hpp"

namespace ttnn::operations::chunk_program {

void bind_chunk_program_operation(nb::module_& mod) {
    nb::class_<ChunkPlan>(mod, "ChunkProgramPlan", "The structural plan of one per-chunk program instance (opaque).")
        .def_prop_ro("kind", [](const ChunkPlan& p) { return static_cast<uint32_t>(p.kind); })
        .def_prop_ro("hash", [](const ChunkPlan& p) { return p.hash; });

    mod.def(
        "chunk_program_plan",
        &ttnn::chunk_program_plan,
        nb::arg("kind"),
        nb::arg("rows"),
        nb::arg("cols"),
        nb::arg("consts"),
        nb::arg("groups"),
        nb::arg("sems"),
        R"doc(The plan of one per-chunk program instance of deepseek_v41_flash_d_p, made once (kinds: 0 page copy,
        1 glued rows, 2 ids reorder, 3 glued ids, 4 derive, 5 owned send, 6 page copies of one page size (NC.9)).)doc");

    ttnn::bind_function<"chunk_program">(
        mod,
        R"doc(
        One call of a per-chunk program of deepseek_v41_flash_d_p (page copy, glued RoPE rows of one or two tables,
        the engine ids into natural / model order, the glued chunk's header derive sections, the owned history send;
        NC.7): the native C++ op of the model's Python prepared programs, the same kernels. In place; returns io[0].

        Args:
            io (List[ttnn.Tensor]): the kind's io tensors (ttnn/cpp/ttnn/operations/chunk_program/chunk_program.hpp).
            plan (ttnn.ChunkProgramPlan): from chunk_program_plan.
            values (List[int]): the call's values (never in the program key).
        )doc",
        &ttnn::chunk_program,
        nb::arg("io"),
        nb::arg("plan"),
        nb::arg("values"));
}

}  // namespace ttnn::operations::chunk_program
