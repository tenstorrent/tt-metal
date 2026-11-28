// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
// SPDX-License-Identifier: Apache-2.0

#include "layer_send_nanobind.hpp"

#include <nanobind/nanobind.h>
#include <nanobind/stl/shared_ptr.h>
#include <nanobind/stl/tuple.h>
#include <nanobind/stl/vector.h>

#include "layer_send.hpp"
#include "ttnn-nanobind/bind_function.hpp"

namespace ttnn::operations::layer_send {

void bind_layer_send_operation(nb::module_& mod) {
    nb::class_<SendPlan>(mod, "LayerSendPlan", "The structural plan of one per-layer send instance (opaque).")
        .def_prop_ro("kind", [](const SendPlan& p) { return static_cast<uint32_t>(p.kind); })
        .def_prop_ro("hash", [](const SendPlan& p) { return p.hash; });

    mod.def(
        "layer_send_plan",
        &ttnn::layer_send_plan,
        nb::arg("kind"),
        nb::arg("rows"),
        nb::arg("cols"),
        nb::arg("wrap_x"),
        nb::arg("wrap_y"),
        nb::arg("hops"),
        nb::arg("consts"),
        nb::arg("sems"),
        nb::arg("pattern"),
        R"doc(The plan of one per-layer send instance of deepseek_v41_flash_d_p, made once at load time (kinds: 0 window,
        1 window fused, 2 snapshot static, 3 snapshot dyn, 4 snapshot multi, 5 glued). hops: per chip (chip-linear) a
        list of (dir, peer row, peer col, link, dst row, dst col).)doc");

    ttnn::bind_function<"layer_send">(
        mod,
        R"doc(
        One call of a per-layer fabric send of deepseek_v41_flash_d_p (the window neighbour send, its fused window
        write, the window snapshot broadcast (static, dynamic, multi-segment) and the glued window send; NC.3): the
        native C++ op of the model's Python prepared programs, the same kernels. In place; returns io[0].

        Args:
            io (List[ttnn.Tensor]): the kind's io tensors (ttnn/cpp/ttnn/operations/layer_send/layer_send.hpp).
            plan (ttnn.LayerSendPlan): from ttnn.layer_send_plan.
            values (List[int]): the call's values (never in the program key).
        )doc",
        &ttnn::layer_send,
        nb::arg("io"),
        nb::arg("plan"),
        nb::arg("values"));
}

}  // namespace ttnn::operations::layer_send
