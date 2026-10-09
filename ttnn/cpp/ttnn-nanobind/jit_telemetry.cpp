// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include "jit_telemetry.hpp"

#include <cstdint>
#include <vector>

#include <nanobind/nanobind.h>
#include <nanobind/stl/string.h>

#include <tt-metalium/experimental/jit_telemetry.hpp>

namespace ttnn::jit_telemetry {

namespace {

namespace jt = tt::tt_metal::experimental::jit_telemetry;

// {name: {"unit", "count", "total", "min", "max"}}
nb::dict to_dict(const std::vector<jt::TokenStats>& stats) {
    nb::dict out;
    for (const auto& s : stats) {
        nb::dict d;
        d["unit"] = s.unit;
        d["count"] = s.count;
        d["total"] = s.total;
        d["min"] = s.min;
        d["max"] = s.max;
        out[s.name.c_str()] = d;
    }
    return out;
}

}  // namespace

void py_module(nb::module_& mod) {
    mod.def("begin_capture", &jt::begin_capture, R"doc(
        Start a JIT telemetry capture and return its id. Prefer ttnn.jit_telemetry.capture().
    )doc");
    mod.def(
        "end_capture",
        [](uint64_t capture_id) { return to_dict(jt::end_capture(capture_id)); },
        nb::arg("capture_id"),
        R"doc(
        Close a capture; returns {token name: {"unit", "count", "total", "min", "max"}} for the
        tokens that recorded into it. Raises if the id is not open.
    )doc");
    mod.def(
        "snapshot",
        []() { return to_dict(jt::snapshot()); },
        R"doc(
        Process-wide values of every token, in the same format as end_capture().
    )doc");
}

}  // namespace ttnn::jit_telemetry
