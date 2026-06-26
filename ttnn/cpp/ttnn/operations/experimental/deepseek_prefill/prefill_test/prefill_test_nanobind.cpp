// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include "prefill_test_nanobind.hpp"

#include <cstdint>
#include <string>

#include <nanobind/nanobind.h>
#include <nanobind/stl/string.h>

#include "layer_completion_consumer.hpp"

namespace ttnn::operations::experimental::deepseek_prefill::prefill_test::detail {

void bind_prefill_test(nb::module_& mod) {
    using ttnn::operations::experimental::deepseek_prefill::LayerCompletionConsumer;

    nb::class_<LayerCompletionConsumer>(mod, "LayerCompletionConsumer")
        .def(
            "__init__",
            [](LayerCompletionConsumer* self,
               const std::string& channel_shm_name,
               uint64_t expected,
               uint32_t connect_timeout_ms,
               uint64_t log_step) {
                new (self) LayerCompletionConsumer(channel_shm_name, expected, connect_timeout_ms, log_step);
            },
            nb::arg("channel_shm_name"),
            nb::arg("expected"),
            nb::arg("connect_timeout_ms") = 30'000u,
            nb::arg("log_step") = 61u,
            "Test/scheduler stand-in: connect to the scheduler counter channel and drain it on a NATIVE "
            "C++ thread (GIL-immune), self-terminating once `expected` completions are drained.")
        .def("stop", &LayerCompletionConsumer::stop, "Idempotent: stop + join + final drain + shutdown channel.")
        .def_prop_ro("total", &LayerCompletionConsumer::total)
        .def_prop_ro("reached_expected", &LayerCompletionConsumer::reached_expected);
}

}  // namespace ttnn::operations::experimental::deepseek_prefill::prefill_test::detail

namespace ttnn::operations::experimental::deepseek_prefill::detail {

void bind_prefill_test(::nanobind::module_& mod) { prefill_test::detail::bind_prefill_test(mod); }

}  // namespace ttnn::operations::experimental::deepseek_prefill::detail
