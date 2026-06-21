// SPDX-License-Identifier: Apache-2.0
#pragma once
#include <nanobind/nanobind.h>
namespace nb = nanobind;
namespace ttnn::operations::experimental::deltanet::detail {
void bind_deltanet_prefill_chunked(nb::module_& mod);
}  // namespace ttnn::operations::experimental::deltanet::detail
