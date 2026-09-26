// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include "ttnn-nanobind/nanobind_fwd.hpp"

// The binding namespace is `hello_world_binding` rather than `hello_world`: the
// public op function is itself named `hello_world` in the enclosing namespace,
// and a namespace and a function may not share a name in the same scope
// (same convention as fft -> fft_binding).
namespace ttnn::operations::experimental::hello_world_binding::detail {

namespace nb = nanobind;

void bind_experimental_hello_world_operation(nb::module_& mod);

}  // namespace ttnn::operations::experimental::hello_world_binding::detail
