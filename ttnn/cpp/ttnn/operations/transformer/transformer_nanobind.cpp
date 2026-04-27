// SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#include "transformer_nanobind.hpp"

#include <nanobind/nanobind.h>

#include "concatenate_heads/concatenate_heads_nanobind.hpp"
#include "split_query_key_value_and_split_heads/split_query_key_value_and_split_heads_nanobind.hpp"

namespace ttnn::operations::transformer {

void py_module(nb::module_& mod) {
    bind_concatenate_heads(mod);
    bind_split_query_key_value_and_split_heads(mod);
}

}  // namespace ttnn::operations::transformer
