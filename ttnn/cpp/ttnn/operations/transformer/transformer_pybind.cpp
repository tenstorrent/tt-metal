// SPDX-FileCopyrightText: © 2024 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include "transformer_pybind.hpp"

#include <pybind11/pybind11.h>
#include <pybind11/stl.h>

#include "ttnn/cpp/pybind11/decorators.hpp"

#include "concatenate_heads/concatenate_heads_pybind.hpp"
#include "attention_softmax/attention_softmax_pybind.hpp"
#include "split_query_key_value_and_split_heads/split_query_key_value_and_split_heads_pybind.hpp"


namespace ttnn::operations::transformer {

void py_module(pybind11::module& module) {
    py_bind_attention_softmax(module);
    py_bind_concatenate_heads(module);
    py_bind_split_query_key_value_and_split_heads(module);
}

}  // namespace ttnn::operations::transformer
