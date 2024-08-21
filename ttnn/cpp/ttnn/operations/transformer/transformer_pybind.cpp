// SPDX-FileCopyrightText: © 2023 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include "transformer_pybind.hpp"

#include <pybind11/pybind11.h>
#include <pybind11/stl.h>

#include "attention_softmax/attention_softmax_pybind.hpp"
#include "concatenate_heads/concatenate_heads_pybind.hpp"
#include "sdpa/sdpa_pybind.hpp"
#include "sdpa_config.hpp"
#include "sdpa_decode/sdpa_decode_pybind.hpp"
#include "sdpa_decode_gqa/sdpa_decode_gqa_pybind.hpp"
#include "split_query_key_value_and_split_heads/split_query_key_value_and_split_heads_pybind.hpp"

namespace ttnn::operations::transformer {

namespace py = pybind11;

void py_module(py::module& module) {
    py::class_<SDPAProgramConfig>(module, "SDPAProgramConfig")
        .def(
            py::init<CoreCoord, std::size_t, std::size_t>(),
            py::kw_only(),
            py::arg("compute_with_storage_grid_size"),
            py::arg("q_chunk_size").noconvert(),
            py::arg("k_chunk_size").noconvert())
        .def_readwrite("compute_with_storage_grid_size", &SDPAProgramConfig::compute_with_storage_grid_size)
        .def_readwrite("q_chunk_size", &SDPAProgramConfig::q_chunk_size)
        .def_readwrite("k_chunk_size", &SDPAProgramConfig::k_chunk_size);

    py_bind_attention_softmax(module);
    py_bind_concatenate_heads(module);
    py_bind_split_query_key_value_and_split_heads(module);

    py_bind_sdpa(module);
    py_bind_sdpa_decode(module);
    py_bind_sdpa_gqa_decode(module);
}

}  // namespace ttnn::operations::transformer
