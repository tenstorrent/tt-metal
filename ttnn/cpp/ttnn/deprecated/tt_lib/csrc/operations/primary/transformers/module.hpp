// SPDX-FileCopyrightText: © 2024 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include "ttnn/deprecated/tt_dnn/op_library/sdpa/sdpa_op.hpp"

#include <pybind11/pybind11.h>
#include <pybind11/stl.h>

namespace py = pybind11;

namespace tt {
namespace operations {
namespace primary {
namespace transformers {


void py_module(py::module& m_transformers) {
    py::class_<SDPADefaultProgramConfig>(m_transformers, "SDPADefaultProgramConfig")
        .def(py::init<>());

    py::class_<SDPAMultiCoreProgramConfig>(m_transformers, "SDPAMultiCoreProgramConfig")
        .def(py::init<CoreCoord, std::size_t, std::size_t>(), py::kw_only(), py::arg("compute_with_storage_grid_size"), py::arg("q_chunk_size").noconvert(), py::arg("k_chunk_size").noconvert())
        .def_readwrite("compute_with_storage_grid_size", &SDPAMultiCoreProgramConfig::compute_with_storage_grid_size)
        .def_readwrite("q_chunk_size", &SDPAMultiCoreProgramConfig::q_chunk_size)
        .def_readwrite("k_chunk_size", &SDPAMultiCoreProgramConfig::k_chunk_size);

    m_transformers.def(
        "scaled_dot_product_attention",
        &scaled_dot_product_attention,
        py::arg("input_tensor_q").noconvert(),
        py::arg("input_tensor_k").noconvert(),
        py::arg("input_tensor_v").noconvert(),
        py::arg("causal_mask").noconvert(),
        py::arg("is_causal").noconvert() = true,
        py::arg("scale").noconvert() = std::nullopt,
        py::arg("output_mem_config").noconvert() = operation::DEFAULT_OUTPUT_MEMORY_CONFIG,
        py::arg("program_config").noconvert() = SDPADefaultProgramConfig{},
        py::arg("compute_kernel_config").noconvert() = std::nullopt,
        py::arg("valid_seq_len").noconvert() = std::nullopt,
        "Causal scaled dot product attention. This API mimicks the PyTorch API of the same name."
        "The implementation is FlashAttention-2 and it currently only supports MQA with causal masking.\n"

        "Q:      [b x nqh x s x dh]"
        "K:      [b x 1   x s x dh]"
        "V:      [b x 1   x s x dh]"
        "mask:   [b x 1   x s x s ]"
        "output: [b x nqh x s x dh]"

        "Mask must be a causal mask with 0s in the lower triangle and -inf in the upper triangle."

        "Accepts a `SDPAMultiCoreProgramConfig` which specifies the grid size and chunk tiles in the Q and K sequence lengths. The op parallelizes over `b`, `nqh`, and Q's `s` dimension."
        );

    m_transformers.def(
        "scaled_dot_product_attention_decode",
        &scaled_dot_product_attention_decode,
        py::arg("input_tensor_q").noconvert(),
        py::arg("input_tensor_k").noconvert(),
        py::arg("input_tensor_v").noconvert(),
        py::arg("cur_pos").noconvert(),
        py::arg("scale").noconvert() = std::nullopt,
        py::arg("output_mem_config").noconvert() = operation::DEFAULT_OUTPUT_MEMORY_CONFIG,
        py::arg("program_config").noconvert() = SDPADefaultProgramConfig{},
        py::arg("compute_kernel_config").noconvert() = std::nullopt,
        "A version of scaled dot product attention specifically for decode."
        "The implementation is Flash-Decode and it currently only supports MQA on decoding single token.\n"

        "Q:      [1 x b x pnh x dh]"
        "K:      [1 x b x   s x dh]"
        "V:      [1 x b x   s x dh]"
        "cur_pos: list of integers of length b"
        "output: [1 x b x pnh x dh]"

        "Accepts a `SDPAMultiCoreProgramConfig` which specifies the grid size and chunk tiles in the K/V/Mask sequence lengths (Q chunk tiles is not used). The op parallelizes over `b` and K/V/Mask's `s` dimension."
        );
    m_transformers.def(
        "scaled_dot_product_attention_decode_gqa",
        &scaled_dot_product_attention_decode_gqa,
        py::arg("input_tensor_q").noconvert(),
        py::arg("input_tensor_k").noconvert(),
        py::arg("input_tensor_v").noconvert(),
        py::arg("cur_pos").noconvert(),
        py::arg("scale").noconvert() = std::nullopt,
        py::arg("output_mem_config").noconvert() = operation::DEFAULT_OUTPUT_MEMORY_CONFIG,
        py::arg("program_config").noconvert() = SDPADefaultProgramConfig{},
        py::arg("compute_kernel_config").noconvert() = std::nullopt,
        "A version of scaled dot product attention specifically for GQA decode."

        "Q:      [1 x qh x b x dh]"
        "K:      [b x kh x s x dh]"
        "V:      [b x kh x s x dh]"
        "cur_pos: list of integers of length b"
        "output: [1 x b x qh x dh]"

        "Accepts a `SDPAMultiCoreProgramConfig` which specifies the grid size and chunk tiles in the K/V/Mask sequence lengths (Q chunk tiles is not used). The op parallelizes over `b` and K/V/Mask's `s` dimension."
        );
}

}  // namespace transformers
}  // namespace primary
}  // namespace operations
}  // namespace tt
