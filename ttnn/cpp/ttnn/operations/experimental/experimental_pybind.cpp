// SPDX-FileCopyrightText: © 2023 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include "experimental_pybind.hpp"

#include <pybind11/pybind11.h>
#include <pybind11/stl.h>

#include "ttnn/operations/experimental/reduction/argmax/argmax_pybind.hpp"
#include "ttnn/operations/experimental/ssm/prefix_scan/prefix_scan_pybind.hpp"
#include "ttnn/operations/experimental/ssm/repeat_and_interleave_eltwise_mul/repeat_and_interleave_eltwise_mul_pybind.hpp"
#include "ttnn/operations/experimental/transformer/transformer_pybind.hpp"
#include "ttnn/operations/experimental/transformer/nlp_create_qkv_heads/nlp_create_qkv_heads_pybind.hpp"

#include "ttnn/operations/experimental/transformer/rotary_embedding/rotary_embedding_pybind.hpp"
#include "ttnn/operations/experimental/transformer/rotary_embedding_llama/rotary_embedding_llama_pybind.hpp"

#include "ttnn/operations/experimental/transformer/rotate_half/rotate_half_pybind.hpp"

namespace ttnn::operations::experimental {

void py_module(py::module& module) {
    // Transformer ops
    transformer::detail::bind_experimental_transformer_operations(module);
    transformer::detail::bind_nlp_create_qkv_heads(module);
    reduction::detail::bind_argmax_operation(module);
    reduction::detail::bind_argmin_operation(module);
    ssm::detail::bind_prefix_scan(module);
    ssm::detail::bind_repeat_and_interleave_eltwise_mul(module);

    transformer::py_bind_rotary_embedding(module);
    transformer::py_bind_rotary_embedding_llama(module);
    transformer::py_bind_rotate_half(module);
}

}  // namespace ttnn::operations::experimental
