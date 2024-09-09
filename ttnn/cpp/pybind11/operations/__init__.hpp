// SPDX-FileCopyrightText: © 2023 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <pybind11/pybind11.h>
#include <pybind11/stl.h>

#include "pybind11/operations/copy.hpp"
#include "pybind11/operations/core.hpp"
#include "pybind11/operations/creation.hpp"
#include "ttnn/operations/ccl/all_gather/all_gather_pybind.hpp"
#include "ttnn/operations/ccl/line_all_gather/line_all_gather_pybind.hpp"
#include "ttnn/operations/ccl/reduce_scatter/reduce_scatter_pybind.hpp"
#include "ttnn/operations/conv/conv2d/conv2d_pybind.hpp"
#include "ttnn/operations/data_movement/data_movement_pybind.hpp"
#include "ttnn/operations/eltwise/binary/binary_pybind.hpp"
#include "ttnn/operations/eltwise/binary_backward/binary_backward_pybind.hpp"
#include "ttnn/operations/eltwise/complex/complex_pybind.hpp"
#include "ttnn/operations/eltwise/complex_unary/complex_unary_pybind.hpp"
#include "ttnn/operations/eltwise/complex_unary_backward/complex_unary_backward_pybind.hpp"
#include "ttnn/operations/eltwise/ternary/ternary_pybind.hpp"
#include "ttnn/operations/eltwise/ternary_backward/ternary_backward_pybind.hpp"
#include "ttnn/operations/eltwise/unary/unary_pybind.hpp"
#include "ttnn/operations/eltwise/unary_backward/unary_backward_pybind.hpp"
#include "ttnn/operations/embedding/embedding_pybind.hpp"
#include "ttnn/operations/embedding_backward/embedding_backward_pybind.hpp"
#include "ttnn/operations/examples/examples_pybind.hpp"
#include "ttnn/operations/experimental/experimental_pybind.hpp"
#include "ttnn/operations/kv_cache/kv_cache_pybind.hpp"
#include "ttnn/operations/loss/loss_pybind.hpp"
#include "ttnn/operations/matmul/matmul_pybind.hpp"
#include "ttnn/operations/moreh/moreh_pybind.hpp"
#include "ttnn/operations/normalization/normalization_pybind.hpp"
#include "ttnn/operations/pool/avgpool/avg_pool_pybind.hpp"
#include "ttnn/operations/pool/downsample/downsample_pybind.hpp"
#include "ttnn/operations/pool/maxpool/max_pool2d_pybind.hpp"
#include "ttnn/operations/pool/maxpool/maxpool_pybind.hpp"
#include "ttnn/operations/pool/upsample/upsample_pybind.hpp"
#include "ttnn/operations/reduction/reduction_pybind.hpp"
#include "ttnn/operations/transformer/transformer_pybind.hpp"

namespace py = pybind11;

namespace ttnn {

namespace operations {

void py_module(py::module& module) {
    auto m_core = module.def_submodule("core", "core operations");
    core::py_module_types(m_core);
    core::py_module(m_core);

    auto m_examples = module.def_submodule("examples", "examples of operations");
    examples::py_module(m_examples);

    auto m_unary = module.def_submodule("unary", "unary operations");
    unary::py_module(m_unary);

    auto m_binary = module.def_submodule("binary", "binary operations");
    binary::py_module(m_binary);

    auto m_binary_backward = module.def_submodule("binary_backward", "binary_backward operations");
    binary_backward::py_module(m_binary_backward);

    auto m_ternary_backward = module.def_submodule("ternary_backward", "ternary_backward operations");
    ternary_backward::py_module(m_ternary_backward);

    auto m_unary_backward = module.def_submodule("unary_backward", "unary_backward operations");
    unary_backward::py_module(m_unary_backward);

    auto m_ccl = module.def_submodule("ccl", "collective communication operations");
    ccl::py_bind_all_gather(m_ccl);
    ccl::py_bind_line_all_gather(m_ccl);
    ccl::py_bind_reduce_scatter(m_ccl);

    auto m_complex = module.def_submodule("complex", "complex tensor creation");
    complex::py_module(m_complex);

    auto m_complex_unary = module.def_submodule("complex_unary", "complex_unary operations");
    complex_unary::py_module(m_complex_unary);

    auto m_complex_unary_backward = module.def_submodule("complex_unary_backward", "complex_unary_backward operations");
    complex_unary_backward::py_module(m_complex_unary_backward);

    auto m_ternary = module.def_submodule("ternary", "ternary operations");
    ternary::py_module(m_ternary);

    auto m_creation = module.def_submodule("creation", "creation operations");
    creation::py_module(m_creation);

    auto m_embedding = module.def_submodule("embedding", "embedding operations");
    embedding::py_module(m_embedding);

    auto m_embedding_backward = module.def_submodule("embedding_backward", "embedding backward operations");
    embedding_backward::py_bind_embedding_backward(m_embedding_backward);

    auto m_loss = module.def_submodule("loss", "loss operations");
    loss::py_bind_loss_functions(m_loss);

    auto m_matmul = module.def_submodule("matmul", "matmul operations");
    matmul::py_module(m_matmul);

    auto m_data_movement = module.def_submodule("data_movement", "data_movement operations");
    data_movement::py_module(m_data_movement);

    auto m_conv2d = module.def_submodule("conv2d", "conv2d operation");
    conv::conv2d::py_bind_conv2d(m_conv2d);

    auto m_pool = module.def_submodule("pool", "pooling  operations");
    pool::py_module(m_pool);
    avgpool::py_module(m_pool);
    upsample::py_module(m_pool);
    downsample::py_bind_downsample(m_pool);

    auto m_normalization = module.def_submodule("normalization", "normalization operations");
    normalization::py_module(m_normalization);

    auto m_transformer = module.def_submodule("transformer", "transformer operations");
    transformer::py_module(m_transformer);

    auto m_reduction = module.def_submodule("reduction", "reduction operations");
    reduction::py_module(m_reduction);

    auto m_kv_cache = module.def_submodule("kv_cache", "KV cache operations");
    kv_cache::py_bind_kv_cache(m_kv_cache);

    auto m_copy = module.def_submodule("copy", "copy operations");
    copy::py_module(m_copy);

    auto m_experimental = module.def_submodule("experimental", "experimental operations");
    experimental::py_module(m_experimental);

    auto m_moreh = module.def_submodule("moreh", "ttnn moreh");
    moreh::py_module(m_moreh);
}

}  // namespace operations

}  // namespace ttnn
