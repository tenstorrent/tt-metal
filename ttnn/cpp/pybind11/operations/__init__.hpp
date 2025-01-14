// SPDX-FileCopyrightText: © 2023 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <pybind11/pybind11.h>
#include <pybind11/stl.h>

#include "pybind11/operations/copy.hpp"
#include "pybind11/operations/core.hpp"
#include "pybind11/operations/creation.hpp"
#include "ttnn/operations/bernoulli/bernoulli_pybind.hpp"
#include "cpp/ttnn/operations/ccl/ccl_pybind.hpp"
#include "ttnn/operations/conv/conv_pybind.hpp"
#include "ttnn/operations/data_movement/data_movement_pybind.hpp"
#include "ttnn/operations/eltwise/binary/binary_pybind.hpp"
#include "ttnn/operations/eltwise/binary_ng/binary_ng_pybind.hpp"
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
#include "ttnn/operations/full/full_pybind.hpp"
#include "ttnn/operations/full_like/full_like_pybind.hpp"
#include "ttnn/operations/index_fill/index_fill_pybind.hpp"
#include "ttnn/operations/kv_cache/kv_cache_pybind.hpp"
#include "ttnn/operations/loss/loss_pybind.hpp"
#include "ttnn/operations/matmul/matmul_pybind.hpp"
#include "ttnn/operations/moreh/moreh_pybind.hpp"
#include "ttnn/operations/normalization/normalization_pybind.hpp"
#include "ttnn/operations/pool/downsample/downsample_pybind.hpp"
#include "ttnn/operations/pool/generic/generic_pools_pybind.hpp"
#include "ttnn/operations/pool/global_avg_pool/global_avg_pool_pybind.hpp"
#include "ttnn/operations/pool/upsample/upsample_pybind.hpp"
#include "ttnn/operations/reduction/reduction_pybind.hpp"
#include "ttnn/operations/sliding_window/sliding_window_pybind.hpp"
#include "ttnn/operations/transformer/transformer_pybind.hpp"
#include "ttnn/operations/prefetcher/prefetcher_pybind.hpp"
#include "ttnn/operations/copy_tensor/copy_tensor_pybind.hpp"
#include "ttnn/operations/uniform/uniform_pybind.hpp"

namespace py = pybind11;

namespace ttnn {

namespace operations {

void py_module(py::module& module) {
    auto m_core = module.def_submodule("core", "core operations");
    core::py_module_types(m_core);
    core::py_module(m_core);

    auto m_examples = module.def_submodule("examples", "examples of operations");
    examples::py_module(m_examples);

    //  Eltwise operations: unary, binary, ternary, backward, complex
    auto m_unary = module.def_submodule("unary", "unary operations");
    unary::py_module(m_unary);

    auto m_binary = module.def_submodule("binary", "binary operations");
    binary::py_module(m_binary);

    auto m_binary_ng = module.def_submodule("binary_ng", "binary_ng operations");
    binary_ng::py_module(m_binary_ng);

    auto m_ternary = module.def_submodule("ternary", "ternary operations");
    ternary::py_module(m_ternary);

    auto m_unary_backward = module.def_submodule("unary_backward", "unary_backward operations");
    unary_backward::py_module(m_unary_backward);

    auto m_binary_backward = module.def_submodule("binary_backward", "binary_backward operations");
    binary_backward::py_module(m_binary_backward);

    auto m_ternary_backward = module.def_submodule("ternary_backward", "ternary_backward operations");
    ternary_backward::py_module(m_ternary_backward);

    auto m_complex = module.def_submodule("complex", "complex tensor creation");
    complex::py_module(m_complex);

    auto m_complex_unary = module.def_submodule("complex_unary", "complex_unary operations");
    complex_unary::py_module(m_complex_unary);

    auto m_complex_unary_backward = module.def_submodule("complex_unary_backward", "complex_unary_backward operations");
    complex_unary_backward::py_module(m_complex_unary_backward);

    auto m_ccl = module.def_submodule("ccl", "collective communication operations");
    ccl::py_module(m_ccl);

    auto m_creation = module.def_submodule("creation", "creation operations");
    creation::py_module(m_creation);

    auto m_embedding = module.def_submodule("embedding", "embedding operations");
    embedding::py_module(m_embedding);

    auto m_embedding_backward = module.def_submodule("embedding_backward", "embedding backward operations");
    embedding_backward::py_bind_embedding_backward(m_embedding_backward);

    auto m_full = module.def_submodule("full", "full operation");
    full::bind_full_operation(m_full);

    auto m_loss = module.def_submodule("loss", "loss operations");
    loss::py_bind_loss_functions(m_loss);

    auto m_matmul = module.def_submodule("matmul", "matmul operations");
    matmul::py_module(m_matmul);

    auto m_data_movement = module.def_submodule("data_movement", "data_movement operations");
    data_movement::py_module(m_data_movement);

    auto m_sliding_window = module.def_submodule("sliding_window", "sliding_window operations");
    sliding_window::py_bind_sliding_window(m_sliding_window);

    auto m_conv2d = module.def_submodule("conv", "Convolution operations");
    conv::py_module(m_conv2d);

    auto m_pool = module.def_submodule("pool", "pooling  operations");
    pool::py_module(m_pool);
    avgpool::py_module(m_pool);
    upsample::py_module(m_pool);
    downsample::py_bind_downsample(m_pool);

    auto m_normalization = module.def_submodule("normalization", "normalization operations");
    normalization::py_module(m_normalization);

    auto m_transformer = module.def_submodule("transformer", "transformer operations");
    transformer::py_module(m_transformer);

    auto m_prefetcher = module.def_submodule("prefetcher", "prefetcher operations");
    prefetcher::py_module(m_prefetcher);

    auto m_copy_tensor = module.def_submodule("copy_tensor", "copy_tensor operations");
    copy_tensor::py_module(m_copy_tensor);

    auto m_reduction = module.def_submodule("reduction", "reduction operations");
    reduction::py_module(m_reduction);

    auto m_kv_cache = module.def_submodule("kv_cache", "KV cache operations");
    kv_cache::py_bind_kv_cache(m_kv_cache);

    auto m_copy = module.def_submodule("copy", "copy operations");
    copy::py_module(m_copy);

    auto m_experimental = module.def_submodule("experimental", "experimental operations");
    experimental::py_module(m_experimental);

    auto m_moreh = module.def_submodule("moreh", "moreh operations");
    moreh::bind_moreh_operations(m_moreh);

    auto m_full_like = module.def_submodule("full_like", "full_like operation");
    full_like::bind_full_like_operation(m_full_like);

    auto m_uniform = module.def_submodule("uniform", "uniform operations");
    uniform::bind_uniform_operation(m_uniform);

    auto m_index_fill = module.def_submodule("index_fill", "index_fill operation");
    index_fill::bind_index_fill_operation(m_index_fill);

    auto m_bernoulli = module.def_submodule("bernoulli", "bernoulli operations");
    bernoulli::bind_bernoulli_operation(m_bernoulli);
}
}  // namespace operations

}  // namespace ttnn
