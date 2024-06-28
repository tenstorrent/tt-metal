// SPDX-FileCopyrightText: © 2023 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <pybind11/pybind11.h>
#include <pybind11/stl.h>

#include "pybind11/operations/ccl.hpp"
#include "pybind11/operations/conv2d.hpp"
#include "pybind11/operations/core.hpp"
#include "pybind11/operations/creation.hpp"
#include "pybind11/operations/embedding.hpp"
#include "pybind11/operations/kv_cache.hpp"
#include "pybind11/operations/matmul.hpp"
#include "pybind11/operations/maxpool2d.hpp"
#include "pybind11/operations/normalization.hpp"
#include "pybind11/operations/pool.hpp"
#include "pybind11/operations/copy.hpp"
#include "pybind11/operations/ternary.hpp"
#include "pybind11/operations/transformer.hpp"

#include "ttnn/operations/eltwise/binary/binary_pybind.hpp"
#include "ttnn/operations/eltwise/unary/unary_pybind.hpp"
#include "ttnn/operations/reduction/reduction_pybind.hpp"
#include "ttnn/operations/eltwise/binary_backward/binary_backward_pybind.hpp"
#include "ttnn/operations/data_movement/data_movement_pybind.hpp"


namespace py = pybind11;

namespace ttnn {

namespace operations {

void py_module(py::module& module) {
    auto m_unary = module.def_submodule("unary", "unary operations");
    unary::py_module(m_unary);

    auto m_binary = module.def_submodule("binary", "binary operations");
    binary::py_module(m_binary);

    auto m_binary_backward = module.def_submodule("binary_backward", "binary_backward operations");
    binary_backward::py_module(m_binary_backward);

    auto m_ternary = module.def_submodule("ternary", "ternary operations");
    ternary::py_module(m_ternary);

    auto m_core = module.def_submodule("core", "core operations");
    core::py_module(m_core);

    auto m_creation = module.def_submodule("creation", "creation operations");
    creation::py_module(m_creation);

    auto m_embedding = module.def_submodule("embedding", "embedding operations");
    embedding::py_module(m_embedding);

    auto m_matmul = module.def_submodule("matmul", "matmul operations");
    matmul::py_module(m_matmul);

    auto m_data_movement = module.def_submodule("data_movement", "data_movement operations");
    data_movement::py_module(m_data_movement);

    auto m_conv2d = module.def_submodule("conv2d", "conv2d operation");
    conv2d::py_module(m_conv2d);

    auto m_maxpool2d = module.def_submodule("maxpool2d", "maxpool 2d operation");
    maxpool2d::py_module(m_maxpool2d);

    auto m_transformer = module.def_submodule("transformer", "transformer operations");
    transformer::py_module(m_transformer);

    auto m_normalization = module.def_submodule("normalization", "normalization operations");
    normalization::py_module(m_normalization);

    auto m_reduction = module.def_submodule("reduction", "reduction operations");
    reduction::py_module(m_reduction);

    auto m_ccl = module.def_submodule("ccl", "collective communication operations");
    ccl::py_module(m_ccl);

    auto m_kv_cache = module.def_submodule("kv_cache", "KV cache operations");
    kv_cache::py_module(m_kv_cache);

    auto m_pool = module.def_submodule("pool", "pool operations");
    pool::py_module(m_pool);

    auto m_copy = module.def_submodule("copy", "copy operations");
    copy::py_module(m_copy);
}

}  // namespace operations

}  // namespace ttnn
