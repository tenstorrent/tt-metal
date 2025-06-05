// SPDX-FileCopyrightText: © 2025 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include "ttnn-pybind/operations/__init__.hpp"

#include <cstdint>

#include <pybind11/pybind11.h>
#include <pybind11/stl.h>

#include "ttnn-pybind/activation.hpp"
#include "ttnn-pybind/cluster.hpp"
#include "ttnn-pybind/core.hpp"
#include "ttnn-pybind/device.hpp"
#include "ttnn-pybind/events.hpp"
#include "ttnn-pybind/fabric.hpp"
#include "ttnn-pybind/global_circular_buffer.hpp"
#include "ttnn-pybind/global_semaphore.hpp"
#include "ttnn-pybind/mesh_socket.hpp"
#include "ttnn-pybind/operations/copy.hpp"
#include "ttnn-pybind/operations/core.hpp"
#include "ttnn-pybind/operations/creation.hpp"
#include "ttnn-pybind/operations/trace.hpp"
#include "ttnn-pybind/profiler.hpp"
#include "ttnn-pybind/program_descriptors.hpp"
#include "ttnn-pybind/reports.hpp"
#include "ttnn-pybind/tensor.hpp"
#include "ttnn-pybind/types.hpp"

#include "ttnn/core.hpp"
#include "ttnn/deprecated/tt_lib/csrc/operations/primary/module.hpp"
#include "ttnn/distributed/distributed_pybind.hpp"
#include "ttnn/graph/graph_pybind.hpp"
#include "ttnn/operations/bernoulli/bernoulli_pybind.hpp"
#include "ttnn/operations/ccl/ccl_pybind.hpp"
#include "ttnn/operations/conv/conv_pybind.hpp"
#include "ttnn/operations/data_movement/data_movement_pybind.hpp"
#include "ttnn/operations/eltwise/binary/binary_pybind.hpp"
#include "ttnn/operations/eltwise/binary_backward/binary_backward_pybind.hpp"
#include "ttnn/operations/eltwise/complex/complex_pybind.hpp"
#include "ttnn/operations/eltwise/complex_unary/complex_unary_pybind.hpp"
#include "ttnn/operations/eltwise/complex_unary_backward/complex_unary_backward_pybind.hpp"
#include "ttnn/operations/eltwise/quantization/quantization_pybind.hpp"
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
#include "ttnn/operations/generic/generic_op_pybind.hpp"
#include "ttnn/operations/index_fill/index_fill_pybind.hpp"
#include "ttnn/operations/kv_cache/kv_cache_pybind.hpp"
#include "ttnn/operations/loss/loss_pybind.hpp"
#include "ttnn/operations/matmul/matmul_pybind.hpp"
#include "ttnn/operations/moreh/moreh_pybind.hpp"
#include "ttnn/operations/normalization/normalization_pybind.hpp"
#include "ttnn/operations/pool/generic/generic_pools_pybind.hpp"
#include "ttnn/operations/pool/global_avg_pool/global_avg_pool_pybind.hpp"
#include "ttnn/operations/pool/upsample/upsample_pybind.hpp"
#include "ttnn/operations/prefetcher/prefetcher_pybind.hpp"
#include "ttnn/operations/reduction/reduction_pybind.hpp"
#include "ttnn/operations/sliding_window/sliding_window_pybind.hpp"
#include "ttnn/operations/transformer/transformer_pybind.hpp"
#include "ttnn/operations/uniform/uniform_pybind.hpp"
#include "ttnn/operations/rand/rand_pybind.hpp"
#include "ttnn/operations/test/test_hang_operation_pybind.hpp"

namespace ttnn::operations {

void py_module(py::module& module) {
    auto m_core = module.def_submodule("core", "core operations");
    core::py_module_types(m_core);
    core::py_module(m_core);

    auto m_trace = module.def_submodule("trace", "trace operations");
    trace::py_module_types(m_trace);
    trace::py_module(m_trace);

    auto m_examples = module.def_submodule("examples", "examples of operations");
    examples::py_module(m_examples);

    //  Eltwise operations: unary, binary, ternary, backward, complex
    auto m_unary = module.def_submodule("unary", "unary operations");
    unary::py_module(m_unary);

    auto m_binary = module.def_submodule("binary", "binary operations");
    binary::py_module(m_binary);

    auto m_quantization = module.def_submodule("quantization", "quantization operations");
    quantization::py_module(m_quantization);

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

    auto m_normalization = module.def_submodule("normalization", "normalization operations");
    normalization::py_module(m_normalization);

    auto m_transformer = module.def_submodule("transformer", "transformer operations");
    transformer::py_module(m_transformer);

    auto m_prefetcher = module.def_submodule("prefetcher", "prefetcher operations");
    prefetcher::py_module(m_prefetcher);

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

    auto m_generic = module.def_submodule("generic", "ttnn generic operation interface");
    generic::bind_generic_operation(m_generic);

    auto m_rand = module.def_submodule("rand", "ttnn rand operation");
    rand::bind_rand_operation(m_rand);
}
}  // namespace ttnn::operations

PYBIND11_MODULE(_ttnn, module) {
    module.doc() = "Python bindings for TTNN";

    /*
    We have to make sure every class and enum is bound before any function that uses it as an argument or a return type.
    So we split the binding calls into two parts: one for classes and enums, and one for functions.
    Another issue to be aware of is that we have to define each shared submodule only once. Therefore, all def_submodule
    calls have to be put in here.
    */

    // MODULES
    auto m_deprecated = module.def_submodule("deprecated", "Deprecated tt_lib bindings");
    auto m_tensor = module.def_submodule("tensor", "ttnn tensor");

    auto m_depr_operations = m_deprecated.def_submodule("operations", "Submodule for experimental operations");
    auto m_primary_ops = m_depr_operations.def_submodule("primary", "Primary operations");

    auto m_graph = module.def_submodule("graph", "Contains graph capture functions");
    auto m_types = module.def_submodule("types", "ttnn Types");
    auto m_activation = module.def_submodule("activation", "ttnn Activation");
    auto m_cluster = module.def_submodule("cluster", "ttnn cluster");
    auto m_core = module.def_submodule("core", "core functions");
    auto m_device = module.def_submodule("device", "ttnn devices");
    auto m_multi_device = module.def_submodule("multi_device", "ttnn multi_device");
    auto m_events = module.def_submodule("events", "ttnn events");
    auto m_global_circular_buffer = module.def_submodule("global_circular_buffer", "ttnn global circular buffer");
    auto m_global_semaphore = module.def_submodule("global_semaphore", "ttnn global semaphore");
    auto m_mesh_socket = module.def_submodule("mesh_socket", "ttnn mesh socket");
    auto m_profiler = module.def_submodule("profiler", "Submodule defining the profiler");
    auto m_reports = module.def_submodule("reports", "ttnn reports");
    auto m_operations = module.def_submodule("operations", "ttnn Operations");
    auto m_fabric = module.def_submodule("fabric", "Fabric instantiation APIs");
    auto m_program_descriptors = module.def_submodule("program_descriptor", "Program descriptors types");

    // TYPES
    ttnn::tensor::tensor_mem_config_module_types(m_tensor);
    ttnn::tensor::pytensor_module_types(m_tensor);
    ttnn::graph::py_graph_module_types(m_graph);

    ttnn::types::py_module_types(m_types);
    ttnn::activation::py_module_types(m_activation);
    ttnn::core::py_module_types(m_core);
    ttnn::device::py_device_module_types(m_device);
    ttnn::fabric::py_bind_fabric_api(m_fabric);
    ttnn::distributed::py_module_types(m_multi_device);
    ttnn::events::py_module_types(m_events);
    ttnn::global_circular_buffer::py_module_types(m_global_circular_buffer);
    ttnn::global_semaphore::py_module_types(m_global_semaphore);
    ttnn::mesh_socket::py_module_types(m_mesh_socket);
    ttnn::reports::py_module_types(m_reports);
    ttnn::program_descriptors::py_module_types(m_program_descriptors);

    // FUNCTIONS / OPERATIONS
    ttnn::tensor::tensor_mem_config_module(m_tensor);
    ttnn::tensor::pytensor_module(m_tensor);
    ttnn::core::py_module(m_core);
    ttnn::graph::py_graph_module(m_graph);

#if defined(TRACY_ENABLE)
    py::function tracy_decorator = py::module::import("tracy.ttnn_profiler_wrapper").attr("callable_decorator");

    tracy_decorator(m_device);
    tracy_decorator(m_tensor);
    tracy_decorator(m_depr_operations);
#endif

    ttnn::types::py_module(m_types);
    ttnn::activation::py_module(m_activation);
    ttnn::cluster::py_cluster_module(m_cluster);
    ttnn::device::py_device_module(m_device);
    ttnn::distributed::py_module(m_multi_device);
    ttnn::events::py_module(m_events);
    ttnn::global_circular_buffer::py_module(m_global_circular_buffer);
    ttnn::global_semaphore::py_module(m_global_semaphore);
    ttnn::mesh_socket::py_module(m_mesh_socket);
    ttnn::profiler::py_module(m_profiler);
    ttnn::reports::py_module(m_reports);

    // ttnn operations have to come before the deprecated ones,
    // because ttnn defines additional type bindings.
    // TODO: pull them out of the ttnn::operations::py_module.
    ttnn::operations::py_module(m_operations);
    tt::operations::primary::py_module(m_primary_ops);

    module.attr("CONFIG") = &ttnn::CONFIG;
    module.def(
        "get_python_operation_id",
        []() -> std::uint64_t { return ttnn::CoreIDs::instance().get_python_operation_id(); },
        "Get operation id");
    module.def(
        "set_python_operation_id",
        [](std::uint64_t id) { ttnn::CoreIDs::instance().set_python_operation_id(id); },
        "Set operation id");
    module.def(
        "fetch_and_increment_python_operation_id",
        []() -> std::uint64_t { return ttnn::CoreIDs::instance().fetch_and_increment_python_operation_id(); },
        "Increment tensor id and return the previously held id");

    module.def(
        "get_tensor_id", []() -> std::uint64_t { return ttnn::CoreIDs::instance().get_tensor_id(); }, "Get tensor id");
    module.def("set_tensor_id", [](std::uint64_t id) { ttnn::CoreIDs::instance().set_tensor_id(id); }, "Set tensor id");
    module.def(
        "fetch_and_increment_tensor_id",
        []() -> std::uint64_t { return ttnn::CoreIDs::instance().fetch_and_increment_tensor_id(); },
        "Increment tensor id and return the previously held id");

    module.def(
        "get_device_operation_id",
        []() -> std::uint64_t { return ttnn::CoreIDs::instance().get_device_operation_id(); },
        "Get device operation id");
    module.def(
        "set_device_operation_id",
        [](std::uint64_t id) { ttnn::CoreIDs::instance().set_device_operation_id(id); },
        "Set device operation id");
    module.def(
        "fetch_and_increment_device_operation_id",
        []() -> std::uint64_t { return ttnn::CoreIDs::instance().fetch_and_increment_device_operation_id(); },
        "Increment device operation id and return the previously held id");
}
