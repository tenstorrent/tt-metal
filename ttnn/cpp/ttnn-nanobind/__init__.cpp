// SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#include "ttnn-nanobind/operations/__init__.hpp"

#include <cstdint>

#include <nanobind/nanobind.h>

#include "ttnn-nanobind/activation.hpp"
#include "ttnn-nanobind/cluster.hpp"
#include "ttnn-nanobind/core.hpp"
#include "ttnn-nanobind/device.hpp"
#include "ttnn-nanobind/events.hpp"
#include "ttnn-nanobind/fabric.hpp"
#include "ttnn-nanobind/global_circular_buffer.hpp"
#include "ttnn-nanobind/global_semaphore.hpp"
#include "ttnn-nanobind/mesh_socket.hpp"
#include "ttnn-nanobind/operations/copy.hpp"
#include "ttnn-nanobind/operations/core.hpp"
#include "ttnn-nanobind/operations/creation.hpp"
#include "ttnn-nanobind/operations/trace.hpp"
#include "ttnn-nanobind/profiler.hpp"
#include "ttnn-nanobind/program_descriptors.hpp"
#include "ttnn-nanobind/tensor_accessor_args.hpp"
#include "ttnn-nanobind/reports.hpp"
#include "ttnn-nanobind/tensor.hpp"
#include "ttnn-nanobind/types.hpp"

#include "ttnn/core.hpp"
// #include "ttnn/deprecated/tt_lib/csrc/operations/primary/module.hpp"
#include "ttnn/distributed/distributed_nanobind.hpp"
#include "ttnn/graph/graph_nanobind.hpp"
#include "ttnn/operations/bernoulli/bernoulli_nanobind.hpp"
#include "ttnn/operations/ccl/ccl_nanobind.hpp"
#include "ttnn/operations/conv/conv_nanobind.hpp"
#include "ttnn/operations/debug/debug_nanobind.hpp"
#include "ttnn/operations/data_movement/data_movement_nanobind.hpp"
#include "ttnn/operations/eltwise/binary/binary_nanobind.hpp"
#include "ttnn/operations/eltwise/binary_backward/binary_backward_nanobind.hpp"
#include "ttnn/operations/eltwise/complex/complex_nanobind.hpp"
#include "ttnn/operations/eltwise/complex_unary/complex_unary_nanobind.hpp"
#include "ttnn/operations/eltwise/complex_unary_backward/complex_unary_backward_nanobind.hpp"
#include "ttnn/operations/eltwise/quantization/quantization_nanobind.hpp"
#include "ttnn/operations/eltwise/ternary/ternary_nanobind.hpp"
#include "ttnn/operations/eltwise/ternary_backward/ternary_backward_nanobind.hpp"
#include "ttnn/operations/eltwise/unary/unary_nanobind.hpp"
#include "ttnn/operations/eltwise/unary_backward/unary_backward_nanobind.hpp"
#include "ttnn/operations/embedding/embedding_nanobind.hpp"
#include "ttnn/operations/embedding_backward/embedding_backward_nanobind.hpp"
#include "ttnn/operations/examples/examples_nanobind.hpp"
#include "ttnn/operations/experimental/experimental_nanobind.hpp"
#include "ttnn/operations/full/full_nanobind.hpp"
#include "ttnn/operations/full_like/full_like_nanobind.hpp"
#include "ttnn/operations/generic/generic_op_nanobind.hpp"
#include "ttnn/operations/index_fill/index_fill_nanobind.hpp"
#include "ttnn/operations/kv_cache/kv_cache_nanobind.hpp"
#include "ttnn/operations/loss/loss_nanobind.hpp"
#include "ttnn/operations/matmul/matmul_nanobind.hpp"
#include "ttnn/operations/moreh/moreh_nanobind.hpp"
#include "ttnn/operations/normalization/normalization_nanobind.hpp"
#include "ttnn/operations/point_to_point/point_to_point_nanobind.hpp"
#include "ttnn/operations/pool/generic/generic_pools_nanobind.hpp"
#include "ttnn/operations/pool/global_avg_pool/global_avg_pool_nanobind.hpp"
#include "ttnn/operations/pool/upsample/upsample_nanobind.hpp"
#include "ttnn/operations/pool/grid_sample/grid_sample_nanobind.hpp"
#include "ttnn/operations/prefetcher/prefetcher_nanobind.hpp"
#include "ttnn/operations/reduction/reduction_nanobind.hpp"
#include "ttnn/operations/sliding_window/sliding_window_nanobind.hpp"
#include "ttnn/operations/transformer/transformer_nanobind.hpp"
#include "ttnn/operations/uniform/uniform_nanobind.hpp"
#include "ttnn/operations/rand/rand_nanobind.hpp"
#include "ttnn/operations/experimental/test/hang_device/hang_device_operation_nanobind.hpp"

namespace nb = nanobind;

namespace ttnn::operations {

void py_module(nb::module_& mod) {
    nb::set_leak_warnings(true);

    auto m_core = mod.def_submodule("core", "core operations");
    core::py_module_types(m_core);
    core::py_module(m_core);

    auto m_trace = mod.def_submodule("trace", "trace operations");
    trace::py_module_types(m_trace);
    trace::py_module(m_trace);

    auto m_examples = mod.def_submodule("examples", "examples of operations");
    examples::py_module(m_examples);

    //  Eltwise operations: unary, binary, ternary, backward, complex
    auto m_unary = mod.def_submodule("unary", "unary operations");
    unary::py_module(m_unary);

    auto m_binary = mod.def_submodule("binary", "binary operations");
    binary::py_module(m_binary);

    auto m_quantization = mod.def_submodule("quantization", "quantization operations");
    quantization::py_module(m_quantization);

    auto m_ternary = mod.def_submodule("ternary", "ternary operations");
    ternary::py_module(m_ternary);

    auto m_unary_backward = mod.def_submodule("unary_backward", "unary_backward operations");
    unary_backward::py_module(m_unary_backward);

    auto m_binary_backward = mod.def_submodule("binary_backward", "binary_backward operations");
    binary_backward::py_module(m_binary_backward);

    auto m_ternary_backward = mod.def_submodule("ternary_backward", "ternary_backward operations");
    ternary_backward::py_module(m_ternary_backward);

    auto m_complex = mod.def_submodule("complex", "complex tensor creation");
    complex::py_module(m_complex);

    auto m_complex_unary = mod.def_submodule("complex_unary", "complex_unary operations");
    complex_unary::py_module(m_complex_unary);

    auto m_complex_unary_backward = mod.def_submodule("complex_unary_backward", "complex_unary_backward operations");
    complex_unary_backward::py_module(m_complex_unary_backward);

    auto m_ccl = mod.def_submodule("ccl", "collective communication operations");
    ccl::py_module(m_ccl);

    auto m_debug = mod.def_submodule("debug", "debug operations");
    debug::py_module(m_debug);

    auto m_creation = mod.def_submodule("creation", "creation operations");
    creation::py_module(m_creation);

    auto m_embedding = mod.def_submodule("embedding", "embedding operations");
    embedding::py_module(m_embedding);

    auto m_embedding_backward = mod.def_submodule("embedding_backward", "embedding backward operations");
    embedding_backward::bind_embedding_backward(m_embedding_backward);

    auto m_full = mod.def_submodule("full", "full operation");
    full::bind_full_operation(m_full);

    auto m_loss = mod.def_submodule("loss", "loss operations");
    loss::bind_loss_functions(m_loss);

    auto m_matmul = mod.def_submodule("matmul", "matmul operations");
    matmul::py_module(m_matmul);

    auto m_data_movement = mod.def_submodule("data_movement", "data_movement operations");
    data_movement::py_module(m_data_movement);

    auto m_sliding_window = mod.def_submodule("sliding_window", "sliding_window operations");
    sliding_window::bind_sliding_window(m_sliding_window);

    auto m_conv2d = mod.def_submodule("conv", "Convolution operations");
    conv::py_module(m_conv2d);

    auto m_pool = mod.def_submodule("pool", "pooling  operations");
    pool::py_module(m_pool);
    avgpool::py_module(m_pool);
    upsample::py_module(m_pool);
    grid_sample::bind_grid_sample(m_pool);

    auto m_normalization = mod.def_submodule("normalization", "normalization operations");
    normalization::py_module(m_normalization);

    auto m_transformer = mod.def_submodule("transformer", "transformer operations");
    transformer::py_module(m_transformer);

    auto m_prefetcher = mod.def_submodule("prefetcher", "prefetcher operations");
    prefetcher::py_module(m_prefetcher);

    auto m_reduction = mod.def_submodule("reduction", "reduction operations");
    reduction::py_module(m_reduction);

    auto m_kv_cache = mod.def_submodule("kv_cache", "KV cache operations");
    kv_cache::bind_kv_cache(m_kv_cache);

    auto m_copy = mod.def_submodule("copy", "copy operations");
    copy::py_module(m_copy);

    auto m_experimental = mod.def_submodule("experimental", "experimental operations");
    experimental::py_module(m_experimental);

    auto m_moreh = mod.def_submodule("moreh", "moreh operations");
    moreh::bind_moreh_operations(m_moreh);

    auto m_full_like = mod.def_submodule("full_like", "full_like operation");
    full_like::bind_full_like_operation(m_full_like);

    auto m_uniform = mod.def_submodule("uniform", "uniform operations");
    uniform::bind_uniform_operation(m_uniform);

    auto m_index_fill = mod.def_submodule("index_fill", "index_fill operation");
    index_fill::bind_index_fill_operation(m_index_fill);

    auto m_bernoulli = mod.def_submodule("bernoulli", "bernoulli operations");
    bernoulli::bind_bernoulli_operation(m_bernoulli);

    auto m_generic = mod.def_submodule("generic", "ttnn generic operation interface");
    generic::bind_generic_operation(m_generic);

    auto m_rand = mod.def_submodule("rand", "ttnn rand operation");
    rand::bind_rand_operation(m_rand);

    auto m_point_to_point = mod.def_submodule("point_to_point", "point_to_point operations");
    point_to_point::bind_point_to_point(m_point_to_point);
}
}  // namespace ttnn::operations

NB_MODULE(_ttnn, mod) {
    mod.doc() = "Python bindings for TTNN";

    /*
    We have to make sure every class and enum is bound before any function that uses it as an argument or a return type.
    So we split the binding calls into two parts: one for classes and enums, and one for functions.
    Another issue to be aware of is that we have to define each shared submodule only once. Therefore, all def_submodule
    calls have to be put in here.
    */

    // MODULES
    auto m_deprecated = mod.def_submodule("deprecated", "Deprecated tt_lib bindings");
    auto m_tensor = mod.def_submodule("tensor", "ttnn tensor");

    auto m_depr_operations = m_deprecated.def_submodule("operations", "Submodule for experimental operations");
    auto m_primary_ops = m_depr_operations.def_submodule("primary", "Primary operations");

    auto m_graph = mod.def_submodule("graph", "Contains graph capture functions");
    auto m_types = mod.def_submodule("types", "ttnn Types");
    auto m_activation = mod.def_submodule("activation", "ttnn Activation");
    auto m_cluster = mod.def_submodule("cluster", "ttnn cluster");
    auto m_core = mod.def_submodule("core", "core functions");
    auto m_device = mod.def_submodule("device", "ttnn devices");
    auto m_multi_device = mod.def_submodule("multi_device", "ttnn multi_device");
    auto m_events = mod.def_submodule("events", "ttnn events");
    auto m_global_circular_buffer = mod.def_submodule("global_circular_buffer", "ttnn global circular buffer");
    auto m_global_semaphore = mod.def_submodule("global_semaphore", "ttnn global semaphore");
    auto m_mesh_socket = mod.def_submodule("mesh_socket", "ttnn mesh socket");
    auto m_profiler = mod.def_submodule("profiler", "Submodule defining the profiler");
    auto m_reports = mod.def_submodule("reports", "ttnn reports");
    auto m_operations = mod.def_submodule("operations", "ttnn Operations");
    auto m_fabric = mod.def_submodule("fabric", "Fabric instantiation APIs");
    auto m_program_descriptors = mod.def_submodule("program_descriptor", "Program descriptors types");
    auto m_tensor_accessor_args = mod.def_submodule("tensor_accessor_args", "Tensor accessor args types");

    // TYPES
    ttnn::tensor::tensor_mem_config_module_types(m_tensor);
    ttnn::tensor::pytensor_module_types(m_tensor);
    ttnn::graph::py_graph_module_types(m_graph);

    ttnn::types::py_module_types(m_types);
    ttnn::activation::py_module_types(m_activation);
    ttnn::cluster::py_cluster_module_types(m_cluster);
    ttnn::core::py_module_types(m_core);
    ttnn::device::py_device_module_types(m_device);
    ttnn::fabric::bind_fabric_api(m_fabric);
    ttnn::distributed::py_module_types(m_multi_device);
    ttnn::events::py_module_types(m_events);
    ttnn::global_circular_buffer::py_module_types(m_global_circular_buffer);
    ttnn::global_semaphore::py_module_types(m_global_semaphore);
    ttnn::mesh_socket::py_module_types(m_mesh_socket);
    ttnn::reports::py_module_types(m_reports);
    ttnn::program_descriptors::py_module_types(m_program_descriptors);
    ttnn::tensor_accessor_args::py_module_types(m_tensor_accessor_args);

    // FUNCTIONS / OPERATIONS
    ttnn::tensor::tensor_mem_config_module(m_tensor);
    ttnn::tensor::pytensor_module(m_tensor);
    ttnn::core::py_module(m_core);
    ttnn::graph::py_graph_module(m_graph);

#if defined(TRACY_ENABLE)
    // https://nanobind.readthedocs.io/en/latest/utilities.html
    // breadcrumbs
    // https://github.com/wjakob/nanobind/discussions/302
    // https://github.com/wjakob/nanobind/discussions/671
    // https://nanobind.readthedocs.io/en/latest/api_core.html#parameterized-wrapper-classes
    // https://nanobind.readthedocs.io/en/latest/api_core.html#_CPPv4N8nanobind3sigE
    nb::callable tracy_decorator = nb::module_::import_("tracy.ttnn_profiler_wrapper").attr("callable_decorator");

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
    ttnn::tensor_accessor_args::py_module(m_tensor_accessor_args);

    // ttnn operations have to come before the deprecated ones,
    // because ttnn defines additional type bindings.
    // TODO: pull them out of the ttnn::operations::py_module.
    ttnn::operations::py_module(m_operations);
    // tt::operations::primary::py_module(m_primary_ops);

    mod.attr("CONFIG") = &ttnn::CONFIG;
    mod.def(
        "get_python_operation_id",
        []() -> std::uint64_t { return ttnn::CoreIDs::instance().get_python_operation_id(); },
        "Get operation id");
    mod.def(
        "set_python_operation_id",
        [](std::uint64_t id) { ttnn::CoreIDs::instance().set_python_operation_id(id); },
        "Set operation id");
    mod.def(
        "fetch_and_increment_python_operation_id",
        []() -> std::uint64_t { return ttnn::CoreIDs::instance().fetch_and_increment_python_operation_id(); },
        "Increment tensor id and return the previously held id");

    mod.def("get_tensor_id", &tt::tt_metal::Tensor::get_tensor_id_counter, "Get the current tensor ID counter value");
    mod.def(
        "set_tensor_id",
        &tt::tt_metal::Tensor::set_tensor_id_counter,
        nb::arg("id"),
        "Set the tensor ID counter to a specific value");
    mod.def(
        "fetch_and_increment_tensor_id",
        &tt::tt_metal::Tensor::next_tensor_id,
        "Atomically fetch and increment the tensor ID counter");

    mod.def(
        "get_device_operation_id",
        []() -> std::uint64_t { return ttnn::CoreIDs::instance().get_device_operation_id(); },
        "Get device operation id");
    mod.def(
        "set_device_operation_id",
        [](std::uint64_t id) { ttnn::CoreIDs::instance().set_device_operation_id(id); },
        "Set device operation id");
    mod.def(
        "fetch_and_increment_device_operation_id",
        []() -> std::uint64_t { return ttnn::CoreIDs::instance().fetch_and_increment_device_operation_id(); },
        "Increment device operation id and return the previously held id");
}
