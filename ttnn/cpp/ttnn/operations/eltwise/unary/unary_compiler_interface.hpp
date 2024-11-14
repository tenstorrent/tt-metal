// SPDX-FileCopyrightText: © 2023 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <exception>
#include <functional>

#include "third_party/json/json.hpp"
#include "tt_metal/common/logger.hpp"
#include "ttnn/compiler_interface/compiler_interface.hpp"
#include "ttnn/cpp/ttnn/tensor/tensor.hpp"
#include "ttnn/graph/graph_processor.hpp"
#include "ttnn/graph/graph_trace_utils.hpp"

namespace ttnn::compiler_interface {

template <auto UnaryFunction>
QueryResponse unary_op_constraints(Device* device, const TensorSpec& input, const TensorSpec& output) {
    // get_op_trace is a lambda that prepares input and output tensors, capturing graph trace of the op without
    // inputs allocation.
    auto get_op_trace = [](Device* device, const TensorSpec& input, const TensorSpec& output) {
        nlohmann::json op_trace;

        // outer graph capture is used to avoid capturing and dispatching of dummy input tensor(s) creation
        {
            ttnn::graph::GraphProcessor::begin_graph_capture(ttnn::graph::GraphProcessor::RunMode::NO_DISPATCH);
            const auto input_tensor = create_device_tensor(input, device);

            // output tensor is created in the inner graph capture to capture its allocation
            {
                ttnn::graph::GraphProcessor::begin_graph_capture(ttnn::graph::GraphProcessor::RunMode::NO_DISPATCH);
                auto output_tensor = UnaryFunction(input_tensor, output.tensor_layout().get_memory_config());
                // close inner graph capture, before output buffer is deallocated
                op_trace = ttnn::graph::GraphProcessor::end_graph_capture();
            }
            // close outer graph capture
            ttnn::graph::GraphProcessor::end_graph_capture();
        }

        // TODO(mbezulj) remove this debug print
        // std::cout << graph::to_graphviz(op_trace) << std::endl;

        return op_trace;
    };

    try {
        auto op_trace = get_op_trace(device, input, output);
        auto interleaved_storage_cores =
            device->compute_with_storage_grid_size().x * device->compute_with_storage_grid_size().y;
        return extract_data_from_trace(op_trace, interleaved_storage_cores);
    } catch (const std::exception& e) {
        tt::log_debug(tt::LogOp, "compiler_interface - error: {}", e.what());
        return QueryResponse{ExecutionStatus::Error, {0, 0, 0}, e.what()};
    }
}

}  // namespace ttnn::compiler_interface
