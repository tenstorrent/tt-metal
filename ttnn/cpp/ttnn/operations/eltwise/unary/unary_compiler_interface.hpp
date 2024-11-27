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
    return op_constraints(device, [&]() {
        const auto input_tensor = create_device_tensor(input, device);

        ttnn::graph::GraphProcessor::begin_graph_capture(ttnn::graph::GraphProcessor::RunMode::NO_DISPATCH);
        auto output_tensor = UnaryFunction(input_tensor, output.tensor_layout().get_memory_config());
        // close inner graph capture, before output buffer is deallocated
        return ttnn::graph::GraphProcessor::end_graph_capture();
    });
};

}  // namespace ttnn::compiler_interface
