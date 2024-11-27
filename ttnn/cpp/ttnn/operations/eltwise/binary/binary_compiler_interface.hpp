// SPDX-FileCopyrightText: © 2023 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <exception>
#include <functional>

#include "third_party/json/json.hpp"
#include "tt_metal/common/logger.hpp"
#include "ttnn/compiler_interface/compiler_interface.hpp"
#include "ttnn/graph/graph_processor.hpp"
#include "ttnn/graph/graph_trace_utils.hpp"

namespace ttnn::compiler_interface {

template <auto BinaryFunction>
QueryResponse binary_op_constraints(
    Device* device, const TensorSpec& input_a, const TensorSpec& input_b, const TensorSpec& output) {
    return op_constraints(device, [&]() {
        const auto input_tensor_a = create_device_tensor(input_a, device);
        const auto input_tensor_b = create_device_tensor(input_b, device);

        ttnn::graph::GraphProcessor::begin_graph_capture(ttnn::graph::GraphProcessor::RunMode::NO_DISPATCH);
        auto output_tensor = BinaryFunction(
            input_tensor_a, input_tensor_b, output.data_type(), output.tensor_layout().get_memory_config());
        // close inner graph capture, before output buffer is deallocated
        return ttnn::graph::GraphProcessor::end_graph_capture();
    });
}

}  // namespace ttnn::compiler_interface
