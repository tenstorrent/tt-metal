// SPDX-FileCopyrightText: © 2025 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include "ttnn/operations/ccl/common/host/command_backend_runtime_args_overrider.hpp"

#include <tt-metalium/runtime_args_data.hpp>
#include <tt-metalium/assert.hpp>

namespace ttnn::ccl {
size_t tensor_address_runtime_args_overrider::add_tensor() {
    size_t tensor_idx = tensor_address_runtime_arg_indices.size();
    tensor_address_runtime_arg_indices.push_back(std::vector<size_t>());
    return tensor_idx;
}

std::vector<size_t> tensor_address_runtime_args_overrider::get_runtime_arg_indices(size_t tensor_idx) const {
    TT_FATAL(
        tensor_idx < tensor_address_runtime_arg_indices.size(),
        "Internal Error. Invalid tensor index when getting runtime arg indices in "
        "tensor_address_runtime_args_overrider");
    return tensor_address_runtime_arg_indices[tensor_idx];
}

void tensor_address_runtime_args_overrider::add_runtime_arg_index(size_t tensor_idx, size_t runtime_arg_index) {
    TT_FATAL(
        tensor_idx < tensor_address_runtime_arg_indices.size(),
        "Invalid tensor index when adding runtime arg index. tensor_idx: {}, highest_available: {}",
        tensor_idx,
        tensor_address_runtime_arg_indices.size());
    tensor_address_runtime_arg_indices[tensor_idx].push_back(runtime_arg_index);
}

void tensor_address_runtime_args_overrider::override_runtime_args(
    size_t tensor_idx, uint32_t new_value, tt::tt_metal::RuntimeArgsData& runtime_args_to_modify) const {
    if (tensor_idx >= tensor_address_runtime_arg_indices.size()) {
        log_trace(tt::LogOp, "Tensor index {} is out of bounds. Skipping override", tensor_idx);
    }
    TT_FATAL(
        tensor_idx < tensor_address_runtime_arg_indices.size(), "Invalid tensor index when overriding runtime args");

    const auto& indices = tensor_address_runtime_arg_indices[tensor_idx];
    TT_FATAL(!indices.empty(), "No runtime arg indices associated with tensor");

    log_trace(tt::LogOp, "Overriding {} runtime args for tensor {} to value {}", indices.size(), tensor_idx, new_value);
    for (size_t idx : indices) {
        TT_FATAL(idx < runtime_args_to_modify.size(), "Runtime arg index out of bounds when overriding args");
        log_trace(tt::LogOp, "\t- {}", idx);
        runtime_args_to_modify[idx] = new_value;
    }
}

size_t tensor_address_runtime_args_overrider::size() const { return tensor_address_runtime_arg_indices.size(); }
}  // namespace ttnn::ccl
