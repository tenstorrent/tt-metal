// SPDX-FileCopyrightText: © 2024 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <optional>

#include "ttnn/operation.hpp"
#include "ttnn/tensor/tensor.hpp"
#include "cpp/ttnn/operations/ccl/ccl_host_types.hpp"

namespace ttnn::operations::experimental::speculative_execution {

struct SwapTensor {
    const uint32_t num_links;
    const uint32_t num_devices;
    const uint32_t device_index;
    const ttnn::ccl::Topology topology;
    std::optional<GlobalSemaphore> semaphore;
    std::optional<IDevice*> forward_device;
    std::optional<IDevice*> backward_device;

    void validate(const std::vector<Tensor>& input_tensors) const;

    std::vector<TensorSpec> compute_output_specs(const std::vector<Tensor>& input_tensors) const;

    operation::ProgramWithCallbacks create_program(
        const std::vector<Tensor>& input_tensors, std::vector<Tensor>& output_tensors) const;
};

}  // namespace ttnn::operations::experimental::speculative_execution
