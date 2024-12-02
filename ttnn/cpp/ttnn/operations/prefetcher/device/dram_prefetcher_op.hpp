// SPDX-FileCopyrightText: © 2023 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <optional>

#include "ttnn/run_operation.hpp"
#include "ttnn/tensor/tensor.hpp"
#include "ttnn/operations/core/core.hpp"

namespace ttnn::operations::dram_prefetcher {

operation::ProgramWithCallbacks dram_prefetcher(const std::vector<Tensor>& tensors);

struct DramPrefetcher {
    std::vector<Tensor> tensors;

    void validate(const std::vector<Tensor>& tensors) const;
    std::vector<ttnn::SimpleShape> compute_output_shapes(const std::vector<Tensor>& tensors) const;
    std::vector<Tensor> create_output_tensors(const std::vector<Tensor>& tensors) const;
    operation::ProgramWithCallbacks create_program(
        const std::vector<Tensor>&, std::vector<Tensor>& output_tensors) const;
};

}  // namespace ttnn::operations::dram_prefetcher
