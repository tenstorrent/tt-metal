// SPDX-FileCopyrightText: © 2023 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include "tensor/tensor.hpp"
#include "tt_dnn/op_library/run_operation.hpp"
#include "tt_metal/host_api.hpp"

namespace tt {

namespace tt_metal {

struct SplitTiled {
    const uint32_t dim;
    const uint32_t num_chunks;
    const MemoryConfig output_mem_config;

    void boiler_plate_asserts(const Tensor &a) const;
    void shape_asserts(const Tensor &a) const;
    Shape get_single_output_shape(const Shape &input_shape) const;
    void validate(const std::vector<Tensor> &input_tensors) const;
    std::vector<Shape> compute_output_shapes(
        const std::vector<Tensor> &input_tensors) const;
    std::vector<Tensor> create_output_tensors(
        const std::vector<Tensor> &input_tensors) const;
    operation::ProgramWithCallbacks create_program(
        const std::vector<Tensor> &input_tensors,
        std::vector<Tensor> &output_tensors) const;
};

}  // namespace tt_metal

}  // namespace tt
