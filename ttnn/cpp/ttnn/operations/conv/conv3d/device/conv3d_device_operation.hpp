#pragma once

#include <array>
#include <cstdint>
#include <optional>
#include <vector>

// #include <tt-metalium/operation.hpp>
#include "ttnn/tensor/tensor.hpp"
// #include "ttnn/decorators.hpp"
#include "ttnn/operation.hpp"

namespace ttnn::operations::conv::conv3d {

struct Conv3dOp {
    uint32_t output_channels;
    std::array<uint32_t, 3> kernel_size;
    std::array<uint32_t, 3> stride;
    std::array<uint32_t, 3> padding;
    std::string padding_mode;
    uint32_t groups;
    MemoryConfig output_mem_config;

    void validate(const std::vector<Tensor>& input_tensors) const;

    std::vector<TensorSpec> compute_output_specs(const std::vector<Tensor>& input_tensors) const;

    operation::ProgramWithCallbacks create_program(
        const std::vector<Tensor>& input_tensors, std::vector<Tensor>& output_tensors) const;
};

}  // namespace ttnn::operations::conv::conv3d
