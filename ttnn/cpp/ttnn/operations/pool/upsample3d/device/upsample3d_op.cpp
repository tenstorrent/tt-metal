// SPDX-FileCopyrightText: © 2024 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include "upsample3d_op.hpp"
#include "upsample3d_program_factory.hpp"

#include "tt-metalium/buffer_types.hpp"
#include "ttnn/tensor/enum_types.hpp"
#include "ttnn/tensor/types.hpp"
#include <tt-metalium/util.hpp>
#include <tt-metalium/work_split.hpp>
#include "ttnn/run_operation.hpp"

namespace ttnn::operations::upsample3d {
using namespace tt;
using namespace tt::tt_metal;

void UpSample3D::validate(const std::vector<Tensor>& input_tensors) const {
    const auto& input_tensor_a = input_tensors.at(0);
    TT_FATAL(input_tensor_a.storage_type() == StorageType::DEVICE, "Operands to upsample3d need to be on device!");
    TT_FATAL(input_tensor_a.buffer() != nullptr, "Operands to upsample3d need to be allocated in buffers on device!");

    // Validate input tensor is 5D
    const auto& input_shape = input_tensor_a.logical_shape();
    TT_FATAL(input_shape.rank() == 5, "Input tensor must be 5D (N, D, H, W, C), got {}D", input_shape.rank());

    // Validate scale factors are positive
    TT_FATAL(scale_factor_d_ > 0, "Scale factor for depth must be positive, got {}", scale_factor_d_);
    TT_FATAL(scale_factor_h_ > 0, "Scale factor for height must be positive, got {}", scale_factor_h_);
    TT_FATAL(scale_factor_w_ > 0, "Scale factor for width must be positive, got {}", scale_factor_w_);

    // Validate mode
    TT_FATAL(mode_ == "nearest", "Only 'nearest' mode is supported for upsample3d, got '{}'", mode_);

    // For now, require ROW_MAJOR layout and interleaved memory
    TT_FATAL(input_tensor_a.layout() == Layout::ROW_MAJOR, "Only ROW_MAJOR layout is supported for upsample3d input");
    TT_FATAL(
        input_tensor_a.memory_config().memory_layout() == TensorMemoryLayout::INTERLEAVED,
        "Only interleaved memory layout is supported for upsample3d input");

    // Validate data type
    TT_FATAL(input_tensor_a.dtype() == DataType::BFLOAT16, "Input tensor data type should be BFLOAT16");
}

std::vector<TensorSpec> UpSample3D::compute_output_specs(const std::vector<Tensor>& input_tensors) const {
    const auto& input_tensor = input_tensors.at(0);
    const auto& input_shape = input_tensor.logical_shape();

    // Compute output shape: [N, D*scale_d, H*scale_h, W*scale_w, C]
    auto output_shape = ttnn::Shape{std::array<uint32_t, 5>{
        input_shape[0],                    // N
        input_shape[1] * scale_factor_d_,  // D * scale_d
        input_shape[2] * scale_factor_h_,  // H * scale_h
        input_shape[3] * scale_factor_w_,  // W * scale_w
        input_shape[4]                     // C
    }};

    return {TensorSpec(
        output_shape, TensorLayout(input_tensor.dtype(), PageConfig(input_tensor.layout()), output_mem_config_))};
}

tt::tt_metal::operation::ProgramWithCallbacks UpSample3D::create_program(
    const std::vector<Tensor>& input_tensors, std::vector<Tensor>& output_tensors) const {
    const auto& input_tensor = input_tensors.at(0);
    const auto& output_tensor = output_tensors.at(0);

    // Extract tensor dimensions and coordinate information
    const auto& input_shape = input_tensor.logical_shape();
    const auto& output_shape = output_tensor.logical_shape();

    const uint32_t N = input_shape[0];
    const uint32_t input_D = input_shape[1];
    const uint32_t input_H = input_shape[2];
    const uint32_t input_W = input_shape[3];
    const uint32_t C = input_shape[4];

    const uint32_t output_D = output_shape[1];
    const uint32_t output_H = output_shape[2];
    const uint32_t output_W = output_shape[3];

    // Coordinate computation and work distribution logic
    // This prepares for kernel implementation in next steps

    // Calculate total work: each output element needs to be computed
    const uint32_t total_output_elements = N * output_D * output_H * output_W * C;

    // For nearest neighbor upsampling, each output coordinate (n,d,h,w,c) maps to
    // input coordinate (n, d/scale_d, h/scale_h, w/scale_w, c)
    // where division is integer division (floor)

    // Calculate how work will be distributed across cores
    // This is preparation for the actual kernel implementation
    const auto compute_grid_size = input_tensor.device()->compute_with_storage_grid_size();
    const uint32_t max_cores = compute_grid_size.x * compute_grid_size.y;

    // Determine work per core - for now just calculate, don't use
    const uint32_t elements_per_core = (total_output_elements + max_cores - 1) / max_cores;

    // Log coordinate computation information for debugging
    // (This would normally be done with proper logging)
    // For now, we'll include this information in comments for the next implementation step

    /*
    Coordinate mapping for nearest neighbor 3D upsampling:

    For each output position (n, out_d, out_h, out_w, c):
    - Input position is (n, out_d / scale_factor_d_, out_h / scale_factor_h_, out_w / scale_factor_w_, c)
    - Where division is integer division (floor)

    Memory layout considerations:
    - Input tensor: [N, D, H, W, C] in row-major order
    - Output tensor: [N, D', H', W', C] in row-major order where D'=D*scale_d, etc.
    - Each core will process a range of output elements

    Work distribution:
    - Total output elements: N * D' * H' * W' * C = {}
    - Available cores: {}
    - Elements per core: {}
    */

    // Run the program with working kernels
    return upsample3d_multi_core_interleaved(
        input_tensor, const_cast<Tensor&>(output_tensor), scale_factor_d_, scale_factor_h_, scale_factor_w_);
}

}  // namespace ttnn::operations::upsample3d
