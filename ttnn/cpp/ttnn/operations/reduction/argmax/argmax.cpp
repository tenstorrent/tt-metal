// SPDX-FileCopyrightText: © 2025 Tenstorrent USA, Inc.
// SPDX-License-Identifier: Apache-2.0

#include "device/argmax_device_operation.hpp"
#include "device/argmax_nc_device_operation.hpp"
#include "device/argmax_utils.hpp"
#include "ttnn/operations/reduction/argmax/argmax.hpp"
#include "ttnn/operations/creation/creation.hpp"
#include "ttnn/tensor/tensor_impl.hpp"
#include "ttnn/operations/data_movement/reshape_view/reshape.hpp"
#include "ttnn/operations/data_movement/copy/copy.hpp"
#include "ttnn/operations/core/to_layout/to_layout_op.hpp"
#include "ttnn/operations/core/compute_kernel/compute_kernel_config.hpp"

#include <utility>

namespace ttnn {

using tt::tt_metal::DataType;
using tt::tt_metal::Layout;

namespace {

bool should_row_major_h_via_tile(
    const Tensor& input, const std::optional<int>& dim, const MemoryConfig& output_memory_config) {
    if (!dim.has_value() || input.layout() != Layout::ROW_MAJOR) {
        return false;
    }
    const int32_t rank = static_cast<int32_t>(input.logical_shape().rank());
    if (rank < 2) {
        return false;
    }
    const int32_t normalized_dim = dim.value() < 0 ? dim.value() + rank : dim.value();
    if (normalized_dim != rank - 2) {
        return false;
    }
    if (input.dtype() != DataType::BFLOAT16 && input.dtype() != DataType::FLOAT32) {
        return false;
    }
    if (input.memory_config().memory_layout() != tt::tt_metal::TensorMemoryLayout::INTERLEAVED) {
        return false;
    }
    if (output_memory_config.memory_layout() != tt::tt_metal::TensorMemoryLayout::INTERLEAVED) {
        return false;
    }
    return true;
}

// Returns true if we should dispatch the reduction to the register-based NC
// path (compute-kernel + DST accumulation). The NC path supports reducing
// along any non-HW dim (i.e., dim < rank - 2) and assumes BFLOAT16 / FLOAT32
// values.
bool should_use_nc_path(const Tensor& input, const std::optional<int>& dim, const MemoryConfig& output_memory_config) {
    if (!dim.has_value()) {
        return false;
    }
    const auto rank = static_cast<int32_t>(input.logical_shape().rank());
    if (rank < 3) {
        // Rank 2 only has H and W; nothing is "non-HW".
        return false;
    }
    const int32_t normalized_dim = dim.value() < 0 ? dim.value() + rank : dim.value();
    if (normalized_dim < 0 || normalized_dim >= rank - 2) {
        return false;
    }
    const auto dtype = input.dtype();
    if (dtype != tt::tt_metal::DataType::BFLOAT16 && dtype != tt::tt_metal::DataType::FLOAT32) {
        return false;
    }
    if (input.memory_config().memory_layout() != tt::tt_metal::TensorMemoryLayout::INTERLEAVED) {
        return false;
    }
    if (output_memory_config.memory_layout() != tt::tt_metal::TensorMemoryLayout::INTERLEAVED) {
        return false;
    }
    return true;
}

// Run the register-based argmax for a non-HW dim. Returns a ROW_MAJOR UINT32
// tensor with the user-visible logical output shape.
Tensor run_argmax_nc(
    const Tensor& input_tensor,
    int dim,
    bool keepdim,
    const MemoryConfig& output_memory_config,
    const std::optional<CoreRangeSet>& sub_core_grids) {
    using tt::tt_metal::Layout;

    // 1) Ensure the input is in TILE layout for the compute kernel.
    Tensor tiled_input = input_tensor;
    if (tiled_input.layout() != Layout::TILE) {
        tiled_input = ttnn::to_layout(tiled_input, Layout::TILE);
    }

    // 2) Run the NC device op. Returns a TILE UINT32 tensor with the reduced
    //    dim's logical size collapsed to 1 (keepdim=True semantics).
    auto compute_kernel_config = init_device_compute_kernel_config(
        tiled_input.device()->arch(), std::nullopt, tt::tt_metal::MathFidelity::HiFi4);
    Tensor tile_out = ttnn::prim::argmax_nc(
        tiled_input,
        /*dim=*/dim,
        /*preallocated_output=*/std::nullopt,
        /*output_mem_config=*/output_memory_config,
        compute_kernel_config,
        sub_core_grids);

    // 3) Convert the TILE UINT32 output back to ROW_MAJOR UINT32.
    Tensor row_major_out = ttnn::to_layout(tile_out, Layout::ROW_MAJOR);

    // 4) Apply keepdim semantics: if keepdim=false, remove the reduced dim.
    if (!keepdim) {
        const auto& logical_shape = row_major_out.logical_shape();
        const int32_t rank = static_cast<int32_t>(logical_shape.rank());
        const int32_t normalized_dim = dim < 0 ? dim + rank : dim;
        ttnn::SmallVector<uint32_t> new_shape;
        new_shape.reserve(rank - 1);
        for (int32_t i = 0; i < rank; ++i) {
            if (i == normalized_dim) {
                continue;
            }
            new_shape.push_back(logical_shape[i]);
        }
        row_major_out = ttnn::reshape(row_major_out, ttnn::Shape(new_shape));
    }

    return row_major_out;
}

}  // namespace

static Tensor zero_volume_argmax(
    const Tensor& input_tensor,
    const std::optional<int>& dim,
    bool keepdim,
    const MemoryConfig& memory_config,
    std::optional<Tensor> optional_output_tensor) {
    auto output_shape = ttnn::Shape(ttnn::prim::get_output_shape(input_tensor, dim, keepdim));
    if (!optional_output_tensor.has_value()) {
        return ttnn::full(
            output_shape,
            0,  // fill_value doesn't matter for zero-volume tensor.
            tt::tt_metal::DataType::UINT32,
            input_tensor.layout(),
            *input_tensor.device(),
            memory_config);
    }

    Tensor& preallocated_tensor = optional_output_tensor.value();
    TT_FATAL(
        preallocated_tensor.logical_shape() == output_shape,
        "Preallocated output tensor has incorrect shape! Got : {}, expected: {}",
        preallocated_tensor.logical_shape(),
        output_shape);

    // Creating result tensor on host and copying to device (there is no direct way to write
    // to a device tensor with a scalar value).
    const TensorSpec& tensor_spec = preallocated_tensor.tensor_spec();
    // Note that allocate_host_buffer() doesn't allow specifying initial value, but that doesn't matter
    // here because the tensor is 0-volume (i.e., it has no elements).
    auto host_buffer = tt::tt_metal::tensor_impl::allocate_host_buffer(tensor_spec);
    Tensor host_tensor(std::move(host_buffer), output_shape, tensor_spec.data_type(), tensor_spec.layout());
    copy_to_device(host_tensor, preallocated_tensor);

    return preallocated_tensor;
}

Tensor argmax(
    const Tensor& input_tensor,
    const std::optional<int>& dim,
    bool keepdim,
    const std::optional<CoreRangeSet>& sub_core_grids,
    const std::optional<MemoryConfig>& memory_config,
    std::optional<Tensor> optional_output_tensor) {
    auto output_memory_config = memory_config.value_or(input_tensor.memory_config());

    TT_FATAL(is_device_tensor(input_tensor), "Input tensor must be on device");
    TT_FATAL(
        !optional_output_tensor.has_value() || is_device_tensor(optional_output_tensor.value()),
        "Preallocated output tensor must be on device");

    const auto& input_shape = input_tensor.logical_shape();
    const auto rank = input_shape.size();
    if (dim.has_value()) {
        if (rank > 0) {
            const int32_t r = static_cast<int32_t>(rank);
            TT_FATAL(
                dim.value() >= -r && dim.value() < r,
                "argmax: Dimension out of range (expected to be in range of [{}, {}], but got {})",
                -r,
                r - 1,
                dim.value());
        } else {
            // Rank 0 (scalar): only the virtual axis 0 / -1 is valid.
            TT_FATAL(
                dim.value() == 0 || dim.value() == -1,
                "argmax: Dimension out of range for scalar tensor (expected 0 or -1, but got {})",
                dim.value());
        }
    }

    if (input_tensor.logical_volume() == 0) [[unlikely]] {
        return zero_volume_argmax(input_tensor, dim, keepdim, output_memory_config, optional_output_tensor);
    }

    if (rank == 0) [[unlikely]] {
        if (!optional_output_tensor.has_value()) {
            return full(
                input_shape,
                /*fill_value=*/0,
                tt::tt_metal::DataType::UINT32,
                input_tensor.layout(),
                *input_tensor.device(),
                output_memory_config);
        }

        Tensor& preallocated_tensor = optional_output_tensor.value();
        TT_FATAL(
            preallocated_tensor.logical_shape() == input_shape,
            "Preallocated output tensor has incorrect shape! Got : {}, expected: {}",
            preallocated_tensor.logical_shape(),
            input_shape);
        // Creating result tensor on host and copying to device (there is no direct way to write
        // to a device tensor with a scalar value).
        const TensorSpec& preallocated_spec = preallocated_tensor.tensor_spec();
        TT_FATAL(
            preallocated_spec.data_type() == DataType::UINT32,
            "Preallocated output tensor must be UINT32 for rank 0 input tensor");
        // Although we only need to store one value, have to account for extra padding
        // in possible tile layout. So host buffer size needs to match device buffer size.
        auto result_vec = std::vector<uint32_t>(
            preallocated_spec.physical_shape().height() * preallocated_spec.physical_shape().width(), 0);
        Tensor host_indices(
            tt::tt_metal::HostBuffer(std::move(result_vec)), input_shape, DataType::UINT32, preallocated_spec.layout());
        copy_to_device(host_indices, preallocated_tensor);

        return preallocated_tensor;
    }

    // Register-based NC path for reductions along any non-HW dimension.
    // Uses DST accumulation (similar to fast_reduce_nc). Supports sub_core_grids.
    if (should_use_nc_path(input_tensor, dim, output_memory_config)) {
        Tensor nc_result = run_argmax_nc(input_tensor, dim.value(), keepdim, output_memory_config, sub_core_grids);
        if (optional_output_tensor.has_value()) {
            // nc_result is already on device; copy_to_device is host → device only.
            ttnn::copy(nc_result, optional_output_tensor.value());
            return optional_output_tensor.value();
        }
        return nc_result;
    }

    if (should_row_major_h_via_tile(input_tensor, dim, output_memory_config)) {
        const Tensor tiled_input = ttnn::to_layout(input_tensor, Layout::TILE);
        return prim::argmax(
            tiled_input,
            DataType::UINT32,
            dim,
            keepdim,
            sub_core_grids,
            output_memory_config,
            std::move(optional_output_tensor));
    }

    return prim::argmax(
        input_tensor,
        tt::tt_metal::DataType::UINT32,
        dim,
        keepdim,
        sub_core_grids,
        output_memory_config,
        std::move(optional_output_tensor));
}

}  // namespace ttnn
