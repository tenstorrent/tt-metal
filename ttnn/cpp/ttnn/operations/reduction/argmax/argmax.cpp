// SPDX-FileCopyrightText: © 2025 Tenstorrent USA, Inc.
// SPDX-License-Identifier: Apache-2.0

#include "device/argmax_device_operation.hpp"
#include "device/argmax_utils.hpp"
#include "ttnn/operations/reduction/argmax/argmax.hpp"
#include "ttnn/operations/creation/creation.hpp"
#include "ttnn/operations/data_movement/permute/permute.hpp"
#include "ttnn/tensor/tensor_impl.hpp"

#include <utility>

namespace ttnn {

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
    bool use_multicore,
    const std::optional<MemoryConfig>& memory_config,
    std::optional<Tensor> optional_output_tensor) {
    auto output_memory_config = memory_config.value_or(input_tensor.memory_config());

    TT_FATAL(is_device_tensor(input_tensor), "Input tensor must be on device");
    TT_FATAL(
        !optional_output_tensor.has_value() || is_device_tensor(optional_output_tensor.value()),
        "Preallocated output tensor must be on device");

    if (input_tensor.logical_volume() == 0) [[unlikely]] {
        return zero_volume_argmax(input_tensor, dim, keepdim, output_memory_config, optional_output_tensor);
    }

    const auto& input_shape = input_tensor.logical_shape();
    auto rank = input_shape.size();
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

    // Device op only supports reducing the last dim. If the user requests a different dim,
    // permute that dim to the end, run argmax on the last dim, and (if keepdim) permute back.
    // Mirrors the workaround in tests/ttnn/unit_tests/operations/matmul/test_argmax.py (L334-338).
    Tensor effective_input = input_tensor;
    std::optional<int> effective_dim = dim;
    int norm_dim = -1;
    bool permuted = false;
    if (dim.has_value()) {
        const int r = static_cast<int>(rank);
        norm_dim = dim.value() < 0 ? dim.value() + r : dim.value();
        if (norm_dim != r - 1) {
            ttnn::SmallVector<int64_t> perm;
            perm.reserve(rank);
            for (int i = 0; i < r; ++i) {
                if (i != norm_dim) {
                    perm.push_back(i);
                }
            }
            perm.push_back(norm_dim);
            effective_input = ttnn::permute(input_tensor, perm, output_memory_config);
            effective_dim = -1;
            permuted = true;
            TT_FATAL(
                !optional_output_tensor.has_value(),
                "Preallocated output tensor is not supported when argmax dim is not the last dim");
        }
    }

    Tensor result = prim::argmax(
        effective_input,
        tt::tt_metal::DataType::UINT32,
        effective_dim,
        keepdim,
        sub_core_grids,
        use_multicore,
        output_memory_config,
        std::move(optional_output_tensor));

    // With keepdim=true, the reduced dim (size 1) stays at the end; move it back to norm_dim.
    // With keepdim=false, the reduced dim is dropped, so the output shape already matches torch.
    if (permuted && keepdim) {
        const int r = static_cast<int>(rank);
        ttnn::SmallVector<int64_t> inv_perm(rank);
        for (int i = 0; i < r; ++i) {
            inv_perm[i] = i;
        }
        for (int i = r - 1; i > norm_dim; --i) {
            inv_perm[i] = i - 1;
        }
        inv_perm[norm_dim] = r - 1;
        result = ttnn::permute(result, inv_perm, output_memory_config);
    }

    return result;
}

}  // namespace ttnn
