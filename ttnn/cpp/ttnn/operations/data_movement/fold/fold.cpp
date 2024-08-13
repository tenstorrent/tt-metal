// SPDX-FileCopyrightText: © 2024 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

// #include "ttnn/deprecated/tt_dnn/op_library/fold/fold_op.hpp"

#include "ttnn/run_operation.hpp"

#include "ttnn/operations/data_movement/transpose/transpose.hpp"
#include "ttnn/cpp/ttnn/operations/data_movement/slice/slice.hpp"
#include "ttnn/cpp/ttnn/deprecated/tt_dnn/op_library/reshape/reshape_op.hpp"
#include "ttnn/cpp/ttnn/operations/data_movement/pad/pad.hpp"

#include "fold.hpp"


namespace ttnn::operations::data_movement {

std::vector<Tensor> fold_with_transpose_(
    const Tensor& input, const std::optional<const tt::tt_metal::Shape>& output_shape, uint8_t stride_h, uint8_t stride_w, uint8_t pad_c, uint8_t pad_h, uint8_t pad_w, uint8_t queue_id) {

    Device * device;

    // Get the device
    if (input.storage_type() != StorageType::DEVICE) {
        device = AutoFormat::GetDefaultDevice();
        TT_ASSERT(device != nullptr, "Requires setting default device if no inputs to op are on device");
    } else {
        device = input.device();
    }

    uint32_t n = input.shape()[0], c = input.shape()[1], h = input.shape()[2], w = input.shape()[3];
    auto padded_c = c + pad_c; // end padding only
    auto padded_h = h + pad_h * 2; // front and end padding
    auto padded_w = w + pad_w * 2; // front and end padding
    auto padded_h32 = tt::round_up(padded_h, TILE_HEIGHT);
    auto padded_w32 = tt::round_up(padded_w, TILE_HEIGHT);

    auto L1_mem_config = tt::tt_metal::MemoryConfig{.memory_layout = tt::tt_metal::TensorMemoryLayout::INTERLEAVED, .buffer_type=BufferType::L1};

    tt::log_debug("input: {}", input.shape());

    // pad input tensor
    tt::tt_metal::Array4D padded_shape = {n, padded_c, padded_h32, padded_w32};
    auto pad_output = ttnn::pad(input, padded_shape, tt::tt_metal::Array4D({0, 0, 0, 0}), 0);

    tt::log_debug("pad_output: {}", pad_output.shape());

    // transpose
    auto transpose_hw_output = ttnn::transpose(pad_output, 2, 3, L1_mem_config);

    tt::log_debug("transpose_hw_output: {}", transpose_hw_output.shape());

    // transpose
    auto transpose_hc_output = ttnn::transpose(transpose_hw_output, 1, 2, L1_mem_config);

    tt::log_debug("transpose_hc_output: {}", transpose_hc_output.shape());

    // reshape
    n = transpose_hc_output.shape()[0], w = transpose_hc_output.shape()[1], c = transpose_hc_output.shape()[2], h = transpose_hc_output.shape()[3];
    auto reshape_hc_output = tt::tt_metal::reshape(transpose_hc_output, n, (w / stride_w), (c * stride_w), h, L1_mem_config);

    tt::log_debug("reshape_hc_output: {}", reshape_hc_output.shape());

    // transpose
    auto transpose_hw_output2 = ttnn::transpose(reshape_hc_output, 2, 3, L1_mem_config);

    tt::log_debug("transpose_hw_output2: {}", transpose_hw_output2.shape());

    // reshape
    n = transpose_hw_output2.shape()[0], w = transpose_hw_output2.shape()[1], h = transpose_hw_output2.shape()[2], c = transpose_hw_output2.shape()[3];
    auto reshape_hw_output = tt::tt_metal::reshape(transpose_hw_output2, n, w, (h / stride_h), (c * stride_h), L1_mem_config);

    tt::log_debug("reshape_hw_output: {}", reshape_hw_output.shape());

    // transpose
    auto transpose_hc_output2 = ttnn::transpose(reshape_hw_output, 1, 2, L1_mem_config);

    tt::log_debug("transpose_hc_output2: {}", transpose_hc_output2.shape());

    std::vector<Tensor> output_tensors;
    if (output_shape.has_value()) {
        // slice
        n = output_shape.value()[0], w = output_shape.value()[1], h = output_shape.value()[2], c = output_shape.value()[3];
        tt::tt_metal::Array4D slice_output_tensor_start = {0, 0, 0, 0};
        tt::tt_metal::Array4D slice_output_tensor_end = {n - 1, w - 1, h - 1, c - 1};
        auto slice_output = ttnn::slice(transpose_hc_output2, slice_output_tensor_start, slice_output_tensor_end, L1_mem_config);

        output_tensors.emplace_back(slice_output);

        tt::log_debug("slice_output: {}", slice_output.shape());
    } else {
        output_tensors.emplace_back(transpose_hc_output2);
    }

    return output_tensors;
}

Tensor fold(uint8_t queue_id,
            const ttnn::Tensor &input_tensor,
            uint8_t stride_h,
            uint8_t stride_w,
            bool use_transpose_as_fold,
            const std::optional<const tt::tt_metal::Shape> &output_shape,
            uint8_t pad_c,
            uint8_t pad_h,
            uint8_t pad_w) {
    if (use_transpose_as_fold) {
        return fold_with_transpose_(input_tensor, output_shape, stride_h, stride_w, pad_c, pad_h, pad_w, queue_id).at(0);
    }
    bool is_sharded = input_tensor.is_sharded();
    Fold::operation_attributes_t op_attr = {.stride_h = stride_h, .stride_w = stride_w, .is_sharded = is_sharded};
    return ttnn::device_operation::run<Fold>(queue_id, op_attr, Fold::tensor_args_t{.input_tensor = input_tensor});
}

Tensor FoldOperation::operator()(uint8_t queue_id,
                                 const ttnn::Tensor &input_tensor,
                                 uint8_t stride_h,
                                 uint8_t stride_w,
                                 bool use_transpose_as_fold,
                                 const std::optional<const tt::tt_metal::Shape> &output_shape,
                                 uint8_t pad_c,
                                 uint8_t pad_h,
                                 uint8_t pad_w) {
    return fold(queue_id, input_tensor, stride_h, stride_w, use_transpose_as_fold, output_shape, pad_c, pad_h, pad_w);
}

Tensor FoldOperation::operator()(const ttnn::Tensor &input_tensor,
                                 uint8_t stride_h,
                                 uint8_t stride_w,
                                 bool use_transpose_as_fold,
                                 const std::optional<const tt::tt_metal::Shape> &output_shape,
                                 uint8_t pad_c,
                                 uint8_t pad_h,
                                 uint8_t pad_w) {
    uint8_t queue_id = 0;
    return fold(queue_id, input_tensor, stride_h, stride_w, use_transpose_as_fold, output_shape, pad_c, pad_h, pad_w);
}
} // namespace ttnn::operations::data_movement
