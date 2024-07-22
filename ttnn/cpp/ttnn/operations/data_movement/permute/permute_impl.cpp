// SPDX-FileCopyrightText: © 2024 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include "ttnn/operations/data_movement/transpose/transpose.hpp"
#include "ttnn/deprecated/tt_dnn/op_library/copy/copy_op.hpp"

#include "tt_metal/host_api.hpp"
#include "tt_metal/common/constants.hpp"
#include "ttnn/deprecated/tt_dnn/op_library/auto_format.hpp"
#include "ttnn/tensor/tensor_utils.hpp"


namespace ttnn {
namespace operations::data_movement {

namespace permute {

Tensor permute_(const Tensor &a, std::vector<uint32_t> dims, const MemoryConfig& output_mem_config) {
    Device * device;

    // Get the device
    if (a.storage_type() != StorageType::DEVICE) {
        device = AutoFormat::GetDefaultDevice();
        TT_ASSERT(device != nullptr, "Requires setting default device if no inputs to op are on device");
    } else {
        device = a.device();
    }

    TT_FATAL(dims.size() == 4, "Only 4D tensor are supported for permute.");
    uint32_t N = dims[0], C = dims[1], H = dims[2], W = dims[3];

    bool pad_n = H == 0 || W == 0;
    bool pad_c = H == 1 || W == 1;
    // Convert tensor back to original
    auto a_pad_shape = AutoFormat::pad_to_tile_shape(a.get_legacy_shape(), pad_c, pad_n);
    auto out_shape = a.get_legacy_shape();
    out_shape = {out_shape[N], out_shape[C], out_shape[H], out_shape[W]};

    auto formatted_input_tensor = a;
    if (!AutoFormat::check_input_tensor_format(a, a_pad_shape)) {
        formatted_input_tensor = AutoFormat::format_input_tensor(a, device, a_pad_shape, 0.0, Layout::TILE);
    }
    auto output = formatted_input_tensor;
    static auto transpose_wh = std::bind(ttnn::transpose, std::placeholders::_1, -2, -1, output_mem_config);
    static auto transpose_hc = std::bind(ttnn::transpose, std::placeholders::_1, 1, -2, output_mem_config);
    static auto transpose_cn = std::bind(ttnn::transpose, std::placeholders::_1, 0, 1, output_mem_config);
    if (N == 0 && C == 1 && H == 2 && W == 3) {
        output = formatted_input_tensor;
    } else if (N == 0 && C == 1 && H == 3 && W == 2) {
        output = transpose_wh(formatted_input_tensor);
    } else if (N == 0 && C == 2 && H == 1 && W == 3) {
        output = transpose_hc(formatted_input_tensor);
    } else if (N == 0 && C == 2 && H == 3 && W == 1) {
        output = transpose_wh(transpose_hc(formatted_input_tensor));
    } else if (N == 0 && C == 3 && H == 1 && W == 2) {
        output = transpose_hc(transpose_wh(formatted_input_tensor));
    } else if (N == 0 && C == 3 && H == 2 && W == 1) {
        output = transpose_wh(transpose_hc(transpose_wh(formatted_input_tensor)));
    } else if (N == 1 && C == 0 && H == 2 && W == 3) {
        output = transpose_cn(formatted_input_tensor);
    } else if (N == 1 && C == 0 && H == 3 && W == 2) {
        output = transpose_wh(transpose_cn(formatted_input_tensor));
    } else if (N == 1 && C == 2 && H == 0 && W == 3) {
        output = transpose_hc(transpose_cn(formatted_input_tensor));
    } else if (N == 1 && C == 2 && H == 3 && W == 0) {
        output = transpose_wh(transpose_hc(transpose_cn(formatted_input_tensor)));
    } else if (N == 1 && C == 3 && H == 0 && W == 2) {
        output = transpose_hc(transpose_wh(transpose_cn(formatted_input_tensor)));
    } else if (N == 1 && C == 3 && H == 2 && W == 0) {
        output = transpose_wh(transpose_hc(transpose_wh(transpose_cn(formatted_input_tensor))));
    } else if (N == 2 && C == 0 && H == 1 && W == 3) {
        output = transpose_cn(transpose_hc(formatted_input_tensor));
    } else if (N == 2 && C == 0 && H == 3 && W == 1) {
        output = transpose_wh(transpose_cn(transpose_hc(formatted_input_tensor)));
    } else if (N == 2 && C == 1 && H == 0 && W == 3) {
        output = transpose_cn(transpose_hc(transpose_cn(formatted_input_tensor)));
    } else if (N == 2 && C == 1 && H == 3 && W == 0) {
        output = transpose_wh(transpose_cn(transpose_hc(transpose_cn(formatted_input_tensor))));
    } else if (N == 2 && C == 3 && H == 0 && W == 1) {
        output = transpose_hc(transpose_wh(transpose_cn(transpose_hc(formatted_input_tensor))));
    } else if (N == 2 && C == 3 && H == 1 && W == 0) {
        output = transpose_wh(transpose_hc(transpose_wh(transpose_cn(transpose_hc(formatted_input_tensor)))));
    } else if (N == 3 && C == 0 && H == 1 && W == 2) {
        output = transpose_cn(transpose_hc(transpose_wh(formatted_input_tensor)));
    } else if (N == 3 && C == 0 && H == 2 && W == 1) {
        output = transpose_wh(transpose_cn(transpose_hc(transpose_wh(formatted_input_tensor))));
    } else if (N == 3 && C == 1 && H == 0 && W == 2) {
        output = transpose_cn(transpose_hc(transpose_cn(transpose_wh(formatted_input_tensor))));
    } else if (N == 3 && C == 1 && H == 2 && W == 0) {
        output = transpose_wh(transpose_cn(transpose_hc(transpose_cn(transpose_wh(formatted_input_tensor)))));
    } else if (N == 3 && C == 2 && H == 0 && W == 1) {
        output = transpose_hc(transpose_wh(transpose_cn(transpose_hc(transpose_wh(formatted_input_tensor)))));
    } else if (N == 3 && C == 2 && H == 1 && W == 0) {
        output = transpose_wh(transpose_hc(transpose_wh(transpose_cn(transpose_hc(transpose_wh(formatted_input_tensor))))));
    } else {
        TT_ASSERT(false, "Illegal permute args");
    }
    return AutoFormat::format_output_tensor(output, out_shape, device, Layout::TILE);
}

Tensor permute_launch(const Tensor &a, std::vector<std::int64_t> dims, const MemoryConfig& output_mem_config) {
    std::vector<Tensor> output_tensors = {Tensor(operation::get_workers_for_op_output({a}))};
    operation::launch_with_autoformat(
        [dims, output_mem_config]  (const std::vector<Tensor>& input_tensors, const std::vector<std::optional<const Tensor>>& optional_input_tensors, const std::vector<std::optional<Tensor>>& optional_output_tensors) mutable -> std::vector<Tensor> {
            auto& a = input_tensors.at(0);
            std::vector<uint32_t> normalized_dims(dims.size());
            std::transform(dims.begin(), dims.end(), normalized_dims.begin(), [a](std::int64_t idx) {return a.get_legacy_shape().get_normalized_index(idx);});
            std::vector<uint32_t> seq_dims(dims.size());
            std::iota(seq_dims.begin(), seq_dims.end(), 0);
            if (normalized_dims == seq_dims) {
                return {AutoFormat::move_tensor_to_mem_config(a, output_mem_config)};
            }
            return {operation::decorate_as_composite(__func__, permute_)(a, normalized_dims, output_mem_config)};
        }, {a}, output_tensors);
    return output_tensors.at(0);
}
}
}
}  // namespace ttnn
