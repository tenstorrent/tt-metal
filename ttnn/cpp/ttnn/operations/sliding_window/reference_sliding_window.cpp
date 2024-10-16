// SPDX-FileCopyrightText: © 2024 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include "ttnn/operations/sliding_window/reference_sliding_window.hpp"

#include <cstdint>
#include <numeric>
#include <tuple>

#include "impl/device/device.hpp"
#include "ttnn/tensor/host_buffer/functions.hpp"
#include "ttnn/tensor/host_buffer/types.hpp"

namespace ttnn::operations::sliding_window {

owned_buffer::Buffer<bfloat16> ref_conv_op(const Tensor &input_padded_tensor,
                                           const Shape &input_nchw_shape,
                                           uint32_t stride_h,
                                           uint32_t stride_w,
                                           const std::vector<float> &filter_vector,
                                           const Shape &filter_pyt_tensor_shape,
                                           const Shape &out_golden_pyt_tensor_shape) {
    uint32_t input_n, input_h, input_w;
    uint32_t filter_h, filter_w;
    uint32_t output_n, output_h, output_w;
    uint32_t out_idx = 0;
    auto input_padded_tensor_buf = owned_buffer::get_as<bfloat16>(input_padded_tensor);

    std::tie(output_n, output_h, output_w) = std::forward_as_tuple(
        out_golden_pyt_tensor_shape[0], out_golden_pyt_tensor_shape[1], out_golden_pyt_tensor_shape[2]);
    std::tie(filter_h, filter_w) = std::forward_as_tuple(filter_pyt_tensor_shape[0], filter_pyt_tensor_shape[1]);
    std::tie(input_n, input_h, input_w) =
        std::forward_as_tuple(input_nchw_shape[0], input_nchw_shape[1], input_nchw_shape[2]);
    auto out_golden_pyt_tensor = owned_buffer::create<bfloat16>(output_n * output_h * output_w);

    std::vector<float> input_window;
    for (int i = 0; i < output_n; i++) {
        for (int k = 0; k < output_h; k++) {
            for (int l = 0; l < output_w; l++) {
                // Get input vector of filter size to calculate convolution.
                auto anchor = i * (input_h * input_w) + k * stride_h * input_w + l * stride_w;
                for (int m = 0; m < filter_h; m++) {
                    for (int n = 0; n < filter_w; n++) {
                        auto idx = anchor + m * input_w * stride_h + n * stride_w;
                        if (idx >= input_padded_tensor_buf.size())
                            input_window.push_back(0);
                        else
                            input_window.push_back(input_padded_tensor_buf[idx].to_float());
                    }
                }
                out_golden_pyt_tensor[out_idx] = bfloat16(static_cast<float>(
                    inner_product(input_window.begin(), input_window.end(), filter_vector.begin(), 0.0)));
                out_idx++;
                input_window.clear();
            }
        }
    }

    return out_golden_pyt_tensor;
}

owned_buffer::Buffer<bfloat16> conv_using_op_trace_metadata(
    const owned_buffer::Buffer<bfloat16> &input_padded_tensor_buf,
    const std::vector<float> &filter_vector,
    const std::vector<uint32_t> &op_trace_metadata,
    uint32_t stride_h,
    uint32_t stride_w,
    uint32_t filter_h,
    uint32_t filter_w,
    uint32_t padded_input_w,
    uint32_t out_tensor_size) {
    auto conv_tensor_buf = owned_buffer::create<bfloat16>(out_tensor_size);
    vector<float> input_window;
    uint32_t out_idx = 0;
    for (auto anchor : op_trace_metadata) {
        for (uint32_t h = 0; h < filter_h; h++) {
            for (uint32_t w = 0; w < filter_w; w++) {
                auto idx = anchor + h * stride_h * padded_input_w + w * stride_w;
                if (idx >= input_padded_tensor_buf.size())
                    input_window.push_back(0);
                else
                    input_window.push_back(input_padded_tensor_buf[idx].to_float());
            }
        }
        conv_tensor_buf[out_idx] = bfloat16(
            static_cast<float>(inner_product(input_window.begin(), input_window.end(), filter_vector.begin(), 0.0)));
        out_idx++;
        input_window.clear();
    }
    return conv_tensor_buf;
}

owned_buffer::Buffer<bfloat16> conv_using_shard_boundaries(
    const owned_buffer::Buffer<bfloat16> &input_padded_tensor_buf,
    const std::vector<float> &filter_vector,
    const std::vector<std::pair<uint32_pair_t, uint32_pair_t>> &shard_boundaries,
    uint32_t stride_h,
    uint32_t stride_w,
    uint32_t padded_input_h,
    uint32_t padded_input_w,
    uint32_t filter_h,
    uint32_t filter_w,
    uint32_t output_h,
    uint32_t output_w,
    uint32_t out_tensor_size) {
    auto conv_tensor_buf = owned_buffer::create<bfloat16>(out_tensor_size);
    std::vector<float> input_window;

    uint32_t output_hw = output_h * output_w;
    uint32_t padded_input_hw = padded_input_h * padded_input_w;
    uint32_t input_idx_strt, input_idx;
    for (auto shard_boundry : shard_boundaries) {
        auto [output_shard_start, output_shard_end] = shard_boundry.first;
        for (auto i = output_shard_start; i <= output_shard_end; i++) {
            for (auto fh = 0; fh < filter_h; fh++) {
                for (auto fw = 0; fw < filter_w; fw++) {
                    input_idx_strt = (i / output_hw) * padded_input_hw +
                                     ((i % output_hw) / output_w) * padded_input_w * stride_h +
                                     (i % output_w) * stride_w;
                    input_idx = input_idx_strt + fh * padded_input_w * stride_h + fw * stride_w;
                    if (input_idx >= input_padded_tensor_buf.size())
                        input_window.push_back(0);
                    else
                        input_window.push_back(input_padded_tensor_buf[input_idx].to_float());
                }
            }
            conv_tensor_buf[i] = bfloat16(static_cast<float>(
                inner_product(input_window.begin(), input_window.end(), filter_vector.begin(), 0.0)));
            input_window.clear();
        }
    }
    return conv_tensor_buf;
}

owned_buffer::Buffer<bfloat16> conv_using_sliding_window_op_config(
    const owned_buffer::Buffer<bfloat16> &input_padded_tensor_buf,
    const vector<float> &filter_vector,
    const std::vector<uint32_t> &op_trace_metadata,
    const vector<std::pair<uint32_pair_t, uint32_pair_t>> &shard_boundaries,
    const vector<std::vector<uint16_t>> &sharded_input_top_left_indices,
    uint32_t input_h,
    uint32_t input_w,
    uint32_t stride_h,
    uint32_t stride_w,
    uint32_t padded_input_w,
    uint32_t filter_h,
    uint32_t filter_w,
    uint32_t out_tensor_size) {
    auto conv_tensor_buf = owned_buffer::create<bfloat16>(out_tensor_size);

    vector<float> input_window;
    uint32_t out_idx = 0;

    for (auto j = 0; j < sharded_input_top_left_indices.size(); j++) {
        auto shard = sharded_input_top_left_indices[j];
        auto [output_shard_start, output_shard_end] = shard_boundaries[j].first;
        for (auto idx : shard) {
            for (auto fh = 0; fh < filter_h; fh++) {
                for (auto fw = 0; fw < filter_w; fw++) {
                    auto input_idx =
                        op_trace_metadata[output_shard_start] + idx + fh * padded_input_w * stride_h + fw * stride_w;
                    if (input_idx >= input_padded_tensor_buf.size())
                        input_window.push_back(0);
                    else
                        input_window.push_back(input_padded_tensor_buf[input_idx].to_float());
                }
            }
            conv_tensor_buf[out_idx] = bfloat16(static_cast<float>(
                inner_product(input_window.begin(), input_window.end(), filter_vector.begin(), 0.0)));
            out_idx++;
            input_window.clear();
        }
    }
    return conv_tensor_buf;
}

std::vector<bool> pad_metadata_from_tensor_metadata(
    const std::vector<std::pair<bool, uint32_pair_t>> &tensor_metadata) {
    vector<bool> ref_pad_metadata;
    for (auto i = 0; i < tensor_metadata.size(); i++) {
        auto is_pad_stick = tensor_metadata[i].first;
        if (is_pad_stick) {
            ref_pad_metadata.push_back(true);
            continue;
        }
        ref_pad_metadata.push_back(false);
    }
    return ref_pad_metadata;
}

std::vector<uint32_t> pad_indices_from_flattened_pad_config(
    const std::vector<std::vector<uint16_t>> &flattened_pad_config,
    const std::vector<std::pair<uint32_pair_t, uint32_pair_t>> &shard_boundaries) {
    std::vector<uint32_t> abs_indices;
    for (auto i = 0; i < shard_boundaries.size(); i++) {
        uint32_pair_t input_boundry = shard_boundaries[i].second;
        uint32_t padded_input_tensor_buf_idx = input_boundry.first;

        std::vector<uint16_t> pad_config = flattened_pad_config[i];
        for (auto j = 0; j < pad_config.size(); j += 2) {
            uint32_t local_idx = padded_input_tensor_buf_idx + pad_config[j];
            uint32_t length = pad_config[j + 1];
            for (auto k = local_idx; k < (local_idx + length); k++) abs_indices.push_back(k);
        }
    }
    return abs_indices;
}

std::vector<uint32_t> input_indices_from_flattened_local_config(
    const std::vector<std::vector<uint16_t>> &flattened_local_config,
    const std::vector<std::pair<uint32_pair_t, uint32_pair_t>> &shard_boundaries) {
    std::vector<uint32_t> abs_indices;
    for (auto i = 0; i < shard_boundaries.size(); i++) {
        uint32_pair_t input_boundry = shard_boundaries[i].second;
        uint32_t padded_input_tensor_buf_idx = input_boundry.first;

        std::vector<uint16_t> local_config = flattened_local_config[i];
        size_t sz = local_config[2];
        for (auto j = 3; j < (sz + 3); j += 3) {
            uint32_t local_idx = padded_input_tensor_buf_idx + local_config[j + 1];
            uint32_t length = local_config[j + 2];
            for (auto k = local_idx; k < (local_idx + length); k++) abs_indices.push_back(k);
        }
    }
    return abs_indices;
}

std::vector<uint32_t> input_indices_from_flattened_remote_config(
    tt::tt_metal::Device *device,
    const std::vector<std::vector<uint16_t>> &flattened_remote_config,
    const std::vector<std::pair<uint32_pair_t, uint32_pair_t>> &shard_boundaries,
    bool remote_read,
    bool is_block_sharded,
    bool transpose_mcast) {
    std::vector<uint32_t> abs_indices;
    auto core_id_to_noc_coords = [is_block_sharded, transpose_mcast, device](uint32_t core_id) -> CoreCoord {
        size_t num_cores_x = device->compute_with_storage_grid_size().x;
        CoreCoord core_coord = is_block_sharded ? (transpose_mcast ? CoreCoord(core_id, 0) : CoreCoord(0, core_id))
                                                : CoreCoord(core_id % num_cores_x, core_id / num_cores_x);
        return device->worker_core_from_logical_core(core_coord);
    };

    std::unordered_map<CoreCoord, uint32_t> phy_to_log_core_map;
    for (uint32_t i = 0; i < shard_boundaries.size(); i++) {
        phy_to_log_core_map[core_id_to_noc_coords(i)] = i;
    }

    for (auto i = 0; i < shard_boundaries.size(); i++) {
        std::vector<uint16_t> remote_config = flattened_remote_config[i];
        CoreCoord coord = {remote_config[0], remote_config[1]};
        uint32_t core_id = phy_to_log_core_map[coord];
        uint32_t padded_input_tensor_buf_idx =
            remote_read ? shard_boundaries[i].second.first : shard_boundaries[core_id].second.first;
        size_t sz = remote_config[2];
        uint32_t local_idx = 3;
        while (sz) {
            for (auto i = local_idx; i < local_idx + sz; i += 3) {
                uint32_t local_idx = padded_input_tensor_buf_idx + remote_config[i + 1];
                uint32_t length = remote_config[i + 2];
                for (auto k = local_idx; k < (local_idx + length); k++) abs_indices.push_back(k);
            }
            local_idx += sz;
            coord = {remote_config[local_idx], remote_config[local_idx + 1]};
            core_id = phy_to_log_core_map[coord];
            padded_input_tensor_buf_idx =
                remote_read ? shard_boundaries[i].second.first : shard_boundaries[core_id].second.first;
            local_idx += 2;
            sz = remote_config[local_idx];
            local_idx++;
        }
    }
    return abs_indices;
}

}  // namespace ttnn::operations::sliding_window
