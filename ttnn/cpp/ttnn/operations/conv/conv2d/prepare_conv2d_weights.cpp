// SPDX-FileCopyrightText: © 2024 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#include "ttnn/operations/conv/conv2d/prepare_conv2d_weights.hpp"
#include "conv2d/conv2d_utils.hpp"
#include "conv2d/conv2d.hpp"
#include "ttnn/operations/conv/conv2d/device/conv2d_device_operation_types.hpp"
#include "ttnn/operations/conv/conv2d/device/conv2d_device_operation.hpp"
#include <tt_stl/assert.hpp>
#include <cstdint>
#include <tt-logger/tt-logger.hpp>
#include "tt-metalium/buffer_types.hpp"
#include "tt-metalium/constants.hpp"
#include "tt-metalium/host_buffer.hpp"
#include "tt-metalium/math.hpp"
#include "tt-metalium/shape.hpp"
#include "ttnn/operations/core/compute_kernel/compute_kernel_config.hpp"
#include "ttnn/operations/sliding_window/sliding_window.hpp"
#include <optional>
#include <tt-metalium/work_split.hpp>
#include "ttnn/operations/core/core.hpp"
#include "ttnn/operations/data_movement/pad/pad.hpp"
#include "ttnn/tensor/host_buffer/functions.hpp"
#include "ttnn/tensor/storage.hpp"
#include "ttnn/tensor/tensor.hpp"
#include "ttnn/tensor/tensor_spec.hpp"
#include "ttnn/tensor/tensor_utils.hpp"
#include "ttnn/tensor/types.hpp"
#include "ttnn/tensor/tensor_ops.hpp"

#include <tt-metalium/bfloat4.hpp>
#include <tt-metalium/bfloat8.hpp>

namespace ttnn::operations::conv {
using namespace tt;
using sliding_window::ParallelConfig;

namespace conv2d {

/**
 * Common 2D threading utility
 * Parallelizes work across two dimensions with configurable thread counts
 */
class WeightLayoutThreader {
private:
    // Runtime flag to disable threading
    static bool threading_enabled;

    static uint32_t get_max_threads_per_dim() {
        if (!threading_enabled) {
            return 1;
        }
        uint32_t hw_concurrency = std::thread::hardware_concurrency();
        // Use sqrt to balance 2D parallelization: total threads = out_threads × in_threads ≈ sqrt(hw_concurrency - 1)²
        // This prevents oversubscription while maintaining good load distribution across both channel dimensions
        return std::max(1u, static_cast<uint32_t>(std::sqrt(hw_concurrency - 1)));
    }

public:
    struct ThreadConfig {
        uint32_t out_threads, in_threads;
        uint32_t out_per_thread, in_per_thread;
        uint32_t out_total, in_total;
    };

    static ThreadConfig calculate_thread_config(uint32_t out_ch, uint32_t in_ch, int MIN_WORK_PER_THREAD = 16) {
        uint32_t max_threads = get_max_threads_per_dim();
        uint32_t out_threads = std::min(max_threads, std::max(1u, out_ch / MIN_WORK_PER_THREAD));
        uint32_t in_threads = std::min(max_threads, std::max(1u, in_ch / MIN_WORK_PER_THREAD));

        return {out_threads, in_threads, out_ch / out_threads, in_ch / in_threads, out_ch, in_ch};
    }

    template <typename Func>
    static void parallel_for_channels(
        uint32_t out_ch, uint32_t in_ch, uint32_t min_work_per_thread, const Func& work_func) {
        auto cfg = calculate_thread_config(out_ch, in_ch, min_work_per_thread);

        if (cfg.out_threads == 1 && cfg.in_threads == 1) {
            work_func(0, 0, 0, out_ch, 0, in_ch);
            return;
        }

        std::vector<std::thread> threads;
        std::exception_ptr exception_caught = nullptr;
        threads.reserve(cfg.out_threads * cfg.in_threads);

        for (uint32_t ot = 0; ot < cfg.out_threads; ++ot) {
            uint32_t o_start = ot * cfg.out_per_thread;
            uint32_t o_end = (ot == cfg.out_threads - 1) ? cfg.out_total : o_start + cfg.out_per_thread;

            for (uint32_t it = 0; it < cfg.in_threads; ++it) {
                uint32_t i_start = it * cfg.in_per_thread;
                uint32_t i_end = (it == cfg.in_threads - 1) ? cfg.in_total : i_start + cfg.in_per_thread;

                threads.emplace_back([=, &exception_caught] {
                    try {
                        work_func(ot, it, o_start, o_end, i_start, i_end);
                    } catch (...) {
                        // catch the first exception and store it
                        if (!exception_caught) {
                            exception_caught = std::current_exception();
                        }
                    }
                });
            }
        }

        // Wait for all threads to complete
        for (auto& t : threads) {
            t.join();
        }

        // Rethrow first exception if one was caught
        if (exception_caught) {
            std::rethrow_exception(exception_caught);
        }
    }
};
// Initialize static member
bool WeightLayoutThreader::threading_enabled = true;

template <typename T>
static tt::tt_metal::HostBuffer create_host_buffer_for_conv_weight(
    tt::tt_metal::HostBuffer data, DataType output_dtype, const ttnn::Shape& output_shape) {
    if (output_dtype == DataType::BFLOAT8_B || output_dtype == DataType::BFLOAT4_B) {
        if constexpr (std::is_same_v<T, float>) {
            // First create a temporary tensor to convert to tiled layout
            auto temp_tensor =
                Tensor(std::move(data), output_shape, DataType::FLOAT32, Layout::ROW_MAJOR).to_layout(Layout::TILE);

            auto output_float_data = tt::tt_metal::host_buffer::get_as<const float>(temp_tensor);
            auto output_packed_data =
                output_dtype == DataType::BFLOAT8_B
                    ? pack_as_bfp8_tiles(output_float_data, /*row_major_input=*/false, /*is_exp_a=*/false)
                    : pack_as_bfp4_tiles(output_float_data, /*row_major_input=*/false, /*is_exp_a=*/false);
            return tt::tt_metal::HostBuffer(std::move(output_packed_data));
        } else {
            TT_THROW("Unsupported data type");
        }
    }
    return tt::tt_metal::host_buffer::get_host_buffer(
        Tensor(std::move(data), output_shape, output_dtype, Layout::ROW_MAJOR).to_layout(Layout::TILE));
}

template <typename T, typename Fn>
Tensor convert_tensor(const Tensor& input_tensor, const Fn& compute, const TensorSpec& output_spec) {
    TT_FATAL(is_cpu_tensor(input_tensor), "convert_tensor only supports cpu tensors");
    return Tensor(input_tensor.host_storage().transform(compute), output_spec, input_tensor.tensor_topology());
}

template <typename Func, typename... Args>
Tensor convert_tensor_to_tiled_layout_common(
    const Tensor& input_tensor,
    std::optional<DataType> output_dtype,
    const std::unordered_map<DataType, Func>& function_map,
    Args&&... args) {
    TT_ASSERT(
        input_tensor.layout() == Layout::ROW_MAJOR &&
        "Tensor(weight/bias) should be in row major layout for conversion to tilized layout.");

    auto entry = function_map.find(input_tensor.dtype());
    if (entry == function_map.end()) {
        TT_THROW("Unsupported data type");
    }
    return entry->second(input_tensor, std::forward<Args>(args)..., output_dtype.value_or(input_tensor.dtype()));
}

template <typename T>
Tensor create_tensor_from_owned_buffer(
    tt::tt_metal::HostBuffer buf, DataType& output_dtype, ttnn::Shape& output_shape) {
    if constexpr (std::is_same_v<T, float>) {
        if (output_dtype == DataType::BFLOAT8_B || output_dtype == DataType::BFLOAT4_B) {
            auto tensor =
                Tensor(std::move(buf), output_shape, DataType::FLOAT32, Layout::ROW_MAJOR).to_layout(Layout::TILE);
            auto output_float_data = tt::tt_metal::host_buffer::get_as<const float>(tensor);
            auto output_packed_data =
                output_dtype == DataType::BFLOAT8_B
                    ? pack_as_bfp8_tiles(output_float_data, /*row_major_input=*/false, /*is_exp_a=*/false)
                    : pack_as_bfp4_tiles(output_float_data, /*row_major_input=*/false, /*is_exp_a=*/false);
            auto output_uint32_buffer = tt::tt_metal::HostBuffer(std::move(output_packed_data));
            return Tensor(std::move(output_uint32_buffer), output_shape, output_dtype, Layout::TILE);
        }
    } else {
        TT_FATAL(
            (output_dtype != DataType::BFLOAT8_B) || (output_dtype != DataType::BFLOAT4_B),
            "Unsupported output datatype");
    }
    auto rm_tensor = Tensor(std::move(buf), output_shape, output_dtype, Layout::ROW_MAJOR);
    return rm_tensor.to_layout(Layout::TILE);
}

template <typename T>
Tensor to_weight_special_padding_tile_layout(
    const Tensor& conv_weight_tensor,
    uint32_t in1_block_h,
    uint32_t in1_block_w,
    bool enable_activation_reuse,
    DataType output_dtype) {
    auto w_shape = conv_weight_tensor.padded_shape();
    uint32_t in1_block_h_datums = in1_block_h * constants::TILE_HEIGHT;
    uint32_t in1_block_w_datums = in1_block_w * constants::TILE_WIDTH;
    auto weight_matrix_cols = w_shape[0];
    // width padding
    if (weight_matrix_cols % in1_block_w_datums != 0) {
        weight_matrix_cols =
            (uint32_t)std::ceil((double)weight_matrix_cols / (double)in1_block_w_datums) * in1_block_w_datums;
    }
    // height padding
    uint32_t inner_dim = enable_activation_reuse ? w_shape[1] * w_shape[2] * w_shape[3] : w_shape[1] * w_shape[3];
    TT_FATAL(
        in1_block_h_datums >= inner_dim,
        "Block height {} must be >= inner dimension {}",
        in1_block_h_datums,
        inner_dim);
    uint32_t block_height_padding = enable_activation_reuse ? 0 : in1_block_h_datums - inner_dim;
    auto weight_matrix_rows =
        enable_activation_reuse ? in1_block_h_datums : ((w_shape[1] * w_shape[3]) + block_height_padding) * w_shape[2];
    ttnn::Shape output_shape{1, 1, weight_matrix_rows, weight_matrix_cols};

    auto compute =
        [&w_shape, in1_block_h, in1_block_w, output_dtype, &output_shape, weight_matrix_cols, block_height_padding](
            const tt::tt_metal::HostBuffer& input_host_buffer) {
            auto input_buffer = tt::tt_metal::host_buffer::get_as<T>(input_host_buffer);

            auto output_buffer = std::vector<T>(output_shape.volume());

            WeightLayoutThreader::parallel_for_channels(
                w_shape[0],
                w_shape[1],
                16,  // Minimum work per thread
                [&output_buffer, &input_buffer, &w_shape, &weight_matrix_cols, &block_height_padding](
                    uint32_t /*out_t*/,
                    uint32_t /*in_t*/,
                    uint32_t k_start,
                    uint32_t k_end,
                    uint32_t c_start,
                    uint32_t c_end) {
                    for (auto r = 0; r < w_shape[2]; r++) {
                        for (auto s = 0; s < w_shape[3]; s++) {
                            for (auto c = c_start; c < c_end; c++) {
                                for (auto k = k_start; k < k_end; k++) {
                                    auto matrix_idx =
                                        k + (c * weight_matrix_cols) + (s * w_shape[1] * weight_matrix_cols) +
                                        (r * ((w_shape[3] * w_shape[1]) + block_height_padding) * weight_matrix_cols);
                                    auto idx = (k * w_shape[1] * w_shape[2] * w_shape[3]) +
                                               (c * w_shape[2] * w_shape[3]) + (r * w_shape[3]) + s;
                                    output_buffer[matrix_idx] = input_buffer[idx];
                                }
                            }
                        }
                    }
                });
            return create_host_buffer_for_conv_weight<T>(
                tt::tt_metal::HostBuffer(std::move(output_buffer)), output_dtype, output_shape);
        };

    const TensorSpec output_spec(
        output_shape, tt::tt_metal::TensorLayout(output_dtype, tt::tt_metal::PageConfig(Layout::TILE), MemoryConfig{}));
    return convert_tensor<T>(conv_weight_tensor, compute, output_spec);
}

template <typename T>
Tensor to_weight_interleaved_mm_layout(const Tensor& conv_weight_tensor, DataType output_dtype) {
    auto w_shape = conv_weight_tensor.padded_shape();
    uint32_t Co = w_shape[0];  // Output channels
    uint32_t Ci = w_shape[1];  // Input channels
    uint32_t Kh = w_shape[2];  // Kernel height
    uint32_t Kw = w_shape[3];  // Kernel width

    // Output shape: [1, 1, KhKwCi, Co]
    uint32_t weight_matrix_rows = Kh * Kw * Ci;
    uint32_t weight_matrix_cols = Co;

    // Pad to tile boundaries
    uint32_t weight_matrix_rows_padded = tt::round_up(weight_matrix_rows, constants::TILE_HEIGHT);
    uint32_t weight_matrix_cols_padded = tt::round_up(weight_matrix_cols, constants::TILE_WIDTH);

    const ttnn::Shape output_shape{1, 1, weight_matrix_rows_padded, weight_matrix_cols_padded};

    auto compute = [&w_shape, weight_matrix_cols_padded, &output_shape, output_dtype](
                       const tt::tt_metal::HostBuffer& input_host_buffer) {
        auto input_buffer = tt::tt_metal::host_buffer::get_as<T>(input_host_buffer);

        auto output_buffer = std::vector<T>(output_shape.volume(), T(0));

        // Convert from [Co, Ci, Kh, Kw] to [1, 1, KhKwCi, Co]
        WeightLayoutThreader::parallel_for_channels(
            w_shape[0],  // Co
            w_shape[1],  // Ci
            16,          // Minimum work per thread
            [&](uint32_t /*out_t*/,
                uint32_t /*in_t*/,
                uint32_t co_start,
                uint32_t co_end,
                uint32_t ci_start,
                uint32_t ci_end) {
                for (auto kh = 0; kh < w_shape[2]; kh++) {
                    for (auto kw = 0; kw < w_shape[3]; kw++) {
                        for (auto ci = ci_start; ci < ci_end; ci++) {
                            for (auto co = co_start; co < co_end; co++) {
                                // Input index: [Co, Ci, Kh, Kw]
                                auto input_idx = (co * w_shape[1] * w_shape[2] * w_shape[3]) +
                                                 (ci * w_shape[2] * w_shape[3]) + (kh * w_shape[3]) + kw;

                                // Output index: [1, 1, KhKwCi, Co]
                                auto output_row = (kh * w_shape[3] * w_shape[1]) + (kw * w_shape[1]) + ci;
                                auto output_idx = (output_row * weight_matrix_cols_padded) + co;

                                output_buffer[output_idx] = input_buffer[input_idx];
                            }
                        }
                    }
                }
            });

        return create_host_buffer_for_conv_weight<T>(
            tt::tt_metal::HostBuffer(std::move(output_buffer)), output_dtype, output_shape);
    };

    const TensorSpec output_spec(
        output_shape, tt::tt_metal::TensorLayout(output_dtype, tt::tt_metal::PageConfig(Layout::TILE), MemoryConfig{}));

    return convert_tensor<T>(conv_weight_tensor, compute, output_spec);
}

template <typename T>
Tensor to_weight_tile_layout(
    const Tensor& conv_weight_tensor, uint32_t in1_block_h, uint32_t in1_block_w, DataType output_dtype) {
    auto w_shape = conv_weight_tensor.padded_shape();
    auto weight_matrix_cols = w_shape[0];
    // width padding
    uint32_t in1_block_w_datums = in1_block_w * constants::TILE_WIDTH;
    if (weight_matrix_cols % in1_block_w_datums != 0) {
        weight_matrix_cols =
            (uint32_t)std::ceil((double)weight_matrix_cols / (double)in1_block_w_datums) * in1_block_w_datums;
    }
    // height padding
    auto weight_matrix_rows = w_shape[1] * w_shape[2] * w_shape[3];
    uint32_t in1_block_h_datums = in1_block_h * constants::TILE_HEIGHT;
    if (weight_matrix_rows % in1_block_h_datums != 0) {
        weight_matrix_rows =
            (uint32_t)std::ceil((double)weight_matrix_rows / (double)in1_block_h_datums) * in1_block_h_datums;
    }
    const ttnn::Shape output_shape{1, 1, weight_matrix_rows, weight_matrix_cols};

    auto compute = [&w_shape, weight_matrix_cols, &output_shape, output_dtype](
                       const tt::tt_metal::HostBuffer& input_host_buffer) {
        auto input_buffer = tt::tt_metal::host_buffer::get_as<T>(input_host_buffer);

        auto output_buffer = std::vector<T>(output_shape.volume());
        WeightLayoutThreader::parallel_for_channels(
            w_shape[0],
            w_shape[1],
            16,  // Minimum work per thread
            [&](uint32_t /*out_t*/,
                uint32_t /*in_t*/,
                uint32_t out_start,
                uint32_t out_end,
                uint32_t in_start,
                uint32_t in_end) {
                for (auto r = 0; r < w_shape[2]; r++) {
                    for (auto s = 0; s < w_shape[3]; s++) {
                        for (auto c = in_start; c < in_end; c++) {
                            for (auto k = out_start; k < out_end; k++) {
                                auto matrix_idx = k + (c * weight_matrix_cols) + (s * w_shape[1] * weight_matrix_cols) +
                                                  (r * w_shape[3] * w_shape[1] * weight_matrix_cols);
                                auto idx = (k * w_shape[1] * w_shape[2] * w_shape[3]) + (c * w_shape[2] * w_shape[3]) +
                                           (r * w_shape[3]) + s;
                                output_buffer[matrix_idx] = input_buffer[idx];
                            }
                        }
                    }
                }
            });
        return create_host_buffer_for_conv_weight<T>(
            tt::tt_metal::HostBuffer(std::move(output_buffer)), output_dtype, output_shape);
    };

    const TensorSpec output_spec(
        output_shape, tt::tt_metal::TensorLayout(output_dtype, tt::tt_metal::PageConfig(Layout::TILE), MemoryConfig{}));

    return convert_tensor<T>(conv_weight_tensor, compute, output_spec);
}

// Converts convolution weights to interleaved MM layout [1, 1, KhKwCi, Co] and tilizes
// Returns a new tensor with layout=Tile
Tensor convert_conv_weight_tensor_to_interleaved_mm_layout(
    const Tensor& conv_weight_tensor, std::optional<DataType> output_dtype) {
    const static std::unordered_map<DataType, std::function<Tensor(const Tensor&, DataType)>>
        to_w_interleaved_mm_layout_map = {
            {DataType::BFLOAT16, &to_weight_interleaved_mm_layout<bfloat16>},
            {DataType::FLOAT32, &to_weight_interleaved_mm_layout<float>},
            {DataType::UINT32, &to_weight_interleaved_mm_layout<uint32_t>}};

    return convert_tensor_to_tiled_layout_common(conv_weight_tensor, output_dtype, to_w_interleaved_mm_layout_map);
}

// Converts convolution weights to tilized 2d matrix layout.
// Returns a new tensor with layout=Tile
Tensor convert_conv_weight_tensor_to_tiled_layout(
    const Tensor& conv_weight_tensor,
    uint32_t in1_block_h,
    uint32_t in1_block_w,
    std::optional<DataType> output_dtype) {
    const static std::unordered_map<DataType, std::function<Tensor(const Tensor&, uint32_t, uint32_t, DataType)>>
        to_w_tile_layout_map = {
            {DataType::BFLOAT16, &to_weight_tile_layout<bfloat16>},
            {DataType::FLOAT32, &to_weight_tile_layout<float>},
            {DataType::UINT32, &to_weight_tile_layout<uint32_t>}};

    return convert_tensor_to_tiled_layout_common(
        conv_weight_tensor, output_dtype, to_w_tile_layout_map, in1_block_h, in1_block_w);
}

template <typename T>
Tensor to_weight_tile_layout_block_sharded(
    const Tensor& conv_weight_tensor,
    uint32_t in_num_channel_shards,
    uint32_t out_num_channel_shards,
    bool full_inner_dim,
    DataType output_dtype) {
    ttnn::Shape w_shape = conv_weight_tensor.padded_shape();
    // Calculate dimensions outside lambda
    uint32_t weight_matrix_cols = w_shape[0];
    TT_ASSERT(weight_matrix_cols % out_num_channel_shards == 0);
    uint32_t conv_output_shard_width = weight_matrix_cols / out_num_channel_shards;
    uint32_t conv_output_shard_width_padded = tt::round_up(conv_output_shard_width, constants::TILE_WIDTH);
    if (conv_output_shard_width < conv_output_shard_width_padded) {
        // width padding for conv output shard padding
        weight_matrix_cols = conv_output_shard_width_padded * out_num_channel_shards;
    }
    uint32_t weight_matrix_rows = w_shape[1] * w_shape[2] * w_shape[3];
    TT_ASSERT(w_shape[1] % in_num_channel_shards == 0);
    uint32_t conv_input_shard_width = w_shape[1] / in_num_channel_shards;
    uint32_t weight_block_height = conv_input_shard_width * w_shape[2] * w_shape[3];

    // Change for case where we use full inner dim vs slicing by kernel height
    uint32_t weight_block_height_padded;
    if (full_inner_dim) {
        // Use full inner dimension - round up the entire block height
        weight_block_height_padded =
            tt::round_up(conv_input_shard_width * w_shape[3] * w_shape[2], constants::TILE_HEIGHT);
    } else {
        // Original logic - slice by kernel height, round up kernel width portion
        weight_block_height_padded =
            tt::round_up(conv_input_shard_width * w_shape[3], constants::TILE_HEIGHT) * w_shape[2];
    }

    if (weight_block_height < weight_block_height_padded) {
        // height padding for non tile multiple block height
        weight_matrix_rows = weight_block_height_padded * in_num_channel_shards;
    }

    ttnn::Shape output_shape{1, 1, weight_matrix_rows, weight_matrix_cols};

    auto compute = [&w_shape,
                    in_num_channel_shards,
                    out_num_channel_shards,
                    output_dtype,
                    &output_shape,
                    weight_matrix_cols,
                    conv_output_shard_width,
                    conv_output_shard_width_padded,
                    conv_input_shard_width,
                    weight_matrix_rows,
                    weight_block_height_padded,
                    full_inner_dim](const tt::tt_metal::HostBuffer& input_host_buffer) {
        auto input_buffer = tt::tt_metal::host_buffer::get_as<T>(input_host_buffer);
        auto output_buffer = std::vector<T>(output_shape.volume(), T(0));  // Initialize with zeros

        // Pre-calculate stride values
        const uint32_t kernel_h = w_shape[2];
        const uint32_t kernel_w = w_shape[3];

        uint32_t height_stride_per_kernel_row =
            tt::round_up(conv_input_shard_width * (full_inner_dim ? kernel_h : 1) * kernel_w, constants::TILE_HEIGHT);

        WeightLayoutThreader::parallel_for_channels(
            out_num_channel_shards,
            in_num_channel_shards,
            1,  // Minimum work per thread
            [&](uint32_t /*out_t*/,
                uint32_t /*in_t*/,
                uint32_t out_start,
                uint32_t out_end,
                uint32_t in_start,
                uint32_t in_end) {
                for (uint32_t ic = in_start; ic < in_end; ic++) {
                    for (uint32_t r = 0; r < kernel_h; r++) {
                        for (uint32_t s = 0; s < kernel_w; s++) {
                            for (uint32_t c_s = 0; c_s < conv_input_shard_width; c_s++) {
                                for (uint32_t oc = out_start; oc < out_end; oc++) {
                                    for (uint32_t k_s = 0; k_s < conv_output_shard_width; k_s++) {
                                        // Calculate matrix row index based on full_inner_dim flag
                                        uint32_t matrix_row;
                                        if (full_inner_dim) {
                                            // When using full inner dim, layout is: [ic_shard][flattened_inner_dim]
                                            // where flattened_inner_dim = r*kernel_w*conv_input_shard_width +
                                            // s*conv_input_shard_width + c_s
                                            uint32_t flattened_inner_idx = (r * kernel_w * conv_input_shard_width) +
                                                                           (s * conv_input_shard_width) + c_s;
                                            matrix_row = ic * weight_block_height_padded + flattened_inner_idx;
                                        } else {
                                            // Original logic - slice by kernel height
                                            matrix_row = ic * weight_block_height_padded +
                                                         r * height_stride_per_kernel_row + s * conv_input_shard_width +
                                                         c_s;
                                        }

                                        uint32_t matrix_col = (oc * conv_output_shard_width_padded) + k_s;
                                        uint32_t matrix_idx = (matrix_row * weight_matrix_cols) + matrix_col;

                                        // Calculate input tensor index [OC][IC][KH][KW]
                                        uint32_t input_oc = (oc * conv_output_shard_width) + k_s;
                                        uint32_t input_ic = (ic * conv_input_shard_width) + c_s;
                                        uint32_t idx = (input_oc * w_shape[1] * w_shape[2] * w_shape[3]) +
                                                       (input_ic * w_shape[2] * w_shape[3]) + (r * w_shape[3]) + s;

                                        // Ensure we're within bounds before writing
                                        if (matrix_idx < output_buffer.size() && input_oc < w_shape[0] &&
                                            input_ic < w_shape[1]) {
                                            output_buffer[matrix_idx] = input_buffer[idx];
                                        }
                                    }
                                }
                            }
                        }
                    }
                }
            });

        return create_host_buffer_for_conv_weight<T>(
            tt::tt_metal::HostBuffer(std::move(output_buffer)), output_dtype, output_shape);
    };

    const TensorSpec output_spec(
        output_shape, tt::tt_metal::TensorLayout(output_dtype, tt::tt_metal::PageConfig(Layout::TILE), MemoryConfig{}));
    return convert_tensor<T>(conv_weight_tensor, compute, output_spec);
}

// Converts convolution weights to tilized 2d matrix layout for block sharded conv.
// Returns a new tensor with layout=Tile
Tensor convert_conv_weight_tensor_to_tiled_layout_block_sharded(
    const Tensor& conv_weight_tensor,
    uint32_t in_num_channel_shards,
    uint32_t out_num_channel_shards,
    bool full_inner_dim,
    std::optional<DataType> output_dtype) {
    const static std::unordered_map<DataType, std::function<Tensor(const Tensor&, uint32_t, uint32_t, bool, DataType)>>
        to_w_tile_layout_map = {
            {DataType::BFLOAT16, &to_weight_tile_layout_block_sharded<bfloat16>},
            {DataType::FLOAT32, &to_weight_tile_layout_block_sharded<float>},
            {DataType::UINT32, &to_weight_tile_layout_block_sharded<uint32_t>}};

    return convert_tensor_to_tiled_layout_common(
        conv_weight_tensor,
        output_dtype,
        to_w_tile_layout_map,
        in_num_channel_shards,
        out_num_channel_shards,
        full_inner_dim);
}

template <typename T>
Tensor to_bias_tile_layout_block_sharded(
    const Tensor& conv_bias_tensor, uint32_t num_channel_shards, DataType output_dtype) {
    auto b_shape = conv_bias_tensor.padded_shape();
    TT_ASSERT(b_shape[0] == 1 && b_shape[1] == 1 && b_shape[2] == 1);

    auto bias_matrix_cols = b_shape[3];
    auto conv_output_shard_width = bias_matrix_cols / num_channel_shards;
    auto conv_output_shard_width_padded =
        (uint32_t)std::ceil((double)conv_output_shard_width / (double)constants::TILE_WIDTH) * constants::TILE_WIDTH;
    if (conv_output_shard_width < conv_output_shard_width_padded) {
        bias_matrix_cols = conv_output_shard_width_padded * num_channel_shards;
    }
    const auto bias_matrix_rows = 32;
    const ttnn::Shape output_shape{1, 1, bias_matrix_rows, bias_matrix_cols};

    auto compute = [&b_shape,
                    num_channel_shards,
                    output_dtype,
                    &output_shape,
                    conv_output_shard_width,
                    conv_output_shard_width_padded](const tt::tt_metal::HostBuffer& input_host_buffer) {
        auto input_buffer = tt::tt_metal::host_buffer::get_as<T>(input_host_buffer);
        auto output_buffer = std::vector<T>(output_shape.volume());
        for (auto oc = 0; oc < num_channel_shards; oc++) {
            for (auto k_s = 0; k_s < conv_output_shard_width; k_s++) {
                auto matrix_idx = (oc * conv_output_shard_width_padded) + k_s;
                auto idx = (oc * conv_output_shard_width) + k_s;
                output_buffer[matrix_idx] = input_buffer[idx];
            }
        }
        return create_host_buffer_for_conv_weight<T>(
            tt::tt_metal::HostBuffer(std::move(output_buffer)), output_dtype, output_shape);
    };

    const TensorSpec output_spec(
        output_shape, tt::tt_metal::TensorLayout(output_dtype, tt::tt_metal::PageConfig(Layout::TILE), MemoryConfig{}));
    return convert_tensor<T>(conv_bias_tensor, compute, output_spec);
}

// Converts convolution bias to tilized 2d matrix layout for block sharded conv.
// Returns a new tensor with layout=Tile
Tensor convert_conv_bias_tensor_to_tiled_layout_block_sharded(
    const Tensor& conv_bias_tensor, uint32_t num_channel_shards, std::optional<DataType> output_dtype) {
    const static std::unordered_map<
        DataType,
        std::function<Tensor(const Tensor&, uint32_t num_channel_shards, DataType output_dtype)>>
        to_b_tile_layout_map = {
            {DataType::BFLOAT16, &to_bias_tile_layout_block_sharded<bfloat16>},
            {DataType::FLOAT32, &to_bias_tile_layout_block_sharded<float>},
            {DataType::UINT32, &to_bias_tile_layout_block_sharded<uint32_t>},
        };
    return convert_tensor_to_tiled_layout_common(
        conv_bias_tensor, output_dtype, to_b_tile_layout_map, num_channel_shards);
}

// Converts convolution weights to tilized 2d matrix layout.
// Returns a new tensor with layout=Tile
Tensor convert_conv_weight_tensor_to_special_padding_tiled_layout(
    const Tensor& conv_weight_tensor,
    uint32_t in1_block_h,
    uint32_t in1_block_w,
    bool enable_activation_reuse,
    std::optional<DataType> output_dtype) {
    const static std::unordered_map<DataType, std::function<Tensor(const Tensor&, uint32_t, uint32_t, bool, DataType)>>
        to_w_tile_layout_map = {
            {DataType::BFLOAT16, &to_weight_special_padding_tile_layout<bfloat16>},
            {DataType::FLOAT32, &to_weight_special_padding_tile_layout<float>},
            {DataType::UINT32, &to_weight_special_padding_tile_layout<uint32_t>}};

    return convert_tensor_to_tiled_layout_common(
        conv_weight_tensor, output_dtype, to_w_tile_layout_map, in1_block_h, in1_block_w, enable_activation_reuse);
}

/*
Helper function to aid in converting grouped weight tensor to ungrouped weight tensor with padded zero channels
*/
template <typename T>
static Tensor conv_group_weight_zero_pad_helper(
    const Tensor& weight,
    const ttnn::Shape& original_weight_shape,
    const ttnn::Shape& output_weight_shape,
    uint32_t num_groups,
    DataType output_dtype) {
    auto pad_weight = [&original_weight_shape, &output_weight_shape, num_groups, output_dtype](
                          const tt::tt_metal::HostBuffer& conv_weight_tensor_host_buffer) {
        auto conv_weight_tensor_buffer = tt::tt_metal::host_buffer::get_as<T>(conv_weight_tensor_host_buffer);
        auto output_buffer = std::vector<T>(output_weight_shape.volume());

        auto original_strides = compute_strides(original_weight_shape);
        auto output_strides = compute_strides(output_weight_shape);

        for (int curr_batch_idx = 0; curr_batch_idx < original_weight_shape[0]; curr_batch_idx++) {
            int new_batch_idx = curr_batch_idx;

            // Find which group_id the filter belongs to - through this, we can compute the offset where the padding
            // should be applied
            auto group_size = original_weight_shape[0] / num_groups;
            auto group_index = curr_batch_idx / group_size;
            auto group_id = std::min(group_index, num_groups - 1);
            int new_channel_start_idx = group_id * original_weight_shape[1];

            for (int j = 0; j < original_weight_shape[1]; j++) {
                for (int k = 0; k < original_weight_shape[2]; k++) {
                    for (int m = 0; m < original_weight_shape[3]; m++) {
                        // Get value from original weight tensor
                        auto value_flat_input_index = tt::tt_metal::compute_flat_indices(
                            ttnn::SmallVector<uint32_t>{curr_batch_idx, j, k, m}, original_strides);
                        auto value = conv_weight_tensor_buffer[value_flat_input_index];

                        // Copy value to output tensor at the adjusted position
                        auto new_channel_idx = new_channel_start_idx + j;
                        auto output_flat_input_index = tt::tt_metal::compute_flat_indices(
                            ttnn::SmallVector<uint32_t>{new_batch_idx, new_channel_idx, k, m}, output_strides);
                        output_buffer[output_flat_input_index] = value;
                    }
                }
            }
        }
        return tt::tt_metal::HostBuffer(std::move(output_buffer));
    };

    const TensorSpec output_spec(
        output_weight_shape,
        tt::tt_metal::TensorLayout(output_dtype, tt::tt_metal::PageConfig(Layout::ROW_MAJOR), MemoryConfig{}));
    return convert_tensor<T>(weight, pad_weight, output_spec);
}

/*
Helper function to aid in converting depthwise weight tensor to broadcasted weight tensor with repeated input channels
*/
template <typename T>
static Tensor conv_depthwise_weight_bcast_helper(
    const Tensor& conv_weight_tensor,
    const ttnn::Shape& original_weight_shape,
    const ttnn::Shape& output_weight_shape,
    DataType output_dtype) {
    auto compute = [&original_weight_shape, &output_weight_shape, output_dtype](
                       const tt::tt_metal::HostBuffer& conv_weight_tensor_host_buffer) {
        auto conv_weight_tensor_buffer = tt::tt_metal::host_buffer::get_as<T>(conv_weight_tensor_host_buffer);
        // Create a new buffer with the output shape
        auto output_buffer = std::vector<T>(output_weight_shape.volume());
        auto original_strides = compute_strides(original_weight_shape);
        auto output_strides = compute_strides(output_weight_shape);

        WeightLayoutThreader::parallel_for_channels(
            output_weight_shape[0],
            output_weight_shape[1],
            16,  // Minimum work per thread
            [&](uint32_t /*out_t*/,
                uint32_t /*in_t*/,
                uint32_t out_start,
                uint32_t out_end,
                uint32_t in_start,
                uint32_t in_end) {
                for (int i = out_start; i < out_end; i++) {
                    for (int j = in_start; j < in_end; j++) {
                        for (int k = 0; k < output_weight_shape[2]; k++) {
                            for (int l = 0; l < output_weight_shape[3]; l++) {
                                auto value_flat_input_index = tt::tt_metal::compute_flat_indices(
                                    ttnn::SmallVector<uint32_t>{i, 0, k, l}, original_strides);
                                auto value = conv_weight_tensor_buffer[value_flat_input_index];
                                auto output_flat_input_index = tt::tt_metal::compute_flat_indices(
                                    ttnn::SmallVector<uint32_t>{i, j, k, l}, output_strides);
                                output_buffer[output_flat_input_index] = value;
                            }
                        }
                    }
                }
            });

        return tt::tt_metal::HostBuffer(std::move(output_buffer));
    };
    const TensorSpec output_spec(
        output_weight_shape,
        tt::tt_metal::TensorLayout(output_dtype, tt::tt_metal::PageConfig(Layout::ROW_MAJOR), MemoryConfig{}));
    return convert_tensor<T>(conv_weight_tensor, compute, output_spec);
}

/*
Converts convolution weights to grouped layout with padded zeros
This function will take in a weight tensor with shape [out_channels, in_channels // groups, H, W] and return a newly
allocated output tensor with shape [out_channels, in_channels, H, W] The extra channels in shape[1] will be padded with
0 - then the entire weight tensor is convolved with the input tensor - equivalent to convolution if the input tensor was
divided into num_groups for each groupped filter
*/
Tensor convert_conv_weight_tensor_to_grouped_layout(
    const Tensor& conv_weight_tensor, uint32_t num_groups, DataType output_dtype) {
    // Define output tensor shape. This is going to be channel dimension of weight tensor * num_groups - this value
    // should match number of input channels being convolved with the weight tensor
    const auto& original_conv_weight_tensor_shape = conv_weight_tensor.logical_shape();
    ttnn::Shape output_conv_weight_tensor_shape{
        original_conv_weight_tensor_shape[0],
        original_conv_weight_tensor_shape[1] * num_groups,
        original_conv_weight_tensor_shape[2],
        original_conv_weight_tensor_shape[3]};

    const static std::
        unordered_map<DataType, std::function<Tensor(const Tensor&, ttnn::Shape, ttnn::Shape, uint32_t, DataType)>>
            to_w_tile_layout_map = {
                {DataType::INT32, &conv_group_weight_zero_pad_helper<int32_t>},
                {DataType::FLOAT32, &conv_group_weight_zero_pad_helper<float>},
                {DataType::BFLOAT16, &conv_group_weight_zero_pad_helper<bfloat16>},
                {DataType::UINT16, &conv_group_weight_zero_pad_helper<uint16_t>},
                {DataType::BFLOAT8_B, &conv_group_weight_zero_pad_helper<float>},
                {DataType::UINT32, &conv_group_weight_zero_pad_helper<uint32_t>},
                {DataType::BFLOAT4_B, &conv_group_weight_zero_pad_helper<uint32_t>},
            };

    if (tt_metal::is_device_tensor(conv_weight_tensor)) {
        log_warning(
            tt::LogOp,
            "Prepare weights for Conv2D with groups > 1 expects weights on host, but they are on device. The op will "
            "move them back to host.");
    }
    return convert_tensor_to_tiled_layout_common(
        tt_metal::is_device_tensor(conv_weight_tensor) ? ttnn::operations::core::from_device(conv_weight_tensor)
                                                       : conv_weight_tensor,
        output_dtype,
        to_w_tile_layout_map,
        original_conv_weight_tensor_shape,
        output_conv_weight_tensor_shape,
        num_groups);
}

/*
Helper function to aid in converting grouped weight tensor for conv_transpose2d
This is different from conv_group_weight_zero_pad_helper because conv_transpose2d weights have shape:
[in_channels, out_channels/groups, H, W] and we need to expand dimension 1 to out_channels
with proper zero padding based on which group each input channel belongs to.
*/
template <typename T>
static Tensor conv_transpose2d_group_weight_zero_pad_helper(
    const Tensor& weight,
    const ttnn::Shape& original_weight_shape,
    const ttnn::Shape& output_weight_shape,
    uint32_t num_groups,
    DataType output_dtype) {
    auto pad_weight = [&original_weight_shape, &output_weight_shape, num_groups, output_dtype](
                          const tt::tt_metal::HostBuffer& conv_weight_tensor_host_buffer) {
        auto conv_weight_tensor_buffer = tt::tt_metal::host_buffer::get_as<T>(conv_weight_tensor_host_buffer);
        auto output_buffer = std::vector<T>(output_weight_shape.volume());

        // For conv_transpose2d: [in_channels, out_channels/groups, H, W] -> [in_channels, out_channels, H, W]
        // Each input channel i belongs to group g = i / (in_channels/groups)
        // The local output channel c (0 to out_channels/groups-1) maps to global output channel:
        // global_out_channel = g * (out_channels/groups) + c
        uint32_t in_channels = original_weight_shape[0];
        uint32_t out_channels_per_group = original_weight_shape[1];
        uint32_t in_channels_per_group = in_channels / num_groups;

        auto original_weight_strides = compute_strides(original_weight_shape);
        auto output_weight_strides = compute_strides(output_weight_shape);
        for (int i = 0; i < original_weight_shape[0]; i++) {  // in_channels
            // Find which group this input channel belongs to
            auto group_id = i / in_channels_per_group;

            for (int c = 0; c < original_weight_shape[1]; c++) {  // out_channels/groups
                // Calculate the global output channel index
                int global_out_channel = group_id * out_channels_per_group + c;

                for (int h = 0; h < original_weight_shape[2]; h++) {
                    for (int w = 0; w < original_weight_shape[3]; w++) {
                        // Get value from original weight tensor
                        auto value_flat_input_index = tt::tt_metal::compute_flat_indices(
                            ttnn::SmallVector<uint32_t>{i, c, h, w}, original_weight_strides);
                        auto value = conv_weight_tensor_buffer[value_flat_input_index];

                        // Copy value to output tensor at the adjusted position
                        auto output_flat_input_index = tt::tt_metal::compute_flat_indices(
                            ttnn::SmallVector<uint32_t>{i, global_out_channel, h, w}, output_weight_strides);
                        output_buffer[output_flat_input_index] = value;
                    }
                }
            }
        }
        return tt::tt_metal::HostBuffer(std::move(output_buffer));
    };

    const TensorSpec output_spec(
        output_weight_shape,
        tt::tt_metal::TensorLayout(output_dtype, tt::tt_metal::PageConfig(Layout::ROW_MAJOR), MemoryConfig{}));
    return convert_tensor<T>(weight, pad_weight, output_spec);
}

/*
Converts convolution weights for conv_transpose2d to grouped layout with padded zeros
This function will take in a weight tensor with shape [in_channels, out_channels // groups, H, W] and return a newly
allocated output tensor with shape [in_channels, out_channels, H, W]
The extra channels in shape[1] will be padded with 0 at the appropriate positions based on group membership.
This is used BEFORE transform_weights_for_conv_transpose2d to properly handle grouped convolutions.
*/
Tensor convert_conv_weight_tensor_to_grouped_layout_for_conv_transpose2d(
    const Tensor& conv_weight_tensor, uint32_t num_groups, DataType output_dtype) {
    // Define output tensor shape
    // Input: [in_channels, out_channels/groups, H, W]
    // Output: [in_channels, out_channels, H, W] where out_channels = (out_channels/groups) * num_groups
    const auto& original_conv_weight_tensor_shape = conv_weight_tensor.logical_shape();
    ttnn::Shape output_conv_weight_tensor_shape{
        original_conv_weight_tensor_shape[0],
        original_conv_weight_tensor_shape[1] * num_groups,
        original_conv_weight_tensor_shape[2],
        original_conv_weight_tensor_shape[3]};

    const static std::
        unordered_map<DataType, std::function<Tensor(const Tensor&, ttnn::Shape, ttnn::Shape, uint32_t, DataType)>>
            to_w_tile_layout_map = {
                {DataType::INT32, &conv_transpose2d_group_weight_zero_pad_helper<int32_t>},
                {DataType::FLOAT32, &conv_transpose2d_group_weight_zero_pad_helper<float>},
                {DataType::BFLOAT16, &conv_transpose2d_group_weight_zero_pad_helper<bfloat16>},
                {DataType::UINT16, &conv_transpose2d_group_weight_zero_pad_helper<uint16_t>},
                {DataType::BFLOAT8_B, &conv_transpose2d_group_weight_zero_pad_helper<float>},
                {DataType::UINT32, &conv_transpose2d_group_weight_zero_pad_helper<uint32_t>},
                {DataType::BFLOAT4_B, &conv_transpose2d_group_weight_zero_pad_helper<uint32_t>},
            };

    if (tt_metal::is_device_tensor(conv_weight_tensor)) {
        log_warning(
            tt::LogOp,
            "Prepare weights for ConvTranspose2D with groups > 1 expects weights on host, but they are on device. The "
            "op will move them back to host.");
    }
    return convert_tensor_to_tiled_layout_common(
        tt_metal::is_device_tensor(conv_weight_tensor) ? ttnn::operations::core::from_device(conv_weight_tensor)
                                                       : conv_weight_tensor,
        output_dtype,
        to_w_tile_layout_map,
        original_conv_weight_tensor_shape,
        output_conv_weight_tensor_shape,
        num_groups);
}

/*
Converts convolution weights to depthwise layout
This function will take in a weight tensor with shape [out_channels, 1, H, W] and return a newly
allocated output tensor with shape [out_channels, act_block_h, H, W] The extra channels in shape[1] are repeated
from the original weight tensor - it would be convolving act_block in conv_matrix in one go
*/
Tensor convert_conv_weight_tensor_to_depthwise_layout(
    const Tensor& conv_weight_tensor, uint32_t act_block_h_ntiles, DataType output_dtype) {
    const auto& original_conv_weight_tensor_shape = conv_weight_tensor.logical_shape();
    uint32_t num_input_channels_to_repeat = act_block_h_ntiles * constants::TILE_HEIGHT;
    ttnn::Shape output_conv_weight_tensor_shape{
        original_conv_weight_tensor_shape[0],
        num_input_channels_to_repeat,
        original_conv_weight_tensor_shape[2],
        original_conv_weight_tensor_shape[3]};

    // Create newly allocated buffer all initialized to 0 depending on the datatype of the weight tensor
    const static std::unordered_map<DataType, std::function<Tensor(const Tensor&, ttnn::Shape, ttnn::Shape, DataType)>>
        to_w_tile_layout_map = {
            {DataType::INT32, &conv_depthwise_weight_bcast_helper<int32_t>},
            {DataType::FLOAT32, &conv_depthwise_weight_bcast_helper<float>},
            {DataType::BFLOAT16, &conv_depthwise_weight_bcast_helper<bfloat16>},
            {DataType::UINT16, &conv_depthwise_weight_bcast_helper<uint16_t>},
            {DataType::BFLOAT8_B, &conv_depthwise_weight_bcast_helper<float>},
            {DataType::UINT32, &conv_depthwise_weight_bcast_helper<uint32_t>},
            {DataType::BFLOAT4_B, &conv_depthwise_weight_bcast_helper<uint32_t>},
        };
    output_dtype = ((output_dtype == DataType::BFLOAT8_B) || (output_dtype == DataType::BFLOAT4_B)) ? DataType::FLOAT32
                                                                                                    : output_dtype;
    if (tt_metal::is_device_tensor(conv_weight_tensor)) {
        log_warning(
            tt::LogOp,
            "Prepare weights for Depthwise Conv1D expects weights on host, but they are on device. The op will move "
            "them back to host.");
    }
    return convert_tensor_to_tiled_layout_common(
        tt_metal::is_device_tensor(conv_weight_tensor) ? ttnn::operations::core::from_device(conv_weight_tensor)
                                                       : conv_weight_tensor,
        output_dtype,
        to_w_tile_layout_map,
        original_conv_weight_tensor_shape,
        output_conv_weight_tensor_shape);
}

static Tensor to_folded_weight_layout(const Tensor& conv_weight_tensor, std::array<uint32_t, 2> stride) {
    auto w_shape = conv_weight_tensor.padded_shape();
    uint32_t out_channels = w_shape[0];
    uint32_t in_channels = w_shape[1];
    uint32_t kernel_h = w_shape[2];
    uint32_t kernel_w = w_shape[3];

    // Get input data type
    auto dtype = conv_weight_tensor.dtype();

    auto pad_h = (stride[0] - (kernel_h % stride[0])) % stride[0];
    auto pad_w = (stride[1] - (kernel_w % stride[1])) % stride[1];

    auto padded_kernel_h = kernel_h + pad_h;
    auto padded_kernel_w = kernel_w + pad_w;

    ttnn::Shape output_shape = ttnn::Shape(
        {out_channels, in_channels * stride[0] * stride[1], padded_kernel_h / stride[0], padded_kernel_w / stride[1]});

    auto fold_weights = [&]<typename T>(const tt::tt_metal::HostStorage& storage) {
        auto folded_storage = storage.transform([&](const tt::tt_metal::HostBuffer& input_host_buffer) {
            auto input_buffer = tt::tt_metal::host_buffer::get_as<T>(input_host_buffer);

            std::vector<T> output_buffer(output_shape.volume(), T(0));
            int new_h = padded_kernel_h / stride[0];
            int new_w = padded_kernel_w / stride[1];
            WeightLayoutThreader::parallel_for_channels(
                out_channels,
                in_channels,
                16,  // Minimum work per thread
                [&](uint32_t /*out_t*/,
                    uint32_t /*in_t*/,
                    uint32_t out_start,
                    uint32_t out_end,
                    uint32_t in_start,
                    uint32_t in_end) {
                    for (auto oc = out_start; oc < out_end; oc++) {
                        for (auto ic = in_start; ic < in_end; ic++) {
                            for (auto kh = 0; kh < kernel_h; kh++) {
                                for (auto kw = 0; kw < kernel_w; kw++) {
                                    uint32_t src_idx = ((((oc * in_channels + ic) * kernel_h) + kh) * kernel_w) + kw;

                                    int sh = kh % stride[0];
                                    int sw = kw % stride[1];

                                    // Calculate new y,x coordinates
                                    int y = kh / stride[0];
                                    int x = kw / stride[1];

                                    // Calculate folded input channel index
                                    int folded_ic_idx = ((sh * stride[1] + sw) * in_channels) + ic;

                                    // Calculate final destination index
                                    int dst_idx = (oc * in_channels * stride[0] * stride[1] * new_h * new_w) +
                                                  (folded_ic_idx * new_h * new_w) + (y * new_w) + x;

                                    output_buffer[dst_idx] = input_buffer[src_idx];
                                }
                            }
                        }
                    }
                });

            return tt::tt_metal::HostBuffer(std::move(output_buffer));
        });
        return Tensor(
            std::move(folded_storage),
            TensorSpec(
                output_shape,
                tt::tt_metal::TensorLayout(dtype, tt::tt_metal::PageConfig(Layout::ROW_MAJOR), MemoryConfig{})),
            conv_weight_tensor.tensor_topology());
    };

    const auto& storage = conv_weight_tensor.host_storage();
    switch (dtype) {
        case DataType::FLOAT32: return fold_weights.template operator()<float>(storage);
        case DataType::BFLOAT16: return fold_weights.template operator()<bfloat16>(storage);
        case DataType::UINT32: return fold_weights.template operator()<uint32_t>(storage);
        case DataType::INT32: return fold_weights.template operator()<int32_t>(storage);
        case DataType::UINT16: return fold_weights.template operator()<uint16_t>(storage);
        case DataType::BFLOAT8_B: return fold_weights.template operator()<float>(storage);
        case DataType::BFLOAT4_B: return fold_weights.template operator()<uint32_t>(storage);
        default:
            TT_THROW(
                "Unsupported input data type for to_folded_weight_layout: {} (type id: {})",
                dtype,
                static_cast<int>(dtype));
    }
}

void validate_host_conv_weights(const ttnn::Tensor& weight_tensor) {
    TT_FATAL(
        !ttnn::has_storage_type_of(weight_tensor, ttnn::DEVICE_STORAGE_TYPE),
        "Host conv weights should be placed on host");
    TT_FATAL(weight_tensor.layout() == Layout::ROW_MAJOR, "Host conv weights layout should be in row_major layout");
    TT_FATAL(weight_tensor.logical_shape().rank() == 4, "Host conv weights should be 4D tensor");
}

void validate_host_conv_bias(const ttnn::Tensor& bias_tensor) {
    TT_FATAL(bias_tensor.logical_shape().rank() == 4, "Host conv bias should be 4D tensor");
    TT_FATAL(bias_tensor.layout() == Layout::ROW_MAJOR, "Host conv bias layout should be in row_major layout");
    const auto& bias_shape = bias_tensor.logical_shape();
    TT_FATAL(bias_shape[0] == 1 && bias_shape[1] == 1 && bias_shape[2] == 1, "Host conv bias shape is not correct");
}

// Validate device conv weights format (minimal validation for main path)
bool is_valid_device_conv_weights(
    const ttnn::Tensor& weight_tensor,
    uint32_t /*in_channels*/,
    uint32_t out_channels,
    const std::optional<DataType>& expected_dtype) {
    if (weight_tensor.layout() != Layout::TILE) {
        return false;
    }

    const auto& shape = weight_tensor.logical_shape();
    if (shape.rank() != 4 || shape[0] != 1 || shape[1] != 1) {
        return false;
    }

    if (shape[3] < out_channels) {
        return false;
    }

    if (expected_dtype.has_value() && weight_tensor.dtype() != expected_dtype.value()) {
        return false;
    }

    return true;
}

// Validate device conv bias format (minimal validation for main path)
bool is_valid_device_conv_bias(
    const ttnn::Tensor& bias_tensor, uint32_t out_channels, const std::optional<DataType>& expected_dtype) {
    if (bias_tensor.layout() != Layout::TILE) {
        return false;
    }

    if (bias_tensor.logical_shape()[3] < out_channels) {
        return false;
    }

    if (expected_dtype.has_value() && bias_tensor.dtype() != expected_dtype.value()) {
        return false;
    }

    return true;
}

static Conv2dBlockConfig get_opt_block_config(
    bool mm_conv,
    uint32_t in_channels,
    uint32_t out_channels,
    uint32_t output_height,
    uint32_t output_width,
    uint32_t batch_size,
    uint32_t input_height,
    uint32_t input_width,
    uint32_t groups,
    std::array<uint32_t, 2> kernel_size,
    std::array<uint32_t, 2> stride,
    std::array<uint32_t, 2> dilation,
    std::array<uint32_t, 4> padding,
    MeshDevice* device,
    Conv2dConfig& conv_config,
    Layout input_layout,
    DataType input_dtype,
    DataType output_dtype,
    const DeviceComputeKernelConfig& compute_config,
    const MemoryConfig& input_memory_config,
    const bool has_bias) {
    auto compute_grid_size = device->compute_with_storage_grid_size();

    conv_config = determine_conv_config_for_auto_shard(
        conv_config,
        mm_conv,
        batch_size,
        in_channels,
        out_channels,
        output_height,
        output_width,
        kernel_size[1],
        input_height,
        input_width,
        compute_grid_size,
        input_layout,
        input_dtype,
        output_dtype,
        input_memory_config,
        kernel_size,
        stride,
        dilation,
        padding,
        groups,
        has_bias,
        compute_config);

    if (input_memory_config.is_sharded() && !conv_config.reshard_if_not_optimal) {
        conv_config.shard_layout = input_memory_config.memory_layout();
    }
    const uint32_t in_channels_alignment = get_input_channels_alignment(
        conv_config.shard_layout.value(),
        input_layout,
        input_memory_config.buffer_type() == BufferType::DRAM,
        mm_conv,
        input_memory_config);

    ParallelConfig parallel_config;
    if (input_memory_config.shard_spec().has_value() && !conv_config.reshard_if_not_optimal) {
        parallel_config = {
            .grid = input_memory_config.shard_spec().value().grid,
            .shard_scheme = input_memory_config.memory_layout(),
            .shard_orientation = input_memory_config.shard_spec().value().orientation};
    } else {
        ShardOrientation shard_orientation =
            conv_config.transpose_shards ? ShardOrientation::COL_MAJOR : ShardOrientation::ROW_MAJOR;

        parallel_config = determine_parallel_config(
            conv_config.shard_layout.value(),
            batch_size,
            in_channels,
            output_height,
            output_width,
            out_channels,
            in_channels_alignment,
            compute_grid_size,
            shard_orientation,
            !mm_conv,
            true,
            true,
            conv_config.act_block_h_override);
    }

    auto output_compute_grid_size = get_output_compute_grid_size(compute_grid_size, conv_config, parallel_config);
    ParallelConfig output_parallel_config = determine_output_parallel_config(
        parallel_config, output_compute_grid_size, out_channels, parallel_config.shard_orientation, mm_conv);

    MemoryConfig conv_out_memory_config = create_sharded_memory_config_from_parallel_config(
        ttnn::Shape(
            {1,
             1,
             batch_size * output_height * output_width,
             tt::round_up(
                 out_channels,
                 get_num_cores_channels_from_parallel_config(output_parallel_config) * tt::constants::TILE_WIDTH)}),
        output_parallel_config,
        tt::constants::TILE_HEIGHT);
    ParallelConfig largest_parallel_config = output_parallel_config.grid.num_cores() > parallel_config.grid.num_cores()
                                                 ? output_parallel_config
                                                 : parallel_config;
    Conv2dParallelizationConfig opt_conv_op_parallel_config =
        determine_conv_op_parallel_config_from_conv_output_mem_config(
            conv_out_memory_config,
            get_num_cores_nhw_from_parallel_config(parallel_config),
            get_num_cores_channels_from_parallel_config(parallel_config),
            get_num_cores_channels_from_parallel_config(output_parallel_config));

    uint32_t in_channels_padded =
        tt::round_up(in_channels, get_num_cores_channels_from_parallel_config(parallel_config) * in_channels_alignment);

    uint32_t nhw_out_padded_ntile_per_core =
        conv_out_memory_config.shard_spec().value().shape[0] / tt::constants::TILE_HEIGHT;

    const bool conv_is_1d_depthwise =
        is_1d_depthwise_conv(groups, in_channels, out_channels, kernel_size[0], kernel_size[1], input_height, has_bias);

    return determine_per_core_conv_block_config(
        parallel_config,
        opt_conv_op_parallel_config,
        in_channels_padded,
        nhw_out_padded_ntile_per_core,
        conv_config.act_block_h_override,
        conv_config.act_block_w_div,
        kernel_size[0],
        kernel_size[1],
        output_width,
        get_fp32_dest_acc_en(compute_config),
        conv_config.full_inner_dim,
        conv_config.enable_activation_reuse,
        conv_is_1d_depthwise);
}

static uint32_t calculate_out_channels_padded(uint32_t out_channels, const ParallelConfig& output_parallel_config) {
    uint32_t output_num_cores_channels = get_num_cores_channels_from_parallel_config(output_parallel_config);
    return tt::round_up(out_channels, output_num_cores_channels * tt::constants::TILE_WIDTH);
}

static Conv2dWeightsBiasPrepConfig setup_conv_prep_config(
    ttnn::MemoryConfig input_memory_config,
    Layout input_layout,
    uint32_t in_channels,
    uint32_t out_channels,
    uint32_t batch_size,
    uint32_t input_height,
    uint32_t input_width,
    std::array<uint32_t, 2> kernel_size,
    std::array<uint32_t, 2> stride,
    std::variant<std::array<uint32_t, 2>, std::array<uint32_t, 4>> padding,
    std::array<uint32_t, 2> dilation,
    bool has_bias,
    uint32_t groups,
    MeshDevice* device,
    Conv2dConfig& conv_config,
    const DeviceComputeKernelConfig& compute_config,
    DataType input_dtype,
    const std::optional<const DataType>& output_dtype,
    const std::optional<const Conv2dSliceConfig>& dram_slice_config_ = std::nullopt) {
    DataType conv_output_dtype = output_dtype.value_or(input_dtype);

    std::array<uint32_t, 4> padding_n4 = sliding_window::get_pair_n4_padding(padding);
    bool mm_conv = use_matmul_for_1x1_conv(kernel_size, stride, padding_n4, dilation, groups, conv_config);
    auto orig_stride = stride;
    const bool is_conv1d = is_1d_conv(kernel_size[0], input_height);
    conv_config.enable_kernel_stride_folding = auto_enable_kernel_folding(
        input_memory_config,
        input_layout,
        input_dtype,
        conv_config.enable_kernel_stride_folding,
        input_height,
        input_width,
        kernel_size,
        stride,
        dilation,
        padding_n4);
    if (conv_config.enable_kernel_stride_folding.value()) {
        auto folding_result = compute_kernel_stride_folding_params(
            input_height, input_width, in_channels, kernel_size, stride, padding_n4, conv_config);

        input_height = folding_result.input_height;
        input_width = folding_result.input_width;
        in_channels = folding_result.in_channels;
        stride = folding_result.stride;
        kernel_size = folding_result.kernel_size;
        mm_conv = folding_result.mm_conv;
    }

    auto [output_height, output_width] =
        calculate_output_image_size({input_height, input_width}, kernel_size, stride, padding_n4, dilation);

    auto conv_execution_path =
        determine_conv2d_execution_path(input_memory_config.buffer_type() == BufferType::L1, dram_slice_config_);
    // Conv1D doesn't support DRAM
    bool is_dram_conv = (conv_execution_path == Conv2dExecutionPath::DRAM) && !is_conv1d;

    if (is_dram_conv && !mm_conv /*DRAM with Matmul doesn't need slicing*/) {
        Tensor dummy_weight_tensor = tt::tt_metal::create_device_tensor(
            tt::tt_metal::TensorSpec(
                ttnn::Shape({out_channels, in_channels / groups, kernel_size[0], kernel_size[1]}),
                tt::tt_metal::TensorLayout(
                    conv_config.weights_dtype.value(),
                    tt::tt_metal::PageConfig(Layout::ROW_MAJOR),
                    MemoryConfig{
                        TensorMemoryLayout::INTERLEAVED,
                        BufferType::DRAM,
                    })),
            device);
        std::optional<Tensor> dummy_bias_tensor = std::nullopt;
        if (has_bias) {
            dummy_bias_tensor = tt::tt_metal::create_device_tensor(
                tt::tt_metal::TensorSpec(
                    ttnn::Shape({1, 1, 1, out_channels}),
                    tt::tt_metal::TensorLayout(
                        conv_config.weights_dtype.value(),
                        tt::tt_metal::PageConfig(Layout::ROW_MAJOR),
                        MemoryConfig{
                            TensorMemoryLayout::INTERLEAVED,
                            BufferType::DRAM,
                        })),
                device);
        }
        auto conv2d_slice_attr = get_conv2d_slice_attr(
            batch_size,
            input_height,
            input_width,
            in_channels,
            out_channels,
            kernel_size,
            stride,
            padding_n4,
            dilation,
            groups,
            input_layout,
            input_dtype,
            conv_output_dtype,
            std::ref(dummy_weight_tensor),
            has_bias ? std::make_optional(std::ref(dummy_bias_tensor.value())) : std::nullopt,
            conv_config,
            compute_config,
            device);

        auto dram_slice_config = op_slicing::determine_slice_config(
            conv2d_slice_attr.get(),
            ttnn::Shape{batch_size, input_height, input_width, in_channels},
            ttnn::Shape{batch_size, output_height, output_width, out_channels},
            dram_slice_config_,
            conv_config.output_layout,
            device);
        log_info(
            tt::LogOp,
            "Auto determined DRAM Slice Config in Prepare Conv2d Weights as {} for {}",
            dram_slice_config,
            conv2d_slice_attr->name());
        if (dram_slice_config.num_slices == 1) {
            log_info(tt::LogOp, "DRAM Slicing is not needed as only one slice is required.");
            is_dram_conv = false;
        }
        uint32_t slice_rounding_value = 1;
        if (conv_config.output_layout == tt_metal::Layout::TILE &&
            dram_slice_config.slice_type == Conv2dSliceConfig::SliceType::DRAM_WIDTH) {
            // In Conv2d DRAM with Outputs in Tile layout, we need to round the slice size to a multiple of TILE_HEIGHT.
            slice_rounding_value = tt::constants::TILE_HEIGHT;
        }

        const uint32_t output_sliced_dim =
            dram_slice_config.slice_type == Conv2dSliceConfig::SliceType::DRAM_HEIGHT ? output_height : output_width;

        TT_FATAL(
            dram_slice_config.num_slices <= output_sliced_dim,
            " Number of slices {} should be less or equal than the dimension being sliced {} in Conv2D DRAM Slicing",
            dram_slice_config.num_slices,
            output_sliced_dim);

        const uint32_t min_output_slice_size =
            tt::div_up(tt::div_up(output_sliced_dim, slice_rounding_value), dram_slice_config.num_slices) *
            slice_rounding_value;

        if (dram_slice_config.slice_type == Conv2dSliceConfig::SliceType::DRAM_HEIGHT) {
            output_height = min_output_slice_size;
            input_height = ((output_height - 1) * stride[0]) + ((kernel_size[0] - 1) * (dilation[0] - 1)) +
                           kernel_size[0] - padding_n4[0];
            padding_n4[1] = 0;  // No padding on bottom for sliced height
        } else {
            output_width = min_output_slice_size;
            input_width = ((output_width - 1) * stride[1]) + ((kernel_size[1] - 1) * (dilation[1] - 1)) +
                          kernel_size[1] - padding_n4[2];
            padding_n4[3] = 0;  // No padding on right for sliced width
        }
        input_memory_config = conv2d_slice_attr->get_input_memory_config(
            {0, 0},                        // Slice Start
            {output_height, output_width}  // Slice End
        );
    }

    auto opt_conv_op_block_config = get_opt_block_config(
        mm_conv,
        in_channels,
        out_channels,
        output_height,
        output_width,
        batch_size,
        input_height,
        input_width,
        groups,
        kernel_size,
        stride,
        dilation,
        padding_n4,
        device,
        conv_config,
        input_layout,
        input_dtype,
        conv_output_dtype,
        compute_config,
        input_memory_config,
        has_bias);

    if (input_memory_config.is_sharded() && !conv_config.reshard_if_not_optimal) {
        conv_config.shard_layout = input_memory_config.memory_layout();
    }

    uint32_t input_channels_alignment = get_input_channels_alignment(
        conv_config.shard_layout.value(), input_layout, is_dram_conv, mm_conv, input_memory_config);

    ParallelConfig parallel_config;
    if (input_memory_config.shard_spec().has_value() && !conv_config.reshard_if_not_optimal) {
        parallel_config = {
            .grid = input_memory_config.shard_spec().value().grid,
            .shard_scheme = input_memory_config.memory_layout(),
            .shard_orientation = input_memory_config.shard_spec().value().orientation};
    } else {
        ShardOrientation shard_orientation =
            conv_config.transpose_shards ? ShardOrientation::COL_MAJOR : ShardOrientation::ROW_MAJOR;

        parallel_config = determine_parallel_config(
            conv_config.shard_layout.value(),
            batch_size,
            in_channels,
            output_height,
            output_width,
            out_channels,
            input_channels_alignment,
            device->compute_with_storage_grid_size(),
            shard_orientation,
            !mm_conv,
            true,
            true,
            conv_config.act_block_h_override);

        auto [input_padded_shape, input_tensor_sharded_memory_config] = determine_input_memory_config(
            conv_config.shard_layout.value(),
            conv_config.transpose_shards ? ShardOrientation::COL_MAJOR : ShardOrientation::ROW_MAJOR,
            batch_size,
            ttnn::Shape({batch_size, input_height, input_width, in_channels}),
            ttnn::Shape({batch_size, output_height, output_width, out_channels}),
            mm_conv,
            device->compute_with_storage_grid_size(),
            input_layout,
            is_dram_conv ? BufferType::DRAM : BufferType::L1,
            parallel_config,
            conv_config.act_block_h_override);

        opt_conv_op_block_config = get_opt_block_config(
            mm_conv,
            in_channels,
            out_channels,
            output_height,
            output_width,
            batch_size,
            input_height,
            input_width,
            groups,
            kernel_size,
            stride,
            dilation,
            padding_n4,
            device,
            conv_config,
            input_layout,
            input_dtype,
            conv_output_dtype,
            compute_config,
            input_tensor_sharded_memory_config,
            has_bias);

        input_channels_alignment = get_input_channels_alignment(
            conv_config.shard_layout.value(), input_layout, false, mm_conv, input_tensor_sharded_memory_config);
    }

    auto output_compute_grid_size =
        get_output_compute_grid_size(device->compute_with_storage_grid_size(), conv_config, parallel_config);
    ParallelConfig output_parallel_config = determine_output_parallel_config(
        parallel_config, output_compute_grid_size, out_channels, parallel_config.shard_orientation, mm_conv);

    const bool auto_shard = !input_memory_config.is_sharded() && !conv_config.shard_layout.has_value();
    return Conv2dWeightsBiasPrepConfig(
        input_channels_alignment,
        conv_config.weights_dtype,
        opt_conv_op_block_config.act_block_w_ntiles,
        opt_conv_op_block_config.out_subblock_w_ntiles,
        parallel_config,
        output_parallel_config,
        groups,
        opt_conv_op_block_config.act_block_h_ntiles,
        input_height,
        input_width,
        mm_conv && (auto_shard || is_dram_conv),
        out_channels,
        has_bias,
        conv_config.enable_kernel_stride_folding.value(),
        conv_config.full_inner_dim,
        conv_config.enable_activation_reuse,
        orig_stride);
}

static ttnn::Tensor prepare_conv_weights_internal(
    const ttnn::Tensor& weight_tensor, Conv2dWeightsBiasPrepConfig& params, MeshDevice* device) {
    ttnn::Tensor weight_tensor_ = weight_tensor;  // tensor to return
    Shape weight_shape = weight_tensor.logical_shape();
    // In case of 1D convolution and 3D weight tensor, reinterpret it as 4D tensor
    if (weight_shape.rank() == 3 && params.input_height == 1) {
        weight_tensor_ = ttnn::reshape(weight_tensor_, Shape({weight_shape[0], weight_shape[1], 1, weight_shape[2]}));
    }
    validate_host_conv_weights(weight_tensor_);
    log_trace(tt::LogOp, "Prepare Conv Weights with params: {}", params);
    const auto& original_weights_shape = weight_tensor_.logical_shape();
    uint32_t original_weights_out_channels = original_weights_shape[0];
    uint32_t original_weights_in_channels = original_weights_shape[1];
    uint32_t original_weights_window_h = original_weights_shape[2];
    uint32_t original_weights_window_w = original_weights_shape[3];

    const bool is_conv1d = is_1d_conv(original_weights_window_h, params.input_height);
    const bool is_conv_1d_depthwise_conv = is_1d_depthwise_conv(
        params.groups,
        original_weights_in_channels * params.groups,
        original_weights_out_channels,
        original_weights_window_h,
        original_weights_window_w,
        params.input_height,
        params.has_bias);
    // Convert weight tensor to 0 padded shape if groups > 1
    if (!is_conv1d and params.groups > 1) {
        weight_tensor_ =
            convert_conv_weight_tensor_to_grouped_layout(weight_tensor_, params.groups, weight_tensor_.dtype());
    } else if (is_conv1d and params.groups > 1) {
        if (is_conv_1d_depthwise_conv) {
            weight_tensor_ = convert_conv_weight_tensor_to_depthwise_layout(
                weight_tensor_, params.act_block_h_ntiles, weight_tensor_.dtype());
            // After depthwise conversion, in_channels = act_block_h_ntiles * TILE_HEIGHT
            // inner_dim = in_channels * kernel_w = act_block_h_ntiles * TILE_HEIGHT * kernel_w
            // weight_block_h_ntiles * TILE_HEIGHT must >= inner_dim
            // So: weight_block_h_ntiles >= act_block_h_ntiles * kernel_w
            params.weight_block_h_ntiles = params.act_block_h_ntiles * original_weights_window_w;
        } else {
            weight_tensor_ =
                convert_conv_weight_tensor_to_grouped_layout(weight_tensor_, params.groups, weight_tensor_.dtype());
        }
    }
    if (params.enable_kernel_stride_folding) {
        weight_tensor_ = to_folded_weight_layout(weight_tensor_, params.stride);
    }
    const auto& weights_shape = weight_tensor_.logical_shape();
    uint32_t out_channels = weights_shape[0];
    uint32_t in_channels = weights_shape[1];
    uint32_t window_h = weights_shape[2];
    uint32_t window_w = weights_shape[3];

    TT_FATAL(
        out_channels == original_weights_out_channels,
        "Weight transformation changed output channels from {} to {}. Update bias preparation logic.",
        original_weights_out_channels,
        out_channels);

    uint32_t in_channels_padded = tt::round_up(in_channels, params.input_channels_alignment);
    uint32_t out_channels_padded = tt::round_up(out_channels, constants::TILE_WIDTH);

    uint32_t out_channel_padding = out_channels_padded - out_channels;

    // for conv op, pad the weights to block shape
    if (params.interleaved_mm_conv) {
        // Use interleaved MM layout conversion: [Co, Ci, Kh, Kw] -> [1, 1, KhKwCi, Co] and tilize
        weight_tensor_ = convert_conv_weight_tensor_to_interleaved_mm_layout(weight_tensor_, weight_tensor_.dtype());
    } else {
        auto input_parallel_config = params.input_parallel_config.value();
        auto output_parallel_config = params.output_parallel_config.value();
        uint32_t input_num_cores_channels = get_num_cores_channels_from_parallel_config(input_parallel_config);
        uint32_t output_num_cores_channels = get_num_cores_channels_from_parallel_config(output_parallel_config);
        in_channels_padded = tt::round_up(in_channels, input_num_cores_channels * params.input_channels_alignment);
        out_channels_padded = calculate_out_channels_padded(out_channels, output_parallel_config);
        out_channel_padding = out_channels_padded - out_channels;
        ttnn::Shape weights_channels_padded_shape({out_channels_padded, in_channels_padded, window_h, window_w});

        weight_tensor_ = ttnn::pad(
            weight_tensor_, weights_channels_padded_shape.to_array_4D(), tt::tt_metal::Array4D({0, 0, 0, 0}), 0);

        if (input_parallel_config.shard_scheme == TensorMemoryLayout::HEIGHT_SHARDED) {
            weight_tensor_ = convert_conv_weight_tensor_to_special_padding_tiled_layout(
                weight_tensor_,
                params.weight_block_h_ntiles,
                params.weight_block_w_ntiles,
                params.enable_activation_reuse,
                weight_tensor_.dtype());
        } else if (input_parallel_config.shard_scheme == TensorMemoryLayout::BLOCK_SHARDED) {
            weight_tensor_ = convert_conv_weight_tensor_to_tiled_layout_block_sharded(
                weight_tensor_,
                input_num_cores_channels,
                output_num_cores_channels,
                params.full_inner_dim,
                weight_tensor_.dtype());
        } else if (input_parallel_config.shard_scheme == TensorMemoryLayout::WIDTH_SHARDED) {
            weight_tensor_ = convert_conv_weight_tensor_to_tiled_layout(
                weight_tensor_, params.weight_block_h_ntiles, params.weight_block_w_ntiles, weight_tensor_.dtype());
        } else {
            TT_THROW("Unsupported conv weights params : {}", params);
        }
    }

    uint32_t weight_matrix_height = in_channels * window_h * window_w;
    TT_FATAL(weight_tensor_.logical_shape()[2] >= weight_matrix_height, " Matrix Height Padding can't be negative");
    ttnn::Shape target_shape({1, 1, weight_matrix_height, out_channels});
    ttnn::Shape padded_target_shape({1, 1, weight_tensor_.logical_shape()[2], out_channels + out_channel_padding});
    weight_tensor_ = ttnn::reshape(weight_tensor_, target_shape, padded_target_shape);
    if (params.weights_bias_dtype.has_value()) {
        weight_tensor_ = ttnn::to_dtype(weight_tensor_, params.weights_bias_dtype.value());
    }

    // Always move parameters to device
    weight_tensor_ = ttnn::operations::core::to_device(weight_tensor_, device, std::nullopt);

    return weight_tensor_;
}

std::optional<ttnn::Tensor> prepare_conv_bias_internal(
    const std::optional<const ttnn::Tensor>& bias_tensor,
    uint32_t out_channels,
    const Conv2dWeightsBiasPrepConfig& params,
    DataType weight_dtype,
    MeshDevice* device) {
    if (!bias_tensor.has_value()) {
        return std::optional<ttnn::Tensor>();
    }

    ttnn::Tensor bias_tensor_ = bias_tensor.value();
    bool is_bias_tensor_is_on_device = tt::tt_metal::is_device_tensor(bias_tensor_);
    if (!is_bias_tensor_is_on_device) {
        TT_FATAL(bias_tensor_.logical_shape()[3] == out_channels, "Bias must have the same length as output channels");
        uint32_t out_channels_padded = tt::round_up(out_channels, constants::TILE_WIDTH);
        if (params.output_parallel_config.has_value()) {
            out_channels_padded = calculate_out_channels_padded(out_channels, params.output_parallel_config.value());
        }
        // Inline the operations from conv_bias_layout_convert
        validate_host_conv_bias(bias_tensor_);
        ttnn::Shape bias_channels_padded_shape(
            {1, 1, 32, round_up(out_channels_padded, params.weight_block_w_ntiles * 32)});
        bias_tensor_ =
            ttnn::pad(bias_tensor_, bias_channels_padded_shape.to_array_4D(), tt::tt_metal::Array4D{0, 0, 0, 0}, 0);
        bias_tensor_ = ttnn::to_layout(bias_tensor_, Layout::TILE);
        if (bias_tensor_.dtype() != weight_dtype) {
            bias_tensor_ = ttnn::to_dtype(bias_tensor_, weight_dtype);
        }
        bias_tensor_ = ttnn::operations::core::to_device(bias_tensor_, device, std::nullopt);
    }
    TT_ASSERT(bias_tensor_.dtype() == weight_dtype, "Bias tensor should be in the same dtype as the weights tensor");

    return bias_tensor_;
}

std::pair<ttnn::Tensor, std::optional<ttnn::Tensor>> prepare_conv_weights_biases_and_move_to_device(
    const ttnn::Tensor& weight_tensor,
    const std::optional<const ttnn::Tensor>& bias_tensor,
    Conv2dWeightsBiasPrepConfig& params,
    MeshDevice* device) {
    // Prepare weights
    ttnn::Tensor weight_tensor_prepared = prepare_conv_weights_internal(weight_tensor, params, device);

    // Prepare bias if provided
    std::optional<ttnn::Tensor> bias_tensor_prepared =
        prepare_conv_bias_internal(bias_tensor, params.out_channels, params, weight_tensor_prepared.dtype(), device);

    return {weight_tensor_prepared, bias_tensor_prepared};
}

ttnn::Tensor prepare_conv_weights(
    const ttnn::Tensor& weight_tensor,
    const ttnn::MemoryConfig& input_memory_config,
    Layout input_layout,
    const std::string& weights_format,
    uint32_t in_channels,
    uint32_t out_channels,
    uint32_t batch_size,
    uint32_t input_height,
    uint32_t input_width,
    std::array<uint32_t, 2> kernel_size,
    std::array<uint32_t, 2> stride,
    std::variant<std::array<uint32_t, 2>, std::array<uint32_t, 4>> padding,
    std::array<uint32_t, 2> dilation,
    const bool has_bias,
    uint32_t groups,
    MeshDevice* device,
    DataType input_dtype,
    const std::optional<const DataType>& output_dtype,
    const std::optional<const Conv2dConfig>& conv_config_,
    const std::optional<const DeviceComputeKernelConfig>& compute_config_,
    const std::optional<const Conv2dSliceConfig>& dram_slice_config_) {
    if (weights_format != "OIHW") {
        log_warning(
            tt::LogOp,
            "PyTorch expects Conv2D Weights in OIHW format, but got {}. If you have passed the correct weights, then "
            "make sure that the weights_format string is set to \"OIHW\".",
            weights_format);
    }

    Conv2dConfig conv_config = conv_config_.value_or(Conv2dConfig());

    if (!conv_config.weights_dtype.has_value()) {
        log_warning(
            tt::LogOp,
            "Conv2D prepare_weights was called with conv_config.weights_dtype not set. \n weights_dtype will be set to "
            "the dtype of the input weights tensor. \n Weights & Bias must be the same dtype, so ensure that "
            "conv_weights_dtype is set to the same dtype before calling prepare_bias.");
        conv_config.weights_dtype = weight_tensor.dtype();
    }

    DeviceComputeKernelConfig compute_config = compute_config_.value_or(
        get_conv_default_compute_kernel_config(device, input_dtype, conv_config.weights_dtype.value()));
    // Use common setup function to get configuration parameters
    Conv2dWeightsBiasPrepConfig params = setup_conv_prep_config(
        input_memory_config,
        input_layout,
        in_channels,
        out_channels,
        batch_size,
        input_height,
        input_width,
        kernel_size,
        stride,
        padding,
        dilation,
        has_bias,
        groups,
        device,
        conv_config,
        compute_config,
        input_dtype,
        output_dtype,
        dram_slice_config_);

    // Use internal API to prepare weights
    return prepare_conv_weights_internal(weight_tensor, params, device);
}

ttnn::Tensor prepare_conv_bias(
    const ttnn::Tensor& bias_tensor,
    const ttnn::MemoryConfig& input_memory_config,
    Layout input_layout,
    uint32_t in_channels,
    uint32_t out_channels,
    uint32_t batch_size,
    uint32_t input_height,
    uint32_t input_width,
    std::array<uint32_t, 2> kernel_size,
    std::array<uint32_t, 2> stride,
    std::variant<std::array<uint32_t, 2>, std::array<uint32_t, 4>> padding,
    std::array<uint32_t, 2> dilation,
    uint32_t groups,
    MeshDevice* device,
    DataType input_dtype,
    const std::optional<const DataType>& output_dtype,
    const std::optional<const Conv2dConfig>& conv_config_,
    const std::optional<const DeviceComputeKernelConfig>& compute_config_,
    const std::optional<const Conv2dSliceConfig>& /*dram_slice_config_*/) {
    TT_FATAL(!ttnn::has_storage_type_of(bias_tensor, ttnn::DEVICE_STORAGE_TYPE), "conv bias should be placed on host");
    Conv2dConfig conv_config = conv_config_.value_or(Conv2dConfig());

    TT_ASSERT(conv_config.weights_dtype.has_value(), "prepare_conv_bias requires conv_config.weights_dtype to be set.");

    DeviceComputeKernelConfig compute_config = compute_config_.value_or(
        get_conv_default_compute_kernel_config(device, input_dtype, conv_config.weights_dtype.value()));

    // Use common setup function to get configuration parameters
    auto params = setup_conv_prep_config(
        input_memory_config,
        input_layout,
        in_channels,
        out_channels,
        batch_size,
        input_height,
        input_width,
        kernel_size,
        stride,
        padding,
        dilation,
        true,  // has_bias = true for bias preparation
        groups,
        device,
        conv_config,
        compute_config,
        input_dtype,
        output_dtype);

    // Use internal API to prepare bias
    auto prepared_bias = prepare_conv_bias_internal(
        std::optional<const ttnn::Tensor>(bias_tensor),
        out_channels,
        params,
        conv_config.weights_dtype.value(),
        device);

    return prepared_bias.value();  // We know bias exists since we passed it
}

}  // namespace conv2d
}  // namespace ttnn::operations::conv
