// SPDX-FileCopyrightText: © 2023 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include "ttnn/core.hpp"
#include "ttnn/types.hpp"
#include "tensor/tensor.hpp"
#include "ttnn/cpp/ttnn/operations/conv2d.hpp"
#include "ttnn/experimental/tt_dnn/op_library/run_operation.hpp"
#include "ttnn/experimental/tt_dnn/op_library/sliding_window_op_infra/sliding_window.hpp"


inline uint32_t ceil_multiple_of(uint32_t n, uint32_t m) {
    return (uint32_t) std::ceil((float) n / m) * m;
}

namespace tt {
namespace tt_metal {

struct MaxPool {
    uint32_t in_n_; // nbatch
    uint32_t in_h_, in_w_;
    uint32_t out_h_, out_w_;
    uint32_t kernel_size_h_, kernel_size_w_;
    uint32_t stride_h_, stride_w_;
    uint32_t pad_h_, pad_w_;
    uint32_t dilation_h_, dilation_w_;
    MemoryConfig out_mem_config_;
    uint32_t nblocks_;
    bool use_multicore_;

    void validate(const std::vector<Tensor> &input_tensors) const;
    std::vector<tt::tt_metal::Shape> compute_output_shapes(const std::vector<Tensor> &input_tensors) const;
    std::vector<Tensor> create_output_tensors(const std::vector<Tensor> &input_tensors) const;
    operation::ProgramWithCallbacks create_program(const std::vector<Tensor>& input_tensors, std::vector<Tensor> &output_tensors) const;
    operation::OpPerformanceModel create_op_performance_model(const std::vector<Tensor>& input_tensors, const std::vector<std::optional<const Tensor>>& optional_input_tensors, const std::vector<std::optional<Tensor>>& optional_output_tensors, const std::vector<Tensor> &output_tensors) const;

    static constexpr auto attribute_names = std::make_tuple(
        "in_n",
        "in_h",
        "in_w",
        "kernel_size_h",
        "kernel_size_w",
        "stride_h",
        "stride_w",
        "pad_h",
        "pad_w",
        "dilation_h",
        "dilation_w",
        "out_mem_config",
        "nblocks",
        "use_multicore");
    const auto attribute_values() const {
        return std::make_tuple(
            std::cref(this->in_n_),
            std::cref(this->in_h_),
            std::cref(this->in_w_),
            std::cref(this->kernel_size_h_),
            std::cref(this->kernel_size_w_),
            std::cref(this->stride_h_),
            std::cref(this->stride_w_),
            std::cref(this->pad_h_),
            std::cref(this->pad_w_),
            std::cref(this->dilation_h_),
            std::cref(this->dilation_w_),
            std::cref(this->out_mem_config_),
            std::cref(this->nblocks_),
            std::cref(this->use_multicore_));
    }
};

operation::ProgramWithCallbacks max_pool_2d_single_core(const Tensor &input, Tensor& output,
                                                        uint32_t in_h, uint32_t in_w,
                                                        uint32_t out_h, uint32_t out_w,
                                                        uint32_t kernel_size_h, uint32_t kernel_size_w,
                                                        uint32_t stride_h, uint32_t stride_w,
                                                        uint32_t pad_h, uint32_t pad_w,
                                                        uint32_t dilation_h, uint32_t dilation_w,
                                                        const MemoryConfig& out_mem_config,
                                                        uint32_t nblocks);
operation::ProgramWithCallbacks max_pool_2d_multi_core(const Tensor &input, Tensor& output,
                                                       uint32_t in_h, uint32_t in_w,
                                                       uint32_t out_h, uint32_t out_w,
                                                       uint32_t kernel_size_h, uint32_t kernel_size_w,
                                                       uint32_t stride_h, uint32_t stride_w,
                                                       uint32_t pad_h, uint32_t pad_w,
                                                       uint32_t dilation_h, uint32_t dilation_w,
                                                       const MemoryConfig& out_mem_config,
                                                       uint32_t nblocks);
operation::ProgramWithCallbacks max_pool_2d_multi_core_generic(const Tensor &input, Tensor& output,
                                                                uint32_t in_h, uint32_t in_w,
                                                                uint32_t out_h, uint32_t out_w,
                                                                uint32_t kernel_size_h, uint32_t kernel_size_w,
                                                                uint32_t stride_h, uint32_t stride_w,
                                                                uint32_t pad_h, uint32_t pad_w,
                                                                uint32_t dilation_h, uint32_t dilation_w,
                                                                const MemoryConfig& out_mem_config,
                                                                uint32_t nblocks);
namespace deprecated {
operation::ProgramWithCallbacks max_pool_2d_multi_core_sharded_with_halo(const Tensor &input, Tensor& output,
                                                                uint32_t in_n, uint32_t in_h, uint32_t in_w,
                                                                uint32_t out_h, uint32_t out_w,
                                                                uint32_t kernel_size_h, uint32_t kernel_size_w,
                                                                uint32_t stride_h, uint32_t stride_w,
                                                                uint32_t pad_h, uint32_t pad_w,
                                                                uint32_t dilation_h, uint32_t dilation_w,
                                                                const MemoryConfig& out_mem_config,
                                                                uint32_t nblocks);
} // namespace deprecated

operation::ProgramWithCallbacks max_pool_2d_multi_core_sharded_with_halo_v2(const Tensor &input,
                                                                const Tensor& reader_indices,
                                                                Tensor& output,
                                                                uint32_t in_n, uint32_t in_h, uint32_t in_w,
                                                                uint32_t out_h, uint32_t out_w,
                                                                uint32_t kernel_size_h, uint32_t kernel_size_w,
                                                                uint32_t stride_h, uint32_t stride_w,
                                                                uint32_t pad_h, uint32_t pad_w,
                                                                uint32_t dilation_h, uint32_t dilation_w,
                                                                const MemoryConfig& out_mem_config,
                                                                uint32_t nblocks);
Tensor max_pool2d(const Tensor &input,
                  uint32_t in_n, uint32_t in_h, uint32_t in_w,
                  uint32_t kernel_size_h, uint32_t kernel_size_w,
                  uint32_t stride_h = 1, uint32_t stride_w = 1,
                  uint32_t pad_h = 0, uint32_t pad_w = 0,               // default: no padding
                  uint32_t dilation_h = 1, uint32_t dilation_w = 1,
                  const MemoryConfig& out_mem_config = operation::DEFAULT_OUTPUT_MEMORY_CONFIG,
                  uint32_t nblocks = 1, bool use_multicore = true);

Tensor max_pool2d_v2(const Tensor &input, const Tensor &reader_indices,
                  uint32_t in_n, uint32_t in_h, uint32_t in_w,
                  uint32_t kernel_size_h, uint32_t kernel_size_w,
                  uint32_t stride_h = 1, uint32_t stride_w = 1,
                  uint32_t pad_h = 0, uint32_t pad_w = 0,               // default: no padding
                  uint32_t dilation_h = 1, uint32_t dilation_w = 1,
                  const MemoryConfig& out_mem_config = operation::DEFAULT_OUTPUT_MEMORY_CONFIG,
                  uint32_t nblocks = 1, bool use_multicore = true);

namespace max_pool_helpers {
uint32_t get_num_cores(const Device* device, uint32_t out_nhw, uint32_t nbatch);
}

// new maxpool uop -- called from the macro-op
struct MaxPoolNew {
    SlidingWindowConfig sliding_window_config_;
    MemoryConfig out_mem_config_;

    void validate(const std::vector<Tensor> &input_tensors) const;
    std::vector<tt::tt_metal::Shape> compute_output_shapes(const std::vector<Tensor> &input_tensors) const;
    std::vector<Tensor> create_output_tensors(const std::vector<Tensor> &input_tensors) const;
    operation::ProgramWithCallbacks create_program(const std::vector<Tensor>& input_tensors, std::vector<Tensor> &output_tensors) const;
    operation::OpPerformanceModel create_op_performance_model(const std::vector<Tensor>& input_tensors, const std::vector<std::optional<const Tensor>>& optional_input_tensors, const std::vector<std::optional<Tensor>>& optional_output_tensors, const std::vector<Tensor> &output_tensors) const;

    static constexpr auto attribute_names = std::make_tuple(
        "sliding_window_config",
        "out_mem_config");
    const auto attribute_values() const {
        return std::make_tuple(
            std::cref(this->sliding_window_config_),
            std::cref(this->out_mem_config_));
    }
};

operation::ProgramWithCallbacks max_pool_2d_multi_core_sharded_with_halo_v2_new(
                                                                const Tensor &input,
                                                                Tensor& output,
                                                                const SlidingWindowConfig& sliding_window_config,
                                                                const MemoryConfig& out_mem_config);

Tensor maxpool2d_new(const Tensor &input,
                        const SlidingWindowConfig& sliding_window_config,
                        uint32_t in_c,
                        const MemoryConfig& out_mem_config = operation::DEFAULT_OUTPUT_MEMORY_CONFIG);

}  // namespace tt_metal
}  // namespace tt

namespace ttnn::operations {
namespace maxpool {

using array2_t = std::array<uint32_t, 2>;

// maxpool macro-op
inline Tensor maxpool2d(const Tensor& input_tensor, uint32_t batch_size, uint32_t input_h, uint32_t input_w, uint32_t channels, array2_t kernel_size, array2_t stride, array2_t padding, array2_t dilation, Device& device) {
    MemoryConfig memory_config = input_tensor.memory_config();
    const auto shard_grid = memory_config.shard_spec.value().grid;
    const auto shard_scheme = memory_config.memory_layout;
    const auto shard_orientation = memory_config.shard_spec.value().orientation;

    TT_FATAL(shard_scheme == TensorMemoryLayout::HEIGHT_SHARDED, "Only height sharded tensors are supported.");
    TT_FATAL(shard_orientation == ShardOrientation::ROW_MAJOR, "Only row major orientation is supported.");

    ParallelConfig parallel_config = conv2d::determine_parallel_config(
                                        shard_scheme == TensorMemoryLayout::HEIGHT_SHARDED,
                                        batch_size,
                                        0,          // in_channels -- not used
                                        input_h,
                                        input_w,
                                        0,          // out_channels -- not used
                                        device,
                                        shard_orientation);
    uint32_t num_cores_nhw = conv2d::get_num_cores_nhw_from_parallel_config(parallel_config);

    SlidingWindowConfig sliding_window_config = SlidingWindowConfig(batch_size,
                                                                    input_h, input_w,
                                                                    kernel_size.at(0), kernel_size.at(1),
                                                                    stride.at(0), stride.at(1),
                                                                    padding.at(0), padding.at(1),
                                                                    dilation.at(0), dilation.at(1),
                                                                    num_cores_nhw,
                                                                    parallel_config.grid);
    uint32_t neg_inf_pad_val = 0xf7ff;  // TODO: double check

    auto haloed_tensor = ttnn::operations::halo::halo_op(input_tensor, sliding_window_config, neg_inf_pad_val, false, parallel_config.shard_orientation == ShardOrientation::COL_MAJOR, 0, memory_config);
    return tt::tt_metal::maxpool2d_new(haloed_tensor, sliding_window_config, channels, memory_config);
}

}  // namespace maxpool
}  // namespace ttnn::operations
