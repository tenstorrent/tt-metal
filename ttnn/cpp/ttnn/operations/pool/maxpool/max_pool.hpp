// SPDX-FileCopyrightText: © 2023 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include "ttnn/core.hpp"
#include "ttnn/types.hpp"
#include "ttnn/tensor/tensor.hpp"
#include "ttnn/operations/conv/conv2d/conv2d.hpp"
#include "ttnn/run_operation.hpp"
#include "ttnn/operations/sliding_window/sliding_window.hpp"


namespace tt {
namespace tt_metal {

inline uint32_t ceil_multiple_of(uint32_t n, uint32_t m) {
    return (uint32_t) std::ceil((float) n / m) * m;
}

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

Tensor max_pool2d_legacy(const Tensor &input, const Tensor &reader_indices,
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

}  // namespace tt_metal
}  // namespace tt
