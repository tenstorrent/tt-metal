// SPDX-FileCopyrightText: © 2023 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include "ttnn/operations/pool/maxpool/max_pool.hpp"

#include <algorithm>
#include <cmath>

#include "detail/util.hpp"
#include "ttnn/tensor/host_buffer/functions.hpp"
#include "ttnn/tensor/tensor_utils.hpp"
#include "ttnn/operations/core/work_split/work_split.hpp"
#include "tt_metal/host_api.hpp"

namespace tt {
namespace tt_metal {

void MaxPool::validate(const std::vector<Tensor> &input_tensors) const {
    const auto& input = input_tensors.at(0);
    TT_FATAL(input.storage_type() == StorageType::DEVICE, "Operands to reshape need to be on device!");
    TT_FATAL(input.buffer() != nullptr , "Operands to reshape need to be allocated in buffers on device!");
    TT_FATAL(input.get_dtype() == DataType::BFLOAT16, "Only BFLOAT16 supported for now");
    TT_FATAL(input.get_layout() == Layout::ROW_MAJOR, "Only ROW_MAJOR supported for now");

    // NOTE: This is not a hard requirement. If need to support non-power-of-2, simply change the address generator in reader to generic one.
    uint32_t in_nbytes_c = (input.get_legacy_shape()[3]) * (input.get_dtype() == DataType::BFLOAT16 ? 2 : 1);
    bool is_pow2 = (in_nbytes_c & (in_nbytes_c - 1)) == 0;
    TT_FATAL(is_pow2, "Row size (nchannels * bytes = {}) should be power of 2 ({}).", in_nbytes_c, is_pow2);

    TT_FATAL(2 * pad_h_ < kernel_size_h_ && 2 * pad_w_ < kernel_size_w_,
              "Total padding along a dim should be less than kernel/window size along same dim");
    TT_FATAL(out_w_ % nblocks_ == 0, "Make sure out_w is divisible by nblocks for now.");

    if (input.memory_config().is_sharded()) {
        TT_FATAL(input.memory_config().memory_layout == TensorMemoryLayout::HEIGHT_SHARDED);
        TT_FATAL(this->use_multicore_);
    } else {
        TT_FATAL(input.memory_config().memory_layout == TensorMemoryLayout::INTERLEAVED);
    }
    if (this->out_mem_config_.is_sharded()) {
        TT_FATAL(this->out_mem_config_.memory_layout == TensorMemoryLayout::HEIGHT_SHARDED);
        TT_FATAL(this->use_multicore_);
    } else {
        TT_FATAL(this->out_mem_config_.memory_layout == TensorMemoryLayout::INTERLEAVED);
    }
}

std::vector<Shape> MaxPool::compute_output_shapes(const std::vector<Tensor> &input_tensors) const {
    // NOTE: Only for RM
    // NOTE2: Assuming { N, 1, H * W, C }
    // NOTE3: Assuming output data type is same as input
    const auto& input = input_tensors.at(0);
    const auto input_shape = input.get_legacy_shape();
    // confirm that the output size supplied to the function matches
    TT_ASSERT(out_h_ == ((in_h_ + 2 * pad_h_ - (dilation_h_ * kernel_size_h_ - 1) - 1) / stride_h_) + 1);
    TT_ASSERT(out_w_ == ((in_w_ + 2 * pad_w_ - (dilation_w_ * kernel_size_w_ - 1) - 1) / stride_w_) + 1);
    uint32_t out_h = out_h_;
    uint32_t out_w = out_w_;
    // need to pad the last dim to TILE_WIDTH
    uint32_t out_c = input_shape[3];
    uint32_t out_c_padded = ceil_multiple_of(out_c, (out_c <= 16) ? 16 : constants::TILE_WIDTH);
    uint32_t out_pagesize = out_c_padded * datum_size(datatype_to_dataformat_converter(input.get_dtype()));
    bool multicore = this->out_mem_config_.shard_spec.has_value();
    if (multicore) {
        uint32_t out_nhw = in_n_ * out_h * out_w;
        uint32_t out_nhw_padded =
            this->out_mem_config_.shard_spec->shape[0] * this->out_mem_config_.shard_spec->num_cores();

        // {1, 1, N * H * W, C}
        const auto out_dims = std::vector<uint32_t>({1, 1, out_nhw_padded, out_c_padded});
        const auto padding = Padding(
            {{0, 0}, {0, 0}, {0, out_nhw_padded - out_nhw}, {0, out_c_padded - out_c}},
            Padding::PadValue::NegativeInfinity);
        auto out_shape = Shape{out_dims, padding};
        return {out_shape};
    } else {
        // Single core still uses the old layout w/ batch unfolded
        uint32_t out_hw = out_h * out_w;

        // {N, 1, H * W, C}
        const auto out_dims = std::vector<uint32_t>({in_n_, 1, out_hw, out_c_padded});
        const auto padding =
            Padding({{0, 0}, {0, 0}, {0, 0}, {0, out_c_padded - out_c}}, Padding::PadValue::NegativeInfinity);
        auto out_shape = Shape{out_dims, padding};
        return {out_shape};
    }
}

std::vector<Tensor> MaxPool::create_output_tensors(const std::vector<Tensor> &inputs) const {
    const auto& input = inputs.at(0);
    if (out_mem_config_.is_sharded()) {
        Shape output_shape = compute_output_shapes(inputs).at(0);
        auto mem_config = this->out_mem_config_;
        if (mem_config.shard_spec.has_value()) {
            mem_config.shard_spec->shape[1] = output_shape[3];
        } else {
            uint32_t nbatch = output_shape[0];
            uint32_t out_nhw = output_shape[0] * output_shape[1] * output_shape[2];
            uint32_t ncores = 1;
            if (input.shard_spec().has_value() && input.shard_spec().value().halo) {
                ncores = input.shard_spec().value().num_cores();
            } else {
                ncores = max_pool_helpers::get_num_cores(input.device(), out_nhw, nbatch);
            }
            uint32_t out_nhw_per_core = out_nhw / ncores;
            CoreRangeSet shard_grid = ttnn::num_cores_to_corerange_set(ncores, input.device()->compute_with_storage_grid_size(), true);
            std::array<uint32_t, 2> shard_shape = {out_nhw_per_core, input.get_legacy_shape()[-1]};
            mem_config.shard_spec = ShardSpec{shard_grid, shard_shape, ShardOrientation::ROW_MAJOR, false};
        }
        return {create_device_tensor(
            output_shape, input.get_dtype(), input.get_layout(), input.device(), mem_config)};
    } else {
        return operation::generic_create_output_tensors(*this, inputs, input.get_dtype(), input.get_layout(), out_mem_config_);
    }
}

operation::ProgramWithCallbacks MaxPool::create_program(const std::vector<Tensor>& inputs, std::vector<Tensor> &outputs) const {
    const auto& input = inputs.at(0);
    auto& output = outputs.at(0);
    if (inputs.size() > 1) {
        const auto& reader_indices = inputs.at(1);
        TT_FATAL(use_multicore_, "UTWHv2 only works with multicore option.");
        TT_FATAL(input.memory_config().is_sharded(), "Input needs to be sharded for UTWHv2");
        return {max_pool_2d_multi_core_sharded_with_halo_v2(
                                    input, reader_indices,
                                    output,
                                    in_n_, in_h_, in_w_,
                                    out_h_, out_w_,
                                    kernel_size_h_, kernel_size_w_,
                                    stride_h_, stride_w_,
                                    pad_h_, pad_w_,
                                    dilation_h_, dilation_w_,
                                    out_mem_config_,
                                    nblocks_)};
    } else {
        if (!use_multicore_) {
            return {max_pool_2d_single_core(input, output,
                                            in_h_, in_w_,
                                            out_h_, out_w_,
                                            kernel_size_h_, kernel_size_w_,
                                            stride_h_, stride_w_,
                                            pad_h_, pad_w_,
                                            dilation_h_, dilation_w_,
                                            out_mem_config_,
                                            nblocks_)};
        } else {
            if (input.memory_config().is_sharded() && input.shard_spec().value().halo) {
                log_fatal("This version of max_pool sharded with halo has been deprecated. Please use v2.");
                return {};
            }
            return {max_pool_2d_multi_core_generic(input, output,
                                        in_h_, in_w_,
                                        out_h_, out_w_,
                                        kernel_size_h_, kernel_size_w_,
                                        stride_h_, stride_w_,
                                        pad_h_, pad_w_,
                                        dilation_h_, dilation_w_,
                                        out_mem_config_,
                                        nblocks_)};
        }
    }
}

Tensor max_pool2d(const Tensor &input,
                  uint32_t in_n, uint32_t in_h, uint32_t in_w,
                  uint32_t kernel_size_h, uint32_t kernel_size_w,
                  uint32_t stride_h, uint32_t stride_w,
                  uint32_t pad_h, uint32_t pad_w,
                  uint32_t dilation_h, uint32_t dilation_w,
                  const MemoryConfig& out_mem_config,
                  uint32_t nblocks,
                  bool use_multicore) {
    TT_ASSERT(dilation_h == 1 && dilation_w == 1 && "Dilation not yet supported in max_pool2d.");
    TT_ASSERT(pad_h < 2 && pad_w < 2 && "Padding > 1 not yet supported.");
    TT_ASSERT(stride_h == stride_w && "Stride should be equal for both H and W for now.");
    // calculate the H and W dims for output
    uint32_t out_h = ((in_h + 2 * pad_h - (dilation_h * kernel_size_h - 1) - 1) / stride_h) + 1;   // floor
    uint32_t out_w = ((in_w + 2 * pad_w - (dilation_w * kernel_size_w - 1) - 1) / stride_w) + 1;   // floor
    return operation::run_without_autoformat(MaxPool{in_n, in_h, in_w,
                                                     out_h, out_w,
                                                     kernel_size_h, kernel_size_w,
                                                     stride_h, stride_w,
                                                     pad_h, pad_w,
                                                     dilation_h, dilation_w,
                                                     out_mem_config,
                                                     nblocks,
                                                     use_multicore},
                                             {input}).at(0);
}

Tensor max_pool2d_legacy(const Tensor &input,
                  const Tensor &reader_indices,
                  uint32_t in_n, uint32_t in_h, uint32_t in_w,
                  uint32_t kernel_size_h, uint32_t kernel_size_w,
                  uint32_t stride_h, uint32_t stride_w,
                  uint32_t pad_h, uint32_t pad_w,
                  uint32_t dilation_h, uint32_t dilation_w,
                  const MemoryConfig& out_mem_config,
                  uint32_t nblocks,
                  bool use_multicore) {
    std::vector<Tensor> output_tensors = {Tensor(operation::get_workers_for_op_output({input, reader_indices}))};
    operation::launch_op(
        [in_n, in_h, in_w, kernel_size_h, kernel_size_w, stride_h, stride_w, pad_h, pad_w, dilation_h, dilation_w, out_mem_config, nblocks, use_multicore]
            (const std::vector<Tensor>& input_tensors, const std::vector<std::optional<const Tensor>>& optional_input_tensors, const std::vector<std::optional<Tensor>>& optional_output_tensors) mutable -> std::vector<Tensor> {
                TT_ASSERT(dilation_h == 1 && dilation_w == 1 && "Dilation not yet supported in max_pool2d.");
                TT_ASSERT(pad_h < 2 && pad_w < 2 && "Padding > 1 not yet supported.");
                TT_ASSERT(stride_h == stride_w && "Stride should be equal for both H and W for now.");
                // calculate the H and W dims for output
                uint32_t out_h = ((in_h + 2 * pad_h - (dilation_h * kernel_size_h - 1) - 1) / stride_h) + 1;   // floor
                uint32_t out_w = ((in_w + 2 * pad_w - (dilation_w * kernel_size_w - 1) - 1) / stride_w) + 1;   // floor
                return operation::run_without_autoformat(MaxPool{in_n, in_h, in_w,
                                                                out_h, out_w,
                                                                kernel_size_h, kernel_size_w,
                                                                stride_h, stride_w,
                                                                pad_h, pad_w,
                                                                dilation_h, dilation_w,
                                                                out_mem_config,
                                                                nblocks,
                                                                use_multicore},
                                                        input_tensors);
            }, {input, reader_indices}, output_tensors);
    return output_tensors.at(0);
}

operation::OpPerformanceModel MaxPool::create_op_performance_model(const std::vector<Tensor>& input_tensors, const std::vector<std::optional<const Tensor>>& optional_input_tensors, const std::vector<std::optional<Tensor>>& optional_output_tensors, const std::vector<Tensor> &output_tensors) const {
    const auto& input = input_tensors.at(0);
    const auto& input_shape = input.get_shape();
    uint32_t batch_size = this->in_n_;
    uint32_t conv_activation_h = this->in_h_;
    uint32_t conv_activation_w = this->in_w_;
    uint32_t conv_activation_c = input_shape[3];
    uint32_t output_channels = input_shape[3];

    uint32_t filter_h = (uint32_t) this->kernel_size_h_;
    uint32_t filter_w = (uint32_t) this->kernel_size_w_;
    uint32_t stride_h = (uint32_t) this->stride_h_;
    uint32_t stride_w = (uint32_t) this->stride_w_;
    uint32_t pad_h = (uint32_t) this->pad_h_;
    uint32_t pad_w = (uint32_t) this->pad_w_;

    // GS specific parameters
    int num_cores = 9 * 12;
    int tensix_mul_adds_per_cycle_lofi = 2048;

    // Calculate output dimensions: relevant for window/stride based OPs (conv, maxpool, downsample)
    int output_height = std::floor((conv_activation_h - filter_h + 2 * pad_h) / stride_h + 1);
    int output_width = std::floor((conv_activation_w - filter_w + 2 * pad_w) / stride_w + 1);

    // Calculate number of mul/add / compare operations
    int64_t num_mul_adds_per_elem = conv_activation_c * filter_h * filter_w; // 1 multiply and 1 add per element
    int64_t num_mul_adds = num_mul_adds_per_elem * output_height * output_width * output_channels * batch_size;

    int ideal_dev_clock_cycles = std::ceil((float)num_mul_adds / (float)(num_cores * tensix_mul_adds_per_cycle_lofi));

    operation::OpPerformanceModel result(input_tensors, output_tensors, ideal_dev_clock_cycles);
    return result;
}


} // namespace tt_metal
} // namespace tt
