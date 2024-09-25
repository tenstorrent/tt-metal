// SPDX-FileCopyrightText: © 2023 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include "ttnn/deprecated/tt_dnn/op_library/moreh_dot/moreh_dot_op.hpp"
#include "ttnn/deprecated/tt_dnn/op_library/moreh_matmul/moreh_matmul_op.hpp"
#include "ttnn/deprecated/tt_dnn/op_library/moreh_helper_functions.hpp"
#include "tt_metal/common/constants.hpp"
#include "tt_metal/host_api.hpp"

using namespace tt::constants;

namespace tt {
namespace operations {
namespace primary {

////////////////////////////////////////////////////////////////////////////
//                         Util
////////////////////////////////////////////////////////////////////////////
namespace {
inline bool is_dot_forward(const Tensor& input, const Tensor& other, bool transpose_input, bool transpose_other) {
    // TODO: non-4d support for dot.
    if (input.get_shape().with_tile_padding().rank() != 4 || other.get_shape().with_tile_padding().rank() != 4) {
        return false;
    }

    if (transpose_input || transpose_other) {
        return false;
    }

    return is_1d_tensor(input) && is_1d_tensor(other) && is_same_shape(input, other);
}

ttnn::Shape compute_output_shape(
    const ttnn::Shape& input_shape_wo_padding, const ttnn::Shape& other_shape_wo_padding, bool transpose_input, bool transpose_other) {
    const auto& input_shape = input_shape_wo_padding.with_tile_padding();
    const auto& other_shape = other_shape_wo_padding.with_tile_padding();

    auto h = (transpose_input) ? (input_shape[-1]) : (input_shape[-2]);
    auto w = (transpose_other) ? (other_shape[-2]) : (other_shape[-1]);
    auto h_wo_padding = (transpose_input) ? (input_shape_wo_padding[-1]) : (input_shape_wo_padding[-2]);
    auto w_wo_padding = (transpose_other) ? (other_shape_wo_padding[-2]) : (other_shape_wo_padding[-1]);

    std::vector<uint32_t> input_dim(tt::tt_metal::MAX_NUM_DIMENSIONS, 1);
    std::vector<uint32_t> other_dim(tt::tt_metal::MAX_NUM_DIMENSIONS, 1);
    get_tensor_dim(input_dim, input_shape);
    get_tensor_dim(other_dim, other_shape);

    int32_t output_rank = std::max(input_shape.rank(), other_shape.rank());
    log_debug(
        LogOp,
        "{}:{} input, other, output rank {}, {}, {}",
        __func__,
        __LINE__,
        input_shape.rank(),
        other_shape.rank(),
        output_rank);

    std::vector<uint32_t> output_dim(output_rank);
    // batch dims
    for (int i = 0; i < output_rank - 2; ++i) {
        int idx = output_rank - 1 - i;
        TT_ASSERT(idx >= 0);
        uint32_t max_dim = std::max(input_dim[idx], other_dim[idx]);
        output_dim[i] = max_dim;
    }
    // matrix dims
    output_dim[output_rank - 2] = h;
    output_dim[output_rank - 1] = w;

    ttnn::Shape output_shape{output_dim};
    auto padding = output_shape.padding();
    // padding for t logmatrix dims
    padding[output_rank - 2] = Padding::PadDimension{0, h - h_wo_padding};
    padding[output_rank - 1] = Padding::PadDimension{0, w - w_wo_padding};
    return {ttnn::Shape(output_shape, padding)};
}

}  // namespace

void get_tensor_dim(std::vector<uint32_t>& dim, const ttnn::Shape& shape) {
    const auto rank = shape.rank();
    for (auto i = 0; i < rank; ++i) {
        auto idx = rank - 1 - i;

        // last 2-dim
        if (idx == rank - 1 || idx == rank - 2) {
            dim[i] = shape[idx] / TILE_HEIGHT;
        } else {
            dim[i] = shape[idx];
        }
    }

    log_debug(LogOp, "rank {}", rank);
    for (auto i = 0; i < tt::tt_metal::MAX_NUM_DIMENSIONS; ++i) {
        log_debug(LogOp, "dim[{}] = {}", i, dim[i]);
    }
}

std::vector<int64_t> find_reduce_dim(const ttnn::Shape& a_shape, const ttnn::Shape& b_shape) {
    std::vector<uint32_t> a_dim(tt::tt_metal::MAX_NUM_DIMENSIONS, 1);
    std::vector<uint32_t> b_dim(tt::tt_metal::MAX_NUM_DIMENSIONS, 1);
    get_tensor_dim(a_dim, a_shape);
    get_tensor_dim(b_dim, b_shape);
    int32_t rank = std::max(a_shape.rank(), b_shape.rank());
    log_debug(LogOp, "find_reduce_dim :{} rank {} a {} b {}", __LINE__, rank, a_shape.rank(), b_shape.rank());
    std::vector<int64_t> dims;
    // batch dims
    for (int i = 0; i < rank - 2; ++i) {
        int idx = rank - 1 - i;
        TT_ASSERT(idx >= 0);
        if (a_dim[idx] != b_dim[idx]) {
            dims.push_back(i);
            log_debug(LogOp, "find_reduce_dim :{} push {} dim", __LINE__, i);
        }
    }
    return dims;
}

bool is_same_batch_dim(const Tensor& tensor_a, const Tensor& tensor_b) {
    // check batch dims
    const auto& a_shape = tensor_a.get_shape().with_tile_padding();
    const auto& b_shape = tensor_b.get_shape().with_tile_padding();
    std::vector<uint32_t> a_dim(tt::tt_metal::MAX_NUM_DIMENSIONS, 1);
    std::vector<uint32_t> b_dim(tt::tt_metal::MAX_NUM_DIMENSIONS, 1);
    get_tensor_dim(a_dim, a_shape);
    get_tensor_dim(b_dim, b_shape);
    for (auto i = 2; i < tt::tt_metal::MAX_NUM_DIMENSIONS; ++i) {
        if (a_dim[i] != b_dim[i]) {
            log_debug(LogOp, "{}:{} {} a_dim {} - b_dim {}", __func__, __LINE__, i, a_dim[i], b_dim[i]);
            return false;
        }
    }
    log_debug(LogOp, "{}:{} batch dims are the same.", __func__, __LINE__);
    return true;
}

operation::ProgramWithCallbacks MorehMatmul::create_program(
    const std::vector<Tensor>& input_tensors,
    const std::vector<std::optional<const Tensor>>& optional_input_tensors,
    std::vector<Tensor>& output_tensors) const {
    const auto& input_tensor = input_tensors.at(0);
    const auto& other_tensor = input_tensors.at(1);
    const auto& output_tensor = output_tensors.at(0);
    const auto& bias_tensor = optional_input_tensors.at(0);
    return moreh_matmul_multi_core(
        input_tensor,
        other_tensor,
        output_tensor,
        bias_tensor,
        this->transpose_input,
        this->transpose_other,
        this->compute_kernel_config);
}

// Must be provided in the case where an optional output tensor was not provided
std::vector<ttnn::Shape> MorehMatmul::compute_output_shapes(const std::vector<Tensor>& input_tensors) const {
    return {compute_output_shape(
        input_tensors.at(0).get_shape(),
        input_tensors.at(1).get_shape(),
        this->transpose_input,
        this->transpose_other)};
}

std::vector<Tensor> MorehMatmul::create_output_tensors(
    const std::vector<Tensor>& input_tensors, const std::vector<std::optional<Tensor>>& output_tensors) const {
    if (!output_tensors.empty() && output_tensors.at(0).has_value()) {
        return {output_tensors.at(0).value()};
    }

    return operation::generic_create_output_tensors(
        *this, input_tensors, input_tensors.at(0).get_dtype(), Layout::TILE, this->output_mem_config);
}

void MorehMatmul::validate_with_output_tensors(
    const std::vector<Tensor>& input_tensors,
    const std::vector<std::optional<const Tensor>>& optional_input_tensors,
    const std::vector<std::optional<Tensor>>& output_tensors) const {
    log_debug(LogOp, "{}:{}", __func__, __LINE__);

    const auto& input = input_tensors.at(0);
    const auto& other = input_tensors.at(1);
    const auto& bias = optional_input_tensors.at(0);
    const auto& output = output_tensors.at(0);

    // validate tensor
    check_tensor(input, "moreh_matmul", "input");
    check_tensor(other, "moreh_matmul", "other");
    check_tensor(output, "moreh_matmul", "output");
    check_tensor(bias, "moreh_matmul", "bias");

    // check matrix dims
    const auto& input_shape = input.get_shape();
    const auto& other_shape = other.get_shape();
    const auto& input_wo_shape = input_shape;
    const auto& other_wo_shape = other_shape;
    uint32_t input_m = (this->transpose_input) ? (input_wo_shape[-1]) : (input_wo_shape[-2]);
    uint32_t input_k = (this->transpose_input) ? (input_wo_shape[-2]) : (input_wo_shape[-1]);
    uint32_t other_k = (this->transpose_other) ? (other_wo_shape[-1]) : (other_wo_shape[-2]);
    uint32_t other_n = (this->transpose_other) ? (other_wo_shape[-2]) : (other_wo_shape[-1]);

    TT_FATAL(input_k == other_k, "k must be the same. input_k {}, other_k {}", input_k, other_k);

    // check batch dims
    std::vector<uint32_t> input_dim(tt::tt_metal::MAX_NUM_DIMENSIONS, 1);
    std::vector<uint32_t> other_dim(tt::tt_metal::MAX_NUM_DIMENSIONS, 1);
    get_tensor_dim(input_dim, input_shape);
    get_tensor_dim(other_dim, other_shape);
    for (auto i = 2; i < tt::tt_metal::MAX_NUM_DIMENSIONS; ++i) {
        if (input_dim[i] != other_dim[i]) {
            TT_FATAL(input_dim[i] == 1 || other_dim[i] ==1, "one of dim must be one. {}th dim input_dim {}, other_dim {}", i, input_dim[i], other_dim[i]);
        }
    }

    // check output dims
    if (output.has_value()) {
        const auto& output_shape = output.value().get_shape();
        const auto& output_wo_shape = output_shape;
        uint32_t output_m = output_wo_shape[-2];
        uint32_t output_n = output_wo_shape[-1];
        TT_FATAL(input_m == output_m, "m must be the same. input_m {}, output_m {}", input_m, output_m);
        TT_FATAL(other_n == output_n, "n must be the same. other_n {}, output_n {}", other_n, output_n);

        std::vector<uint32_t> output_dim(tt::tt_metal::MAX_NUM_DIMENSIONS, 1);
        get_tensor_dim(output_dim, output_shape);

        for (auto i = 2; i < tt::tt_metal::MAX_NUM_DIMENSIONS; ++i) {
            TT_FATAL(std::max(input_dim[i], other_dim[i]) == output_dim[i], "{}th max(input_dim[i], other_dim[i]) {} must be the same as output_dim[i] {}", i, std::max(input_dim[i], other_dim[i]), output_dim[i]);
        }
    }

    // check bias size
    if (bias.has_value()) {
        const auto& bias_wo_shape = bias.value().get_shape();
        uint32_t bias_rank = bias_wo_shape.rank();
        uint32_t bias_w = bias_wo_shape[-1];
        TT_FATAL(bias_rank == 2, "bias rank {} must be 2 (tilized).", bias_rank);
        TT_FATAL(bias_w == 1 || bias_w == other_n, "bias_w must be one or the same as other_n. bias_w {}, other_n {}", bias_w, other_n);
    }
}

Tensor moreh_matmul_(
    const Tensor& input,
    const Tensor& other,
    bool transpose_input,
    bool transpose_other,
    const std::optional<Tensor>& output,
    const std::optional<Tensor>& bias,
    const MemoryConfig& output_mem_config,
    std::optional<const ttnn::DeviceComputeKernelConfig> compute_kernel_config) {
    log_debug(LogOp, "{}:{} run matmul {} {}", __func__, __LINE__, transpose_input, transpose_other);

    TT_FATAL(input.storage_type() == StorageType::DEVICE, "Error");
    auto kernel_config_val = init_device_compute_kernel_config(input.device()->arch(), compute_kernel_config, MathFidelity::HiFi4);

    std::vector<Tensor> output_tensors = {Tensor(operation::get_workers_for_op_output({input, other}, {bias}))};

    operation::launch_op(
        [output_mem_config, transpose_input, transpose_other, kernel_config_val](
            const std::vector<Tensor>& input_tensors,
            const std::vector<std::optional<const Tensor>>& optional_input_tensors,
            const std::vector<std::optional<Tensor>>& optional_output_tensors) mutable -> std::vector<Tensor> {
            return operation::run(
                MorehMatmul{
                    .output_mem_config = output_mem_config,
                    .transpose_input = transpose_input,
                    .transpose_other = transpose_other,
                    .compute_kernel_config = kernel_config_val},
                input_tensors,
                optional_input_tensors,
                optional_output_tensors);
        },
        {input, other},
        output_tensors,
        {bias},
        {output});

    return output_tensors.at(0);
}

Tensor moreh_matmul(
    const Tensor& input,
    const Tensor& other,
    bool transpose_input,
    bool transpose_other,
    const std::optional<const Tensor> output,
    const std::optional<const Tensor> bias,
    const MemoryConfig& output_mem_config,
    std::optional<const ttnn::DeviceComputeKernelConfig> compute_kernel_config) {

    // TODO(seunghwan100): Add the argument "output_tensor" to moreh_dot.
    if (is_dot_forward(input, other, transpose_input, transpose_other)) {
        TT_ASSERT(!bias.has_value());
        return moreh_dot(input, other, output_mem_config);
    }
    return moreh_matmul_(input, other, transpose_input, transpose_other, output, bias, output_mem_config, compute_kernel_config);
}

}  // namespace primary
}  // namespace operations
}  // namespace tt
