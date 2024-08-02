// SPDX-FileCopyrightText: © 2024 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include "ttnn/deprecated/tt_dnn/op_library/composite/composite_ops.hpp"
#include "ttnn/deprecated/tt_dnn/op_library/auto_format.hpp"
#include "ttnn/operations/data_movement/concat/concat.hpp"
#include "ttnn/deprecated/tt_dnn/op_library/copy/copy_op.hpp"
#include "ttnn/deprecated/tt_dnn/op_library/math.hpp"
#include "ttnn/deprecated/tt_dnn/op_library/optimizer/optimizer_ops.hpp"
#include "ttnn/deprecated/tt_dnn/op_library/reduce/reduce_op.hpp"
#include "ttnn/deprecated/tt_dnn/op_library/reshape/reshape_op.hpp"
#include "ttnn/tensor/tensor_impl.hpp"
#include "ttnn/tensor/tensor_utils.hpp"
#include "ttnn/operations/data_movement/slice/slice.hpp"
#include "ttnn/operations/data_movement/permute/permute.hpp"
#include "tt_numpy/functions.hpp"
#include "ttnn/operations/eltwise/unary/unary_composite.hpp"
#include "ttnn/cpp/ttnn/operations/eltwise/ternary/where.hpp"

#include "ttnn/operations/eltwise/binary/binary.hpp"
#include "ttnn/operations/eltwise/unary/unary.hpp"
#include "ttnn/operations/copy.hpp"
#include "ttnn/operations/matmul/matmul.hpp"
#include "ttnn/operations/eltwise/unary/unary_composite.hpp"
#include "ttnn/operations/eltwise/binary/binary_composite.hpp"

namespace tt {

namespace tt_metal {

Tensor mk_zero_tensor_like(
    uint8_t queue_id,
    const Tensor& reference_tensor,
    const MemoryConfig& output_mem_config,
    std::optional<Tensor> output_tensor = std::nullopt) {
    const DataType& dtype =
        output_tensor.has_value() ? output_tensor.value().get_dtype() : reference_tensor.get_dtype();
    Tensor zero = ttnn::operations::creation::create_scalar(0.0f, dtype, Layout::TILE, reference_tensor.device());
    return ttnn::multiply(queue_id, reference_tensor, zero, std::nullopt, output_mem_config, output_tensor);
}

Tensor mk_zero_tensor_like(
    const Tensor& reference_tensor,
    const MemoryConfig& output_mem_config,
    std::optional<Tensor> output_tensor = std::nullopt) {
    uint8_t default_queue_id = 0;
    return mk_zero_tensor_like(default_queue_id, reference_tensor, output_mem_config, output_tensor);
}

// TODO: enable zeroes(), ones() and eye() type functions on-device using this type of logic
template <typename T>
Tensor mk_filled_tensor_like(
    const Tensor& reference_tensor,
    T val,
    const MemoryConfig& output_mem_config,
    std::optional<Tensor> output_tensor = std::nullopt,
    uint8_t queue_id = 0) {
    const DataType& dtype =
        output_tensor.has_value() ? output_tensor.value().get_dtype() : reference_tensor.get_dtype();
    Tensor k = ttnn::operations::creation::create_scalar(val, dtype, Layout::TILE, reference_tensor.device());
    Tensor zero_like = mk_zero_tensor_like(reference_tensor, output_mem_config);
    if (output_tensor.has_value()) {
        return ttnn::add(queue_id, zero_like, k, std::nullopt, output_mem_config, output_tensor);
    } else {
        return ttnn::add(queue_id, zero_like, k, std::nullopt, output_mem_config);
    }
}

// Function: softshrink
// Ref: https://pytorch.org/docs/stable/generated/torch.nn.Softshrink.html
Tensor _softshrink(const Tensor& a, float param, const MemoryConfig& output_mem_config) {
    TT_ASSERT(param >= 0);
    Tensor t_a_plus_param = ttnn::add(a, param, std::nullopt, output_mem_config);
    Tensor t1 = ttnn::multiply(ttnn::ltz(t_a_plus_param, output_mem_config), t_a_plus_param, std::nullopt, output_mem_config);
    t_a_plus_param.deallocate();
    Tensor t_a_minus_param = ttnn::subtract(a, param, std::nullopt, output_mem_config);
    Tensor t2 =
        ttnn::multiply(ttnn::gtz(t_a_minus_param, output_mem_config), t_a_minus_param, std::nullopt, output_mem_config);
    t_a_minus_param.deallocate();
    return ttnn::add(t1, t2, std::nullopt, output_mem_config);
}
Tensor softshrink(const Tensor& a, float param, const MemoryConfig& output_mem_config) {
    return operation::decorate_as_composite(__func__, _softshrink)(a, param, output_mem_config);
}

// Function: hardshrink
// Ref: https://pytorch.org/docs/stable/generated/torch.nn.Hardshrink.html
Tensor _hardshrink(const Tensor& a, float param, const MemoryConfig& output_mem_config) {
    TT_ASSERT(param >= 0);
    Tensor t1 = ttnn::multiply(ttnn::ltz(ttnn::add(a, param)), a, std::nullopt, output_mem_config);
    Tensor t2 = ttnn::multiply(ttnn::gtz(ttnn::subtract(a, param)), a, std::nullopt, output_mem_config);
    return ttnn::add(t1, t2, std::nullopt, output_mem_config);
}
Tensor hardshrink(const Tensor& a, float param, const MemoryConfig& output_mem_config) {
    return operation::decorate_as_composite(__func__, _hardshrink)(a, param, output_mem_config);
}


// ELU :
//  Theano defines it as,
//  return tensor.switch(x > 0, x, alpha * tensor.expm1(x))

// rpow: y = k**(a) = exp( a**log(k) )
Tensor rpow(const Tensor& a, float k, const MemoryConfig& output_mem_config) {
    TT_ASSERT(k > 0.0, "rpow cannot be calcualted for non-positive numbers");
    float log_k = logf(k);

    Tensor scalar = ttnn::operations::creation::create_scalar(log_k, a.get_dtype(), Layout::TILE, a.device());
    Tensor result = ttnn::multiply(a, scalar, std::nullopt, output_mem_config);
    scalar.deallocate();
    return ttnn::exp(result, false, output_mem_config);
}

Tensor _logical_andi(const Tensor& input_a, float immediate, const MemoryConfig& output_mem_config) {
    if (std::fpclassify(immediate) == FP_ZERO) {
        return full_like(input_a, immediate, output_mem_config);
    } else {
        return ttnn::nez(input_a);
    }
}
Tensor logical_andi(const Tensor& input_a, float immediate, const MemoryConfig& output_mem_config) {
    return operation::decorate_as_composite(__func__, _logical_andi)(input_a, immediate, output_mem_config);
}


// ldexp(input,other)=input * (2^other)
Tensor _ldexp(const Tensor& input_a, const Tensor& input_b, const MemoryConfig& output_mem_config) {
    Tensor result = ttnn::multiply(input_a, ttnn::exp2(input_b, output_mem_config), std::nullopt, output_mem_config);
    return result;
}

Tensor _unary_rdiv_trunc(
    float value,
    const Tensor& input,
    const MemoryConfig& output_mem_config) {
    auto arch = input.device()->arch();
    TT_FATAL(arch == tt::ARCH::WORMHOLE_B0, "Op is only supported on Wormhole");
    Tensor result = ttnn::multiply(ttnn::full_like(input, value), ttnn::reciprocal(input));
    return ttnn::trunc(result);
}
Tensor unary_rdiv_trunc(
    float value,
    const Tensor& input,
    const MemoryConfig& output_mem_config) {
    return operation::decorate_as_composite(__func__, _unary_rdiv_trunc)(value, input, output_mem_config);
}

Tensor is_odd(const Tensor& input, const MemoryConfig& output_mem_config) {
    Tensor result = ttnn::multiply(input, (1.0f/2.0f));
    Tensor floor_res = ttnn::floor(result);
    return ttnn::ne(result, floor_res);
}

Tensor _rfloor_div(float value, const Tensor& input, const MemoryConfig& output_mem_config) {
    Tensor result = ttnn::multiply(ttnn::full_like(input, value), ttnn::reciprocal(input));
    return ttnn::floor(result, output_mem_config);
}
Tensor rfloor_div(float value, const Tensor& input, const MemoryConfig& output_mem_config) {
    return operation::decorate_as_composite(__func__, _rfloor_div)(value, input, output_mem_config);
}

// logit(input, eps)=log(input / 1 - input)
Tensor _logit(const Tensor& input_a, float eps, const MemoryConfig& output_mem_config) {
    Tensor t_eps = full_like(input_a, eps, output_mem_config);
    Tensor t1m_eps = full_like(input_a, (1 - eps), output_mem_config);
    Tensor logit_input = ttnn::where(
        ttnn::ltz(t_eps, output_mem_config),
        input_a,
        ttnn::where(
            ttnn::lt(input_a, t_eps, std::nullopt, output_mem_config),
            t_eps,
            ttnn::where(ttnn::gt(input_a, t1m_eps, std::nullopt, output_mem_config), t1m_eps, input_a, output_mem_config),
            output_mem_config),
        output_mem_config);
    t_eps.deallocate();
    t1m_eps.deallocate();
    Tensor linput_m1 = ttnn::rsub(logit_input, 1.0, output_mem_config);
    Tensor log_input =
        ttnn::multiply(logit_input, ttnn::reciprocal(linput_m1, output_mem_config), std::nullopt, output_mem_config);
    linput_m1.deallocate();
    Tensor t_inf =
        ttnn::multiply(ttnn::sign(input_a, output_mem_config), std::numeric_limits<float>::infinity(), std::nullopt, output_mem_config);
    Tensor logit_result = ttnn::where(
        ttnn::eq(logit_input, 1.0, std::nullopt, output_mem_config),
        t_inf,
        ttnn::where(ttnn::ltz(log_input, output_mem_config), std::nanf(" "), ttnn::log(log_input, output_mem_config), output_mem_config),
        output_mem_config);
    return logit_result;
}
Tensor logit(const Tensor& input_a, float eps, const MemoryConfig& output_mem_config) {
    return operation::decorate_as_composite(__func__, _logit)(input_a, eps, output_mem_config);
}

// logical_xori
Tensor _logical_xori(const Tensor& input_a, float value, const MemoryConfig& output_mem_config) {
    if (std::fpclassify(value) == FP_ZERO) {
        return ttnn::nez(input_a);
    } else {
        return ttnn::eqz(input_a);  // eqz( input_a ) = not( nez( input_a ) )
    }
}
Tensor logical_xori(const Tensor& input_a, float value, const MemoryConfig& output_mem_config) {
    return operation::decorate_as_composite(__func__, _logical_xori)(input_a, value, output_mem_config);
}

// Celu
// torch.where(x > 0, x, alpha * (torch.exp(x / alpha) - 1))
Tensor _celu(const Tensor& input_a, float alpha, const MemoryConfig& output_mem_config) {
    float recip_val = 1.0f / alpha;
    using ttnn::operations::unary::UnaryWithParam;
    using ttnn::operations::unary::UnaryOpType;
    std::vector<UnaryWithParam> ops_chain = {
    UnaryWithParam{UnaryOpType::MUL_UNARY_SFPU, recip_val},
    UnaryWithParam{UnaryOpType::EXP, 1.0f},
    UnaryWithParam{UnaryOpType::SUB_UNARY_SFPU, 1.0f}, UnaryWithParam{UnaryOpType::MUL_UNARY_SFPU, alpha} };

    Tensor result = ttnn::unary_chain(input_a, ops_chain, output_mem_config);
    result = ttnn::where(ttnn::gtz(input_a, output_mem_config), input_a, result, output_mem_config);
    return result;
}
Tensor celu(const Tensor& input_a, float alpha, const MemoryConfig& output_mem_config) {
    return operation::decorate_as_composite(__func__, _celu)(input_a, alpha, output_mem_config);
}


using HWFunctionT = std::function<Tensor(const Tensor& y, const MemoryConfig&)>;
Tensor _make_global_from_hw_impl(
    HWFunctionT fn, const Tensor& y, const MemoryConfig& output_mem_config = operation::DEFAULT_OUTPUT_MEMORY_CONFIG) {
    const Shape s_orig = y.get_legacy_shape();
    TT_FATAL(s_orig.rank() == 4, "Cannot support non-rank 4 Tensor");

    // format to HW
    Tensor y_hw = reshape(y, 1, 1, s_orig[2], s_orig[3] * s_orig[1] * s_orig[0], output_mem_config);

    // compute @fn
    Tensor z_0 = fn(y_hw, output_mem_config);
    TT_FATAL(y_hw.get_legacy_shape() == z_0.get_legacy_shape(), "shape match");
    y_hw.deallocate();

    // reformat
    Tensor z_1 = reshape(z_0, s_orig[0], s_orig[1], s_orig[2], s_orig[3], output_mem_config);
    z_0.deallocate();

    return z_1;
}

// Global Norm
Tensor _normalize_global(const Tensor& y, const MemoryConfig& output_mem_config) {
    return _make_global_from_hw_impl(ttnn::normalize_hw, y, output_mem_config);
}
Tensor normalize_global(const Tensor& y, const MemoryConfig& output_mem_config) {
    return operation::decorate_as_composite(__func__, _normalize_global)(y, output_mem_config);
}


// on-device tensor creation 0s like @reference_tensor
Tensor zeros_like(
    uint8_t queue_id,
    const Tensor& reference_tensor,
    const MemoryConfig& output_mem_config,
    std::optional<Tensor> output_tensor) {
    return mk_zero_tensor_like(reference_tensor, output_mem_config, output_tensor);
}
Tensor zeros_like(
    const Tensor& reference_tensor, const MemoryConfig& output_mem_config, std::optional<Tensor> output_tensor) {
    uint8_t default_queue_id = 0;
    return mk_zero_tensor_like(default_queue_id, reference_tensor, output_mem_config, output_tensor);
}

// on-device tensor creation 1s like @reference_tensor
Tensor ones_like(const Tensor& reference_tensor, const MemoryConfig& output_mem_config) {
    return mk_filled_tensor_like(reference_tensor, 1.0f, output_mem_config);
}

// on-device tensor creation with value like @reference_tensor
Tensor full_like(
    const Tensor& reference_tensor,
    float value,
    const MemoryConfig& output_mem_config,
    std::optional<Tensor> output_tensor) {
    uint8_t default_queue_id = 0;
    return mk_filled_tensor_like(reference_tensor, value, output_mem_config, output_tensor, default_queue_id);
}
Tensor full_like(
    uint8_t queue_id,
    const Tensor& reference_tensor,
    float value,
    const MemoryConfig& output_mem_config,
    std::optional<Tensor> output_tensor) {
    return mk_filled_tensor_like(reference_tensor, value, output_mem_config, output_tensor, queue_id);
}

// hardtanh
Tensor _hardtanh(
    const Tensor& a, float low /* = -1.0f */, float high /* = +1.0f */, const MemoryConfig& output_mem_config) {
    return ttnn::clip(a, low, high, output_mem_config);
}
Tensor hardtanh(
    const Tensor& a, float low /* = -1.0f */, float high /* = +1.0f */, const MemoryConfig& output_mem_config) {
    return operation::decorate_as_composite(__func__, _hardtanh)(a, low, high, output_mem_config);
}

// on-device tensor creation 0s with shape
Tensor zeros(
    const Shape shape, DataType data_type, Layout layout, Device* device, const MemoryConfig& output_mem_config) {
    return tt::numpy::zeros(shape, data_type, layout, device, output_mem_config);
}

Tensor empty(
    const Shape shape, DataType data_type, Layout layout, Device* device, const MemoryConfig& output_mem_config) {
    return create_device_tensor(shape, data_type, layout, device, output_mem_config);
}

// on-device tensor creation 1s with shape
Tensor ones(
    const Shape shape, DataType data_type, Layout layout, Device* device, const MemoryConfig& output_mem_config) {
    return tt::numpy::ones(shape, data_type, layout, device, output_mem_config);
}

// on-device tensor creation with shape and filled with value
Tensor full(
    const Shape shape,
    float value,
    DataType data_type,
    Layout layout,
    Device* device,
    const MemoryConfig& output_mem_config) {
    return tt::numpy::full(shape, value, data_type, layout, device, output_mem_config);
}

// on-device with increment
Tensor arange(int32_t start, int32_t end, int32_t step, Device* device, const MemoryConfig& output_mem_config) {
    return tt::numpy::arange<bfloat16>(start, end, step, Layout::ROW_MAJOR, device, output_mem_config);
}



}  // namespace tt_metal

}  // namespace tt
