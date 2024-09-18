// SPDX-FileCopyrightText: © 2024 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0


#include "unary_composite_op.hpp"

#include <functional>
#include <optional>

#include "third_party/magic_enum/magic_enum.hpp"
#include "tt_metal/common/bfloat16.hpp"
#include "ttnn/operations/data_movement/reshape/reshape.hpp"
#include "ttnn/operations/data_movement/bcast/bcast.hpp"
#include "ttnn/operations/functions.hpp"
#include "ttnn/operations/data_movement/slice/slice.hpp"
#include "ttnn/operations/eltwise/unary/unary_composite.hpp"
#include "ttnn/operations/eltwise/binary/binary_composite.hpp"
#include "ttnn/operations/eltwise/ternary/ternary_composite.hpp"
#include "ttnn/operations/creation.hpp"
#include "ttnn/operations/reduction/generic/generic_reductions.hpp"
#include "ttnn/run_operation.hpp"
#include "ttnn/types.hpp"
#include "ttnn/operations/data_movement/bcast/bcast.hpp"

namespace ttnn::operations::unary{

Tensor _deg2rad(const Tensor& input_tensor, const std::optional<MemoryConfig>& output_mem_config) {
    return ttnn::multiply(input_tensor, (float)(M_PI / 180.0), std::nullopt, output_mem_config.value_or(input_tensor.memory_config()));
}

Tensor _rad2deg(const Tensor& input_tensor, const std::optional<MemoryConfig>& output_mem_config) {
    return ttnn::multiply(input_tensor, (float)(180.0 / M_PI), std::nullopt, output_mem_config.value_or(input_tensor.memory_config()));
}

// // tanhshrink(x) = x - tanh(x)
Tensor _tanhshrink(const Tensor& x, const std::optional<MemoryConfig>& output_mem_config) {
    Tensor tan_x = ttnn::tanh(x, output_mem_config);
    Tensor result = ttnn::subtract(x, tan_x, std::nullopt, output_mem_config);
    return result;
}

// power - floating point exponent
Tensor _power(uint8_t queue_id, const Tensor& input_a, float exponent, const std::optional<MemoryConfig>& output_mem_config, std::optional<Tensor> output_tensor) {
    TT_FATAL(exponent >= 0.0f, "works for positive exponents only");
    const uint32_t exponent_floor = static_cast<uint32_t>(std::floor(exponent));
    if (static_cast<float>(exponent_floor) == exponent) {
        if(output_tensor.has_value()){
            ttnn::power(queue_id,input_a, exponent_floor, output_mem_config, output_tensor);
            return output_tensor.value();
        }
        return ttnn::power(queue_id, input_a, exponent_floor, output_mem_config);
    }
    const float exponent_trunc = exponent - static_cast<float>(exponent_floor);
    Tensor pow_trunc_log = ttnn::multiply(queue_id, ttnn::log(queue_id, input_a, output_mem_config), exponent_trunc, std::nullopt, output_mem_config);
    Tensor pow_frac = ttnn::exp(queue_id, pow_trunc_log, false, output_mem_config);
    pow_trunc_log.deallocate();
    float t_nan = std::nanf("");
    Tensor result = ttnn::multiply(queue_id, ttnn::power(queue_id, input_a, exponent_floor, output_mem_config), pow_frac, std::nullopt, output_mem_config);
    // To handle negative inputs:
    // in torch For -ve inputs with float exponent power returns nan
    auto output_memory_config = output_tensor.has_value() ? output_tensor.value().memory_config() : output_mem_config.value_or(input_a.memory_config());
    result = ttnn::where(ttnn::ltz(queue_id, input_a, output_mem_config), t_nan, result, output_memory_config, output_tensor);
    return result;
}

// power - integer exponent
Tensor _power(uint8_t queue_id, const Tensor& input, uint32_t exponent, const std::optional<MemoryConfig>& output_mem_config, std::optional<Tensor> output_tensor) {
    return ttnn::power(queue_id, input, exponent, output_mem_config, output_tensor);
}

// acosh(x) = log(x + sqrt(x^2 - 1))
Tensor _acosh(const Tensor& input_a, const std::optional<MemoryConfig>& output_mem_config) {
   Tensor t_one = ttnn::full_like(input_a, 1.0f);
   Tensor t_result(input_a);
   {
       Tensor ln_res(input_a);
       {
           Tensor x_abs = ttnn::abs(input_a, output_mem_config);
           Tensor x_sq_m1(input_a);
           {
               Tensor x_sq = ttnn::square(x_abs, output_mem_config);
               x_sq_m1 = ttnn::subtract(x_sq, 1.0f, std::nullopt, output_mem_config);
           }
           ln_res = ttnn::log(
               ttnn::add(x_abs, ttnn::sqrt(x_sq_m1, output_mem_config), std::nullopt, output_mem_config), output_mem_config);
       }
       // To handle inputs <= 1
       // input < 1, output is nan
       // input > 1, output is acosh(input)
       Tensor nan_res = ttnn::multiply(
           ttnn::le(input_a, t_one, std::nullopt, output_mem_config), input_a.device()->sfpu_nan(), std::nullopt, output_mem_config);
       t_result = ttnn::multiply(
           ttnn::gt(input_a, t_one, std::nullopt, output_mem_config), ln_res, std::nullopt, output_mem_config);
       t_result = ttnn::add(nan_res, t_result, std::nullopt, output_mem_config);
   }
   // input == 1, output is 0
   Tensor result = ttnn::where(ttnn::eq(input_a, t_one, std::nullopt, output_mem_config), 0.0f, t_result);
   return result;
}

// asinh(x) = log(x + sqrt(x^2 + 1))
Tensor _asinh(const Tensor& input_a, const std::optional<MemoryConfig>& output_mem_config) {
   Tensor ln_res(input_a);
   {
       Tensor x_abs = ttnn::abs(input_a, output_mem_config);
       Tensor x_sq_p1(input_a);
       {
           Tensor x_sq = ttnn::square(input_a, output_mem_config);
           x_sq_p1 = ttnn::add(x_sq, 1.0f, std::nullopt, output_mem_config);
       }
       ln_res =
           ttnn::log(ttnn::add(x_abs, ttnn::sqrt(x_sq_p1, output_mem_config), std::nullopt, output_mem_config), output_mem_config);
   }
   // input is negative, output is -asinh(input)
   Tensor result = ttnn::where(input_a, ln_res, ttnn::neg(ln_res, output_mem_config));
   return result;
}

// atanh[x] = 0.5 * ln((1 + x) / (1 - x))
Tensor _atanh(const Tensor& input_a, const std::optional<MemoryConfig>& output_mem_config) {
   Tensor comp_result(input_a);
   {
       Tensor nr_term(input_a);
       {
           Tensor pos_x = ttnn::add(input_a, 1.0f, std::nullopt, output_mem_config);
           Tensor neg_x = ttnn::subtract(input_a, 1.0f, std::nullopt, output_mem_config);
           nr_term = ttnn::log(
               ttnn::multiply(
                   pos_x, ttnn::reciprocal(ttnn::neg(neg_x, output_mem_config), output_mem_config), std::nullopt, output_mem_config),
               output_mem_config);
       }
       comp_result = ttnn::multiply(nr_term, 0.5f, std::nullopt, output_mem_config);
   }
   // Input is -1 > value > 1, output is nan
   // Input is -1 < value < 1, output is atanh(input)
   float t_nan = std::nanf("");
   Tensor abs_temp = ttnn::subtract(ttnn::abs(input_a, output_mem_config), 1.0f, std::nullopt, output_mem_config);
   Tensor result = ttnn::where(ttnn::ltz(abs_temp, output_mem_config), comp_result, t_nan);
   return result;
}

// cbrt(a) = pow(a,1/3) or (cbrt(a))**3 = a.
//         = exp[ (1/3)*log[a] ]
Tensor _cbrt(const Tensor& input_tensor, const std::optional<MemoryConfig>& output_mem_config) {
   constexpr float scale = (float)(1.0 / 3.0);
   Tensor t_ln_input =
       ttnn::log(ttnn::abs(input_tensor, output_mem_config), output_mem_config);  // negative log is not useful here
   Tensor t1 = ttnn::multiply(t_ln_input, scale, std::nullopt, output_mem_config);
   t_ln_input.deallocate();
   Tensor t2 = ttnn::exp(t1, false, output_mem_config);
   t1.deallocate();
   Tensor t3 = ttnn::multiply(t2, ttnn::sign(input_tensor, output_mem_config), std::nullopt, output_mem_config);
   return t3;
}

// cosh[x] = (exp[x] + exp[-x])/2
Tensor _cosh(const Tensor& input_a, const std::optional<MemoryConfig>& output_mem_config) {
   Tensor e_pos_x = ttnn::exp(input_a, false, output_mem_config);
   Tensor e_neg_x = ttnn::exp(ttnn::neg(input_a, output_mem_config), false, output_mem_config);
   Tensor nr_term = ttnn::add(e_pos_x, e_neg_x, std::nullopt, output_mem_config);
   e_pos_x.deallocate();
   e_neg_x.deallocate();
   Tensor scalar = ttnn::full_like(input_a, 0.5f);
   return ttnn::multiply(nr_term, scalar, std::nullopt, output_mem_config);
}

// TODO: In future will uplift the op once the floor and tan has supported.
// digamma support for the range of (1, inf)
Tensor _digamma(const Tensor& input_a, const std::optional<MemoryConfig>& output_mem_config) {
   Tensor t_log_out = ttnn::log(input_a, output_mem_config);  // negative log is not useful here

   // 1/2(z)
   Tensor output = ttnn::multiply(ttnn::reciprocal(input_a, output_mem_config), 0.5f, std::nullopt, output_mem_config);
   Tensor tmp = ttnn::square(ttnn::reciprocal(input_a, output_mem_config), output_mem_config);
   Tensor val_square = tmp;
   // (1/12) * x^2
   output = ttnn::subtract(output, ttnn::multiply(tmp, 0.083333333f), std::nullopt, output_mem_config);

   // (1/120) * x^4
   tmp = ttnn::multiply(tmp, val_square, std::nullopt, output_mem_config);
   output =
       ttnn::add(output, ttnn::multiply(tmp, 0.008333333333333333f, std::nullopt, output_mem_config), std::nullopt, output_mem_config);

   //(1/252) * x^6
   tmp = ttnn::multiply(tmp, val_square, std::nullopt, output_mem_config);
   output = ttnn::subtract(
       output, ttnn::multiply(tmp, 0.003968253968253968f, std::nullopt, output_mem_config), std::nullopt, output_mem_config);

   // (1/240) *x^8
   tmp = ttnn::multiply(tmp, val_square, std::nullopt, output_mem_config);
   output =
       ttnn::add(output, ttnn::multiply(tmp, 0.004166666666666667f, std::nullopt, output_mem_config), std::nullopt, output_mem_config);

   //(1/132) * x^10
   tmp = ttnn::multiply(tmp, val_square, std::nullopt, output_mem_config);
   output = ttnn::subtract(
       output, ttnn::multiply(tmp, 0.007575757575757576), std::nullopt, output_mem_config);

   //(691/32760) * x^12
   tmp = ttnn::multiply(tmp, val_square, std::nullopt, output_mem_config);
   output =
       ttnn::add(output, ttnn::multiply(tmp, 0.021092796092796094, std::nullopt, output_mem_config), std::nullopt, output_mem_config);

   //(1/12) * x^14
   tmp = ttnn::multiply(tmp, val_square, std::nullopt, output_mem_config);
   output =
       ttnn::subtract(output, ttnn::multiply(tmp, 0.08333333333333333, std::nullopt, output_mem_config), std::nullopt, output_mem_config);

   return ttnn::subtract(t_log_out, output, std::nullopt, output_mem_config);
}

Tensor _lgamma(const Tensor& x,  const std::optional<MemoryConfig>& output_mem_config) {
    Tensor result(x);
    {
        Tensor t(x);
        {
            Tensor temp_log(x);
            {
                Tensor temp(x);
                Tensor input = ttnn::subtract(x, 1.0f, std::nullopt, output_mem_config);
                {
                    Tensor z1 = ttnn::multiply(
                        ttnn::reciprocal(ttnn::add(input, 1.0f, std::nullopt, output_mem_config), output_mem_config),
                        76.18009172947146f,
                        std::nullopt,
                        output_mem_config);
                    temp = ttnn::add(z1, 1.0f, std::nullopt, output_mem_config);

                    z1 = ttnn::multiply(
                        ttnn::reciprocal(ttnn::add(input, 2.0f, std::nullopt, output_mem_config), output_mem_config),
                        -86.50532032941677f,
                        std::nullopt,
                        output_mem_config);
                    temp = ttnn::add(temp, z1, std::nullopt, output_mem_config);

                    z1 = ttnn::multiply(
                        ttnn::reciprocal(ttnn::add(input, 3.0f, std::nullopt, output_mem_config), output_mem_config),
                        24.01409824083091f,
                        std::nullopt,
                        output_mem_config);
                    temp = ttnn::add(temp, z1, std::nullopt, output_mem_config);

                    z1 = ttnn::multiply(
                        ttnn::reciprocal(ttnn::add(input, 4.0f, std::nullopt, output_mem_config), output_mem_config),
                        -1.231739572450155f,
                        std::nullopt,
                        output_mem_config);
                    temp = ttnn::add(temp, z1, std::nullopt, output_mem_config);

                    z1 = ttnn::multiply(
                        ttnn::reciprocal(ttnn::add(input, 5.0f, std::nullopt, output_mem_config), output_mem_config),
                        0.1208650973866179e-2f,
                        std::nullopt,
                        output_mem_config);
                    temp = ttnn::add(temp, z1, std::nullopt, output_mem_config);

                    z1 = ttnn::multiply(
                        ttnn::reciprocal(ttnn::add(input, 6.0f, std::nullopt, output_mem_config), output_mem_config),
                        -0.5395239384953e-5f,
                        std::nullopt,
                        output_mem_config);
                    temp = ttnn::add(temp, z1, std::nullopt, output_mem_config);
                }
                {
                    Tensor t_log(x);
                    {
                        t = ttnn::add(input, 5.5f, std::nullopt, output_mem_config);
                        t_log = ttnn::log(t, output_mem_config);
                    }
                    temp_log = ttnn::log(temp, output_mem_config);
                    result = ttnn::add(
                        ttnn::multiply(
                            ttnn::add(input, 0.5f, std::nullopt, output_mem_config), t_log, std::nullopt, output_mem_config),
                        0.918938531357171f,
                        std::nullopt,
                        output_mem_config);
                }
            }
            result = ttnn::add(result, temp_log, std::nullopt, output_mem_config);
        }
        result = ttnn::subtract(result, t, std::nullopt, output_mem_config);
        {
            {
                Tensor t_one = ttnn::full_like(x, 1.0f);
                result = ttnn::where(ttnn::eq(x, t_one, std::nullopt, output_mem_config), 0.0f, result);
            }
            {
                Tensor t_two = ttnn::full_like(x, 2.0f);
                result = ttnn::where(ttnn::eq(x, t_two, std::nullopt, output_mem_config), 0.0f, result);
            }
        }
    }
    return result;
}

// log1p 1
// use transformation y = log(1.0 + x) by broadcast
Tensor _log1p(const Tensor& x, const std::optional<MemoryConfig>& output_mem_config) {
    Tensor t_one = ttnn::full_like(x, 1.0f);
    Tensor x_1 = ttnn::add(t_one, x, std::nullopt, output_mem_config);
    Tensor result_log1p = ttnn::log(x_1, output_mem_config);
    return result_log1p;
}

// mish[x] = x*tanh[softplus[x]]
// use transformation y = x*tanh[softplus[x]] by broadcast
// Ref: https://krutikabapat.github.io/Swish-Vs-Mish-Latest-Activation-Functions/
Tensor _mish(const Tensor& x, const std::optional<MemoryConfig>& output_mem_config) {
    std::vector<Tensor> output_tensors = {Tensor(operation::get_workers_for_op_output({x}))};
    operation::launch_op(
        [output_mem_config](
            const std::vector<Tensor>& input_tensors,
            const std::vector<std::optional<const Tensor>>& optional_input_tensors,
            const std::vector<std::optional<Tensor>>& optional_output_tensors) mutable -> std::vector<Tensor> {
            const auto& x = input_tensors.at(0);
            Tensor sp_x = ttnn::softplus(x, 1.0f, 20.0f, output_mem_config);
            Tensor tanh_x = ttnn::tanh(sp_x, output_mem_config);
            sp_x.deallocate();
            Tensor mish_x = ttnn::multiply(x, tanh_x, std::nullopt, output_mem_config);
            return {mish_x};
        },
        {x},
        output_tensors);
    return output_tensors.at(0);
}

// multivariate log-gamma function
// Ref : https://pytorch.org/docs/stable/special.html#torch.special.multigammaln
Tensor _multigammaln(const Tensor& x, const std::optional<MemoryConfig>& output_mem_config) {
    Tensor result = _lgamma(x, output_mem_config);
    result = ttnn::add(
        result, _lgamma(ttnn::subtract(x, 0.5f, std::nullopt, output_mem_config), output_mem_config), std::nullopt, output_mem_config);
    result = ttnn::add(
        result, _lgamma(ttnn::subtract(x, 1.0f, std::nullopt, output_mem_config), output_mem_config), std::nullopt, output_mem_config);
    result = ttnn::add(
        result, _lgamma(ttnn::subtract(x, 1.5f, std::nullopt, output_mem_config), output_mem_config), std::nullopt, output_mem_config);
    result = ttnn::add(result, 3.434189657547f, std::nullopt, output_mem_config);
    return result;
}

Tensor _sinh(const Tensor& input_a, const std::optional<MemoryConfig>& output_mem_config) {
    Tensor e_pos_x = ttnn::exp(input_a, false, output_mem_config);
    Tensor e_neg_x = ttnn::exp(ttnn::neg(input_a, output_mem_config), false, output_mem_config);
    Tensor nr_term = ttnn::subtract(e_pos_x, e_neg_x, std::nullopt, output_mem_config);
    e_pos_x.deallocate();
    e_neg_x.deallocate();
    Tensor scalar = ttnn::full_like(input_a, 0.5f);
    return ttnn::multiply(nr_term, scalar, std::nullopt, output_mem_config);
}

// Function: softsign
// Ref: https://pytorch.org/docs/stable/generated/torch.nn.Softsign.html
Tensor _softsign(const Tensor& a, const std::optional<MemoryConfig>& output_mem_config) {
     Tensor result =ttnn::multiply(
        a,
        ttnn::reciprocal(ttnn::add(ttnn::abs(a, output_mem_config), 1.0f, std::nullopt, output_mem_config), output_mem_config),
        std::nullopt,
        output_mem_config);
        return result;
}

Tensor _swish(const Tensor& a, const std::optional<MemoryConfig>& output_mem_config) {
    // x / (1.0f + exp(-x))
    return ttnn::silu(a);
}

Tensor ExecuteTrunc::invoke(uint8_t queue_id, const Tensor& input, const std::optional<MemoryConfig>& output_mem_config, std::optional<Tensor> output_tensor) {
    auto arch = input.device()->arch();
    output_tensor = output_tensor.value_or(ttnn::empty_like(input));
    TT_FATAL(arch != tt::ARCH::GRAYSKULL, "Op is not supported on Grayskull");
    Tensor floor_res = ttnn::floor(queue_id, input, output_mem_config);
    ttnn::where(queue_id, ttnn::ne(queue_id, input, floor_res), ttnn::add(queue_id, floor_res, 1.0f, std::nullopt, output_mem_config), floor_res, output_mem_config, output_tensor);
    ttnn::where(queue_id, ttnn::gtz(queue_id, input, output_mem_config), floor_res, output_tensor.value(), output_mem_config, output_tensor);
    return output_tensor.value();
}

Tensor ExecuteTrunc::invoke(const Tensor& input, const std::optional<MemoryConfig>& output_mem_config, std::optional<Tensor> output_tensor) {
    return ExecuteTrunc::invoke(DefaultQueueId, input, output_mem_config, output_tensor);
}

// Function variance of whole tensor.
// Tensor variance(const Tensor& y,const Tensor& mean_y);
Tensor _variance_impl(
    const Tensor& y,
    const Tensor& mean_y,
    Tensor& y_minus_mean_y,
    const std::optional<MemoryConfig>& output_mem_config) {
    std::vector<int> dims = { 2, 3 };
    constexpr float correction = 0.0f;
    auto shape_wh = y.get_legacy_shape();
    float scale = 1.0f / ((float)(shape_wh[3] * shape_wh[2]) - correction);
    Tensor sqr_y_minus_mean_y = ttnn::square(y_minus_mean_y, output_mem_config);
    return ttnn::sum(sqr_y_minus_mean_y, dims, true, std::nullopt, std::nullopt, scale);
}
Tensor _variance_impl(const Tensor& y, const Tensor& mean_y, const std::optional<MemoryConfig>& output_mem_config) {
    Tensor y_minus_mean_y = ttnn::bcast(0, y, mean_y, ttnn::BcastOpMath::SUB, ttnn::BcastOpDim::HW);
    return _variance_impl(y, mean_y, y_minus_mean_y, output_mem_config);
}

Tensor _variance(const Tensor& y, const std::optional<MemoryConfig>& output_mem_config) {
    auto output_memory_config = output_mem_config.value_or(y.memory_config());
    std::vector<int> dims = { 2, 3 };
    Tensor mean_y = ttnn::mean(y, dims, true);
    return _variance_impl(y, mean_y, output_memory_config);
}

// Function std
// compute standard deviation of tensor y = sqrt( E((y-<y>)^2)/ y.volume() )
//  Ref: torch.std
Tensor _std(const Tensor& y, const Tensor& mean_y, const std::optional<MemoryConfig>& output_mem_config) {
    return ttnn::sqrt(_variance_impl(y, mean_y, output_mem_config));
}

Tensor _std(const Tensor& y, const Tensor& mean_y, Tensor& y_minus_mean_y, const std::optional<MemoryConfig>& output_mem_config) {
    return ttnn::sqrt(_variance_impl(y, mean_y, y_minus_mean_y, output_mem_config));
}

Tensor _std_overload(const Tensor& y, const std::optional<MemoryConfig>&  output_mem_config) {
    return ttnn::sqrt(_variance(y, output_mem_config));
}

// Function normalize
// use transformation y = (y - mean(y))/std(y) by broadcast
Tensor _normalize(const Tensor& y, const std::optional<MemoryConfig>& output_mem_config) {
    std::vector<int> dims = { 2, 3 };
    Tensor mean_y = ttnn::mean(y, dims, true);
    Tensor y_minus_mean_y = ttnn::bcast(0, y, mean_y, ttnn::BcastOpMath::SUB, ttnn::BcastOpDim::HW);
    Tensor std_y = _std(y, mean_y, y_minus_mean_y, output_mem_config);
    Tensor recip_std_y = ttnn::reciprocal(std_y, output_mem_config);
    Tensor z = ttnn::multiply(y_minus_mean_y, recip_std_y, std::nullopt, output_mem_config);
    return z;
}

// Function Hard Sigmoid
//     Ref: https://github.com/Theano/Theano/blob/master/theano/tensor/nnet/sigm.py
//
//     slope = tensor.constant(0.2, dtype=out_dtype)
//     shift = tensor.constant(0.5, dtype=out_dtype)
//
//     x1 = (x * slope) + shift
//     y = tensor.clip(x1, 0, 1)
//
// PyTorch version:
// hard sigmoid(x) = { x <= -3: 0, x >= +3: +3, x/6 + 0.5 otherwise}
Tensor _hardsigmoid(const Tensor& a, float value_1, float value_2, const std::optional<MemoryConfig>& output_mem_config) {
   Tensor a_t = ttnn::full_like(a,value_1);
   Tensor b_t = ttnn::full_like(a,value_2);
   Tensor a_mac = ttnn::mac(a, a_t, b_t);  // multiply and add.
   Tensor a_clip = relu_max(a_mac, 1.0f);
   return a_clip;
}

// Function @hard_swish
// use transformation y = x * hardsigmoid( x ) by broadcast
// Ref: PyTorch
// hard swish(x) = x*hardsigmoid(x,scale,shift)
Tensor _hardswish(const Tensor& a, float value_1, float value_2, const std::optional<MemoryConfig>& output_mem_config) {
   Tensor a_sigmoid = _hardsigmoid(a, value_1, value_2, output_mem_config);
   Tensor result_sq = ttnn::multiply(a_sigmoid, a, std::nullopt, output_mem_config);
   return result_sq;
}

// Function Clip
// use clip y = min( max( x, min_value), max_value) by broadcast
// Ref: https://pytorch.org/docs/stable/generated/torch.clamp.html#torch.clamp
Tensor _clip(const Tensor& a, float low, float high, const std::optional<MemoryConfig>& output_mem_config) {
    auto output_memory_config = output_mem_config.value_or(a.memory_config());
    const Tensor h_const = full_like(a, high);
    Tensor a_max = ttnn::minimum(a, h_const, output_memory_config);
    if (low == 0.0f) {
        return ttnn::relu(a_max, output_memory_config);
    } else {
        const Tensor l_const = full_like(a, low);
        return ttnn::maximum(a_max, l_const, output_memory_config);
    }
}

// clamp
Tensor _clamp(const Tensor& a, float low, float high, const std::optional<MemoryConfig>& output_mem_config) {
    return _clip(a, low, high, output_mem_config);
}

// hardtanh
Tensor _hardtanh(
    const Tensor& a, float low /* = -1.0f */, float high /* = +1.0f */, const std::optional<MemoryConfig>& output_mem_config) {
        auto output_memory_config = output_mem_config.value_or(a.memory_config());
    return _clip(a, low, high, output_memory_config);
}

// Theano defines this differently...
/**
 *
 *   alpha = 1.6732632423543772848170429916717
 *    scale = 1.0507009873554804934193349852946
 *    return scale * elu(x, alpha)
 *
 */
// Function Selu - scaled exponential linear
// use transformation y = scale *(max(0,x)) + min(0,alpha * (exp(X)-1)) by broadcast
// Ref: https://pytorch.org/docs/stable/generated/torch.nn.SELU.html
Tensor _selu(const Tensor& x, const float scale, const float alpha, const std::optional<MemoryConfig>& output_mem_config) {
    // term 2
    Tensor x_Exp = ttnn::exp(x, false, output_mem_config);
    Tensor x_Exp_minus_1 =ttnn::subtract(x_Exp , -1.0f, std::nullopt, output_mem_config);
    x_Exp.deallocate();
    Tensor result_t2_ = ttnn::multiply(x_Exp_minus_1, alpha, std::nullopt, output_mem_config);
    x_Exp_minus_1.deallocate();
    Tensor result_term2 =
        ttnn::multiply(ttnn::gtz(result_t2_, output_mem_config), result_t2_, std::nullopt, output_mem_config);
    result_t2_.deallocate();

    // term 1
    Tensor x_relu = ttnn::relu(x, output_mem_config);
    Tensor result_term1 = ttnn::multiply(x_relu, scale, std::nullopt, output_mem_config);
    x_relu.deallocate();
    Tensor result_selu = ttnn::add(result_term1, result_term2, std::nullopt, output_mem_config);

    return result_selu;
}

// threshold(a,t,v) = (a <= t)*v + (a > t)*a
Tensor _threshold(const Tensor& input_tensor, float threshold, float value, const std::optional<MemoryConfig>& output_mem_config) {
    Tensor t0 = ttnn::subtract(input_tensor, threshold, std::nullopt, output_mem_config);
    Tensor t1 = ttnn::multiply(ttnn::lez(t0), value, std::nullopt, output_mem_config);
    Tensor t2 = ttnn::multiply(ttnn::gtz(t0, output_mem_config), input_tensor, std::nullopt, output_mem_config);
    return ttnn::add(t1, t2, std::nullopt, output_mem_config);
}

std::vector<Tensor> split_tensor_for_glu(const Tensor& input_a, int32_t dim, const std::optional<MemoryConfig>& output_mem_config) {
    std::vector<Tensor> t_split;
    tt::tt_metal::LegacyShape inshape(input_a.get_legacy_shape());
    TT_FATAL(((inshape[dim] / 2) % tt::constants::TILE_WIDTH == 0), "Split tensor dimension should be in full tile");
    std::vector<uint32_t> s_a = {0, 0, 0, 0};
    std::vector<uint32_t> e_a = {input_a.get_legacy_shape()[0], inshape[1], inshape[2], inshape[3] / 2};

    std::vector<uint32_t> s_b = {0, 0, 0, inshape[3] / 2};
    std::vector<uint32_t> e_b = {inshape[0], inshape[1], inshape[2], inshape[3]};

    auto step = std::vector<uint32_t>({1,1,1,1});
    Tensor t_a = ttnn::slice(DefaultQueueId, input_a, s_a, e_a, step, output_mem_config);
    Tensor t_b = ttnn::slice(DefaultQueueId, input_a, s_b, e_b, step, output_mem_config);

    t_split.emplace_back(t_a);
    t_split.emplace_back(t_b);

    return t_split;
}

// Gated Linear Unit activation: matmul(split[0],sigmoid(split[1]))
 Tensor _glu(const Tensor& input_a, int32_t dim , const std::optional<MemoryConfig>& output_mem_config) {
    TT_ASSERT(dim == -1 || dim == 3, "last dim GLU only supported at this time ");
    if (dim == -1)
        dim = 3;
    std::vector<Tensor> ab = split_tensor_for_glu(input_a, dim, output_mem_config);
    Tensor sigmoid_b = ttnn::sigmoid(ab[1], output_mem_config);
    Tensor glu_result = ttnn::multiply(ab[0], sigmoid_b, std::nullopt, output_mem_config);
    return glu_result;
}

// ReLU Gated Linear Unit activation: matmul(split[0],relu(split[1]))
Tensor _reglu(
    const Tensor& input_a,
    int32_t dim,
    const std::optional<MemoryConfig>& output_mem_config) {
    TT_ASSERT(dim == -1 || dim == 3, "last dim REGLU only supported at this time ");
    if (dim == -1)
        dim = 3;
    std::vector<Tensor> ab = split_tensor_for_glu(input_a, dim, output_mem_config);
    Tensor relu_b = ttnn::relu(ab[1], output_mem_config);
    Tensor reglu_result = ttnn::multiply(ab[0], relu_b, std::nullopt, output_mem_config);
    return reglu_result;
}

// Gaussian Error Gated Linear Unit activation: matmul(split[0],gelu(split[1]))
Tensor _geglu(
    const Tensor& input_a,
    int32_t dim,
    const std::optional<MemoryConfig>& output_mem_config ) {
    TT_ASSERT(dim == -1 || dim == 3, "last dim GEGLU only supported at this time ");
    if (dim == -1)
        dim = 3;

    std::vector<Tensor> ab = split_tensor_for_glu(input_a, dim, output_mem_config);

    constexpr bool fast_appx = true;
    Tensor gelu_b = ttnn::gelu(ab[1], fast_appx, output_mem_config);
    Tensor geglu_result = ttnn::multiply(ab[0], gelu_b, std::nullopt, output_mem_config);
    return geglu_result;
}

// Swish Gated Linear Unit activation: matmul(split[0],swish(split[1]))
Tensor _swiglu(
    const Tensor& input_a,
    int32_t dim,
    const std::optional<MemoryConfig>& output_mem_config ) {
    TT_ASSERT(dim == -1 || dim == 3, "last dim SWIGLU only supported at this time ");
    if (dim == -1)
        dim = 3;

    std::vector<Tensor> ab = split_tensor_for_glu(input_a, dim, output_mem_config);

    Tensor swish_b = _swish(ab[1], output_mem_config);
    Tensor swiglu_result = ttnn::multiply(ab[0], swish_b, std::nullopt, output_mem_config);
    return swiglu_result;
}

// tril : select lower triangular region of input matrix
Tensor _tril(const Tensor& input_a, int32_t diag, const std::optional<MemoryConfig>&  output_mem_config) {
    Tensor index_l = tt::numpy::index_tril<::bfloat16>(
        input_a.get_legacy_shape(), diag, DataType::BFLOAT16, Layout::TILE, input_a.device(), output_mem_config.value());
    return ttnn::multiply(input_a, index_l, std::nullopt, output_mem_config);
}

// triu : select upper triangular region of input matrix
Tensor _triu(const Tensor& input_a, int32_t diag, const std::optional<MemoryConfig>&  output_mem_config) {
    Tensor index_u = tt::numpy::index_triu<::bfloat16>(
        input_a.get_legacy_shape(), diag, DataType::BFLOAT16, Layout::TILE, input_a.device(), output_mem_config.value());
    return ttnn::multiply(input_a, index_u, std::nullopt, output_mem_config);
}

Tensor is_odd(const Tensor& input, const std::optional<MemoryConfig>& output_mem_config) {
    Tensor result = ttnn::multiply(input, (1.0f/2.0f), std::nullopt, output_mem_config);
    Tensor floor_res = ttnn::floor(result, output_mem_config);
    return ttnn::ne(result, floor_res, std::nullopt, output_mem_config);
}

Tensor _round(const Tensor& input, int32_t decimals, const std::optional<MemoryConfig>&  output_mem_config) {
    auto arch = input.device()->arch();
    TT_FATAL(arch == tt::ARCH::WORMHOLE_B0, "Op is only supported on Wormhole");
    Tensor floor_res = ttnn::floor(input, output_mem_config);
    if (decimals != 0) {  // TODO: For decimal value!=0
        Tensor power_10 =
            ttnn::power(ttnn::full_like(input, 10.0f), decimals, output_mem_config);
        Tensor rounded_non_half = ttnn::floor(
            ttnn::add(ttnn::multiply(input, power_10, std::nullopt, output_mem_config), 0.5, std::nullopt, output_mem_config),
            output_mem_config);
        rounded_non_half = ttnn::div(rounded_non_half, power_10);
        return rounded_non_half;
    } else {  // Bankers' Rounding
        Tensor rounded_non_half = ttnn::floor(
            ttnn::add(
                input,
                ttnn::where(ttnn::logical_and(ttnn::ge(input, 0.4), ttnn::le(input, 0.5)), 0.4f, 0.5f, output_mem_config.value()),
                std::nullopt,
                output_mem_config),
            output_mem_config.value());
        Tensor fractional_part = ttnn::subtract(input, floor_res, std::nullopt, output_mem_config);
        Tensor is_half = ttnn::eq(fractional_part, 0.5, std::nullopt, output_mem_config);
        Tensor rounded_half =
            ttnn::add(floor_res, is_odd(floor_res, output_mem_config), std::nullopt, output_mem_config);
        return ttnn::where(is_half, rounded_half, rounded_non_half, output_mem_config.value());
    }
}


// polygamma support for the range of input(1, 10) and n(1, 10)
Tensor _polygamma(const Tensor& input_a, int32_t k, const std::optional<MemoryConfig>& output_mem_config) {
    float k_der = 1.0f + k;
    float fact_val = std::tgamma(k_der);
    float pos_neg = 1.0f;
    if (k == 2 || k == 4 || k == 6 || k == 8 || k == 10) {
        pos_neg = -1.0f;
    }
    Tensor temp(input_a);
    {
        Tensor z1 = ttnn::reciprocal(ttnn::power(input_a, k_der, output_mem_config), output_mem_config);
        temp = z1;
        for (int idx = 1; idx < 11; idx++) {
            z1 = ttnn::reciprocal(ttnn::power(ttnn::add(input_a, idx, std::nullopt, output_mem_config), k_der, output_mem_config), output_mem_config);
            temp = ttnn::add(temp, z1, std::nullopt, output_mem_config);
        }
    }
    fact_val *= pos_neg;
    return ttnn::multiply(temp, fact_val, std::nullopt, output_mem_config);
}

//rdiv
Tensor ExecuteRdiv::invoke(uint8_t queue_id, const Tensor& input_tensor, float value, const std::string& round_mode, const std::optional<MemoryConfig>& memory_config, std::optional<Tensor> optional_output_tensor) {
    float t_inf = std::numeric_limits<float>::infinity();
    Tensor recip_result = ttnn::reciprocal(queue_id, input_tensor, memory_config, optional_output_tensor);
    Tensor result = ttnn::multiply(queue_id, recip_result, value, std::nullopt, memory_config, optional_output_tensor);

    if(round_mode == "trunc"){
        result = ttnn::trunc(result);
     }
     else if(round_mode == "floor"){
        result = ttnn::floor(result);
     }
    return ttnn::where(ttnn::eqz(queue_id, input_tensor, memory_config), t_inf, result, memory_config, optional_output_tensor);
}

// Function: hardshrink
// Ref: https://pytorch.org/docs/stable/generated/torch.nn.Hardshrink.html
Tensor _hardshrink(const Tensor& a, float param, const std::optional<MemoryConfig>& output_mem_config) {
    TT_ASSERT(param >= 0);
    Tensor t1 = ttnn::multiply(ttnn::ltz(ttnn::add(a, param, std::nullopt, output_mem_config)), a, std::nullopt, output_mem_config);
    Tensor t2 = ttnn::multiply(ttnn::gtz(ttnn::subtract(a, param, std::nullopt, output_mem_config)), a, std::nullopt, output_mem_config);
    return ttnn::add(t1, t2, std::nullopt, output_mem_config);
}

// Function: softshrink
// Ref: https://pytorch.org/docs/stable/generated/torch.nn.Softshrink.html
Tensor _softshrink(const Tensor& a, float param, const std::optional<MemoryConfig>& output_mem_config) {
    TT_ASSERT(param >= 0);
    Tensor t_a_plus_param = ttnn::add(a, param, std::nullopt, output_mem_config);
    Tensor t1 = ttnn::multiply(ttnn::ltz(t_a_plus_param, output_mem_config), t_a_plus_param, std::nullopt, output_mem_config);
    t_a_plus_param.deallocate();
    Tensor t_a_minus_param = ttnn::subtract(a, param, std::nullopt, output_mem_config);
    Tensor t2 = ttnn::multiply(ttnn::gtz(t_a_minus_param, output_mem_config), t_a_minus_param, std::nullopt, output_mem_config);
    t_a_minus_param.deallocate();
    return ttnn::add(t1, t2, std::nullopt, output_mem_config);
}



// logit(input, eps)=log(input / 1 - input)
Tensor _logit(const Tensor& input_a, float eps, const std::optional<MemoryConfig>& output_mem_config) {
    Tensor t_eps = ttnn::full_like(input_a, eps);
    Tensor t1m_eps = ttnn::full_like(input_a, (1 - eps));
    Tensor logit_input = ttnn::where(
        ttnn::ltz(t_eps, output_mem_config),
        input_a,
        ttnn::where(
            ttnn::lt(input_a, t_eps, std::nullopt, output_mem_config),
            t_eps,
            ttnn::where(ttnn::gt(input_a, t1m_eps, std::nullopt, output_mem_config), t1m_eps, input_a)
            )
        );
    Tensor linput_m1 = ttnn::rsub(logit_input, 1.0, output_mem_config);
    Tensor log_input = ttnn::multiply(logit_input, ttnn::reciprocal(linput_m1, output_mem_config), std::nullopt, output_mem_config);
    linput_m1.deallocate();
    Tensor t_inf = ttnn::multiply(ttnn::sign(input_a, output_mem_config), input_a.device()->sfpu_inf(), std::nullopt, output_mem_config);
    Tensor logit_result = ttnn::where(
        ttnn::eq(logit_input, 1.0, std::nullopt, output_mem_config),
        t_inf,
        ttnn::where(ttnn::ltz(log_input, output_mem_config), input_a.device()->sfpu_nan(), ttnn::log(log_input, output_mem_config))
        );
    return logit_result;
}

// Celu
// torch.where(x > 0, x, alpha * (torch.exp(x / alpha) - 1))
Tensor _celu(const Tensor& input_a, float alpha, const std::optional<MemoryConfig>& output_mem_config) {
    float recip_val = 1.0f / alpha;
    using ttnn::operations::unary::UnaryWithParam;
    using ttnn::operations::unary::UnaryOpType;
    std::vector<UnaryWithParam> ops_chain = {
    UnaryWithParam{UnaryOpType::MUL_UNARY_SFPU, recip_val},
    UnaryWithParam{UnaryOpType::EXP, 1.0f},
    UnaryWithParam{UnaryOpType::SUB_UNARY_SFPU, 1.0f}, UnaryWithParam{UnaryOpType::MUL_UNARY_SFPU, alpha} };

    Tensor result = ttnn::unary_chain(input_a, ops_chain, output_mem_config);
    result = ttnn::where(ttnn::gtz(input_a, output_mem_config), input_a, result);
    return result;
}

// // tanhshrink(x) = x - tanh(x)
Tensor _logical_not_(const Tensor& x, const std::optional<MemoryConfig>& output_mem_config) {
    return ttnn::logical_not(x, output_mem_config, x);
}

// rpow: y = k**(a) = exp( a**log(k) )
Tensor _rpow(const Tensor& a, float k, const std::optional<MemoryConfig>& output_mem_config) {
    TT_ASSERT(k > 0.0, "rpow cannot be calcualted for non-positive numbers");
    float log_k = logf(k);

    Tensor result = ttnn::multiply(a, log_k);
    return ttnn::exp(result, false);
}

using HWFunctionT = std::function<Tensor(const Tensor& y, const std::optional<MemoryConfig>&)>;
Tensor _make_global_from_hw_impl(HWFunctionT fn, const Tensor& y,  const std::optional<MemoryConfig>& output_mem_config) {
    TT_FATAL(y.get_legacy_shape().rank() == 4, "Cannot support non-rank 4 Tensor");

    // format to HW
    Tensor y_hw = ttnn::reshape_on_device(
        y, 1, 1, y.get_legacy_shape()[2], y.get_legacy_shape()[3] * y.get_legacy_shape()[1] * y.get_legacy_shape()[0]);

    // compute @fn
    Tensor z_0 = fn(y_hw, output_mem_config);
    TT_FATAL(y_hw.get_legacy_shape() == z_0.get_legacy_shape(), "shape match");
    y_hw.deallocate();

    // reformat
    Tensor z_1 = ttnn::reshape_on_device(
        z_0, y.get_legacy_shape()[0], y.get_legacy_shape()[1], y.get_legacy_shape()[2], y.get_legacy_shape()[3]);
    z_0.deallocate();

    return z_1;
}

// Global Norm
Tensor _normalize_global(const Tensor& y,  const std::optional<MemoryConfig>& output_mem_config) {
    return _make_global_from_hw_impl(_normalize, y, output_mem_config);
}

Tensor _frac(const Tensor& input, const std::optional<MemoryConfig>& output_mem_config) {
    auto arch = input.device()->arch();
    TT_FATAL(arch == tt::ARCH::WORMHOLE_B0, "Op is only supported on Wormhole");
    Tensor trunc_res = ttnn::trunc(input);
    Tensor result = ttnn::subtract(input, trunc_res, std::nullopt, output_mem_config);
    return result;
}

}  // namespace ttnn::operations::unary
